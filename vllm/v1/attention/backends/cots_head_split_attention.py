# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""COTS experimental TP-style KV head-split attention helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm._custom_ops import (
    cots_gqa_bf16_decode_attention,
    cots_gqa_bf16_prefill_attention,
    cots_gqa_bf16_scatter_suffix_kv,
)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash


@dataclass
class CotsHeadSplitAttentionMetadata:
    """CPU-owned GQA group metadata for one attention layer call.

    The implementation uses vLLM's normal logical block ids for both devices.
    GPU KV pages store only the leading GPU-owned groups; CPU KV pages store
    the trailing CPU-owned groups.
    """

    cpu_key_cache: torch.Tensor
    cpu_value_cache: torch.Tensor
    cpu_block_table: torch.Tensor
    cpu_seq_lens: torch.Tensor
    gpu_kv_heads: int
    cpu_kv_heads: int
    q_heads_per_kv: int
    cpu_query_start: int
    cpu_query_heads: int
    is_decode: bool
    num_actual_tokens: int
    query_cpu: torch.Tensor | None = None
    output_cpu: torch.Tensor | None = None
    output_lse_cpu: torch.Tensor | None = None
    prefill_query_to_seq_cpu: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    routed_query_cpu: torch.Tensor | None = None
    routed_key_cpu: torch.Tensor | None = None
    routed_value_cpu: torch.Tensor | None = None
    routed_qkv_sidecar: object | None = None


def _copy_cpu_head_slice(
    tensor: torch.Tensor, metadata: CotsHeadSplitAttentionMetadata
) -> torch.Tensor:
    start = metadata.gpu_kv_heads
    end = start + metadata.cpu_kv_heads
    return tensor[:, start:end, :].detach().to(device="cpu")


def cots_head_split_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
) -> None:
    """Store CPU-owned trailing GQA groups in the CPU head-split cache."""

    num_tokens = int(slot_mapping.shape[0])
    if num_tokens <= 0:
        return
    if metadata.routed_key_cpu is not None and metadata.routed_value_cpu is not None:
        key_cpu = metadata.routed_key_cpu[:num_tokens].contiguous()
        value_cpu = metadata.routed_value_cpu[:num_tokens].contiguous()
    else:
        key_cpu = _copy_cpu_head_slice(key[:num_tokens], metadata).contiguous()
        value_cpu = _copy_cpu_head_slice(value[:num_tokens], metadata).contiguous()
    slot_cpu = slot_mapping[:num_tokens].detach().to(device="cpu", dtype=torch.long)
    valid = slot_cpu >= 0
    if not bool(valid.any().item()):
        return
    if not bool(valid.all().item()):
        key_cpu = key_cpu[valid].contiguous()
        value_cpu = value_cpu[valid].contiguous()
        slot_cpu = slot_cpu[valid].contiguous()

    block_size = int(metadata.cpu_key_cache.shape[2])
    block_ids = torch.div(slot_cpu, block_size, rounding_mode="floor").contiguous()
    block_offsets = torch.remainder(slot_cpu, block_size).contiguous()
    cots_gqa_bf16_scatter_suffix_kv(
        key_cpu,
        value_cpu,
        block_ids,
        block_offsets,
        metadata.cpu_key_cache,
        metadata.cpu_value_cache,
    )


def cots_head_split_gpu_kv_cache_update(
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
) -> None:
    """Store only GPU-owned leading GQA groups in the GPU KV cache."""

    key_cache, value_cache = kv_cache.unbind(0)
    gpu_kv_heads = int(metadata.gpu_kv_heads)
    if key_cache.shape[-2] != gpu_kv_heads:
        raise RuntimeError(
            "COTS head-split GPU KV cache has unexpected head geometry: "
            f"cache_heads={key_cache.shape[-2]}, expected={gpu_kv_heads}"
        )
    reshape_and_cache_flash(
        key[:, :gpu_kv_heads, :].contiguous(),
        value[:, :gpu_kv_heads, :].contiguous(),
        key_cache,
        value_cache,
        slot_mapping,
        layer.impl.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )


def cots_head_split_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
    *,
    softmax_scale: float,
) -> None:
    """Run CPU decode attention for the CPU-owned GQA groups."""

    if not metadata.is_decode:
        return

    num_tokens = int(metadata.cpu_seq_lens.shape[0])
    if num_tokens <= 0:
        return

    q_start = metadata.cpu_query_start
    q_end = q_start + metadata.cpu_query_heads
    if metadata.routed_query_cpu is not None:
        query_cpu = metadata.routed_query_cpu[:num_tokens].contiguous()
    else:
        query_view = query[:num_tokens, q_start:q_end, :]
        query_cpu = metadata.query_cpu
        if query_cpu is None or tuple(query_cpu.shape) != tuple(query_view.shape):
            query_cpu = query_view.detach().to(device="cpu").contiguous()
        else:
            query_cpu.copy_(query_view.detach(), non_blocking=False)

    output_cpu = metadata.output_cpu
    if output_cpu is None or tuple(output_cpu.shape) != tuple(query_cpu.shape):
        output_cpu = torch.empty_like(query_cpu, device="cpu")
    output_lse_cpu = metadata.output_lse_cpu
    expected_lse_shape = (metadata.cpu_query_heads, num_tokens)
    if (
        output_lse_cpu is None
        or tuple(output_lse_cpu.shape) != expected_lse_shape
        or output_lse_cpu.dtype != torch.float32
    ):
        output_lse_cpu = torch.empty(
            expected_lse_shape, dtype=torch.float32, device="cpu"
        )

    cots_gqa_bf16_decode_attention(
        query_cpu,
        metadata.cpu_key_cache,
        metadata.cpu_value_cache,
        metadata.cpu_block_table,
        metadata.cpu_seq_lens,
        float(softmax_scale),
        output_cpu,
        output_lse_cpu,
    )

    if metadata.routed_qkv_sidecar is not None:
        from vllm.model_executor.offloader import get_offloader

        register_output = getattr(
            get_offloader(), "register_head_split_attention_output", None
        )
        if register_output is None:
            raise RuntimeError("COTS head-split routed output has no offloader")
        register_output(
            output=output,
            qkv_sidecar=metadata.routed_qkv_sidecar,
            output_cpu=output_cpu,
        )
    else:
        if output_cpu.is_pinned() and output.is_cuda:
            output_view = get_accelerator_view_from_cpu_tensor(output_cpu)
        else:
            output_view = output_cpu.to(device=output.device, non_blocking=True)
        output[:num_tokens, q_start:q_end, :].copy_(output_view)


def cots_head_split_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
    *,
    softmax_scale: float,
) -> None:
    """Run CPU prefill attention for the CPU-owned GQA groups."""

    if metadata.is_decode:
        return
    num_tokens = int(metadata.num_actual_tokens)
    if num_tokens <= 0:
        return
    query_to_seq = metadata.prefill_query_to_seq_cpu
    seq_lens = metadata.prefill_seq_lens_cpu
    if query_to_seq is None or seq_lens is None:
        raise RuntimeError("COTS head-split prefill metadata is missing")

    q_start = metadata.cpu_query_start
    q_end = q_start + metadata.cpu_query_heads
    if metadata.routed_query_cpu is not None:
        query_cpu = metadata.routed_query_cpu[:num_tokens].contiguous()
    else:
        query_view = query[:num_tokens, q_start:q_end, :]
        query_cpu = query_view.detach().to(device="cpu").contiguous()
    output_cpu = torch.empty_like(query_cpu, device="cpu")
    output_lse_cpu = torch.empty(
        (metadata.cpu_query_heads, num_tokens),
        dtype=torch.float32,
        device="cpu",
    )

    cots_gqa_bf16_prefill_attention(
        query_cpu,
        metadata.cpu_key_cache,
        metadata.cpu_value_cache,
        metadata.cpu_block_table,
        query_to_seq,
        seq_lens,
        float(softmax_scale),
        output_cpu,
        output_lse_cpu,
    )

    if metadata.routed_qkv_sidecar is not None:
        from vllm.model_executor.offloader import get_offloader

        register_output = getattr(
            get_offloader(), "register_head_split_attention_output", None
        )
        if register_output is None:
            raise RuntimeError("COTS head-split routed output has no offloader")
        register_output(
            output=output,
            qkv_sidecar=metadata.routed_qkv_sidecar,
            output_cpu=output_cpu,
        )
    else:
        if output_cpu.is_pinned() and output.is_cuda:
            output_view = get_accelerator_view_from_cpu_tensor(output_cpu)
        else:
            output_view = output_cpu.to(device=output.device, non_blocking=True)
        output[:num_tokens, q_start:q_end, :].copy_(output_view)
