# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""COTS experimental TP-style KV head-split attention helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from vllm._custom_ops import (
    cots_gqa_bf16_decode_attention,
    cots_gqa_bf16_prefill_attention,
    cots_gqa_bf16_scatter_suffix_kv,
)
from vllm.utils.cots_diag import COUNTERS_ENABLED as _COTS_COUNTERS_ENABLED
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash


def _timer_start() -> int:
    return time.perf_counter_ns() if _COTS_COUNTERS_ENABLED else 0


def _timing(name: str, start_ns: int) -> None:
    if start_ns == 0:
        return
    from vllm.model_executor.offloader import cots_ops

    cots_ops.add_python_timing(name, time.perf_counter_ns() - start_ns)


def _counter(name: str, value: int = 1) -> None:
    if not _COTS_COUNTERS_ENABLED:
        return
    from vllm.model_executor.offloader import cots_ops

    cots_ops.add_python_counter(name, int(value))


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _phase_for_tokens(num_tokens: int) -> str:
    return "decode" if int(num_tokens) <= 128 else "prefill"


@dataclass
class CotsHeadSplitKVPrefetchDescriptor:
    """Runtime block plan for 3-way head-split KV prefetch.

    ``source_block_ids`` are logical CPU KV block ids that must be copied into
    the compact prefetch workspace for this forward. Destination block ids are
    the dense ``0..num_active_blocks-1`` ids used by ``compact_block_table``.
    The descriptor is metadata-only; the streamer later owns the H2D copy.
    """

    source_key_cache: torch.Tensor
    source_value_cache: torch.Tensor
    source_block_ids: torch.Tensor
    destination_block_ids: torch.Tensor
    compact_block_table: torch.Tensor
    layer_idx: int
    num_active_blocks: int
    max_active_blocks: int
    block_size: int
    prefetch_cpu_kv_start: int
    prefetch_kv_heads: int


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
    cpu_slot_mapping: torch.Tensor
    cpu_seq_lens: torch.Tensor
    gpu_kv_heads: int
    cpu_kv_heads: int
    q_heads_per_kv: int
    cpu_query_start: int
    cpu_query_heads: int
    is_decode: bool
    num_actual_tokens: int
    layer_idx: int = -1
    prefetch_kv_heads: int = 0
    cpu_compute_kv_heads: int = 0
    prefetch_query_start: int = 0
    prefetch_query_heads: int = 0
    cpu_compute_query_start: int = 0
    cpu_compute_query_heads: int = 0
    prefetch_cpu_kv_start: int = 0
    cpu_compute_cpu_kv_start: int = 0
    query_cpu: torch.Tensor | None = None
    output_cpu: torch.Tensor | None = None
    output_lse_cpu: torch.Tensor | None = None
    prefill_query_to_seq_cpu: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    kv_prefetch: CotsHeadSplitKVPrefetchDescriptor | None = None
    routed_query_cpu: torch.Tensor | None = None
    routed_key_cpu: torch.Tensor | None = None
    routed_value_cpu: torch.Tensor | None = None
    routed_qkv_sidecar: object | None = None


def _copy_kv_head_slice(
    tensor: torch.Tensor,
    *,
    head_start: int,
    num_heads: int,
    valid_gpu: torch.Tensor | None = None,
) -> torch.Tensor:
    end = int(head_start) + int(num_heads)
    view = tensor[:, int(head_start) : end, :]
    if valid_gpu is not None:
        view = view[valid_gpu]
    return view.detach().to(device="cpu").contiguous()


def _scatter_cpu_kv_rows(
    *,
    key_cpu: torch.Tensor,
    value_cpu: torch.Tensor,
    block_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_head_start: int,
) -> None:
    num_heads = int(key_cpu.shape[1])
    if num_heads <= 0:
        return
    key_cache_view = key_cache.narrow(1, int(cache_head_start), num_heads)
    value_cache_view = value_cache.narrow(1, int(cache_head_start), num_heads)
    cots_gqa_bf16_scatter_suffix_kv(
        key_cpu,
        value_cpu,
        block_ids,
        block_offsets,
        key_cache_view,
        value_cache_view,
    )


def _cpu_compute_kv_heads(metadata: CotsHeadSplitAttentionMetadata) -> int:
    if int(metadata.cpu_compute_kv_heads) == 0 and int(metadata.prefetch_kv_heads) == 0:
        return int(metadata.cpu_kv_heads)
    return int(metadata.cpu_compute_kv_heads)


def _cpu_compute_cpu_kv_start(metadata: CotsHeadSplitAttentionMetadata) -> int:
    if (
        int(metadata.cpu_compute_cpu_kv_start) == 0
        and int(metadata.prefetch_kv_heads) == 0
    ):
        return 0
    return int(metadata.cpu_compute_cpu_kv_start)


def cots_head_split_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
) -> None:
    """Store CPU-owned trailing GQA groups in the CPU head-split cache."""

    total_start = _timer_start()
    del slot_mapping
    num_tokens = int(metadata.num_actual_tokens)
    if num_tokens <= 0:
        return
    if int(metadata.layer_idx) >= 0:
        from vllm.model_executor.offloader import get_offloader

        wait_writeback = getattr(
            get_offloader(), "wait_head_split_kv_prefetch_writeback", None
        )
        if wait_writeback is not None:
            wait_writeback(int(metadata.layer_idx))
    prepare_start = _timer_start()
    slot_cpu = metadata.cpu_slot_mapping[:num_tokens]
    if slot_cpu.device.type != "cpu":
        raise RuntimeError(
            "COTS head-split CPU KV update requires CPU slot mapping, "
            f"got {slot_cpu.device}"
        )
    valid = slot_cpu >= 0
    if not bool(valid.any().item()):
        return
    valid_all = bool(valid.all().item())
    slot_cpu = slot_cpu[valid].contiguous()
    _timing("head_split_kv_update_prepare", prepare_start)

    block_size = int(metadata.cpu_key_cache.shape[2])
    block_ids = torch.div(slot_cpu, block_size, rounding_mode="floor").contiguous()
    block_offsets = torch.remainder(slot_cpu, block_size).contiguous()

    valid_gpu = None if valid_all else valid.to(device=key.device, non_blocking=True)
    prefetch_kv_heads = int(metadata.prefetch_kv_heads)
    cpu_compute_kv_heads = _cpu_compute_kv_heads(metadata)
    prefetch_cpu_kv_start = int(metadata.prefetch_cpu_kv_start)
    cpu_compute_cpu_kv_start = _cpu_compute_cpu_kv_start(metadata)
    used_routed_kv = (
        metadata.routed_key_cpu is not None
        and metadata.routed_value_cpu is not None
        and cpu_compute_kv_heads > 0
    )

    prefetch_key_cpu: torch.Tensor | None = None
    prefetch_value_cpu: torch.Tensor | None = None
    prefetch_writeback_deferred = False
    if prefetch_kv_heads > 0:
        prefetch_writeback_deferred = (
            metadata.kv_prefetch is not None and key.device.type == "cuda"
        )
        if not prefetch_writeback_deferred:
            head_start = int(metadata.gpu_kv_heads) + prefetch_cpu_kv_start
            prefetch_key_cpu = _copy_kv_head_slice(
                key[:num_tokens],
                head_start=head_start,
                num_heads=prefetch_kv_heads,
                valid_gpu=valid_gpu,
            )
            prefetch_value_cpu = _copy_kv_head_slice(
                value[:num_tokens],
                head_start=head_start,
                num_heads=prefetch_kv_heads,
                valid_gpu=valid_gpu,
            )

    cpu_key_cpu: torch.Tensor | None = None
    cpu_value_cpu: torch.Tensor | None = None
    if cpu_compute_kv_heads > 0:
        if (
            metadata.routed_key_cpu is not None
            and metadata.routed_value_cpu is not None
        ):
            cpu_key_cpu = metadata.routed_key_cpu[:num_tokens].contiguous()
            cpu_value_cpu = metadata.routed_value_cpu[:num_tokens].contiguous()
            if int(cpu_key_cpu.shape[1]) != cpu_compute_kv_heads:
                raise RuntimeError(
                    "COTS head-split routed CPU KV has wrong head count: "
                    f"got={cpu_key_cpu.shape[1]}, expected={cpu_compute_kv_heads}"
                )
            if not valid_all:
                cpu_key_cpu = cpu_key_cpu[valid].contiguous()
                cpu_value_cpu = cpu_value_cpu[valid].contiguous()
        else:
            head_start = int(metadata.gpu_kv_heads) + cpu_compute_cpu_kv_start
            cpu_key_cpu = _copy_kv_head_slice(
                key[:num_tokens],
                head_start=head_start,
                num_heads=cpu_compute_kv_heads,
                valid_gpu=valid_gpu,
            )
            cpu_value_cpu = _copy_kv_head_slice(
                value[:num_tokens],
                head_start=head_start,
                num_heads=cpu_compute_kv_heads,
                valid_gpu=valid_gpu,
            )

    scatter_start = _timer_start()
    if prefetch_key_cpu is not None and prefetch_value_cpu is not None:
        _scatter_cpu_kv_rows(
            key_cpu=prefetch_key_cpu,
            value_cpu=prefetch_value_cpu,
            block_ids=block_ids,
            block_offsets=block_offsets,
            key_cache=metadata.cpu_key_cache,
            value_cache=metadata.cpu_value_cache,
            cache_head_start=prefetch_cpu_kv_start,
        )
    if cpu_key_cpu is not None and cpu_value_cpu is not None:
        _scatter_cpu_kv_rows(
            key_cpu=cpu_key_cpu,
            value_cpu=cpu_value_cpu,
            block_ids=block_ids,
            block_offsets=block_offsets,
            key_cache=metadata.cpu_key_cache,
            value_cache=metadata.cpu_value_cache,
            cache_head_start=cpu_compute_cpu_kv_start,
        )
    _timing("head_split_kv_scatter", scatter_start)
    if _COTS_COUNTERS_ENABLED:
        phase = _phase_for_tokens(num_tokens)
        element_bytes = _dtype_nbytes(key.dtype)
        head_dim = int(metadata.cpu_key_cache.shape[-1])
        prefetch_bytes = (
            int(slot_cpu.shape[0]) * prefetch_kv_heads * head_dim * 2 * element_bytes
        )
        prefetch_sync_bytes = 0 if prefetch_writeback_deferred else prefetch_bytes
        cpu_compute_bytes = (
            int(slot_cpu.shape[0]) * cpu_compute_kv_heads * head_dim * 2 * element_bytes
        )
        kv_bytes = prefetch_sync_bytes + cpu_compute_bytes
        d2h_bytes = prefetch_sync_bytes + (0 if used_routed_kv else cpu_compute_bytes)
        _counter("head_split_kv_update_tokens", num_tokens)
        _counter(f"head_split_kv_update_{phase}_tokens", num_tokens)
        _counter("head_split_kv_update_layers")
        _counter(f"head_split_kv_update_{phase}_layers")
        _counter("head_split_kv_update_cpu_kv_heads", metadata.cpu_kv_heads)
        _counter("head_split_kv_update_d2h_bytes", d2h_bytes)
        _counter(f"head_split_kv_update_{phase}_d2h_bytes", d2h_bytes)
        _counter("head_split_kv_update_scatter_bytes", kv_bytes)
        _counter(f"head_split_kv_update_{phase}_scatter_bytes", kv_bytes)
        _counter("head_split_kv_update_valid_rows", int(slot_cpu.shape[0]))
        _counter(f"head_split_kv_update_{phase}_valid_rows", int(slot_cpu.shape[0]))
        _timing(f"head_split_kv_update_prepare_{phase}", prepare_start)
        _timing(f"head_split_kv_scatter_{phase}", scatter_start)
        _timing(f"head_split_kv_update_total_{phase}", total_start)
    _timing("head_split_kv_update_total", total_start)


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


def _cpu_compute_geometry(
    metadata: CotsHeadSplitAttentionMetadata,
) -> tuple[int, int, int, int]:
    """Return (kv_start, kv_heads, query_start, query_heads) for CPU attention."""

    kv_heads = int(metadata.cpu_compute_kv_heads)
    if kv_heads == 0 and int(metadata.prefetch_kv_heads) == 0:
        kv_heads = int(metadata.cpu_kv_heads)
    query_heads = int(metadata.cpu_compute_query_heads)
    if query_heads == 0 and int(metadata.prefetch_query_heads) == 0:
        query_heads = int(metadata.cpu_query_heads)
    kv_start = int(metadata.cpu_compute_cpu_kv_start)
    query_start = int(metadata.cpu_compute_query_start)
    if query_start == 0 and int(metadata.prefetch_query_heads) == 0:
        query_start = int(metadata.cpu_query_start)
    return kv_start, kv_heads, query_start, query_heads


def _prepare_cpu_query(
    *,
    query: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
    num_tokens: int,
    q_start: int,
    q_end: int,
    query_heads: int,
) -> tuple[torch.Tensor, bool]:
    """Return compact CPU query rows for the CPU-compute KV groups."""

    if metadata.routed_query_cpu is not None:
        query_cpu = metadata.routed_query_cpu[:num_tokens, :query_heads, :].contiguous()
        if int(query_cpu.shape[1]) != int(query_heads):
            raise RuntimeError(
                "COTS head-split routed CPU query has wrong head count: "
                f"got={int(query_cpu.shape[1])}, expected={int(query_heads)}"
            )
        return query_cpu, True

    query_view = query[:num_tokens, q_start:q_end, :]
    query_cpu = metadata.query_cpu
    if query_cpu is None or tuple(query_cpu.shape) != tuple(query_view.shape):
        query_cpu = query_view.detach().to(device="cpu").contiguous()
    else:
        query_cpu.copy_(query_view.detach(), non_blocking=False)
    return query_cpu, False


def _register_or_copy_cpu_attention_output(
    *,
    output: torch.Tensor,
    metadata: CotsHeadSplitAttentionMetadata,
    output_cpu: torch.Tensor,
    num_tokens: int,
    q_start: int,
    q_end: int,
) -> None:
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
        return

    if output_cpu.numel() == 0:
        return
    if output_cpu.is_pinned() and output.is_cuda:
        output_view = get_accelerator_view_from_cpu_tensor(output_cpu)
    else:
        output_view = output_cpu.to(device=output.device, non_blocking=True)
    output[:num_tokens, q_start:q_end, :].copy_(output_view)


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

    total_start = _timer_start()
    kv_start, kv_heads, q_start, query_heads = _cpu_compute_geometry(metadata)
    q_end = q_start + query_heads
    if kv_heads <= 0 or query_heads <= 0:
        output_cpu = torch.empty(
            (num_tokens, 0, query.shape[-1]),
            dtype=query.dtype,
            device="cpu",
        )
        _register_or_copy_cpu_attention_output(
            output=output,
            metadata=metadata,
            output_cpu=output_cpu,
            num_tokens=num_tokens,
            q_start=q_start,
            q_end=q_end,
        )
        return

    query_start = _timer_start()
    query_cpu, used_routed_query = _prepare_cpu_query(
        query=query,
        metadata=metadata,
        num_tokens=num_tokens,
        q_start=q_start,
        q_end=q_end,
        query_heads=query_heads,
    )
    _timing("head_split_decode_query_prepare", query_start)

    output_cpu = metadata.output_cpu
    if output_cpu is None or tuple(output_cpu.shape) != tuple(query_cpu.shape):
        output_cpu = torch.empty_like(query_cpu, device="cpu")
    output_lse_cpu = metadata.output_lse_cpu
    expected_lse_shape = (query_heads, num_tokens)
    if (
        output_lse_cpu is None
        or tuple(output_lse_cpu.shape) != expected_lse_shape
        or output_lse_cpu.dtype != torch.float32
    ):
        output_lse_cpu = torch.empty(
            expected_lse_shape, dtype=torch.float32, device="cpu"
        )

    attention_start = _timer_start()
    key_cache = metadata.cpu_key_cache.narrow(1, kv_start, kv_heads)
    value_cache = metadata.cpu_value_cache.narrow(1, kv_start, kv_heads)
    cots_gqa_bf16_decode_attention(
        query_cpu,
        key_cache,
        value_cache,
        metadata.cpu_block_table,
        metadata.cpu_seq_lens,
        float(softmax_scale),
        output_cpu,
        output_lse_cpu,
    )
    _timing("head_split_decode_attention_kernel", attention_start)

    output_start = _timer_start()
    _register_or_copy_cpu_attention_output(
        output=output,
        metadata=metadata,
        output_cpu=output_cpu,
        num_tokens=num_tokens,
        q_start=q_start,
        q_end=q_end,
    )
    _timing("head_split_decode_output_route", output_start)
    if _COTS_COUNTERS_ENABLED:
        element_bytes = _dtype_nbytes(query_cpu.dtype)
        query_bytes = int(query_cpu.numel()) * element_bytes
        _counter("head_split_decode_attention_tokens", num_tokens)
        _counter("head_split_decode_attention_layers")
        _counter("head_split_decode_attention_cpu_query_heads", query_heads)
        _counter(
            "head_split_decode_query_d2h_bytes",
            0 if used_routed_query else query_bytes,
        )
        _counter(
            "head_split_decode_output_h2d_bytes",
            0 if metadata.routed_qkv_sidecar is not None else query_bytes,
        )
    _timing("head_split_decode_attention_total", total_start)


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

    total_start = _timer_start()
    kv_start, kv_heads, q_start, query_heads = _cpu_compute_geometry(metadata)
    q_end = q_start + query_heads
    if kv_heads <= 0 or query_heads <= 0:
        output_cpu = torch.empty(
            (num_tokens, 0, query.shape[-1]),
            dtype=query.dtype,
            device="cpu",
        )
        _register_or_copy_cpu_attention_output(
            output=output,
            metadata=metadata,
            output_cpu=output_cpu,
            num_tokens=num_tokens,
            q_start=q_start,
            q_end=q_end,
        )
        return

    query_start = _timer_start()
    query_cpu, used_routed_query = _prepare_cpu_query(
        query=query,
        metadata=metadata,
        num_tokens=num_tokens,
        q_start=q_start,
        q_end=q_end,
        query_heads=query_heads,
    )
    _timing("head_split_prefill_query_prepare", query_start)
    output_cpu = torch.empty_like(query_cpu, device="cpu")
    output_lse_cpu = torch.empty(
        (query_heads, num_tokens),
        dtype=torch.float32,
        device="cpu",
    )

    attention_start = _timer_start()
    key_cache = metadata.cpu_key_cache.narrow(1, kv_start, kv_heads)
    value_cache = metadata.cpu_value_cache.narrow(1, kv_start, kv_heads)
    cots_gqa_bf16_prefill_attention(
        query_cpu,
        key_cache,
        value_cache,
        metadata.cpu_block_table,
        query_to_seq,
        seq_lens,
        float(softmax_scale),
        output_cpu,
        output_lse_cpu,
    )
    _timing("head_split_prefill_attention_kernel", attention_start)

    output_start = _timer_start()
    _register_or_copy_cpu_attention_output(
        output=output,
        metadata=metadata,
        output_cpu=output_cpu,
        num_tokens=num_tokens,
        q_start=q_start,
        q_end=q_end,
    )
    _timing("head_split_prefill_output_route", output_start)
    if _COTS_COUNTERS_ENABLED:
        element_bytes = _dtype_nbytes(query_cpu.dtype)
        query_bytes = int(query_cpu.numel()) * element_bytes
        _counter("head_split_prefill_attention_tokens", num_tokens)
        _counter("head_split_prefill_attention_layers")
        _counter(
            "head_split_prefill_attention_cpu_query_heads",
            query_heads,
        )
        _counter(
            "head_split_prefill_query_d2h_bytes",
            0 if used_routed_query else query_bytes,
        )
        _counter(
            "head_split_prefill_output_h2d_bytes",
            0 if metadata.routed_qkv_sidecar is not None else query_bytes,
        )
    _timing("head_split_prefill_attention_total", total_start)
