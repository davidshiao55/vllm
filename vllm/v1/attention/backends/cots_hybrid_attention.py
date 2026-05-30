# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""COTS Phase 2 hybrid CPU/GPU decode attention.

This module owns the routing primitive for the validated Phase 2 path:

    GPU prefix paged attention + COTS CPU suffix attention + online-softmax merge

The worker populates CPU suffix KV and suffix-local metadata before this path
runs. Keeping the routing primitive separate makes the runtime boundary explicit
and testable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm._custom_ops import cots_gqa_bf16_scatter_suffix_kv
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.merge_attn_states import (
    merge_attn_states,
    merge_attn_states_indexed,
)
from vllm.v1.metrics.stats import CotsHybridKVStats


def _cuda_event_timing_enabled() -> bool:
    return (
        os.environ.get("VLLM_COTS_DIAG") == "1"
        or os.environ.get("VLLM_COTS_HYBRID_CUDA_TIMING") == "1"
    )


def _make_cuda_timing_events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


@dataclass
class CotsHybridDecodeMetadata:
    """CPU suffix metadata for one eager decode attention call.

    Attributes:
        cpu_key_cache: CPU suffix K cache in COTS layout
            ``[num_cpu_blocks, num_kv_heads, block_size, head_dim]``.
        cpu_value_cache: CPU suffix V cache with the same layout.
        cpu_block_table: Suffix-local CPU block table, int32 CPU tensor
            ``[num_reqs, max_suffix_blocks]``.
        cpu_seq_lens: Suffix-local sequence lengths, int32 CPU tensor
            ``[num_reqs]``.
        split_blocks: Static block-aligned split point. GPU prefix attention
            consumes ``gpu_block_table[:, :split_blocks]``.
    """

    cpu_key_cache: torch.Tensor
    cpu_value_cache: torch.Tensor
    cpu_block_table: torch.Tensor
    cpu_seq_lens: torch.Tensor
    split_blocks: int
    metrics: CotsHybridKVStats | None = None
    staged_query_cpu: torch.Tensor | None = None
    staged_qkv_cpu: torch.Tensor | None = None
    query_copy_stream: torch.cuda.Stream | None = None
    query_ready_event: torch.cuda.Event | None = None
    staged_query_valid: bool = False
    staged_qkv_valid: bool = False
    staged_key_cpu: torch.Tensor | None = None
    staged_value_cpu: torch.Tensor | None = None
    kv_copy_stream: torch.cuda.Stream | None = None
    kv_ready_event: torch.cuda.Event | None = None
    staged_kv_valid: bool = False
    suffix_out_cpu: torch.Tensor | None = None
    suffix_lse_cpu: torch.Tensor | None = None
    suffix_submit_stream: torch.cuda.Stream | None = None
    suffix_submit_done_event: torch.cuda.Event | None = None
    suffix_read_done_event: torch.cuda.Event | None = None
    staging_done_event: torch.cuda.Event | None = None
    scatter_req_indices: torch.Tensor | None = None
    scatter_block_ids: torch.Tensor | None = None
    scatter_block_offsets: torch.Tensor | None = None
    req_indices_gpu: torch.Tensor | None = None
    scatter_source_indices: torch.Tensor | None = None
    scatter_source_indices_gpu: torch.Tensor | None = None
    prefix_source_indices: torch.Tensor | None = None
    prefix_req_indices_gpu: torch.Tensor | None = None
    prefix_seq_lens_cpu: torch.Tensor | None = None
    suffix_attention_runner: CotsSuffixAttentionRunner | None = None
    suffix_attention_task_id: int = 0
    suffix_attention_snapshot_inputs: bool = True


class CotsSuffixAttentionRunner(Protocol):
    """Execution boundary for Phase 2 CPU suffix attention.

    The production runner uses stream-ordered submit/sync custom ops backed by
    cudaLaunchHostFunc and the same suffix-local CPU cache metadata in eager and
    graph modes.
    """

    kind: str

    def set_runtime_counts(self, num_tokens: int, scatter_count: int) -> None: ...

    def get_counters(self) -> dict[str, int]: ...

    def reset_counters(self) -> None: ...

    def run_gqa_bf16_suffix_attention(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        output_lse: torch.Tensor,
        cuda_anchor: torch.Tensor | None = None,
        task_id: int = 0,
        scatter_block_ids: torch.Tensor | None = None,
        scatter_block_offsets: torch.Tensor | None = None,
        scatter_key_cpu: torch.Tensor | None = None,
        scatter_value_cpu: torch.Tensor | None = None,
        scatter_from_qkv: bool = False,
        scatter_from_separate_kv: bool = False,
        snapshot_inputs: bool = True,
        submit_stream: torch.cuda.Stream | None = None,
        submit_ready_event: torch.cuda.Event | None = None,
        submit_done_event: torch.cuda.Event | None = None,
        sync_after_submit: bool = True,
    ) -> None: ...

    def sync_gqa_bf16_suffix_attention(
        self,
        *,
        cuda_anchor: torch.Tensor,
        task_id: int = 0,
    ) -> None: ...


@dataclass
class CotsHybridSuffixState:
    suffix_runner: CotsSuffixAttentionRunner
    suffix_out_cpu: torch.Tensor
    suffix_lse_cpu: torch.Tensor
    cpu_suffix_start: float
    cuda_anchor: torch.Tensor
    task_id: int
    sync_required: bool


def can_submit_cots_hybrid_suffix_early(
    hybrid_metadata: CotsHybridDecodeMetadata,
) -> bool:
    # Internal all-suffix overlap path. The coalesced-prefix mixed-row path
    # passes precomputed prefix output and intentionally does not use this.
    return (
        hybrid_metadata.suffix_attention_runner is not None
        and hybrid_metadata.suffix_attention_runner.kind == "native_prepared"
        and hybrid_metadata.staged_query_valid
        and hybrid_metadata.suffix_submit_stream is not None
        and hybrid_metadata.suffix_submit_done_event is not None
        and not torch.cuda.is_current_stream_capturing()
    )


def _source_indices_for_device(
    hybrid_metadata: CotsHybridDecodeMetadata,
    device: torch.device,
) -> torch.Tensor | None:
    source_indices = hybrid_metadata.scatter_source_indices
    if source_indices is None:
        return None
    cached = hybrid_metadata.scatter_source_indices_gpu
    if cached is not None and cached.device == device:
        return cached
    return source_indices.to(device=device, non_blocking=True)


class CotsPreparedNativeSuffixAttentionRunner:
    """Prepared host-callback runner for CPU suffix attention.

    Python owns the stable CPU query/KV metadata; the CPU suffix attention task
    itself is submitted through stream-ordered custom ops backed by
    ``cudaLaunchHostFunc``.
    """

    kind = "native_prepared"

    def __init__(self, *, num_tasks: int) -> None:
        try:
            from vllm import _cots_C
        except ImportError as e:
            raise RuntimeError(
                "CotsPreparedNativeSuffixAttentionRunner requires the "
                "`vllm._cots_C` extension. Rebuild vLLM before enabling "
                "COTS hybrid KV."
            ) from e
        from vllm.v1.attention.backends import cots_suffix_attention_ops

        self._ops = cots_suffix_attention_ops
        self._runner_id = cots_suffix_attention_ops.register_suffix_infer(
            _cots_C.CotsSuffixAttentionInfer()
        )
        self._num_tasks = int(num_tasks)
        cots_suffix_attention_ops.install_suffix_infer(
            self._runner_id, n_tasks=self._num_tasks
        )
        cots_suffix_attention_ops.install_suffix_wait_kernel_sync(
            self._runner_id, n_tasks=self._num_tasks
        )
        self._owns_registry_entry = True

    def wait_kernel_sync_installed(self, task_id: int = 0) -> bool:
        return self._ops.suffix_wait_kernel_sync_installed(
            self._runner_id, int(task_id)
        )

    def wait_kernel_slots(self, task_id: int = 0) -> tuple[int, int]:
        return self._ops.suffix_wait_kernel_slots(self._runner_id, int(task_id))

    def get_counters(self) -> dict[str, int]:
        return self._ops.get_suffix_counters(self._runner_id)

    def reset_counters(self) -> None:
        self._ops.reset_suffix_counters(self._runner_id)

    def set_runtime_counts(self, num_tokens: int, scatter_count: int) -> None:
        self._ops.set_suffix_runtime_counts(
            self._runner_id, int(num_tokens), int(scatter_count)
        )

    def run_gqa_bf16_suffix_attention(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        output_lse: torch.Tensor,
        cuda_anchor: torch.Tensor | None = None,
        task_id: int = 0,
        scatter_block_ids: torch.Tensor | None = None,
        scatter_block_offsets: torch.Tensor | None = None,
        scatter_key_cpu: torch.Tensor | None = None,
        scatter_value_cpu: torch.Tensor | None = None,
        scatter_from_qkv: bool = False,
        scatter_from_separate_kv: bool = False,
        snapshot_inputs: bool = True,
        submit_stream: torch.cuda.Stream | None = None,
        submit_ready_event: torch.cuda.Event | None = None,
        submit_done_event: torch.cuda.Event | None = None,
        sync_after_submit: bool = True,
    ) -> None:
        if cuda_anchor is None:
            raise RuntimeError(
                "native_prepared suffix attention requires a CUDA anchor"
            )
        if query.device.type != "cpu":
            raise RuntimeError("prepared suffix query must be a CPU tensor")
        if query.dtype != torch.bfloat16:
            raise RuntimeError(f"prepared suffix query must be BF16, got {query.dtype}")
        if query.dim() != 3:
            raise RuntimeError(
                "prepared suffix query must have shape [B, H, D], got "
                f"{tuple(query.shape)}"
            )
        num_q_heads = int(query.shape[1])
        head_dim = int(query.shape[2])
        if num_q_heads <= 0:
            raise RuntimeError("prepared suffix query must have at least one head")
        if head_dim != 128:
            raise RuntimeError(
                "prepared suffix attention currently supports head_dim=128, "
                f"got {head_dim}"
            )
        if query.stride(-1) != 1:
            raise RuntimeError("prepared suffix query head_dim stride must be 1")
        if key_cache.device.type != "cpu" or value_cache.device.type != "cpu":
            raise RuntimeError("prepared suffix KV caches must be CPU tensors")
        if key_cache.shape != value_cache.shape:
            raise RuntimeError("prepared suffix key/value cache shapes must match")
        if key_cache.dim() != 4:
            raise RuntimeError(
                "prepared suffix KV cache must have shape "
                "[blocks, num_kv_heads, block_size, 128], got "
                f"{tuple(key_cache.shape)}"
            )
        num_kv_heads = int(key_cache.shape[1])
        cache_head_dim = int(key_cache.shape[3])
        if num_kv_heads <= 0:
            raise RuntimeError("prepared suffix KV cache must have at least one head")
        if cache_head_dim != head_dim:
            raise RuntimeError(
                "prepared suffix KV cache head_dim must match query, got "
                f"{cache_head_dim} vs {head_dim}"
            )
        if num_q_heads % num_kv_heads != 0:
            raise RuntimeError(
                "prepared suffix attention requires num_q_heads divisible by "
                f"num_kv_heads, got {num_q_heads} and {num_kv_heads}"
            )
        heads_per_kv = num_q_heads // num_kv_heads
        if heads_per_kv > 8:
            raise RuntimeError(
                "prepared suffix attention supports at most 8 query heads per "
                f"KV head, got {heads_per_kv}"
            )
        if output.shape != query.shape or output.dtype != torch.bfloat16:
            raise RuntimeError("prepared suffix output must match BF16 query shape")
        if (
            output_lse.shape != (num_q_heads, query.shape[0])
            or output_lse.dtype != torch.float32
        ):
            raise RuntimeError(
                "prepared suffix output_lse must have shape [num_q_heads, B] "
                "and FP32 dtype"
            )
        if not block_table.is_contiguous() or not seq_lens.is_contiguous():
            raise RuntimeError(
                "prepared suffix block_table/seq_lens must be contiguous"
            )
        if (
            block_table.shape[0] != query.shape[0]
            or seq_lens.shape[0] != query.shape[0]
        ):
            raise RuntimeError("prepared suffix metadata batch must match query batch")

        if scatter_from_qkv and scatter_from_separate_kv:
            raise RuntimeError(
                "prepared suffix scatter source must be QKV or separate K/V, not both"
            )

        scatter_count = 0
        scatter_block_ids_ptr = 0
        scatter_block_offsets_ptr = 0
        scatter_key_ptr = 0
        scatter_value_ptr = 0
        if scatter_from_qkv or scatter_from_separate_kv:
            if scatter_block_ids is None or scatter_block_offsets is None:
                raise RuntimeError("prepared suffix scatter requires block ids/offsets")
            if (
                scatter_block_ids.device.type != "cpu"
                or scatter_block_offsets.device.type != "cpu"
            ):
                raise RuntimeError("prepared suffix scatter metadata must be CPU")
            if (
                scatter_block_ids.dtype != torch.long
                or scatter_block_offsets.dtype != torch.long
            ):
                raise RuntimeError("prepared suffix scatter metadata must be int64")
            if (
                not scatter_block_ids.is_contiguous()
                or not scatter_block_offsets.is_contiguous()
            ):
                raise RuntimeError(
                    "prepared suffix scatter metadata must be contiguous"
                )
            if scatter_block_ids.numel() != scatter_block_offsets.numel():
                raise RuntimeError("prepared suffix scatter metadata lengths differ")
            scatter_count = int(scatter_block_ids.numel())
            scatter_block_ids_ptr = int(scatter_block_ids.data_ptr())
            scatter_block_offsets_ptr = int(scatter_block_offsets.data_ptr())

        if scatter_from_qkv:
            combined_heads = num_q_heads + 2 * num_kv_heads
            if (
                query.stride(1) != head_dim
                or query.stride(0) < combined_heads * head_dim
            ):
                raise RuntimeError(
                    "prepared suffix QKV scatter requires a [B, Q+2KV, D] "
                    "staging layout with Q as the leading view"
                )
        elif scatter_from_separate_kv:
            if scatter_key_cpu is None or scatter_value_cpu is None:
                raise RuntimeError(
                    "prepared suffix separate K/V scatter requires key/value buffers"
                )
            if (
                scatter_key_cpu.device.type != "cpu"
                or scatter_value_cpu.device.type != "cpu"
            ):
                raise RuntimeError(
                    "prepared suffix separate K/V scatter buffers must be CPU"
                )
            if (
                scatter_key_cpu.dtype != torch.bfloat16
                or scatter_value_cpu.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "prepared suffix separate K/V scatter buffers must be BF16"
                )
            expected_kv_shape = (query.shape[0], num_kv_heads, head_dim)
            if (
                scatter_key_cpu.shape != expected_kv_shape
                or scatter_value_cpu.shape != expected_kv_shape
            ):
                raise RuntimeError(
                    "prepared suffix separate K/V scatter buffers must have shape "
                    f"[B, {num_kv_heads}, {head_dim}], got "
                    f"{tuple(scatter_key_cpu.shape)} and "
                    f"{tuple(scatter_value_cpu.shape)}"
                )
            if (
                not scatter_key_cpu.is_contiguous()
                or not scatter_value_cpu.is_contiguous()
            ):
                raise RuntimeError(
                    "prepared suffix separate K/V scatter buffers must be contiguous"
                )
            scatter_key_ptr = int(scatter_key_cpu.data_ptr())
            scatter_value_ptr = int(scatter_value_cpu.data_ptr())

        infer = self._ops.lookup_suffix_infer(
            self._runner_id, "cots_prepared_suffix_attention"
        )
        infer.populate_task(
            int(task_id),
            int(query.data_ptr()),
            int(query.shape[0]),
            num_q_heads,
            num_kv_heads,
            head_dim,
            int(query.stride(0)),
            int(query.stride(1)),
            int(query.stride(2)),
            int(key_cache.data_ptr()),
            int(key_cache.shape[0]),
            int(key_cache.shape[2]),
            int(value_cache.data_ptr()),
            int(block_table.data_ptr()),
            int(block_table.shape[1]),
            int(seq_lens.data_ptr()),
            int(output.data_ptr()),
            int(output_lse.data_ptr()),
            scatter_block_ids_ptr,
            scatter_block_offsets_ptr,
            scatter_key_ptr,
            scatter_value_ptr,
            scatter_count,
            bool(scatter_from_qkv),
            bool(scatter_from_separate_kv),
            bool(snapshot_inputs),
            float(scale),
        )
        if submit_stream is not None:
            if submit_done_event is None:
                raise RuntimeError("async suffix submit requires a submit_done_event")
            if submit_ready_event is not None:
                submit_stream.wait_event(submit_ready_event)
            with torch.cuda.stream(submit_stream):
                torch.ops.vllm.cots_suffix_attn_submit(
                    cuda_anchor, self._runner_id, int(task_id)
                )
                submit_done_event.record(submit_stream)
            if sync_after_submit:
                torch.cuda.current_stream(cuda_anchor.device).wait_event(
                    submit_done_event
                )
        else:
            torch.ops.vllm.cots_suffix_attn_submit(
                cuda_anchor, self._runner_id, int(task_id)
            )
        if sync_after_submit:
            self.sync_gqa_bf16_suffix_attention(
                cuda_anchor=cuda_anchor,
                task_id=task_id,
            )

    def sync_gqa_bf16_suffix_attention(
        self,
        *,
        cuda_anchor: torch.Tensor,
        task_id: int = 0,
    ) -> None:
        torch.ops.vllm.cots_suffix_attn_sync(cuda_anchor, self._runner_id, int(task_id))

    def close(self) -> None:
        if not getattr(self, "_owns_registry_entry", False):
            return
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.current_stream().synchronize()
            self._ops.sync_suffix_infer_blocking(self._runner_id)
        finally:
            self._ops.unregister_suffix_infer(self._runner_id)
            self._owns_registry_entry = False

    def __del__(self) -> None:
        try:
            if getattr(self, "_owns_registry_entry", False):
                self._ops.unregister_suffix_infer(self._runner_id)
                self._owns_registry_entry = False
        except Exception:
            pass


def cots_hybrid_stage_query(
    query: torch.Tensor,
    hybrid_metadata: CotsHybridDecodeMetadata,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
) -> None:
    """Stage decode Q, and when needed K/V, to pinned CPU memory.

    The CPU suffix kernel needs Q for attention and the current token's K/V for
    the CPU suffix cache. We first try the compact combined-QKV copy. If the
    compiled model materialized Q, K, and V as separate tensors, we stage K/V
    here as separate D2H copies so CUDA graph capture records them outside the
    opaque ``unified_kv_cache_update`` custom op.
    """

    staged = hybrid_metadata.staged_query_cpu
    stream = hybrid_metadata.query_copy_stream
    event = hybrid_metadata.query_ready_event
    hybrid_metadata.staged_query_valid = False
    hybrid_metadata.staged_qkv_valid = False
    if staged is None or stream is None or event is None:
        return
    if not query.is_cuda or not staged.is_pinned():
        return

    source_indices = _source_indices_for_device(hybrid_metadata, query.device)
    if source_indices is not None:
        query = query.index_select(0, source_indices)
        if key is not None:
            key = key.index_select(0, source_indices)
        if value is not None:
            value = value.index_select(0, source_indices)

    batch = hybrid_metadata.cpu_seq_lens.shape[0]
    if query.shape[0] < batch:
        raise ValueError(
            "COTS hybrid Q staging expected at least one Q row per request: "
            f"query={tuple(query.shape)}, batch={batch}"
        )
    staged_qkv = hybrid_metadata.staged_qkv_cpu
    if staged_qkv is not None and staged_qkv.is_pinned():
        total_heads = staged_qkv.shape[1]
        if (
            query.stride(-1) == 1
            and query.stride(-2) == query.shape[-1]
            and query.stride(0) == total_heads * query.shape[-1]
            and query.storage_offset() == 0
        ):
            try:
                qkv_view = query.as_strided(
                    (batch, total_heads, query.shape[-1]),
                    (query.stride(0), query.stride(1), query.stride(2)),
                )
            except RuntimeError:
                qkv_view = None
            if qkv_view is not None and staged_qkv[:batch].shape == qkv_view.shape:
                current_stream = torch.cuda.current_stream(query.device)
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    staged_qkv[:batch].copy_(qkv_view.detach(), non_blocking=True)
                    event.record(stream)
                query.record_stream(stream)
                hybrid_metadata.staged_query_valid = True
                hybrid_metadata.staged_qkv_valid = True
                if hybrid_metadata.metrics is not None:
                    hybrid_metadata.metrics.q_d2h_bytes += query[:batch].nbytes
                return

    staged_view = staged[:batch]
    if staged_view.shape != query[:batch].shape or staged_view.dtype != query.dtype:
        raise ValueError(
            "COTS hybrid Q staging buffer shape/dtype mismatch: "
            f"staged={tuple(staged_view.shape)} {staged_view.dtype}, "
            f"query={tuple(query[:batch].shape)} {query.dtype}"
        )

    key_view = None
    value_view = None
    kv_event = hybrid_metadata.kv_ready_event
    staged_key = hybrid_metadata.staged_key_cpu
    staged_value = hybrid_metadata.staged_value_cpu
    can_stage_separate_kv = (
        key is not None
        and value is not None
        and key.is_cuda
        and value.is_cuda
        and staged_key is not None
        and staged_value is not None
        and kv_event is not None
        and staged_key.is_pinned()
        and staged_value.is_pinned()
    )
    if can_stage_separate_kv:
        assert key is not None
        assert value is not None
        assert staged_key is not None
        assert staged_value is not None
        assert kv_event is not None
        key_view = staged_key[:batch]
        value_view = staged_value[:batch]
        if (
            key_view.shape != key[:batch].shape
            or value_view.shape != value[:batch].shape
        ):
            raise ValueError(
                "COTS hybrid K/V staging buffer shape mismatch: "
                f"key_staged={tuple(key_view.shape)}, key={tuple(key[:batch].shape)}, "
                f"value_staged={tuple(value_view.shape)}, "
                f"value={tuple(value[:batch].shape)}"
            )

    current_stream = torch.cuda.current_stream(query.device)
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        staged_view.copy_(query[:batch].detach(), non_blocking=True)
        event.record(stream)
        if (
            key_view is not None
            and value_view is not None
            and key is not None
            and value is not None
        ):
            key_view.copy_(key[:batch].detach(), non_blocking=True)
            value_view.copy_(value[:batch].detach(), non_blocking=True)
            assert kv_event is not None
            kv_event.record(stream)
            hybrid_metadata.staged_kv_valid = True
    query.record_stream(stream)
    if (
        key_view is not None
        and value_view is not None
        and key is not None
        and value is not None
    ):
        key.record_stream(stream)
        value.record_stream(stream)
    hybrid_metadata.staged_query_valid = True
    if hybrid_metadata.metrics is not None:
        hybrid_metadata.metrics.q_d2h_bytes += staged_view.nbytes
        if key_view is not None and value_view is not None:
            hybrid_metadata.metrics.kv_d2h_bytes += key_view.nbytes + value_view.nbytes


def _scatter_cpu_kv_to_suffix_cache(
    key_cpu: torch.Tensor,
    value_cpu: torch.Tensor,
    hybrid_metadata: CotsHybridDecodeMetadata,
) -> None:
    req_indices = hybrid_metadata.scatter_req_indices
    block_ids = hybrid_metadata.scatter_block_ids
    block_offsets = hybrid_metadata.scatter_block_offsets
    if req_indices is not None and block_ids is not None and block_offsets is not None:
        if block_ids.numel() == 0:
            return
        num_kv_heads = hybrid_metadata.cpu_key_cache.shape[1]
        head_dim = hybrid_metadata.cpu_key_cache.shape[3]
        if (
            req_indices.numel() == key_cpu.shape[0]
            and key_cpu.shape[1:] == (num_kv_heads, head_dim)
            and value_cpu.shape[1:] == (num_kv_heads, head_dim)
            and head_dim == 128
        ):
            cots_gqa_bf16_scatter_suffix_kv(
                key_cpu,
                value_cpu,
                block_ids,
                block_offsets,
                hybrid_metadata.cpu_key_cache,
                hybrid_metadata.cpu_value_cache,
            )
            return
        else:
            key_src = key_cpu[req_indices]
            value_src = value_cpu[req_indices]
        hybrid_metadata.cpu_key_cache[block_ids, :, block_offsets, :] = key_src
        hybrid_metadata.cpu_value_cache[block_ids, :, block_offsets, :] = value_src
        return

    batch = hybrid_metadata.cpu_seq_lens.shape[0]
    block_size = hybrid_metadata.cpu_key_cache.shape[2]
    for req_index in range(batch):
        suffix_len = int(hybrid_metadata.cpu_seq_lens[req_index].item())
        if suffix_len <= 0:
            continue
        local_pos = suffix_len - 1
        block_col = local_pos // block_size
        block_offset = local_pos % block_size
        block_id = int(hybrid_metadata.cpu_block_table[req_index, block_col].item())
        hybrid_metadata.cpu_key_cache[block_id, :, block_offset, :].copy_(
            key_cpu[req_index]
        )
        hybrid_metadata.cpu_value_cache[block_id, :, block_offset, :].copy_(
            value_cpu[req_index]
        )


def cots_hybrid_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    hybrid_metadata: CotsHybridDecodeMetadata,
) -> None:
    """Mirror the current eager decode token's K/V into the CPU suffix cache.

    CUDA K/V is staged into pinned CPU buffers asynchronously. The scatter into
    the final block cache is completed by ``cots_hybrid_decode_attention`` after
    GPU prefix attention has been launched, which lets the tiny D2H transfer
    overlap with useful GPU work. CPU inputs keep the old immediate path for
    focused unit tests.
    """

    batch = hybrid_metadata.cpu_seq_lens.shape[0]
    source_indices = hybrid_metadata.scatter_source_indices
    if source_indices is not None:
        if key.is_cuda:
            source_indices = _source_indices_for_device(hybrid_metadata, key.device)
            assert source_indices is not None
        key = key.index_select(0, source_indices)
        value = value.index_select(0, source_indices)
    if key.shape[0] < batch or value.shape[0] < batch:
        raise ValueError(
            "COTS hybrid KV update expected at least one K/V row per request: "
            f"key={tuple(key.shape)}, value={tuple(value.shape)}, batch={batch}"
        )

    if hybrid_metadata.staged_kv_valid and not hybrid_metadata.staged_qkv_valid:
        return

    hybrid_metadata.staged_kv_valid = False
    staged_key = hybrid_metadata.staged_key_cpu
    staged_value = hybrid_metadata.staged_value_cpu
    stream = hybrid_metadata.kv_copy_stream
    event = hybrid_metadata.kv_ready_event
    if hybrid_metadata.staged_qkv_valid and hybrid_metadata.staged_qkv_cpu is not None:
        hybrid_metadata.staged_kv_valid = True
        if hybrid_metadata.metrics is not None:
            hybrid_metadata.metrics.kv_d2h_bytes += (
                key[:batch].nbytes + value[:batch].nbytes
            )
        return
    if (
        key.is_cuda
        and value.is_cuda
        and staged_key is not None
        and staged_value is not None
        and stream is not None
        and event is not None
        and staged_key.is_pinned()
        and staged_value.is_pinned()
    ):
        key_view = staged_key[:batch]
        value_view = staged_value[:batch]
        if (
            key_view.shape != key[:batch].shape
            or value_view.shape != value[:batch].shape
        ):
            raise ValueError(
                "COTS hybrid KV staging buffer shape mismatch: "
                f"key_staged={tuple(key_view.shape)}, key={tuple(key[:batch].shape)}, "
                f"value_staged={tuple(value_view.shape)}, "
                f"value={tuple(value[:batch].shape)}"
            )
        current_stream = torch.cuda.current_stream(key.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            key_view.copy_(key[:batch].detach(), non_blocking=True)
            value_view.copy_(value[:batch].detach(), non_blocking=True)
            event.record(stream)
        key.record_stream(stream)
        value.record_stream(stream)
        hybrid_metadata.staged_kv_valid = True
        if hybrid_metadata.metrics is not None:
            hybrid_metadata.metrics.kv_d2h_bytes += key_view.nbytes + value_view.nbytes
        return

    key_cpu = (
        key[:batch].detach().to(device="cpu", dtype=hybrid_metadata.cpu_key_cache.dtype)
    )
    value_cpu = (
        value[:batch]
        .detach()
        .to(device="cpu", dtype=hybrid_metadata.cpu_value_cache.dtype)
    )
    if hybrid_metadata.metrics is not None:
        hybrid_metadata.metrics.kv_d2h_bytes += key_cpu.nbytes + value_cpu.nbytes
    _scatter_cpu_kv_to_suffix_cache(key_cpu, value_cpu, hybrid_metadata)


def _finish_staged_kv_cache_update(
    hybrid_metadata: CotsHybridDecodeMetadata,
    batch: int,
    ready_already_synchronized: bool = False,
) -> None:
    if not hybrid_metadata.staged_kv_valid:
        return
    ready_event = (
        hybrid_metadata.query_ready_event
        if hybrid_metadata.staged_qkv_valid
        else hybrid_metadata.kv_ready_event
    )
    if ready_event is not None and not ready_already_synchronized:
        wait_start = time.perf_counter()
        ready_event.synchronize()
        if hybrid_metadata.metrics is not None:
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
            hybrid_metadata.metrics.cpu_suffix_wait_ms += wait_ms
            if hybrid_metadata.staged_qkv_valid:
                hybrid_metadata.metrics.qkv_ready_wait_ms += wait_ms
            else:
                hybrid_metadata.metrics.kv_ready_wait_ms += wait_ms
    if hybrid_metadata.staged_qkv_valid and hybrid_metadata.staged_qkv_cpu is not None:
        num_kv_heads = hybrid_metadata.cpu_key_cache.shape[1]
        num_q_heads = hybrid_metadata.staged_qkv_cpu.shape[1] - 2 * num_kv_heads
        key_cpu = hybrid_metadata.staged_qkv_cpu[
            :batch, num_q_heads : num_q_heads + num_kv_heads, :
        ]
        value_cpu = hybrid_metadata.staged_qkv_cpu[
            :batch, num_q_heads + num_kv_heads :, :
        ]
    else:
        assert hybrid_metadata.staged_key_cpu is not None
        assert hybrid_metadata.staged_value_cpu is not None
        key_cpu = hybrid_metadata.staged_key_cpu[:batch]
        value_cpu = hybrid_metadata.staged_value_cpu[:batch]
    _scatter_cpu_kv_to_suffix_cache(key_cpu, value_cpu, hybrid_metadata)
    hybrid_metadata.staged_kv_valid = False
    hybrid_metadata.staged_qkv_valid = False


def submit_cots_hybrid_suffix_attention(
    *,
    query: torch.Tensor,
    hybrid_metadata: CotsHybridDecodeMetadata,
    batch: int,
    num_query_heads: int,
    softmax_scale: float,
    sync_after_submit: bool,
) -> CotsHybridSuffixState:
    suffix_runner = hybrid_metadata.suffix_attention_runner
    if suffix_runner is None:
        raise RuntimeError("COTS hybrid decode metadata has no suffix runner")
    if hybrid_metadata.cpu_seq_lens.shape[0] != batch:
        raise ValueError("cpu_seq_lens batch dimension must match query")

    device = query.device
    qkv_scatter_in_runner = False
    separate_kv_scatter_in_runner = False
    scatter_key_cpu = None
    scatter_value_cpu = None
    qkv_ready_already_synchronized = False
    suffix_submit_stream = None
    suffix_submit_ready_event = None
    suffix_submit_done_event = None
    if hybrid_metadata.staged_query_valid:
        assert (
            hybrid_metadata.staged_query_cpu is not None
            or hybrid_metadata.staged_qkv_cpu is not None
        )
        query_event = hybrid_metadata.query_ready_event
        kv_event = hybrid_metadata.kv_ready_event
        if query_event is not None and not (
            hybrid_metadata.staged_kv_valid and query_event is kv_event
        ):
            if (
                suffix_runner.kind == "native_prepared"
                and hybrid_metadata.suffix_submit_stream is not None
                and not torch.cuda.is_current_stream_capturing()
            ):
                suffix_submit_stream = hybrid_metadata.suffix_submit_stream
                suffix_submit_ready_event = query_event
                suffix_submit_done_event = hybrid_metadata.suffix_submit_done_event
            elif suffix_runner.kind == "native_prepared":
                torch.cuda.current_stream(device).wait_event(query_event)
                qkv_ready_already_synchronized = True
            else:
                wait_start = time.perf_counter()
                query_event.synchronize()
                qkv_ready_already_synchronized = hybrid_metadata.staged_qkv_valid
                if hybrid_metadata.metrics is not None:
                    wait_ms = (time.perf_counter() - wait_start) * 1000.0
                    hybrid_metadata.metrics.cpu_suffix_wait_ms += wait_ms
                    hybrid_metadata.metrics.qkv_ready_wait_ms += wait_ms
        if (
            hybrid_metadata.staged_qkv_valid
            and hybrid_metadata.staged_qkv_cpu is not None
        ):
            query_cpu = hybrid_metadata.staged_qkv_cpu[:batch, :num_query_heads, :]
            qkv_scatter_in_runner = (
                suffix_runner.kind == "native_prepared"
                and hybrid_metadata.scatter_block_ids is not None
                and hybrid_metadata.scatter_block_offsets is not None
                and hybrid_metadata.scatter_block_ids.numel() > 0
            )
        else:
            assert hybrid_metadata.staged_query_cpu is not None
            query_cpu = hybrid_metadata.staged_query_cpu[:batch]
            separate_kv_scatter_in_runner = (
                suffix_runner.kind == "native_prepared"
                and hybrid_metadata.staged_kv_valid
                and hybrid_metadata.staged_key_cpu is not None
                and hybrid_metadata.staged_value_cpu is not None
                and hybrid_metadata.scatter_block_ids is not None
                and hybrid_metadata.scatter_block_offsets is not None
                and hybrid_metadata.scatter_block_ids.numel() > 0
            )
            if separate_kv_scatter_in_runner:
                if kv_event is not None:
                    if suffix_submit_stream is not None:
                        suffix_submit_ready_event = kv_event
                    else:
                        torch.cuda.current_stream(device).wait_event(kv_event)
                staged_key_cpu = hybrid_metadata.staged_key_cpu
                staged_value_cpu = hybrid_metadata.staged_value_cpu
                assert staged_key_cpu is not None
                assert staged_value_cpu is not None
                scatter_key_cpu = staged_key_cpu[:batch]
                scatter_value_cpu = staged_value_cpu[:batch]
    else:
        query_cpu = query[:batch].detach().to("cpu")
        if hybrid_metadata.metrics is not None:
            hybrid_metadata.metrics.q_d2h_bytes += query[:batch].nbytes
    if not (qkv_scatter_in_runner or separate_kv_scatter_in_runner):
        _finish_staged_kv_cache_update(
            hybrid_metadata,
            batch,
            ready_already_synchronized=qkv_ready_already_synchronized,
        )

    suffix_out_cpu = (
        hybrid_metadata.suffix_out_cpu[:batch]
        if hybrid_metadata.suffix_out_cpu is not None
        else torch.empty(
            query_cpu.shape,
            dtype=query_cpu.dtype,
            device="cpu",
            pin_memory=True,
        )
    )
    suffix_lse_cpu = (
        hybrid_metadata.suffix_lse_cpu[:, :batch]
        if hybrid_metadata.suffix_lse_cpu is not None
        else torch.empty(
            (num_query_heads, batch),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
    )
    cpu_suffix_start = time.perf_counter()
    suffix_kwargs = dict(
        query=query_cpu,
        key_cache=hybrid_metadata.cpu_key_cache,
        value_cache=hybrid_metadata.cpu_value_cache,
        block_table=hybrid_metadata.cpu_block_table,
        seq_lens=hybrid_metadata.cpu_seq_lens,
        scale=softmax_scale,
        output=suffix_out_cpu,
        output_lse=suffix_lse_cpu,
        cuda_anchor=query,
        task_id=hybrid_metadata.suffix_attention_task_id,
        scatter_block_ids=(
            hybrid_metadata.scatter_block_ids
            if (qkv_scatter_in_runner or separate_kv_scatter_in_runner)
            else None
        ),
        scatter_block_offsets=(
            hybrid_metadata.scatter_block_offsets
            if (qkv_scatter_in_runner or separate_kv_scatter_in_runner)
            else None
        ),
        scatter_key_cpu=scatter_key_cpu,
        scatter_value_cpu=scatter_value_cpu,
        scatter_from_qkv=qkv_scatter_in_runner,
        scatter_from_separate_kv=separate_kv_scatter_in_runner,
        snapshot_inputs=hybrid_metadata.suffix_attention_snapshot_inputs,
    )
    if suffix_submit_stream is not None:
        suffix_kwargs["submit_stream"] = suffix_submit_stream
        suffix_kwargs["submit_ready_event"] = suffix_submit_ready_event
        suffix_kwargs["submit_done_event"] = suffix_submit_done_event
    if not sync_after_submit:
        suffix_kwargs["sync_after_submit"] = False
    suffix_runner.run_gqa_bf16_suffix_attention(**suffix_kwargs)
    if qkv_scatter_in_runner or separate_kv_scatter_in_runner:
        hybrid_metadata.staged_kv_valid = False
        if qkv_scatter_in_runner:
            hybrid_metadata.staged_qkv_valid = False
    return CotsHybridSuffixState(
        suffix_runner=suffix_runner,
        suffix_out_cpu=suffix_out_cpu,
        suffix_lse_cpu=suffix_lse_cpu,
        cpu_suffix_start=cpu_suffix_start,
        cuda_anchor=query,
        task_id=hybrid_metadata.suffix_attention_task_id,
        sync_required=not sync_after_submit,
    )


def cots_hybrid_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    gpu_key_cache: torch.Tensor,
    gpu_value_cache: torch.Tensor,
    gpu_block_table: torch.Tensor,
    hybrid_metadata: CotsHybridDecodeMetadata,
    softmax_scale: float,
    fa_version: int,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    precomputed_prefix_out: torch.Tensor | None = None,
    precomputed_prefix_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the COTS hybrid decode attention primitive.

    Phase 2 restrictions are intentionally tight: BF16 GQA decode with
    head_dim=128, no ALiBi/sliding-window/sinks/FP8. The surrounding
    FlashAttentionImpl gate enforces the feature flag and broader attention
    restrictions before calling here.
    """

    if query.dim() != 3:
        raise ValueError(f"query must be [B, H, D], got {tuple(query.shape)}")
    num_query_heads = int(query.shape[1])
    head_dim = int(query.shape[2])
    if num_query_heads <= 0:
        raise ValueError("query must have at least one head")
    if head_dim != 128:
        raise ValueError(
            f"COTS hybrid decode currently supports head_dim=128, got {head_dim}"
        )
    if query.dtype != torch.bfloat16:
        raise ValueError(f"query must be bfloat16, got {query.dtype}")
    if not query.is_cuda:
        raise ValueError("query must be CUDA-resident for GPU prefix attention")
    if (
        hybrid_metadata.cpu_key_cache.dim() != 4
        or hybrid_metadata.cpu_value_cache.dim() != 4
    ):
        raise ValueError(
            "CPU suffix KV caches must have shape "
            "[blocks, num_kv_heads, block_size, head_dim]"
        )
    if hybrid_metadata.cpu_key_cache.shape != hybrid_metadata.cpu_value_cache.shape:
        raise ValueError("CPU suffix key/value cache shapes must match")
    num_kv_heads = int(hybrid_metadata.cpu_key_cache.shape[1])
    cache_head_dim = int(hybrid_metadata.cpu_key_cache.shape[3])
    if num_kv_heads <= 0:
        raise ValueError("CPU suffix KV cache must have at least one head")
    if cache_head_dim != head_dim:
        raise ValueError(
            "CPU suffix KV cache head_dim must match query, got "
            f"{cache_head_dim} vs {head_dim}"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "COTS hybrid decode requires num_query_heads divisible by "
            f"num_kv_heads, got {num_query_heads} and {num_kv_heads}"
        )
    heads_per_kv = num_query_heads // num_kv_heads
    if heads_per_kv > 8:
        raise ValueError(
            "COTS hybrid decode supports at most 8 query heads per KV head, "
            f"got {heads_per_kv}"
        )
    if output.shape != query.shape:
        raise ValueError(
            f"output shape {tuple(output.shape)} must match query {tuple(query.shape)}"
        )
    if hybrid_metadata.split_blocks <= 0:
        raise ValueError("split_blocks must be > 0 for COTS hybrid decode")

    full_output = output
    if (precomputed_prefix_out is None) != (precomputed_prefix_lse is None):
        raise ValueError("precomputed prefix output and LSE must be provided together")
    has_precomputed_prefix = precomputed_prefix_out is not None

    source_indices = hybrid_metadata.scatter_source_indices
    source_indices_gpu = None
    batch = query.shape[0]
    if source_indices is not None:
        source_indices_gpu = _source_indices_for_device(hybrid_metadata, query.device)
        assert source_indices_gpu is not None
        batch = source_indices.numel()
        query_already_staged = hybrid_metadata.staged_query_valid
        if not has_precomputed_prefix or not query_already_staged:
            query = query.index_select(0, source_indices_gpu)
        if not has_precomputed_prefix:
            output = torch.empty(
                (batch, num_query_heads, head_dim),
                dtype=query.dtype,
                device=query.device,
            )

    device = query.device

    prefix_len = None
    cu_query_lens = None
    prefix_kv_lens = None
    if not has_precomputed_prefix:
        block_size = gpu_key_cache.shape[-3]
        prefix_len = hybrid_metadata.split_blocks * block_size
        if gpu_block_table.shape[0] != batch:
            req_indices_gpu = hybrid_metadata.req_indices_gpu
            if req_indices_gpu is None:
                raise ValueError(
                    "gpu_block_table batch dimension must match query batch when "
                    "no active request indices are provided: "
                    f"{gpu_block_table.shape[0]} vs {batch}"
                )
            gpu_block_table = gpu_block_table.index_select(0, req_indices_gpu[:batch])
        if gpu_block_table.shape[1] < hybrid_metadata.split_blocks:
            raise ValueError(
                "gpu_block_table does not contain enough prefix blocks: "
                f"{gpu_block_table.shape[1]} < {hybrid_metadata.split_blocks}"
            )
        cu_query_lens = torch.arange(batch + 1, dtype=torch.int32, device=device)
        prefix_kv_lens = torch.full(
            (batch,), prefix_len, dtype=torch.int32, device=device
        )
    if hybrid_metadata.cpu_seq_lens.shape[0] != batch:
        raise ValueError("cpu_seq_lens batch dimension must match query")

    cuda_timing = (
        not has_precomputed_prefix
        and hybrid_metadata.metrics is not None
        and _cuda_event_timing_enabled()
        and torch.cuda.is_available()
    )
    current_stream = torch.cuda.current_stream(device) if cuda_timing else None
    prefix_start = prefix_end = None
    if cuda_timing:
        prefix_start, prefix_end = _make_cuda_timing_events()

    early_suffix_state = None
    if (
        early_suffix_state is None
        and not has_precomputed_prefix
        and can_submit_cots_hybrid_suffix_early(hybrid_metadata)
    ):
        early_suffix_state = submit_cots_hybrid_suffix_attention(
            query=query,
            hybrid_metadata=hybrid_metadata,
            batch=batch,
            num_query_heads=num_query_heads,
            softmax_scale=softmax_scale,
            sync_after_submit=False,
        )

    if precomputed_prefix_out is not None:
        assert precomputed_prefix_lse is not None
        prefix_out = precomputed_prefix_out
        prefix_lse = precomputed_prefix_lse
        if source_indices_gpu is None:
            expected_prefix_shape = query.shape
            expected_lse_shape = (num_query_heads, batch)
        else:
            expected_prefix_shape = full_output.shape
            expected_lse_shape = (num_query_heads, full_output.shape[0])
        if prefix_out.shape != expected_prefix_shape:
            raise ValueError(
                "precomputed prefix output has unexpected shape, got "
                f"{tuple(prefix_out.shape)} vs {tuple(expected_prefix_shape)}"
            )
        if prefix_lse.shape != expected_lse_shape:
            raise ValueError(
                "precomputed prefix LSE has unexpected shape, got "
                f"{tuple(prefix_lse.shape)} vs {expected_lse_shape}"
            )
    else:
        if cuda_timing:
            assert current_stream is not None
            assert prefix_start is not None
            prefix_start.record(current_stream)
        assert prefix_len is not None
        assert cu_query_lens is not None
        assert prefix_kv_lens is not None
        descale_shape = (batch, gpu_key_cache.shape[-2])
        prefix_out, prefix_lse = flash_attn_varlen_func(
            q=query,
            k=gpu_key_cache,
            v=gpu_value_cache,
            out=None,
            cu_seqlens_q=cu_query_lens,
            seqused_k=prefix_kv_lens,
            max_seqlen_q=1,
            max_seqlen_k=prefix_len,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=gpu_block_table[:, : hybrid_metadata.split_blocks],
            return_softmax_lse=True,
            fa_version=fa_version,
            q_descale=q_descale.expand(descale_shape)
            if q_descale is not None
            else None,
            k_descale=k_descale.expand(descale_shape)
            if k_descale is not None
            else None,
            v_descale=v_descale.expand(descale_shape)
            if v_descale is not None
            else None,
            num_splits=1,
        )
    if cuda_timing:
        assert current_stream is not None
        assert prefix_end is not None
        prefix_end.record(current_stream)

    # Suffix path: materialize the tiny query on CPU, compute suffix
    # attention, then expose output/LSE back to GPU for merge. Q and K/V are
    # preferably staged before this call so their waits can overlap with GPU
    # prefix attention.
    if early_suffix_state is None:
        suffix_state = submit_cots_hybrid_suffix_attention(
            query=query,
            hybrid_metadata=hybrid_metadata,
            batch=batch,
            num_query_heads=num_query_heads,
            softmax_scale=softmax_scale,
            sync_after_submit=True,
        )
    else:
        suffix_state = early_suffix_state
    suffix_runner = suffix_state.suffix_runner
    suffix_out_cpu = suffix_state.suffix_out_cpu
    suffix_lse_cpu = suffix_state.suffix_lse_cpu
    cpu_suffix_start = suffix_state.cpu_suffix_start
    if cuda_timing and prefix_start is not None and prefix_end is not None:
        prefix_end.synchronize()
        assert hybrid_metadata.metrics is not None
        hybrid_metadata.metrics.gpu_prefix_attn_ms += prefix_start.elapsed_time(
            prefix_end
        )
    if suffix_state.sync_required:
        suffix_runner.sync_gqa_bf16_suffix_attention(
            cuda_anchor=suffix_state.cuda_anchor,
            task_id=suffix_state.task_id,
        )
    if hybrid_metadata.metrics is not None:
        hybrid_metadata.metrics.hybrid_decode_calls += 1
        hybrid_metadata.metrics.cpu_suffix_attn_ms += (
            time.perf_counter() - cpu_suffix_start
        ) * 1000.0

    if hybrid_metadata.metrics is not None:
        hybrid_metadata.metrics.kv_uva_h2d_bytes += (
            suffix_out_cpu.nbytes + suffix_lse_cpu.nbytes
        )
    # The suffix buffers are CPU-written pinned memory consumed by the GPU via
    # UVA. The CPU writes complete before the merge kernel is launched.
    suffix_out_gpu = get_accelerator_view_from_cpu_tensor(suffix_out_cpu)
    suffix_lse_gpu = get_accelerator_view_from_cpu_tensor(suffix_lse_cpu)
    merge_start = merge_end = None
    if cuda_timing:
        assert current_stream is not None
        merge_start, merge_end = _make_cuda_timing_events()
        merge_start.record(current_stream)
    if source_indices_gpu is not None and has_precomputed_prefix:
        merge_attn_states_indexed(
            full_output,
            prefix_out,
            prefix_lse,
            suffix_out_gpu,
            suffix_lse_gpu,
            source_indices_gpu,
        )
    else:
        merge_attn_states(
            output,
            prefix_out,
            prefix_lse,
            suffix_out_gpu,
            suffix_lse_gpu,
        )
    if cuda_timing:
        assert merge_start is not None and merge_end is not None
        merge_end.record(current_stream)
        merge_end.synchronize()
        assert hybrid_metadata.metrics is not None
        hybrid_metadata.metrics.hybrid_merge_ms += merge_start.elapsed_time(merge_end)
    if hybrid_metadata.suffix_read_done_event is not None:
        hybrid_metadata.suffix_read_done_event.record(torch.cuda.current_stream(device))
    if hybrid_metadata.staging_done_event is not None:
        hybrid_metadata.staging_done_event.record(torch.cuda.current_stream(device))
    if source_indices_gpu is not None:
        if not has_precomputed_prefix:
            full_output.index_copy_(0, source_indices_gpu, output)
        return full_output
    return output
