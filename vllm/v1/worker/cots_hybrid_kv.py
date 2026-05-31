# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side COTS hybrid KV state.

Phase 2 keeps vLLM's scheduler policy mostly unchanged and mirrors suffix
K/V into a planner-sized CPU pool. This store owns only worker-local state:
per-layer CPU cache tensors, staging buffers, and the suffix-local block table
consumed by the COTS CPU attention kernel. CPU block ownership belongs to the
scheduler/KV-manager side.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.cots_hybrid_attention import (
    CotsHybridDecodeMetadata,
    NativeCotsSuffixAttentionRunner,
)
from vllm.v1.metrics.stats import CotsHybridKVStats

if TYPE_CHECKING:
    from vllm.model_executor.offloader.base import ForwardDispatchInfo


class CotsHybridKVStore:
    """CPU suffix KV cache mirror for eager COTS hybrid decode.

    Block IDs are supplied by the scheduler/KV manager and are shared across
    layers: CPU block 0 in every layer stores the same logical CPU KV block.
    The store treats the block table as authoritative and does not own cache
    policy or request/block ownership.
    """

    def __init__(
        self,
        *,
        layer_names: Sequence[str],
        split_blocks: int,
        cpu_pool_bytes: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        max_num_reqs: int,
        max_model_len: int,
        pin_memory: bool,
        num_query_heads: int,
        enable_suffix_read_events: bool = True,
    ) -> None:
        if not layer_names:
            raise ValueError("COTS hybrid KV requires at least one attention layer")
        if split_blocks <= 0:
            raise ValueError(f"split_blocks must be positive, got {split_blocks}")
        if cpu_pool_bytes <= 0:
            raise ValueError(f"cpu_pool_bytes must be positive, got {cpu_pool_bytes}")
        if dtype != torch.bfloat16:
            raise ValueError(
                "COTS hybrid KV Phase 2 supports only BF16 CPU suffix attention"
            )

        self.layer_names = tuple(layer_names)
        self.split_blocks = int(split_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_size = int(head_size)
        self.dtype = dtype
        self.pin_memory = bool(pin_memory)
        self.enable_suffix_read_events = bool(enable_suffix_read_events)
        self.num_query_heads = int(num_query_heads)
        self.split_tokens = self.split_blocks * self.block_size
        self.max_suffix_blocks_per_req = max(
            cdiv(max_model_len, self.block_size) - self.split_blocks, 0
        )

        dtype_bytes = torch.empty((), dtype=dtype).element_size()
        bytes_per_block_per_layer = (
            2 * self.block_size * num_kv_heads * head_size * dtype_bytes
        )
        num_cpu_blocks = cpu_pool_bytes // (
            bytes_per_block_per_layer * len(self.layer_names)
        )
        if num_cpu_blocks <= 0:
            raise ValueError(
                "COTS hybrid KV CPU pool is too small for one suffix block per "
                "layer: "
                f"pool_bytes={cpu_pool_bytes}, "
                f"bytes_per_block_per_layer={bytes_per_block_per_layer}, "
                f"num_layers={len(self.layer_names)}"
            )

        self.num_cpu_blocks = int(num_cpu_blocks)
        self._key_caches: dict[str, torch.Tensor] = {}
        self._value_caches: dict[str, torch.Tensor] = {}
        cache_shape = (
            self.num_cpu_blocks,
            int(num_kv_heads),
            self.block_size,
            int(head_size),
        )
        for layer_name in self.layer_names:
            self._key_caches[layer_name] = torch.empty(
                cache_shape, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self._value_caches[layer_name] = torch.empty(
                cache_shape, dtype=dtype, device="cpu", pin_memory=pin_memory
            )

        self._cpu_block_table = torch.empty(
            (max_num_reqs, self.max_suffix_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_seq_lens = torch.empty(
            (max_num_reqs,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        # Stable CPU metadata for graph replay. Prepared suffix tasks capture
        # raw pointers, so rebuilds must update these buffers in-place instead
        # of returning freshly allocated scatter tensors.
        self._scatter_req_indices = torch.empty(
            (max_num_reqs,),
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._scatter_block_ids = torch.empty(
            (max_num_reqs,),
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._scatter_block_offsets = torch.empty(
            (max_num_reqs,),
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        # The native suffix task ids are partitioned by layer/slot/batch-size.
        # Four slots are the validated Phase 2 setting.
        self._num_staging_slots = 4
        self._query_staging_slots = {
            layer_name: [
                torch.empty(
                    (max_num_reqs, self.num_query_heads, self.head_size),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for _ in range(self._num_staging_slots)
            ]
            for layer_name in self.layer_names
        }
        self._qkv_staging_slots = {
            layer_name: [
                torch.empty(
                    (
                        max_num_reqs,
                        self.num_query_heads + 2 * self.num_kv_heads,
                        self.head_size,
                    ),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for _ in range(self._num_staging_slots)
            ]
            for layer_name in self.layer_names
        }
        self._key_staging_slots = {
            layer_name: [
                torch.empty(
                    (max_num_reqs, self.num_kv_heads, self.head_size),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for _ in range(self._num_staging_slots)
            ]
            for layer_name in self.layer_names
        }
        self._value_staging_slots = {
            layer_name: [
                torch.empty(
                    (max_num_reqs, self.num_kv_heads, self.head_size),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for _ in range(self._num_staging_slots)
            ]
            for layer_name in self.layer_names
        }
        self._query_copy_streams = {
            layer_name: (
                torch.cuda.Stream()
                if pin_memory and torch.cuda.is_available()
                else None
            )
            for layer_name in self.layer_names
        }
        self._query_ready_events = {
            layer_name: (
                torch.cuda.Event(blocking=False)
                if pin_memory and torch.cuda.is_available()
                else None
            )
            for layer_name in self.layer_names
        }
        self._kv_copy_streams = self._query_copy_streams
        self._kv_ready_events = {
            layer_name: (
                torch.cuda.Event(blocking=False)
                if pin_memory and torch.cuda.is_available()
                else None
            )
            for layer_name in self.layer_names
        }
        self._suffix_submit_done_events = {
            layer_name: [
                torch.cuda.Event(blocking=False)
                if pin_memory and torch.cuda.is_available()
                else None
                for _ in range(self._num_staging_slots)
            ]
            for layer_name in self.layer_names
        }
        self._suffix_out_slots = {
            layer_name: [
                torch.empty(
                    (max_num_reqs, self.num_query_heads, self.head_size),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for _ in range(self._num_staging_slots)
            ]
            for layer_name in self.layer_names
        }
        self._suffix_lse_slots: dict[str, list[dict[int, torch.Tensor]]] = {
            layer_name: [{} for _ in range(self._num_staging_slots)]
            for layer_name in self.layer_names
        }
        self._suffix_out_overflow_buffers: dict[str, dict[int, torch.Tensor]] = {
            layer_name: {} for layer_name in self.layer_names
        }
        self._suffix_lse_buffers: dict[str, dict[int, torch.Tensor]] = {
            layer_name: {} for layer_name in self.layer_names
        }
        self._suffix_read_events: dict[str, dict[int, torch.cuda.Event]] = {
            layer_name: {} for layer_name in self.layer_names
        }
        self._staging_reuse_events: dict[tuple[str, int], torch.cuda.Event] = {}
        self._next_staging_slot = {layer_name: 0 for layer_name in self.layer_names}
        self._common_cache_key: tuple[object, ...] | None = None
        self._common_cache_value: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None
        self._runtime_stats = CotsHybridKVStats()
        self._max_num_reqs = int(max_num_reqs)
        # Decode graph buckets keep one prepared task per captured batch size
        # and staging slot. CPU suffix prefill can row-expand a request into
        # more rows than the decode batch cap; those eager-only calls reuse one
        # overflow task per layer so we do not install thousands of wait-kernel
        # slots.
        self._suffix_attention_tasks_per_layer = (
            self._num_staging_slots * self._max_num_reqs + 1
        )
        self._suffix_attention_runner = NativeCotsSuffixAttentionRunner(
            num_tasks=len(self.layer_names) * self._suffix_attention_tasks_per_layer
        )
        self._suffix_counter_stats_enabled = (
            os.environ.get("VLLM_COTS_DIAG") == "1"
            or os.environ.get("VLLM_COTS_SUFFIX_COUNTERS") == "1"
        )
        if self._suffix_counter_stats_enabled:
            self._suffix_attention_runner.reset_counters()
        self._suffix_attention_task_base_ids = {
            layer_name: i * self._suffix_attention_tasks_per_layer
            for i, layer_name in enumerate(self.layer_names)
        }
        self._suffix_attention_overflow_task_ids = {
            layer_name: base + self._num_staging_slots * self._max_num_reqs
            for layer_name, base in self._suffix_attention_task_base_ids.items()
        }
        self._last_live_counts: tuple[int, int] | None = None
        self._submit_timing_stats: dict[str, float] = {}

    def reset_submit_timing_stats(self) -> None:
        self._submit_timing_stats = {}

    def drain_submit_timing_stats(self) -> dict[str, float]:
        stats = self._submit_timing_stats
        self._submit_timing_stats = {}
        return stats

    def set_live_counts(self, num_tokens: int, scatter_count: int) -> None:
        live_counts = (int(num_tokens), int(scatter_count))
        if live_counts == self._last_live_counts:
            return
        self._suffix_attention_runner.set_runtime_counts(*live_counts)
        self._last_live_counts = live_counts

    def live_counts_are_zero(self) -> bool:
        return self._last_live_counts == (0, 0)

    def positions_have_suffix(
        self, positions_cpu: Sequence[int], num_tokens: int
    ) -> bool:
        count = min(int(num_tokens), len(positions_cpu))
        return any(int(positions_cpu[i]) >= self.split_tokens for i in range(count))

    def on_dispatch(
        self,
        info: ForwardDispatchInfo,
        *,
        positions_cpu: Sequence[int] | None,
    ) -> None:
        num_actual_tokens = int(info.num_tokens_unpadded)
        if num_actual_tokens <= 0:
            self.set_live_counts(0, 0)
            return
        if positions_cpu is None:
            suffix_rows = num_actual_tokens
        else:
            count = min(num_actual_tokens, len(positions_cpu))
            suffix_rows = sum(
                1 for i in range(count) if int(positions_cpu[i]) >= self.split_tokens
            )
        # Decode graph buckets use live-count overrides for padded static tasks.
        # Row-expanded CPU suffix prefill can exceed the decode bucket capacity;
        # those overflow tasks are populated with exact live row/scatter counts,
        # so publishing an override here can break still-draining callbacks from
        # smaller prepared tasks.
        if suffix_rows > self._max_num_reqs:
            self.set_live_counts(-1, -1)
        else:
            # Decode path: every live suffix row scatters exactly one
            # current-token K/V row. Future multi-token/spec paths should
            # publish scatter separately.
            self.set_live_counts(suffix_rows, suffix_rows)

    def invalidate_common_metadata_cache(self) -> None:
        # CPU block ownership lives in the scheduler/KV manager. The worker only
        # drops cached metadata that may reference old physical CPU block IDs.
        self._common_cache_key = None
        self._common_cache_value = None

    def close(self) -> None:
        self._last_live_counts = None
        self._suffix_attention_runner.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def drain_runtime_stats(self) -> CotsHybridKVStats | None:
        stats = self._runtime_stats
        if self._suffix_counter_stats_enabled:
            self._drain_suffix_runner_counters(stats)
        if not stats.has_worker_activity():
            return None
        self._runtime_stats = CotsHybridKVStats()
        return stats

    def _drain_suffix_runner_counters(self, stats: CotsHybridKVStats) -> None:
        counters = self._suffix_attention_runner.get_counters()
        self._suffix_attention_runner.reset_counters()
        stats.suffix_submit_prepare_ms += (
            counters.get("suffix_submit_prepare_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_submit_metadata_snapshot_ms += (
            counters.get("suffix_submit_metadata_snapshot_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_submit_launch_hostfunc_ms += (
            counters.get("suffix_submit_launch_hostfunc_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_dispatch_cb_ms += (
            counters.get("suffix_dispatch_cb_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_dispatch_snapshot_ms += (
            counters.get("suffix_dispatch_cb_snapshot_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_dispatch_enqueue_ms += (
            counters.get("suffix_dispatch_cb_enqueue_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_worker_busy_ms += (
            counters.get("suffix_worker_busy_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_worker_queue_wait_ms += (
            counters.get("suffix_worker_queue_wait_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_worker_scatter_ms += (
            counters.get("suffix_worker_scatter_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_worker_attention_ms += (
            counters.get("suffix_worker_attention_total_ns", 0) / 1_000_000.0
        )
        stats.suffix_worker_requested_num_threads = max(
            stats.suffix_worker_requested_num_threads,
            counters.get("suffix_worker_requested_num_threads", 0),
        )
        stats.suffix_worker_observed_num_threads = max(
            stats.suffix_worker_observed_num_threads,
            counters.get("suffix_worker_observed_num_threads", 0),
        )
        stats.suffix_wait_kernel_launches += counters.get(
            "suffix_wait_kernel_launch_count", 0
        )
        stats.suffix_wait_kernel_immediate += counters.get(
            "suffix_wait_kernel_immediate_resume_count", 0
        )
        stats.suffix_wait_kernel_lagging += counters.get(
            "suffix_wait_kernel_lagging_wait_count", 0
        )
        stats.suffix_wait_kernel_spin_iters += counters.get(
            "suffix_wait_kernel_spin_iters_total", 0
        )
        stats.suffix_worker_capacity_rows += counters.get(
            "suffix_worker_capacity_rows", 0
        )
        stats.suffix_worker_live_rows += counters.get("suffix_worker_live_rows", 0)
        stats.suffix_worker_padded_rows += counters.get("suffix_worker_padded_rows", 0)
        stats.suffix_worker_scatter_rows += counters.get(
            "suffix_worker_scatter_rows", 0
        )
        stats.suffix_worker_zero_live_count += counters.get(
            "suffix_worker_zero_live_count", 0
        )

    def _synchronize_static_metadata_reuse(self) -> None:
        if not (self.pin_memory and torch.cuda.is_available()):
            return
        if not self._staging_reuse_events:
            return
        wait_start = time.perf_counter()
        for event in self._staging_reuse_events.values():
            event.synchronize()
        self._runtime_stats.cpu_suffix_read_wait_ms += (
            time.perf_counter() - wait_start
        ) * 1000.0

    def _empty_cpu_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=self.pin_memory)

    def _common_decode_cache_key(
        self,
        *,
        req_ids: Sequence[str],
        seq_lens_cpu: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        is_prefilling_cpu: torch.Tensor | None,
        max_query_len: int,
        num_actual_tokens: int,
        cpu_block_ids_by_req: Sequence[tuple[list[int], ...] | None] | None,
    ) -> tuple[object, ...]:
        num_reqs = len(req_ids)
        prefill_key: tuple[bool, ...] | None = None
        if is_prefilling_cpu is not None:
            prefill_key = tuple(bool(v) for v in is_prefilling_cpu[:num_reqs].tolist())
        cpu_blocks_key = None
        if cpu_block_ids_by_req is not None:
            cpu_blocks_key = tuple(
                None
                if req_blocks is None
                else tuple(tuple(group) for group in req_blocks)
                for req_blocks in cpu_block_ids_by_req[:num_reqs]
            )
        return (
            tuple(req_ids),
            tuple(int(v) for v in seq_lens_cpu[:num_reqs].tolist()),
            tuple(int(v) for v in prompt_lens_cpu[:num_reqs].tolist()),
            prefill_key,
            int(max_query_len),
            int(num_actual_tokens),
            cpu_blocks_key,
        )

    def _build_common_decode_metadata(
        self,
        *,
        req_ids: Sequence[str],
        seq_lens_cpu: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        is_prefilling_cpu: torch.Tensor | None,
        max_query_len: int,
        num_actual_tokens: int,
        cpu_block_ids_by_req: Sequence[tuple[list[int], ...] | None] | None,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        del prompt_lens_cpu, is_prefilling_cpu
        num_rows = len(req_ids)
        if max_query_len != 1 or num_actual_tokens != num_rows:
            return None

        seq_lens = seq_lens_cpu[:num_rows]
        if torch.any(seq_lens < self.split_tokens):
            return None

        suffix_lens = torch.clamp(seq_lens - self.split_tokens, min=0).to(torch.int32)
        max_suffix_blocks = max(
            1,
            max(
                cdiv(int(suffix_len.item()), self.block_size)
                for suffix_len in suffix_lens
            ),
        )
        if max_suffix_blocks > self.max_suffix_blocks_per_req:
            raise RuntimeError(
                "COTS hybrid KV suffix exceeds the configured per-request table: "
                f"needed={max_suffix_blocks}, "
                f"max={self.max_suffix_blocks_per_req}"
            )

        if num_rows <= self._max_num_reqs:
            self._synchronize_static_metadata_reuse()
            block_table = self._cpu_block_table[:num_rows]
            cpu_seq_lens = self._cpu_seq_lens[:num_rows]
        else:
            block_table = self._empty_cpu_tensor(
                (num_rows, self.max_suffix_blocks_per_req), dtype=torch.int32
            )
            cpu_seq_lens = self._empty_cpu_tensor((num_rows,), dtype=torch.int32)
        block_table.fill_(0)
        cpu_seq_lens.copy_(suffix_lens)

        for row_index, req_id in enumerate(req_ids):
            target_blocks = cdiv(int(suffix_lens[row_index].item()), self.block_size)
            if cpu_block_ids_by_req is None:
                req_blocks = []
            else:
                req_cpu_block_ids = cpu_block_ids_by_req[row_index]
                req_blocks = [] if req_cpu_block_ids is None else req_cpu_block_ids[0]
            if len(req_blocks) < target_blocks:
                raise RuntimeError(
                    "COTS hybrid KV scheduler/worker CPU block mismatch: "
                    f"request {req_id!r} needs {target_blocks} suffix blocks, "
                    f"got {len(req_blocks)}"
                )
            used_req_blocks = req_blocks[:target_blocks]
            invalid_blocks = [
                int(block_id)
                for block_id in used_req_blocks
                if int(block_id) < 0 or int(block_id) >= self.num_cpu_blocks
            ]
            if invalid_blocks:
                bad_block_id = invalid_blocks[0]
                raise RuntimeError(
                    "COTS hybrid KV scheduler/worker CPU block mismatch: "
                    f"request {req_id!r} has out-of-range CPU suffix block id "
                    f"{bad_block_id}; valid range is [0, {self.num_cpu_blocks})"
                )
            if target_blocks:
                block_table[row_index, :target_blocks] = torch.tensor(
                    used_req_blocks, dtype=torch.int32
                )

        valid = suffix_lens > 0
        valid_indices = (
            torch.nonzero(valid, as_tuple=False).flatten().to(dtype=torch.long)
        )
        scatter_count = int(valid_indices.numel())
        if scatter_count <= self._max_num_reqs:
            scatter_req_indices = self._scatter_req_indices[:scatter_count]
            scatter_block_ids = self._scatter_block_ids[:scatter_count]
            scatter_block_offsets = self._scatter_block_offsets[:scatter_count]
        else:
            scatter_req_indices = self._empty_cpu_tensor(
                (scatter_count,), dtype=torch.long
            )
            scatter_block_ids = self._empty_cpu_tensor(
                (scatter_count,), dtype=torch.long
            )
            scatter_block_offsets = self._empty_cpu_tensor(
                (scatter_count,), dtype=torch.long
            )
        if scatter_count > 0:
            scatter_req_indices.copy_(valid_indices)
            local_positions = suffix_lens[scatter_req_indices] - 1
            scatter_block_cols = (local_positions // self.block_size).to(
                dtype=torch.long
            )
            scatter_block_offsets.copy_(
                (local_positions % self.block_size).to(dtype=torch.long)
            )
            scatter_block_ids.copy_(
                block_table[scatter_req_indices, scatter_block_cols].to(
                    dtype=torch.long
                )
            )

        # CUDA graphs replay padded decode batches. Dynamic row-expanded prefill
        # uses overflow buffers/tasks and is intentionally eager-only for now.
        if num_rows < self._max_num_reqs:
            self._cpu_seq_lens[num_rows:].zero_()
            self._cpu_block_table[num_rows:].zero_()
        if scatter_count < self._max_num_reqs:
            self._scatter_req_indices[scatter_count:].zero_()
            self._scatter_block_ids[scatter_count:].zero_()
            self._scatter_block_offsets[scatter_count:].zero_()

        return (
            block_table,
            cpu_seq_lens,
            scatter_req_indices,
            scatter_block_ids,
            scatter_block_offsets,
        )

    def _prepare_layer_decode_buffers(
        self,
        layer_name: str,
        batch: int,
    ) -> tuple[Any, ...]:
        can_use_static_staging = batch <= self._max_num_reqs
        staging_done_event = None
        suffix_read_event = None
        suffix_attention_snapshot_inputs = True

        if can_use_static_staging:
            slot_start = self._next_staging_slot[layer_name]
            slot_id = slot_start
            if self.pin_memory and torch.cuda.is_available():
                selected_event = None
                for offset in range(self._num_staging_slots):
                    candidate = (slot_start + offset) % self._num_staging_slots
                    candidate_event = self._staging_reuse_events.get(
                        (layer_name, candidate)
                    )
                    if candidate_event is None or candidate_event.query():
                        slot_id = candidate
                        selected_event = candidate_event
                        break
                else:
                    slot_id = slot_start
                    selected_event = self._staging_reuse_events.get(
                        (layer_name, slot_id)
                    )
                    if selected_event is not None:
                        staging_wait_start = time.perf_counter()
                        selected_event.synchronize()
                        self._runtime_stats.cpu_suffix_read_wait_ms += (
                            time.perf_counter() - staging_wait_start
                        ) * 1000.0
                if selected_event is None:
                    selected_event = torch.cuda.Event(blocking=False)
                    self._staging_reuse_events[(layer_name, slot_id)] = selected_event
                staging_done_event = selected_event
                suffix_attention_snapshot_inputs = False
            self._next_staging_slot[layer_name] = (
                slot_id + 1
            ) % self._num_staging_slots

            staged_query_cpu = self._query_staging_slots[layer_name][slot_id][:batch]
            staged_qkv_cpu = self._qkv_staging_slots[layer_name][slot_id][:batch]
            staged_key_cpu = self._key_staging_slots[layer_name][slot_id][:batch]
            staged_value_cpu = self._value_staging_slots[layer_name][slot_id][:batch]
            suffix_out_cpu = self._suffix_out_slots[layer_name][slot_id][:batch]
            suffix_lse_by_batch = self._suffix_lse_slots[layer_name][slot_id]
            suffix_lse_cpu = suffix_lse_by_batch.get(batch)
            if suffix_lse_cpu is None:
                suffix_lse_cpu = torch.empty(
                    (self.num_query_heads, batch),
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                suffix_lse_by_batch[batch] = suffix_lse_cpu
            suffix_task_id = (
                self._suffix_attention_task_base_ids[layer_name]
                + slot_id * self._max_num_reqs
                + batch
                - 1
            )
            query_copy_stream = self._query_copy_streams[layer_name]
            query_ready_event = self._query_ready_events[layer_name]
            kv_copy_stream = self._kv_copy_streams[layer_name]
            kv_ready_event = self._kv_ready_events[layer_name]
            suffix_submit_stream = self._query_copy_streams[layer_name]
            suffix_submit_done_event = self._suffix_submit_done_events[layer_name][
                slot_id
            ]
        else:
            suffix_lse_by_batch = self._suffix_lse_buffers[layer_name]
            suffix_lse_cpu = suffix_lse_by_batch.get(batch)
            if suffix_lse_cpu is None:
                suffix_lse_cpu = torch.empty(
                    (self.num_query_heads, batch),
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                suffix_lse_by_batch[batch] = suffix_lse_cpu
            if (
                self.pin_memory
                and self.enable_suffix_read_events
                and torch.cuda.is_available()
            ):
                suffix_read_event = self._suffix_read_events[layer_name].get(batch)
                if suffix_read_event is None:
                    suffix_read_event = torch.cuda.Event(blocking=False)
                    self._suffix_read_events[layer_name][batch] = suffix_read_event
                else:
                    read_wait_start = time.perf_counter()
                    suffix_read_event.synchronize()
                    self._runtime_stats.cpu_suffix_read_wait_ms += (
                        time.perf_counter() - read_wait_start
                    ) * 1000.0

            # Row-expanded CPU suffix prefill is eager-only in this first pass.
            # Use synchronous D2H staging and a per-layer overflow native task.
            # Unlike captured decode buckets, the overflow task is populated
            # with its exact live row/scatter counts. Clear the graph-style
            # live-count override so it cannot apply to a smaller prepared task.
            self.set_live_counts(-1, -1)
            staged_query_cpu = None
            staged_qkv_cpu = None
            staged_key_cpu = None
            staged_value_cpu = None
            suffix_out_by_batch = self._suffix_out_overflow_buffers[layer_name]
            suffix_out_cpu = suffix_out_by_batch.get(batch)
            if suffix_out_cpu is None:
                suffix_out_cpu = self._empty_cpu_tensor(
                    (batch, self.num_query_heads, self.head_size), dtype=self.dtype
                )
                suffix_out_by_batch[batch] = suffix_out_cpu
            suffix_task_id = self._suffix_attention_overflow_task_ids[layer_name]
            query_copy_stream = None
            query_ready_event = None
            kv_copy_stream = None
            kv_ready_event = None
            suffix_submit_stream = None
            suffix_submit_done_event = None

        return (
            staged_query_cpu,
            staged_qkv_cpu,
            query_copy_stream,
            query_ready_event,
            staged_key_cpu,
            staged_value_cpu,
            kv_copy_stream,
            kv_ready_event,
            suffix_out_cpu,
            suffix_lse_cpu,
            suffix_submit_stream,
            suffix_submit_done_event,
            suffix_read_event,
            staging_done_event,
            suffix_task_id,
            suffix_attention_snapshot_inputs,
        )

    def build_decode_metadata_from_common(
        self,
        *,
        layer_name: str,
        common_metadata: CotsHybridDecodeMetadata,
    ) -> CotsHybridDecodeMetadata:
        if layer_name not in self._key_caches:
            raise KeyError(f"Unknown COTS hybrid KV layer: {layer_name}")
        batch = int(common_metadata.cpu_seq_lens.shape[0])
        (
            staged_query_cpu,
            staged_qkv_cpu,
            query_copy_stream,
            query_ready_event,
            staged_key_cpu,
            staged_value_cpu,
            kv_copy_stream,
            kv_ready_event,
            suffix_out_cpu,
            suffix_lse_cpu,
            suffix_submit_stream,
            suffix_submit_done_event,
            suffix_read_event,
            staging_done_event,
            suffix_task_id,
            suffix_attention_snapshot_inputs,
        ) = self._prepare_layer_decode_buffers(layer_name, batch)
        return CotsHybridDecodeMetadata(
            cpu_key_cache=self._key_caches[layer_name],
            cpu_value_cache=self._value_caches[layer_name],
            cpu_block_table=common_metadata.cpu_block_table,
            cpu_seq_lens=common_metadata.cpu_seq_lens,
            split_blocks=self.split_blocks,
            metrics=self._runtime_stats,
            staged_query_cpu=staged_query_cpu,
            staged_qkv_cpu=staged_qkv_cpu,
            query_copy_stream=query_copy_stream,
            query_ready_event=query_ready_event,
            staged_key_cpu=staged_key_cpu,
            staged_value_cpu=staged_value_cpu,
            kv_copy_stream=kv_copy_stream,
            kv_ready_event=kv_ready_event,
            suffix_out_cpu=suffix_out_cpu,
            suffix_lse_cpu=suffix_lse_cpu,
            suffix_submit_stream=suffix_submit_stream,
            suffix_submit_done_event=suffix_submit_done_event,
            suffix_read_done_event=suffix_read_event,
            staging_done_event=staging_done_event,
            scatter_req_indices=common_metadata.scatter_req_indices,
            scatter_block_ids=common_metadata.scatter_block_ids,
            scatter_block_offsets=common_metadata.scatter_block_offsets,
            req_indices_gpu=common_metadata.req_indices_gpu,
            scatter_source_indices=common_metadata.scatter_source_indices,
            scatter_source_indices_gpu=common_metadata.scatter_source_indices_gpu,
            prefix_source_indices=common_metadata.prefix_source_indices,
            prefix_req_indices_gpu=common_metadata.prefix_req_indices_gpu,
            prefix_seq_lens_cpu=common_metadata.prefix_seq_lens_cpu,
            suffix_attention_runner=self._suffix_attention_runner,
            suffix_attention_task_id=suffix_task_id,
            suffix_attention_snapshot_inputs=suffix_attention_snapshot_inputs,
        )

    def build_decode_metadata(
        self,
        *,
        layer_name: str,
        req_ids: Sequence[str],
        seq_lens_cpu: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        is_prefilling_cpu: torch.Tensor | None,
        max_query_len: int,
        num_actual_tokens: int,
        cpu_block_ids_by_req: Sequence[tuple[list[int], ...] | None] | None = None,
        req_indices_cpu: Sequence[int] | None = None,
        req_indices_gpu: torch.Tensor | None = None,
        positions_cpu: Sequence[int] | None = None,
    ) -> CotsHybridDecodeMetadata | None:
        """Return per-layer CPU suffix metadata, or None before the split."""

        submit_timing = os.environ.get("VLLM_COTS_HYBRID_SUBMIT_TIMING") == "1"
        submit_timing_last = time.perf_counter() if submit_timing else 0.0

        def _mark_submit_timing(bucket: str) -> None:
            nonlocal submit_timing_last
            if not submit_timing:
                return
            now = time.perf_counter()
            self._submit_timing_stats[bucket] = (
                self._submit_timing_stats.get(bucket, 0.0)
                + (now - submit_timing_last) * 1000.0
            )
            submit_timing_last = now

        if submit_timing:
            self._submit_timing_stats["calls"] = (
                self._submit_timing_stats.get("calls", 0.0) + 1.0
            )

        if layer_name not in self._key_caches:
            raise KeyError(f"Unknown COTS hybrid KV layer: {layer_name}")
        if len(req_ids) == 0 or num_actual_tokens <= 0:
            return None

        num_reqs = len(req_ids)
        prefix_source_indices = None
        prefix_req_indices_gpu = None
        prefix_seq_lens_cpu = None
        scatter_source_indices_gpu = None
        if req_indices_cpu is None:
            if num_actual_tokens != num_reqs:
                return None
            token_rows = list(range(num_reqs))
            active_req_indices = token_rows
            active_req_indices_gpu = None
        else:
            all_token_rows = list(range(num_actual_tokens))

            def _row_uses_cpu_suffix(row: int) -> bool:
                if cpu_block_ids_by_req is None:
                    return True
                req_idx = int(req_indices_cpu[row])
                if req_idx < 0 or req_idx >= num_reqs:
                    return False
                req_blocks = cpu_block_ids_by_req[req_idx]
                return req_blocks is not None and any(req_blocks)

            if positions_cpu is None:
                token_rows = [i for i in all_token_rows if _row_uses_cpu_suffix(i)]
                prefix_rows = [i for i in all_token_rows if not _row_uses_cpu_suffix(i)]
            else:
                token_rows = [
                    i
                    for i in all_token_rows
                    if int(positions_cpu[i]) >= self.split_tokens
                    and _row_uses_cpu_suffix(i)
                ]
                prefix_rows = [
                    i
                    for i in all_token_rows
                    if int(positions_cpu[i]) < self.split_tokens
                    or not _row_uses_cpu_suffix(i)
                ]
            active_req_indices = [int(req_indices_cpu[i]) for i in token_rows]
            if any(idx < 0 or idx >= num_reqs for idx in active_req_indices):
                return None
            if prefix_rows:
                prefix_source_indices = torch.tensor(prefix_rows, dtype=torch.long)
                if positions_cpu is None:
                    prefix_seq_lens_values = [
                        int(seq_lens_cpu[int(req_indices_cpu[i])].item())
                        for i in prefix_rows
                    ]
                else:
                    prefix_seq_lens_values = [
                        int(positions_cpu[i]) + 1 for i in prefix_rows
                    ]
                prefix_seq_lens_cpu = torch.tensor(
                    prefix_seq_lens_values,
                    dtype=torch.int32,
                )
                if req_indices_gpu is not None:
                    prefix_rows_gpu = torch.tensor(
                        prefix_rows, dtype=torch.long, device=req_indices_gpu.device
                    )
                    prefix_req_indices_gpu = req_indices_gpu.index_select(
                        0, prefix_rows_gpu
                    )
                else:
                    prefix_req_indices_gpu = torch.tensor(
                        [int(req_indices_cpu[i]) for i in prefix_rows],
                        dtype=torch.long,
                    )
            active_req_indices_gpu = None
            token_rows_gpu = None
            if req_indices_gpu is not None:
                token_rows_gpu = torch.tensor(
                    token_rows, dtype=torch.long, device=req_indices_gpu.device
                )
                active_req_indices_gpu = req_indices_gpu.index_select(0, token_rows_gpu)
            if len(token_rows) == num_actual_tokens and active_req_indices == list(
                range(num_reqs)
            ):
                active_req_indices_gpu = None
        _mark_submit_timing("row_filter_ms")
        if positions_cpu is None and max_query_len != 1:
            return None
        if (
            positions_cpu is None
            and is_prefilling_cpu is not None
            and torch.any(is_prefilling_cpu[:num_reqs])
        ):
            return None
        if not token_rows:
            return None
        _mark_submit_timing("guards_ms")

        active_req_ids = [req_ids[idx] for idx in active_req_indices]
        effective_max_query_len = 1 if positions_cpu is not None else max_query_len
        active_index_tensor = torch.tensor(active_req_indices, dtype=torch.long)
        active_prompt_lens_cpu = prompt_lens_cpu.index_select(0, active_index_tensor)
        if positions_cpu is None:
            active_seq_lens_cpu = seq_lens_cpu.index_select(0, active_index_tensor)
            active_is_prefilling_cpu = (
                None
                if is_prefilling_cpu is None
                else is_prefilling_cpu.index_select(0, active_index_tensor)
            )
        else:
            active_positions_cpu = torch.tensor(
                [int(positions_cpu[i]) for i in token_rows],
                dtype=torch.int32,
            )
            active_seq_lens_cpu = active_positions_cpu + 1
            active_is_prefilling_cpu = active_positions_cpu < active_prompt_lens_cpu
        active_cpu_block_ids_by_req = (
            None
            if cpu_block_ids_by_req is None
            else [cpu_block_ids_by_req[idx] for idx in active_req_indices]
        )
        _mark_submit_timing("active_tensors_ms")

        common_key = self._common_decode_cache_key(
            req_ids=active_req_ids,
            seq_lens_cpu=active_seq_lens_cpu,
            prompt_lens_cpu=active_prompt_lens_cpu,
            is_prefilling_cpu=active_is_prefilling_cpu,
            max_query_len=effective_max_query_len,
            num_actual_tokens=len(active_req_ids),
            cpu_block_ids_by_req=active_cpu_block_ids_by_req,
        )
        _mark_submit_timing("cache_key_ms")
        if common_key == self._common_cache_key:
            common = self._common_cache_value
            if submit_timing:
                self._submit_timing_stats["common_cache_hits"] = (
                    self._submit_timing_stats.get("common_cache_hits", 0.0) + 1.0
                )
        else:
            if submit_timing:
                self._submit_timing_stats["common_cache_misses"] = (
                    self._submit_timing_stats.get("common_cache_misses", 0.0) + 1.0
                )
            common = self._build_common_decode_metadata(
                req_ids=active_req_ids,
                seq_lens_cpu=active_seq_lens_cpu,
                prompt_lens_cpu=active_prompt_lens_cpu,
                is_prefilling_cpu=active_is_prefilling_cpu,
                max_query_len=effective_max_query_len,
                num_actual_tokens=len(active_req_ids),
                cpu_block_ids_by_req=active_cpu_block_ids_by_req,
            )
            self._common_cache_key = common_key
            self._common_cache_value = common
        _mark_submit_timing("common_ms")
        if common is None:
            return None

        (
            block_table,
            cpu_seq_lens,
            scatter_req_indices,
            scatter_block_ids,
            scatter_block_offsets,
        ) = common
        batch = len(active_req_ids)
        (
            staged_query_cpu,
            staged_qkv_cpu,
            query_copy_stream,
            query_ready_event,
            staged_key_cpu,
            staged_value_cpu,
            kv_copy_stream,
            kv_ready_event,
            suffix_out_cpu,
            suffix_lse_cpu,
            suffix_submit_stream,
            suffix_submit_done_event,
            suffix_read_event,
            staging_done_event,
            suffix_task_id,
            suffix_attention_snapshot_inputs,
        ) = self._prepare_layer_decode_buffers(layer_name, batch)
        _mark_submit_timing("staging_ms")
        scatter_source_indices = None
        if not (
            len(token_rows) == num_actual_tokens
            and token_rows == list(range(num_actual_tokens))
        ):
            scatter_source_indices = torch.tensor(token_rows, dtype=torch.long)
            if req_indices_gpu is not None:
                scatter_source_indices_gpu = (
                    token_rows_gpu
                    if token_rows_gpu is not None
                    else torch.tensor(
                        token_rows, dtype=torch.long, device=req_indices_gpu.device
                    )
                )
        metadata = CotsHybridDecodeMetadata(
            cpu_key_cache=self._key_caches[layer_name],
            cpu_value_cache=self._value_caches[layer_name],
            cpu_block_table=block_table,
            cpu_seq_lens=cpu_seq_lens,
            split_blocks=self.split_blocks,
            metrics=self._runtime_stats,
            staged_query_cpu=staged_query_cpu,
            staged_qkv_cpu=staged_qkv_cpu,
            query_copy_stream=query_copy_stream,
            query_ready_event=query_ready_event,
            staged_key_cpu=staged_key_cpu,
            staged_value_cpu=staged_value_cpu,
            kv_copy_stream=kv_copy_stream,
            kv_ready_event=kv_ready_event,
            suffix_out_cpu=suffix_out_cpu,
            suffix_lse_cpu=suffix_lse_cpu,
            suffix_submit_stream=suffix_submit_stream,
            suffix_submit_done_event=suffix_submit_done_event,
            suffix_read_done_event=suffix_read_event,
            staging_done_event=staging_done_event,
            scatter_req_indices=scatter_req_indices,
            scatter_block_ids=scatter_block_ids,
            scatter_block_offsets=scatter_block_offsets,
            req_indices_gpu=active_req_indices_gpu,
            scatter_source_indices=scatter_source_indices,
            scatter_source_indices_gpu=scatter_source_indices_gpu,
            prefix_source_indices=prefix_source_indices,
            prefix_req_indices_gpu=prefix_req_indices_gpu,
            prefix_seq_lens_cpu=prefix_seq_lens_cpu,
            suffix_attention_runner=self._suffix_attention_runner,
            suffix_attention_task_id=suffix_task_id,
            suffix_attention_snapshot_inputs=suffix_attention_snapshot_inputs,
        )
        _mark_submit_timing("metadata_obj_ms")
        return metadata
