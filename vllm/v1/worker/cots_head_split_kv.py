# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side COTS TP-style head-split KV state."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch
from cots.snap import gqa_num_cpu_groups

from vllm.utils.cots_diag import COUNTERS_ENABLED as _COTS_COUNTERS_ENABLED
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.cots_head_split_attention import (
    CotsHeadSplitAttentionMetadata,
    CotsHeadSplitKVPrefetchDescriptor,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


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


class CotsHeadSplitKVStore:
    """CPU KV mirror for the experimental TP-style GQA head split.

    GPU KV pages are sized for the leading GPU-owned groups. CPU KV pages use
    the same logical block ids and store the trailing CPU-owned groups.
    """

    def __init__(
        self,
        *,
        layer_names: Sequence[str],
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        num_query_heads: int,
        head_size: int,
        dtype: torch.dtype,
        f_cpu_kv_store: float,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        pin_memory: bool,
        kv_head_prefetch_enabled: bool = False,
        kv_prefetch_max_active_blocks: int = 0,
        kv_group_plan_by_bucket: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        if not layer_names:
            raise ValueError("COTS head-split KV requires at least one layer")
        if dtype != torch.bfloat16:
            raise ValueError("COTS head-split KV supports only BF16 KV cache")
        if head_size != 128:
            raise ValueError(
                "COTS head-split KV CPU attention currently supports head_dim=128, "
                f"got {head_size}"
            )
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "COTS head-split KV requires GQA-compatible heads: "
                f"num_query_heads={num_query_heads}, num_kv_heads={num_kv_heads}"
            )

        cpu_kv_heads = gqa_num_cpu_groups(f_cpu_kv_store, num_kv_heads=num_kv_heads)
        if cpu_kv_heads <= 0:
            raise ValueError(
                "cots.f_cpu_kv_store snaps to zero CPU GQA groups. Increase the "
                "fraction or disable head-split KV."
            )
        if cpu_kv_heads >= num_kv_heads:
            raise ValueError(
                "COTS head-split KV requires at least one GPU-owned "
                "GQA group. Use f_cpu_kv_store < 1.0."
            )

        self.layer_names = tuple(layer_names)
        self._layer_index_by_name = {
            layer_name: idx for idx, layer_name in enumerate(self.layer_names)
        }
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.num_query_heads = int(num_query_heads)
        self.cpu_kv_heads = int(cpu_kv_heads)
        self.gpu_kv_heads = self.num_kv_heads - self.cpu_kv_heads
        self.q_heads_per_kv = self.num_query_heads // self.num_kv_heads
        self.cpu_query_start = self.gpu_kv_heads * self.q_heads_per_kv
        self.cpu_query_heads = self.cpu_kv_heads * self.q_heads_per_kv
        self.kv_head_prefetch_enabled = bool(kv_head_prefetch_enabled)
        self.kv_prefetch_max_active_blocks = int(kv_prefetch_max_active_blocks)
        self.kv_group_plan_by_bucket = dict(kv_group_plan_by_bucket or {})
        if self.kv_head_prefetch_enabled:
            if self.kv_prefetch_max_active_blocks <= 0:
                raise ValueError(
                    "COTS head-split KV prefetch requires "
                    "kv_prefetch_max_active_blocks > 0"
                )
            if not self.kv_group_plan_by_bucket:
                raise ValueError(
                    "COTS head-split KV prefetch requires a compact "
                    "KV group plan by dispatch bucket"
                )
            for (
                bucket,
                (cpu_compute, prefetch),
            ) in self.kv_group_plan_by_bucket.items():
                if int(cpu_compute) + int(prefetch) != self.cpu_kv_heads:
                    raise ValueError(
                        "COTS head-split KV prefetch plan must cover all "
                        "CPU-owned KV groups: "
                        f"bucket={bucket}, C={cpu_compute}, P={prefetch}, "
                        f"A={self.cpu_kv_heads}"
                    )
        self.head_size = int(head_size)
        self.dtype = dtype
        self.pin_memory = bool(pin_memory)
        self.max_blocks_per_req = cdiv(int(max_model_len), self.block_size)
        self._max_num_reqs = int(max_num_reqs)
        self._max_num_tokens = int(max_num_tokens)

        cache_shape = (
            self.num_blocks,
            self.cpu_kv_heads,
            self.block_size,
            self.head_size,
        )
        self._key_caches: dict[str, torch.Tensor] = {}
        self._value_caches: dict[str, torch.Tensor] = {}
        for layer_name in self.layer_names:
            self._key_caches[layer_name] = torch.empty(
                cache_shape, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self._value_caches[layer_name] = torch.empty(
                cache_shape, dtype=dtype, device="cpu", pin_memory=pin_memory
            )

        self._cpu_block_table = torch.empty(
            (self._max_num_reqs, self.max_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_seq_lens = torch.empty(
            (self._max_num_reqs,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_slot_mapping = torch.empty(
            (self._max_num_tokens,),
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._query_staging: dict[str, torch.Tensor] = {}
        self._output_staging: dict[str, torch.Tensor] = {}
        self._lse_staging: dict[str, torch.Tensor] = {}
        self._prefill_query_to_seq = torch.empty(
            (self._max_num_tokens,),
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._prefill_seq_lens = torch.empty(
            (self._max_num_tokens,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._prefetch_block_positions = torch.arange(
            self.max_blocks_per_req,
            dtype=torch.long,
            device="cpu",
        )
        self._prefetch_active_mask = torch.empty(
            (self._max_num_reqs, self.max_blocks_per_req),
            dtype=torch.bool,
            device="cpu",
            pin_memory=pin_memory,
        )
        compact_rows = max(1, self._max_num_reqs)
        compact_cols = max(1, self.max_blocks_per_req)
        self._prefetch_compact_block_table = torch.empty(
            (compact_rows, compact_cols),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        source_capacity = max(1, self.kv_prefetch_max_active_blocks)
        self._prefetch_source_block_ids = torch.empty(
            (source_capacity,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._prefetch_destination_block_ids = torch.arange(
            source_capacity,
            dtype=torch.int32,
            device="cpu",
        )
        for layer_name in self.layer_names:
            self._query_staging[layer_name] = torch.empty(
                (self._max_num_reqs, self.cpu_query_heads, self.head_size),
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._output_staging[layer_name] = torch.empty(
                (self._max_num_reqs, self.cpu_query_heads, self.head_size),
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._lse_staging[layer_name] = torch.empty(
                (self.cpu_query_heads, self._max_num_reqs),
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            )

    def layer_index(self, layer_name: str) -> int:
        try:
            return self._layer_index_by_name[layer_name]
        except KeyError as exc:
            raise KeyError(f"Unknown COTS head-split KV layer: {layer_name}") from exc

    @staticmethod
    def _cpu_tensor(
        value: torch.Tensor | np.ndarray | Sequence[int],
        *,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise RuntimeError(
                    f"COTS head-split {name} must be CPU-native; got {value.device}"
                )
            return value.to(dtype=dtype) if value.dtype != dtype else value
        return torch.as_tensor(value, dtype=dtype, device="cpu")

    def _copy_cpu_block_table(
        self,
        block_table_cpu: torch.Tensor | np.ndarray,
        *,
        num_reqs: int,
    ) -> torch.Tensor:
        block_table = self._cpu_tensor(
            block_table_cpu, dtype=torch.int32, name="block table"
        )
        max_blocks = int(block_table.shape[1])
        if max_blocks > self.max_blocks_per_req:
            raise RuntimeError(
                "COTS head-split KV block table exceeds CPU metadata capacity: "
                f"needed={max_blocks}, max={self.max_blocks_per_req}"
            )
        if num_reqs <= self._max_num_reqs:
            dst = self._cpu_block_table[:num_reqs, :max_blocks]
        else:
            dst = torch.empty(
                (num_reqs, max_blocks),
                dtype=torch.int32,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        dst.copy_(block_table[:num_reqs])
        return dst

    def _build_cpu_slot_mapping(
        self,
        *,
        block_table_cpu: torch.Tensor,
        query_start_loc_cpu: torch.Tensor | np.ndarray,
        positions_cpu: torch.Tensor | np.ndarray | Sequence[int],
        num_reqs: int,
        num_actual_tokens: int,
        total_cp_world_size: int = 1,
        total_cp_rank: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> torch.Tensor:
        if num_actual_tokens <= self._max_num_tokens:
            slot_mapping = self._cpu_slot_mapping[:num_actual_tokens]
        else:
            slot_mapping = torch.empty(
                (num_actual_tokens,),
                dtype=torch.long,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        slot_mapping.fill_(PAD_SLOT_ID)
        if num_actual_tokens <= 0:
            return slot_mapping

        starts = self._cpu_tensor(
            query_start_loc_cpu, dtype=torch.long, name="query_start_loc"
        )[: num_reqs + 1]
        positions = self._cpu_tensor(positions_cpu, dtype=torch.long, name="positions")[
            :num_actual_tokens
        ]
        if int(positions.numel()) < num_actual_tokens:
            raise RuntimeError(
                "COTS head-split CPU positions are shorter than active tokens: "
                f"got={int(positions.numel())}, needed={num_actual_tokens}"
            )

        total_cp_world_size = int(total_cp_world_size)
        total_cp_rank = int(total_cp_rank)
        cp_kv_cache_interleave_size = int(cp_kv_cache_interleave_size)
        if total_cp_world_size <= 0 or cp_kv_cache_interleave_size <= 0:
            raise RuntimeError(
                "COTS head-split CPU slot mapping received invalid CP geometry: "
                f"world={total_cp_world_size}, "
                f"interleave={cp_kv_cache_interleave_size}"
            )
        virtual_block_size = self.block_size * total_cp_world_size

        for req_idx in range(num_reqs):
            start = int(starts[req_idx].item())
            end = int(starts[req_idx + 1].item())
            if end <= start:
                continue
            if start < 0 or end > num_actual_tokens:
                raise RuntimeError(
                    "COTS head-split CPU query_start_loc is outside token range: "
                    f"req={req_idx}, start={start}, end={end}, "
                    f"num_actual_tokens={num_actual_tokens}"
                )
            pos = positions[start:end]
            block_indices = torch.div(
                pos, virtual_block_size, rounding_mode="floor"
            ).to(dtype=torch.long)
            block_numbers = block_table_cpu[req_idx].index_select(0, block_indices)
            block_numbers = block_numbers.to(dtype=torch.long)

            virtual_offsets = pos - block_indices * virtual_block_size
            is_local = (
                torch.div(
                    virtual_offsets,
                    cp_kv_cache_interleave_size,
                    rounding_mode="floor",
                )
                % total_cp_world_size
            ) == total_cp_rank
            local_offsets = torch.div(
                virtual_offsets,
                total_cp_world_size * cp_kv_cache_interleave_size,
                rounding_mode="floor",
            ) * cp_kv_cache_interleave_size + torch.remainder(
                virtual_offsets, cp_kv_cache_interleave_size
            )
            slot_ids = block_numbers * self.block_size + local_offsets
            slot_mapping[start:end] = torch.where(
                is_local,
                slot_ids,
                torch.full_like(slot_ids, PAD_SLOT_ID),
            )
        return slot_mapping

    def _build_prefill_token_metadata(
        self,
        *,
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        num_reqs: int,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_actual_tokens <= self._max_num_tokens:
            query_to_seq = self._prefill_query_to_seq[:num_actual_tokens]
            per_token_seq_lens = self._prefill_seq_lens[:num_actual_tokens]
        else:
            query_to_seq = torch.empty(
                (num_actual_tokens,),
                dtype=torch.long,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            per_token_seq_lens = torch.empty(
                (num_actual_tokens,),
                dtype=torch.int32,
                device="cpu",
                pin_memory=self.pin_memory,
            )

        starts = self._cpu_tensor(
            query_start_loc_cpu, dtype=torch.long, name="query_start_loc"
        )[: num_reqs + 1]
        final_seq_lens = self._cpu_tensor(
            seq_lens_cpu, dtype=torch.int32, name="seq_lens"
        )[:num_reqs]
        for req_idx in range(num_reqs):
            start = int(starts[req_idx].item())
            end = int(starts[req_idx + 1].item())
            if end <= start:
                continue
            q_len = end - start
            final_seq_len = int(final_seq_lens[req_idx].item())
            computed_len = final_seq_len - q_len
            if computed_len < 0:
                raise RuntimeError(
                    "COTS head-split prefill metadata has negative computed "
                    f"length for request {req_idx}: final_seq_len={final_seq_len}, "
                    f"query_len={q_len}"
                )
            query_to_seq[start:end].fill_(req_idx)
            per_token_seq_lens[start:end].copy_(
                torch.arange(
                    computed_len + 1,
                    final_seq_len + 1,
                    dtype=torch.int32,
                    device="cpu",
                )
            )
        return query_to_seq, per_token_seq_lens

    def _copy_seq_lens(
        self,
        seq_lens_cpu: torch.Tensor,
        *,
        num_reqs: int,
    ) -> torch.Tensor:
        if num_reqs <= self._max_num_reqs:
            dst = self._cpu_seq_lens[:num_reqs]
        else:
            dst = torch.empty(
                (num_reqs,),
                dtype=torch.int32,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        seq_lens = self._cpu_tensor(seq_lens_cpu, dtype=torch.int32, name="seq_lens")
        dst.copy_(seq_lens[:num_reqs])
        return dst

    def build_metadata(
        self,
        *,
        layer_name: str,
        block_table_cpu: torch.Tensor | np.ndarray,
        seq_lens_cpu: torch.Tensor,
        is_prefilling_cpu: torch.Tensor | None,
        query_start_loc_cpu: torch.Tensor,
        positions_cpu: torch.Tensor | np.ndarray | Sequence[int],
        max_query_len: int,
        num_actual_tokens: int,
        num_reqs: int,
        total_cp_world_size: int = 1,
        total_cp_rank: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> CotsHeadSplitAttentionMetadata:
        timer_start = _timer_start()
        if layer_name not in self._key_caches:
            raise KeyError(f"Unknown COTS head-split KV layer: {layer_name}")
        if num_reqs <= 0:
            raise ValueError("COTS head-split KV metadata requires at least one req")

        is_decode = max_query_len == 1 and num_actual_tokens == num_reqs
        if is_decode and is_prefilling_cpu is not None:
            is_decode = not bool(is_prefilling_cpu[:num_reqs].any().item())

        cpu_block_table = self._copy_cpu_block_table(block_table_cpu, num_reqs=num_reqs)
        cpu_slot_mapping = self._build_cpu_slot_mapping(
            block_table_cpu=cpu_block_table,
            query_start_loc_cpu=query_start_loc_cpu,
            positions_cpu=positions_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            total_cp_world_size=total_cp_world_size,
            total_cp_rank=total_cp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        cpu_seq_lens = self._copy_seq_lens(seq_lens_cpu, num_reqs=num_reqs)

        query_cpu = output_cpu = output_lse_cpu = None
        prefill_query_to_seq = None
        prefill_seq_lens = None
        if is_decode and num_reqs <= self._max_num_reqs:
            query_cpu = self._query_staging[layer_name][:num_reqs]
            output_cpu = self._output_staging[layer_name][:num_reqs]
            output_lse_cpu = self._lse_staging[layer_name][:, :num_reqs]
        elif not is_decode:
            prefill_query_to_seq, prefill_seq_lens = self._build_prefill_token_metadata(
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens_cpu=seq_lens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_actual_tokens,
            )

        group_plan = self._metadata_group_plan_kwargs(num_actual_tokens)
        kv_prefetch = self._build_kv_prefetch_descriptor(
            layer_name=layer_name,
            cpu_block_table=cpu_block_table,
            cpu_seq_lens=cpu_seq_lens,
            group_plan=group_plan,
        )

        metadata = CotsHeadSplitAttentionMetadata(
            cpu_key_cache=self._key_caches[layer_name],
            cpu_value_cache=self._value_caches[layer_name],
            cpu_block_table=cpu_block_table,
            cpu_slot_mapping=cpu_slot_mapping,
            cpu_seq_lens=cpu_seq_lens,
            gpu_kv_heads=self.gpu_kv_heads,
            cpu_kv_heads=self.cpu_kv_heads,
            q_heads_per_kv=self.q_heads_per_kv,
            cpu_query_start=self.cpu_query_start,
            cpu_query_heads=self.cpu_query_heads,
            is_decode=is_decode,
            num_actual_tokens=num_actual_tokens,
            layer_idx=self.layer_index(layer_name),
            **group_plan,
            query_cpu=query_cpu,
            output_cpu=output_cpu,
            output_lse_cpu=output_lse_cpu,
            prefill_query_to_seq_cpu=prefill_query_to_seq,
            prefill_seq_lens_cpu=prefill_seq_lens,
            kv_prefetch=kv_prefetch,
        )
        if _COTS_COUNTERS_ENABLED:
            _counter("head_split_metadata_common_tokens", num_actual_tokens)
            _counter("head_split_metadata_common_reqs", num_reqs)
            _counter("head_split_metadata_common_decode_calls", 1 if is_decode else 0)
            _counter("head_split_metadata_common_prefill_calls", 0 if is_decode else 1)
        _timing("head_split_metadata_common", timer_start)
        return metadata

    def _default_kv_group_plan(self) -> tuple[int, int]:
        return self.cpu_kv_heads, 0

    def _kv_group_plan_for_tokens(self, num_actual_tokens: int) -> tuple[int, int]:
        if not self.kv_head_prefetch_enabled:
            return self._default_kv_group_plan()
        for bucket in sorted(self.kv_group_plan_by_bucket):
            if int(num_actual_tokens) <= int(bucket):
                return self.kv_group_plan_by_bucket[bucket]
        return self.kv_group_plan_by_bucket[max(self.kv_group_plan_by_bucket)]

    def _metadata_group_plan_kwargs(self, num_actual_tokens: int) -> dict[str, int]:
        cpu_compute_kv_heads, prefetch_kv_heads = self._kv_group_plan_for_tokens(
            num_actual_tokens
        )
        if int(cpu_compute_kv_heads) + int(prefetch_kv_heads) != self.cpu_kv_heads:
            raise RuntimeError(
                "COTS head-split KV group plan must cover all CPU-owned "
                f"KV heads: C={cpu_compute_kv_heads}, P={prefetch_kv_heads}, "
                f"A={self.cpu_kv_heads}"
            )
        prefetch_query_heads = prefetch_kv_heads * self.q_heads_per_kv
        cpu_compute_query_heads = cpu_compute_kv_heads * self.q_heads_per_kv
        return {
            "prefetch_kv_heads": prefetch_kv_heads,
            "cpu_compute_kv_heads": cpu_compute_kv_heads,
            "prefetch_query_start": self.cpu_query_start,
            "prefetch_query_heads": prefetch_query_heads,
            "cpu_compute_query_start": self.cpu_query_start + prefetch_query_heads,
            "cpu_compute_query_heads": cpu_compute_query_heads,
            "prefetch_cpu_kv_start": 0,
            "cpu_compute_cpu_kv_start": prefetch_kv_heads,
        }

    def _active_block_ids(
        self,
        *,
        cpu_block_table: torch.Tensor,
        cpu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timer_start = _timer_start()
        num_reqs = int(cpu_seq_lens.shape[0])
        max_blocks = int(cpu_block_table.shape[1])
        blocks_per_req = torch.div(
            cpu_seq_lens.to(dtype=torch.long) + self.block_size - 1,
            self.block_size,
            rounding_mode="floor",
        )
        max_needed = int(blocks_per_req.max().item()) if num_reqs > 0 else 0
        if max_needed > max_blocks:
            raise RuntimeError(
                "COTS head-split KV prefetch descriptor exceeds block table "
                f"width: needed={max_needed}, available={max_blocks}"
            )

        if num_reqs <= self._max_num_reqs and max_blocks <= self.max_blocks_per_req:
            active_mask = self._prefetch_active_mask[:num_reqs, :max_blocks]
        else:
            active_mask = torch.empty(
                (num_reqs, max_blocks), dtype=torch.bool, device="cpu"
            )
        active_mask.copy_(
            self._prefetch_block_positions[:max_blocks].unsqueeze(0)
            < blocks_per_req.unsqueeze(1)
        )
        active = cpu_block_table[:num_reqs, :max_blocks][active_mask]
        if active.numel() == 0:
            _timing("head_split_kv_prefetch_active_blocks", timer_start)
            return (
                self._prefetch_source_block_ids[:0],
                self._prefetch_destination_block_ids[:0],
                active_mask,
            )

        unique_blocks = torch.unique(active.to(dtype=torch.int32), sorted=True)
        num_active = int(unique_blocks.numel())
        if num_active > self.kv_prefetch_max_active_blocks:
            raise RuntimeError(
                "COTS head-split KV prefetch active block set exceeds workload "
                "contract: "
                f"active_blocks={num_active}, "
                f"max_active_blocks={self.kv_prefetch_max_active_blocks}"
            )
        source_block_ids = self._prefetch_source_block_ids[:num_active]
        source_block_ids.copy_(unique_blocks)
        destination_block_ids = self._prefetch_destination_block_ids[:num_active]
        if _COTS_COUNTERS_ENABLED:
            _counter("head_split_kv_prefetch_active_blocks", num_active)
        _timing("head_split_kv_prefetch_active_blocks", timer_start)
        return source_block_ids, destination_block_ids, active_mask

    def _build_kv_prefetch_descriptor(
        self,
        *,
        layer_name: str,
        cpu_block_table: torch.Tensor,
        cpu_seq_lens: torch.Tensor,
        group_plan: dict[str, int],
    ) -> CotsHeadSplitKVPrefetchDescriptor | None:
        prefetch_kv_heads = int(group_plan["prefetch_kv_heads"])
        if not self.kv_head_prefetch_enabled or prefetch_kv_heads <= 0:
            return None

        timer_start = _timer_start()
        num_reqs = int(cpu_seq_lens.shape[0])
        max_blocks = int(cpu_block_table.shape[1])
        source_block_ids, destination_block_ids, active_mask = self._active_block_ids(
            cpu_block_table=cpu_block_table,
            cpu_seq_lens=cpu_seq_lens,
        )
        num_active = int(source_block_ids.shape[0])
        if num_reqs <= self._max_num_reqs and max_blocks <= self.max_blocks_per_req:
            compact_block_table = self._prefetch_compact_block_table[
                :num_reqs, :max_blocks
            ]
        else:
            compact_block_table = torch.empty(
                (num_reqs, max_blocks), dtype=torch.int32, device="cpu"
            )
        compact_block_table.fill_(0)
        if num_active > 0:
            source_long = source_block_ids.to(dtype=torch.long)
            table_long = cpu_block_table[:num_reqs, :max_blocks].to(dtype=torch.long)
            compact_all = torch.searchsorted(source_long, table_long)
            compact_block_table[active_mask] = compact_all[active_mask].to(
                dtype=torch.int32
            )

        descriptor = CotsHeadSplitKVPrefetchDescriptor(
            source_key_cache=self._key_caches[layer_name],
            source_value_cache=self._value_caches[layer_name],
            source_block_ids=source_block_ids,
            destination_block_ids=destination_block_ids,
            compact_block_table=compact_block_table,
            layer_idx=self.layer_index(layer_name),
            num_active_blocks=num_active,
            max_active_blocks=self.kv_prefetch_max_active_blocks,
            block_size=self.block_size,
            prefetch_cpu_kv_start=int(group_plan["prefetch_cpu_kv_start"]),
            prefetch_kv_heads=prefetch_kv_heads,
        )
        if _COTS_COUNTERS_ENABLED:
            element_bytes = int(torch.empty((), dtype=self.dtype).element_size())
            prefetch_bytes = (
                num_active
                * self.block_size
                * prefetch_kv_heads
                * self.head_size
                * 2
                * element_bytes
            )
            _counter("head_split_kv_prefetch_descriptor_blocks", num_active)
            _counter("head_split_kv_prefetch_descriptor_bytes", prefetch_bytes)
        _timing("head_split_kv_prefetch_descriptor", timer_start)
        return descriptor

    def build_metadata_from_common(
        self,
        *,
        layer_name: str,
        common_metadata: CotsHeadSplitAttentionMetadata,
    ) -> CotsHeadSplitAttentionMetadata:
        timer_start = _timer_start()
        if layer_name not in self._key_caches:
            raise KeyError(f"Unknown COTS head-split KV layer: {layer_name}")
        num_reqs = int(common_metadata.cpu_seq_lens.shape[0])
        query_cpu = output_cpu = output_lse_cpu = None
        if common_metadata.is_decode and num_reqs <= self._max_num_reqs:
            query_cpu = self._query_staging[layer_name][:num_reqs]
            output_cpu = self._output_staging[layer_name][:num_reqs]
            output_lse_cpu = self._lse_staging[layer_name][:, :num_reqs]
        metadata = CotsHeadSplitAttentionMetadata(
            cpu_key_cache=self._key_caches[layer_name],
            cpu_value_cache=self._value_caches[layer_name],
            cpu_block_table=common_metadata.cpu_block_table,
            cpu_slot_mapping=common_metadata.cpu_slot_mapping,
            cpu_seq_lens=common_metadata.cpu_seq_lens,
            gpu_kv_heads=self.gpu_kv_heads,
            cpu_kv_heads=self.cpu_kv_heads,
            q_heads_per_kv=self.q_heads_per_kv,
            cpu_query_start=self.cpu_query_start,
            cpu_query_heads=self.cpu_query_heads,
            is_decode=common_metadata.is_decode,
            num_actual_tokens=common_metadata.num_actual_tokens,
            layer_idx=self.layer_index(layer_name),
            prefetch_kv_heads=common_metadata.prefetch_kv_heads,
            cpu_compute_kv_heads=common_metadata.cpu_compute_kv_heads,
            prefetch_query_start=common_metadata.prefetch_query_start,
            prefetch_query_heads=common_metadata.prefetch_query_heads,
            cpu_compute_query_start=common_metadata.cpu_compute_query_start,
            cpu_compute_query_heads=common_metadata.cpu_compute_query_heads,
            prefetch_cpu_kv_start=common_metadata.prefetch_cpu_kv_start,
            cpu_compute_cpu_kv_start=common_metadata.cpu_compute_cpu_kv_start,
            query_cpu=query_cpu,
            output_cpu=output_cpu,
            output_lse_cpu=output_lse_cpu,
            prefill_query_to_seq_cpu=common_metadata.prefill_query_to_seq_cpu,
            prefill_seq_lens_cpu=common_metadata.prefill_seq_lens_cpu,
            kv_prefetch=(
                None
                if common_metadata.kv_prefetch is None
                else replace(
                    common_metadata.kv_prefetch,
                    layer_idx=self.layer_index(layer_name),
                    source_key_cache=self._key_caches[layer_name],
                    source_value_cache=self._value_caches[layer_name],
                )
            ),
        )
        if _COTS_COUNTERS_ENABLED:
            _counter("head_split_metadata_layer_tokens", metadata.num_actual_tokens)
            _counter("head_split_metadata_layer_layers")
        _timing("head_split_metadata_layer", timer_start)
        return metadata
