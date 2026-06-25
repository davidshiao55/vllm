# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side COTS TP-style head-split KV state."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from cots.snap import gqa_num_cpu_groups

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.cots_head_split_attention import (
    CotsHeadSplitAttentionMetadata,
)


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
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.num_query_heads = int(num_query_heads)
        self.cpu_kv_heads = int(cpu_kv_heads)
        self.gpu_kv_heads = self.num_kv_heads - self.cpu_kv_heads
        self.q_heads_per_kv = self.num_query_heads // self.num_kv_heads
        self.cpu_query_start = self.gpu_kv_heads * self.q_heads_per_kv
        self.cpu_query_heads = self.cpu_kv_heads * self.q_heads_per_kv
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

    def _copy_block_table(
        self,
        block_table: torch.Tensor,
        *,
        num_reqs: int,
    ) -> torch.Tensor:
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
        dst.copy_(block_table[:num_reqs].to(device="cpu", dtype=torch.int32))
        return dst

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

        starts = query_start_loc_cpu[: num_reqs + 1].to(device="cpu")
        final_seq_lens = seq_lens_cpu[:num_reqs].to(device="cpu", dtype=torch.int32)
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
        dst.copy_(seq_lens_cpu[:num_reqs].to(device="cpu", dtype=torch.int32))
        return dst

    def build_metadata(
        self,
        *,
        layer_name: str,
        block_table: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        is_prefilling_cpu: torch.Tensor | None,
        query_start_loc_cpu: torch.Tensor,
        max_query_len: int,
        num_actual_tokens: int,
        num_reqs: int,
    ) -> CotsHeadSplitAttentionMetadata:
        if layer_name not in self._key_caches:
            raise KeyError(f"Unknown COTS head-split KV layer: {layer_name}")
        if num_reqs <= 0:
            raise ValueError("COTS head-split KV metadata requires at least one req")

        is_decode = max_query_len == 1 and num_actual_tokens == num_reqs
        if is_decode and is_prefilling_cpu is not None:
            is_decode = not bool(is_prefilling_cpu[:num_reqs].any().item())

        cpu_block_table = self._copy_block_table(block_table, num_reqs=num_reqs)
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

        return CotsHeadSplitAttentionMetadata(
            cpu_key_cache=self._key_caches[layer_name],
            cpu_value_cache=self._value_caches[layer_name],
            cpu_block_table=cpu_block_table,
            cpu_seq_lens=cpu_seq_lens,
            gpu_kv_heads=self.gpu_kv_heads,
            cpu_kv_heads=self.cpu_kv_heads,
            q_heads_per_kv=self.q_heads_per_kv,
            cpu_query_start=self.cpu_query_start,
            cpu_query_heads=self.cpu_query_heads,
            is_decode=is_decode,
            num_actual_tokens=num_actual_tokens,
            query_cpu=query_cpu,
            output_cpu=output_cpu,
            output_lse_cpu=output_lse_cpu,
            prefill_query_to_seq_cpu=prefill_query_to_seq,
            prefill_seq_lens_cpu=prefill_seq_lens,
        )

    def build_metadata_from_common(
        self,
        *,
        layer_name: str,
        common_metadata: CotsHeadSplitAttentionMetadata,
    ) -> CotsHeadSplitAttentionMetadata:
        if layer_name not in self._key_caches:
            raise KeyError(f"Unknown COTS head-split KV layer: {layer_name}")
        num_reqs = int(common_metadata.cpu_seq_lens.shape[0])
        query_cpu = output_cpu = output_lse_cpu = None
        if common_metadata.is_decode and num_reqs <= self._max_num_reqs:
            query_cpu = self._query_staging[layer_name][:num_reqs]
            output_cpu = self._output_staging[layer_name][:num_reqs]
            output_lse_cpu = self._lse_staging[layer_name][:, :num_reqs]
        return CotsHeadSplitAttentionMetadata(
            cpu_key_cache=self._key_caches[layer_name],
            cpu_value_cache=self._value_caches[layer_name],
            cpu_block_table=common_metadata.cpu_block_table,
            cpu_seq_lens=common_metadata.cpu_seq_lens,
            gpu_kv_heads=self.gpu_kv_heads,
            cpu_kv_heads=self.cpu_kv_heads,
            q_heads_per_kv=self.q_heads_per_kv,
            cpu_query_start=self.cpu_query_start,
            cpu_query_heads=self.cpu_query_heads,
            is_decode=common_metadata.is_decode,
            num_actual_tokens=common_metadata.num_actual_tokens,
            query_cpu=query_cpu,
            output_cpu=output_cpu,
            output_lse_cpu=output_lse_cpu,
            prefill_query_to_seq_cpu=common_metadata.prefill_query_to_seq_cpu,
            prefill_seq_lens_cpu=common_metadata.prefill_seq_lens_cpu,
        )
