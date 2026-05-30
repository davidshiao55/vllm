# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

pytest.importorskip("vllm._cots_C")

from vllm._custom_ops import cots_gqa_bf16_suffix_attention  # noqa: E402
from vllm.v1.attention.backends.cots_hybrid_attention import (  # noqa: E402
    CotsHybridDecodeMetadata,
    cots_hybrid_decode_attention,
    cots_hybrid_kv_cache_update,
    cots_hybrid_stage_query,
)
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version  # noqa: E402
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="COTS hybrid attention composition test requires CUDA merge.",
)


def _flatten_suffix_blocks(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    # COTS suffix KV layout is [blocks, kv_heads, block_size, head_dim].
    return cache[block_ids].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:seq_len]


def _flatten_gpu_prefix_blocks(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    # FlashAttention paged KV layout is [blocks, block_size, kv_heads, head_dim].
    return cache[block_ids].reshape(-1, 4, 128)[:seq_len]


def _decode_attn_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []

    for seq_idx in range(query.size(0)):
        k = key[seq_idx].float()
        v = value[seq_idx].float()
        k = torch.repeat_interleave(k, repeats=7, dim=1)
        v = torch.repeat_interleave(v, repeats=7, dim=1)

        q = query[seq_idx].float() * scale
        logits = torch.einsum("hd,thd->ht", q, k)
        lses.append(torch.logsumexp(logits, dim=-1))
        attn = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", attn, v).to(query.dtype))

    return torch.stack(outputs, dim=0), torch.stack(lses, dim=1).contiguous()


def _decode_attn_full_reference(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_key_cache: torch.Tensor,
    suffix_value_cache: torch.Tensor,
    block_table: torch.Tensor,
    suffix_lens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []

    for seq_idx, suffix_len_tensor in enumerate(suffix_lens):
        suffix_len = int(suffix_len_tensor.item())
        if suffix_len > 0:
            num_suffix_blocks = math.ceil(suffix_len / suffix_key_cache.size(2))
            block_ids = block_table[seq_idx, :num_suffix_blocks].long()
            suffix_key = _flatten_suffix_blocks(suffix_key_cache, block_ids, suffix_len)
            suffix_value = _flatten_suffix_blocks(
                suffix_value_cache, block_ids, suffix_len
            )
            k = torch.cat([prefix_key[seq_idx], suffix_key], dim=0)
            v = torch.cat([prefix_value[seq_idx], suffix_value], dim=0)
        else:
            k = prefix_key[seq_idx]
            v = prefix_value[seq_idx]

        k = torch.repeat_interleave(k.float(), repeats=7, dim=1)
        v = torch.repeat_interleave(v.float(), repeats=7, dim=1)
        q = query[seq_idx].float() * scale
        logits = torch.einsum("hd,thd->ht", q, k)
        lses.append(torch.logsumexp(logits, dim=-1))
        attn = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", attn, v).to(query.dtype))

    return torch.stack(outputs, dim=0), torch.stack(lses, dim=1).contiguous()


class _DirectSuffixRunner:
    kind = "test_direct"

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
    ) -> None:
        del cuda_anchor, task_id, scatter_block_ids, scatter_block_offsets
        del scatter_key_cpu, scatter_value_cpu
        del scatter_from_qkv, scatter_from_separate_kv, snapshot_inputs
        cots_gqa_bf16_suffix_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=output,
            output_lse=output_lse,
        )


@pytest.mark.parametrize(
    ("prefix_len", "suffix_lens_list", "block_size"),
    [
        (64, [0, 1, 17, 63], 16),
        (64, [1, 1, 1, 1], 16),
        (257, [16, 129, 255], 128),
    ],
)
def test_cots_hybrid_attention_merge_matches_full_reference(
    prefix_len: int,
    suffix_lens_list: list[int],
    block_size: int,
) -> None:
    torch.manual_seed(2027)
    dtype = torch.bfloat16
    batch = len(suffix_lens_list)
    max_suffix_blocks = max(
        1, max(math.ceil(suffix_len / block_size) for suffix_len in suffix_lens_list)
    )
    num_suffix_blocks = batch * max_suffix_blocks
    scale = 128**-0.5

    query = torch.randn(batch, 28, 128, dtype=dtype)
    prefix_key = torch.randn(batch, prefix_len, 4, 128, dtype=dtype)
    prefix_value = torch.randn_like(prefix_key)
    suffix_key_cache = torch.randn(num_suffix_blocks, 4, block_size, 128, dtype=dtype)
    suffix_value_cache = torch.randn_like(suffix_key_cache)
    block_table = torch.arange(num_suffix_blocks, dtype=torch.int32).reshape(
        batch, max_suffix_blocks
    )
    suffix_lens = torch.tensor(suffix_lens_list, dtype=torch.int32)

    prefix_out, prefix_lse = _decode_attn_dense(
        query.cuda(),
        prefix_key.cuda(),
        prefix_value.cuda(),
        scale,
    )

    suffix_out_cpu = torch.empty_like(query)
    suffix_lse_cpu = torch.empty(28, batch, dtype=torch.float32)
    cots_gqa_bf16_suffix_attention(
        query=query,
        key_cache=suffix_key_cache,
        value_cache=suffix_value_cache,
        block_table=block_table,
        seq_lens=suffix_lens,
        scale=scale,
        output=suffix_out_cpu,
        output_lse=suffix_lse_cpu,
    )

    hybrid_out = torch.empty_like(prefix_out)
    hybrid_lse = torch.empty_like(prefix_lse)
    merge_attn_states(
        hybrid_out,
        prefix_out.contiguous(),
        prefix_lse.contiguous(),
        suffix_out_cpu.cuda(),
        suffix_lse_cpu.cuda(),
        hybrid_lse,
    )
    torch.accelerator.synchronize()

    ref_out, ref_lse = _decode_attn_full_reference(
        query,
        prefix_key,
        prefix_value,
        suffix_key_cache,
        suffix_value_cache,
        block_table,
        suffix_lens,
        scale,
    )

    torch.testing.assert_close(hybrid_out.cpu(), ref_out, atol=4.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(hybrid_lse.cpu(), ref_lse, atol=3.0e-2, rtol=1.0e-3)


@pytest.mark.parametrize(
    ("prefix_len", "suffix_lens_list", "block_size"),
    [
        (64, [0, 1, 17, 63], 16),
        (64, [1, 1, 1, 1], 16),
        (256, [16, 129, 255], 128),
    ],
)
def test_cots_hybrid_decode_routing_matches_full_reference(
    prefix_len: int,
    suffix_lens_list: list[int],
    block_size: int,
) -> None:
    torch.manual_seed(2028)
    dtype = torch.bfloat16
    batch = len(suffix_lens_list)
    split_blocks = prefix_len // block_size
    assert split_blocks * block_size == prefix_len
    max_suffix_blocks = max(
        1, max(math.ceil(suffix_len / block_size) for suffix_len in suffix_lens_list)
    )
    num_prefix_blocks = batch * split_blocks
    num_suffix_blocks = batch * max_suffix_blocks
    scale = 128**-0.5

    query = torch.randn(batch, 28, 128, dtype=dtype)
    gpu_key_cache = torch.randn(num_prefix_blocks, block_size, 4, 128, dtype=dtype)
    gpu_value_cache = torch.randn_like(gpu_key_cache)
    gpu_block_table = torch.arange(num_prefix_blocks, dtype=torch.int32).reshape(
        batch, split_blocks
    )
    suffix_key_cache = torch.randn(num_suffix_blocks, 4, block_size, 128, dtype=dtype)
    suffix_value_cache = torch.randn_like(suffix_key_cache)
    suffix_block_table = torch.arange(num_suffix_blocks, dtype=torch.int32).reshape(
        batch, max_suffix_blocks
    )
    suffix_lens = torch.tensor(suffix_lens_list, dtype=torch.int32)

    prefix_key = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_key_cache, gpu_block_table[i].long(), prefix_len
            )
            for i in range(batch)
        ],
        dim=0,
    )
    prefix_value = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_value_cache, gpu_block_table[i].long(), prefix_len
            )
            for i in range(batch)
        ],
        dim=0,
    )

    query_gpu = query.cuda()
    hybrid_out = torch.empty_like(query_gpu)
    hybrid_metadata = CotsHybridDecodeMetadata(
        cpu_key_cache=suffix_key_cache,
        cpu_value_cache=suffix_value_cache,
        cpu_block_table=suffix_block_table,
        cpu_seq_lens=suffix_lens,
        split_blocks=split_blocks,
        staged_query_cpu=torch.empty_like(query, device="cpu", pin_memory=True),
        query_copy_stream=torch.cuda.Stream(),
        query_ready_event=torch.cuda.Event(blocking=False),
        suffix_attention_runner=_DirectSuffixRunner(),
    )
    cots_hybrid_stage_query(query_gpu, hybrid_metadata)
    cots_hybrid_decode_attention(
        output=hybrid_out,
        query=query_gpu,
        gpu_key_cache=gpu_key_cache.cuda(),
        gpu_value_cache=gpu_value_cache.cuda(),
        gpu_block_table=gpu_block_table.cuda(),
        hybrid_metadata=hybrid_metadata,
        softmax_scale=scale,
        fa_version=get_flash_attn_version(),
    )
    torch.accelerator.synchronize()

    ref_out, _ = _decode_attn_full_reference(
        query,
        prefix_key,
        prefix_value,
        suffix_key_cache,
        suffix_value_cache,
        suffix_block_table,
        suffix_lens,
        scale,
    )

    torch.testing.assert_close(hybrid_out.cpu(), ref_out, atol=4.0e-2, rtol=2.0e-2)


def test_cots_hybrid_prefill_rows_share_request_prefix_and_match_reference() -> None:
    torch.manual_seed(2031)
    dtype = torch.bfloat16
    q_rows = 4
    prefix_len = 64
    block_size = 16
    split_blocks = prefix_len // block_size
    suffix_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    scale = 128**-0.5

    query = torch.randn(q_rows, 28, 128, dtype=dtype)
    gpu_key_cache = torch.randn(split_blocks, block_size, 4, 128, dtype=dtype)
    gpu_value_cache = torch.randn_like(gpu_key_cache)
    gpu_block_table = torch.arange(split_blocks, dtype=torch.int32).reshape(
        1, split_blocks
    )
    suffix_key_cache = torch.randn(1, 4, block_size, 128, dtype=dtype)
    suffix_value_cache = torch.randn_like(suffix_key_cache)
    suffix_block_table = torch.zeros(q_rows, 1, dtype=torch.int32)

    prefix_key_one = _flatten_gpu_prefix_blocks(
        gpu_key_cache, gpu_block_table[0].long(), prefix_len
    )
    prefix_value_one = _flatten_gpu_prefix_blocks(
        gpu_value_cache, gpu_block_table[0].long(), prefix_len
    )
    prefix_key = prefix_key_one.unsqueeze(0).expand(q_rows, -1, -1, -1)
    prefix_value = prefix_value_one.unsqueeze(0).expand(q_rows, -1, -1, -1)

    query_gpu = query.cuda()
    hybrid_out = torch.empty_like(query_gpu)
    hybrid_metadata = CotsHybridDecodeMetadata(
        cpu_key_cache=suffix_key_cache,
        cpu_value_cache=suffix_value_cache,
        cpu_block_table=suffix_block_table,
        cpu_seq_lens=suffix_lens,
        split_blocks=split_blocks,
        staged_query_cpu=torch.empty_like(query, device="cpu", pin_memory=True),
        query_copy_stream=torch.cuda.Stream(),
        query_ready_event=torch.cuda.Event(blocking=False),
        req_indices_gpu=torch.zeros(q_rows, dtype=torch.long, device="cuda"),
        suffix_attention_runner=_DirectSuffixRunner(),
    )
    cots_hybrid_stage_query(query_gpu, hybrid_metadata)
    cots_hybrid_decode_attention(
        output=hybrid_out,
        query=query_gpu,
        gpu_key_cache=gpu_key_cache.cuda(),
        gpu_value_cache=gpu_value_cache.cuda(),
        gpu_block_table=gpu_block_table.cuda(),
        hybrid_metadata=hybrid_metadata,
        softmax_scale=scale,
        fa_version=get_flash_attn_version(),
    )
    torch.accelerator.synchronize()

    ref_out, _ = _decode_attn_full_reference(
        query,
        prefix_key,
        prefix_value,
        suffix_key_cache,
        suffix_value_cache,
        suffix_block_table,
        suffix_lens,
        scale,
    )

    torch.testing.assert_close(hybrid_out.cpu(), ref_out, atol=4.0e-2, rtol=2.0e-2)


def test_cots_hybrid_decode_overwrites_only_suffix_rows_in_mixed_batch() -> None:
    torch.manual_seed(2030)
    dtype = torch.bfloat16
    total_rows = 3
    active_rows = torch.tensor([0, 2], dtype=torch.long)
    active_batch = active_rows.numel()
    prefix_len = 64
    block_size = 16
    split_blocks = prefix_len // block_size
    suffix_lens = torch.tensor([1, 17], dtype=torch.int32)
    max_suffix_blocks = 2
    scale = 128**-0.5

    query = torch.randn(total_rows, 28, 128, dtype=dtype)
    gpu_key_cache = torch.randn(
        total_rows * split_blocks, block_size, 4, 128, dtype=dtype
    )
    gpu_value_cache = torch.randn_like(gpu_key_cache)
    gpu_block_table = torch.arange(
        total_rows * split_blocks, dtype=torch.int32
    ).reshape(total_rows, split_blocks)
    suffix_key_cache = torch.randn(
        active_batch * max_suffix_blocks, 4, block_size, 128, dtype=dtype
    )
    suffix_value_cache = torch.randn_like(suffix_key_cache)
    suffix_block_table = torch.arange(
        active_batch * max_suffix_blocks, dtype=torch.int32
    ).reshape(active_batch, max_suffix_blocks)

    prefix_key = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_key_cache, gpu_block_table[int(row)].long(), prefix_len
            )
            for row in active_rows
        ],
        dim=0,
    )
    prefix_value = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_value_cache, gpu_block_table[int(row)].long(), prefix_len
            )
            for row in active_rows
        ],
        dim=0,
    )

    query_gpu = query.cuda()
    original_out = torch.randn_like(query_gpu)
    hybrid_out = original_out.clone()
    hybrid_metadata = CotsHybridDecodeMetadata(
        cpu_key_cache=suffix_key_cache,
        cpu_value_cache=suffix_value_cache,
        cpu_block_table=suffix_block_table,
        cpu_seq_lens=suffix_lens,
        split_blocks=split_blocks,
        staged_query_cpu=torch.empty(
            active_batch, 28, 128, dtype=dtype, device="cpu", pin_memory=True
        ),
        query_copy_stream=torch.cuda.Stream(),
        query_ready_event=torch.cuda.Event(blocking=False),
        req_indices_gpu=active_rows.cuda(),
        scatter_source_indices=active_rows,
        suffix_attention_runner=_DirectSuffixRunner(),
    )
    cots_hybrid_stage_query(query_gpu, hybrid_metadata)
    cots_hybrid_decode_attention(
        output=hybrid_out,
        query=query_gpu,
        gpu_key_cache=gpu_key_cache.cuda(),
        gpu_value_cache=gpu_value_cache.cuda(),
        gpu_block_table=gpu_block_table.cuda(),
        hybrid_metadata=hybrid_metadata,
        softmax_scale=scale,
        fa_version=get_flash_attn_version(),
    )
    torch.accelerator.synchronize()

    ref_out, _ = _decode_attn_full_reference(
        query[active_rows],
        prefix_key,
        prefix_value,
        suffix_key_cache,
        suffix_value_cache,
        suffix_block_table,
        suffix_lens,
        scale,
    )

    torch.testing.assert_close(
        hybrid_out[active_rows].cpu(), ref_out, atol=4.0e-2, rtol=2.0e-2
    )
    torch.testing.assert_close(hybrid_out[1], original_out[1], atol=0, rtol=0)


class _PrefixOnlySuffixRunner:
    kind = "test_prefix_only"

    def __init__(self) -> None:
        self.calls = 0
        self.seen_seq_lens: list[int] | None = None

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
    ) -> None:
        del query, key_cache, value_cache, block_table, scale, cuda_anchor, task_id
        del scatter_block_ids, scatter_block_offsets, scatter_key_cpu
        del scatter_value_cpu, scatter_from_qkv, scatter_from_separate_kv
        del snapshot_inputs
        self.calls += 1
        self.seen_seq_lens = [int(v) for v in seq_lens.tolist()]
        output.zero_()
        output_lse.fill_(float("-inf"))


class _RecordingPreparedSuffixRunner:
    kind = "native_prepared"

    def __init__(self) -> None:
        self.calls = 0
        self.scatter_from_qkv = False
        self.scatter_block_ids: torch.Tensor | None = None
        self.scatter_block_offsets: torch.Tensor | None = None
        self.query_stride: tuple[int, ...] | None = None

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
    ) -> None:
        del key_cache, value_cache, block_table, seq_lens, scale, cuda_anchor, task_id
        self.calls += 1
        del scatter_key_cpu, scatter_value_cpu, scatter_from_separate_kv
        del snapshot_inputs
        self.scatter_from_qkv = scatter_from_qkv
        self.scatter_block_ids = scatter_block_ids
        self.scatter_block_offsets = scatter_block_offsets
        self.query_stride = tuple(query.stride())
        output.zero_()
        output_lse.fill_(float("-inf"))


def test_cots_hybrid_decode_uses_metadata_suffix_runner() -> None:
    torch.manual_seed(2033)
    dtype = torch.bfloat16
    batch = 2
    prefix_len = 64
    block_size = 16
    split_blocks = prefix_len // block_size
    suffix_lens = torch.tensor([1, 17], dtype=torch.int32)
    max_suffix_blocks = 2
    scale = 128**-0.5

    query = torch.randn(batch, 28, 128, dtype=dtype)
    gpu_key_cache = torch.randn(batch * split_blocks, block_size, 4, 128, dtype=dtype)
    gpu_value_cache = torch.randn_like(gpu_key_cache)
    gpu_block_table = torch.arange(batch * split_blocks, dtype=torch.int32).reshape(
        batch, split_blocks
    )
    suffix_key_cache = torch.empty(
        batch * max_suffix_blocks, 4, block_size, 128, dtype=dtype
    )
    suffix_value_cache = torch.empty_like(suffix_key_cache)
    suffix_block_table = torch.arange(
        batch * max_suffix_blocks, dtype=torch.int32
    ).reshape(batch, max_suffix_blocks)
    prefix_key = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_key_cache, gpu_block_table[i].long(), prefix_len
            )
            for i in range(batch)
        ],
        dim=0,
    )
    prefix_value = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_value_cache, gpu_block_table[i].long(), prefix_len
            )
            for i in range(batch)
        ],
        dim=0,
    )

    runner = _PrefixOnlySuffixRunner()
    query_gpu = query.cuda()
    hybrid_out = torch.empty_like(query_gpu)
    hybrid_metadata = CotsHybridDecodeMetadata(
        cpu_key_cache=suffix_key_cache,
        cpu_value_cache=suffix_value_cache,
        cpu_block_table=suffix_block_table,
        cpu_seq_lens=suffix_lens,
        split_blocks=split_blocks,
        staged_query_cpu=torch.empty_like(query, device="cpu", pin_memory=True),
        query_copy_stream=torch.cuda.Stream(),
        query_ready_event=torch.cuda.Event(blocking=False),
        suffix_attention_runner=runner,
    )

    cots_hybrid_stage_query(query_gpu, hybrid_metadata)
    cots_hybrid_decode_attention(
        output=hybrid_out,
        query=query_gpu,
        gpu_key_cache=gpu_key_cache.cuda(),
        gpu_value_cache=gpu_value_cache.cuda(),
        gpu_block_table=gpu_block_table.cuda(),
        hybrid_metadata=hybrid_metadata,
        softmax_scale=scale,
        fa_version=get_flash_attn_version(),
    )
    torch.accelerator.synchronize()

    prefix_ref, _ = _decode_attn_dense(
        query_gpu, prefix_key.cuda(), prefix_value.cuda(), scale
    )
    assert runner.calls == 1
    assert runner.seen_seq_lens == [1, 17]
    torch.testing.assert_close(
        hybrid_out.cpu(), prefix_ref.cpu(), atol=4.0e-2, rtol=2.0e-2
    )


def test_cots_hybrid_stage_query_uses_combined_qkv_storage() -> None:
    batch = 4
    qkv_gpu = torch.randn(
        batch,
        36,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_gpu = qkv_gpu[:, :28, :]

    hybrid_metadata = CotsHybridDecodeMetadata(
        cpu_key_cache=torch.empty(1, 4, 16, 128, dtype=torch.bfloat16),
        cpu_value_cache=torch.empty(1, 4, 16, 128, dtype=torch.bfloat16),
        cpu_block_table=torch.zeros(batch, 1, dtype=torch.int32),
        cpu_seq_lens=torch.ones(batch, dtype=torch.int32),
        split_blocks=1,
        staged_query_cpu=torch.empty(
            batch, 28, 128, dtype=torch.bfloat16, pin_memory=True
        ),
        staged_qkv_cpu=torch.empty(
            batch, 36, 128, dtype=torch.bfloat16, pin_memory=True
        ),
        query_copy_stream=torch.cuda.Stream(),
        query_ready_event=torch.cuda.Event(blocking=False),
    )

    cots_hybrid_stage_query(query_gpu, hybrid_metadata)
    assert hybrid_metadata.staged_query_valid
    assert hybrid_metadata.staged_qkv_valid
    assert hybrid_metadata.query_ready_event is not None
    hybrid_metadata.query_ready_event.synchronize()

    torch.testing.assert_close(
        hybrid_metadata.staged_qkv_cpu, qkv_gpu.cpu(), atol=0, rtol=0
    )


def test_cots_hybrid_decode_native_prepared_routes_qkv_scatter_to_runner() -> None:
    torch.manual_seed(2036)
    dtype = torch.bfloat16
    batch = 2
    prefix_len = 64
    block_size = 16
    split_blocks = prefix_len // block_size
    scale = 128**-0.5

    qkv_gpu = torch.randn(batch, 36, 128, dtype=dtype, device="cuda")
    query_gpu = qkv_gpu[:, :28, :]
    gpu_key_cache = torch.randn(batch * split_blocks, block_size, 4, 128, dtype=dtype)
    gpu_value_cache = torch.randn_like(gpu_key_cache)
    gpu_block_table = torch.arange(batch * split_blocks, dtype=torch.int32).reshape(
        batch, split_blocks
    )
    cpu_key_cache = torch.zeros(batch, 4, block_size, 128, dtype=dtype)
    cpu_value_cache = torch.zeros_like(cpu_key_cache)
    suffix_block_table = torch.arange(batch, dtype=torch.int32).reshape(batch, 1)
    suffix_lens = torch.ones(batch, dtype=torch.int32)
    scatter_block_ids = torch.arange(batch, dtype=torch.long)
    scatter_block_offsets = torch.zeros(batch, dtype=torch.long)
    runner = _RecordingPreparedSuffixRunner()

    prefix_key = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_key_cache, gpu_block_table[i].long(), prefix_len
            )
            for i in range(batch)
        ],
        dim=0,
    )
    prefix_value = torch.stack(
        [
            _flatten_gpu_prefix_blocks(
                gpu_value_cache, gpu_block_table[i].long(), prefix_len
            )
            for i in range(batch)
        ],
        dim=0,
    )

    hybrid_metadata = CotsHybridDecodeMetadata(
        cpu_key_cache=cpu_key_cache,
        cpu_value_cache=cpu_value_cache,
        cpu_block_table=suffix_block_table,
        cpu_seq_lens=suffix_lens,
        split_blocks=split_blocks,
        staged_query_cpu=torch.empty(batch, 28, 128, dtype=dtype, pin_memory=True),
        staged_qkv_cpu=torch.empty(batch, 36, 128, dtype=dtype, pin_memory=True),
        query_copy_stream=torch.cuda.Stream(),
        query_ready_event=torch.cuda.Event(blocking=False),
        scatter_block_ids=scatter_block_ids,
        scatter_block_offsets=scatter_block_offsets,
        suffix_attention_runner=runner,
    )

    cots_hybrid_stage_query(query_gpu, hybrid_metadata)
    cots_hybrid_kv_cache_update(
        qkv_gpu[:, 28:32, :], qkv_gpu[:, 32:36, :], hybrid_metadata
    )
    assert hybrid_metadata.staged_kv_valid
    hybrid_out = torch.empty_like(query_gpu)
    cots_hybrid_decode_attention(
        output=hybrid_out,
        query=query_gpu,
        gpu_key_cache=gpu_key_cache.cuda(),
        gpu_value_cache=gpu_value_cache.cuda(),
        gpu_block_table=gpu_block_table.cuda(),
        hybrid_metadata=hybrid_metadata,
        softmax_scale=scale,
        fa_version=get_flash_attn_version(),
    )
    torch.accelerator.synchronize()

    prefix_ref, _ = _decode_attn_dense(
        query_gpu, prefix_key.cuda(), prefix_value.cuda(), scale
    )
    assert runner.calls == 1
    assert runner.scatter_from_qkv
    assert runner.query_stride == (36 * 128, 128, 1)
    assert runner.scatter_block_ids is scatter_block_ids
    assert runner.scatter_block_offsets is scatter_block_offsets
    assert not hybrid_metadata.staged_kv_valid
    assert not hybrid_metadata.staged_qkv_valid
    torch.testing.assert_close(
        cpu_key_cache, torch.zeros_like(cpu_key_cache), atol=0, rtol=0
    )
    torch.testing.assert_close(
        cpu_value_cache, torch.zeros_like(cpu_value_cache), atol=0, rtol=0
    )
    torch.testing.assert_close(
        hybrid_out.cpu(), prefix_ref.cpu(), atol=4.0e-2, rtol=2.0e-2
    )
