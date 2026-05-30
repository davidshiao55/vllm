# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

pytest.importorskip("vllm._cots_C")

from vllm._custom_ops import (  # noqa: E402
    cots_gqa_bf16_scatter_suffix_kv,
    cots_gqa_bf16_suffix_attention,
)

MODEL_SHAPES = [
    pytest.param(28, 4, 128, id="qwen2.5-7b"),
    pytest.param(32, 8, 128, id="llama3-8b"),
]


def _flatten_blocks(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    # COTS suffix KV layout is [blocks, kv_heads, block_size, head_dim].
    num_kv_heads = cache.shape[1]
    head_dim = cache.shape[3]
    return (
        cache[block_ids]
        .permute(0, 2, 1, 3)
        .reshape(-1, num_kv_heads, head_dim)[:seq_len]
    )


def _reference_gqa_suffix_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = torch.zeros_like(query)
    lse = torch.full(
        (query.size(1), query.size(0)),
        float("-inf"),
        dtype=torch.float32,
    )

    repeats = query.size(1) // key_cache.size(1)
    for seq_idx, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor.item())
        if seq_len == 0:
            continue

        num_blocks = math.ceil(seq_len / key_cache.size(2))
        block_ids = block_table[seq_idx, :num_blocks].long()
        k = _flatten_blocks(key_cache, block_ids, seq_len).float()
        v = _flatten_blocks(value_cache, block_ids, seq_len).float()
        k = torch.repeat_interleave(k, repeats=repeats, dim=1)
        v = torch.repeat_interleave(v, repeats=repeats, dim=1)

        q = query[seq_idx].float() * scale
        logits = torch.einsum("hd,thd->ht", q, k)
        lse[:, seq_idx] = torch.logsumexp(logits, dim=-1)
        attn = torch.softmax(logits, dim=-1)
        outputs[seq_idx] = torch.einsum("ht,thd->hd", attn, v).to(query.dtype)

    return outputs, lse


@pytest.mark.parametrize(("num_q_heads", "num_kv_heads", "head_dim"), MODEL_SHAPES)
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize(
    "seq_lens_list",
    [
        [0, 1, 17, 63],
        [1, 1, 1, 1],
        [128, 128],
        [129, 257, 511],
    ],
)
def test_cots_gqa_suffix_attention_matches_reference(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    seq_lens_list: list[int],
) -> None:
    torch.manual_seed(2026)
    dtype = torch.bfloat16
    batch = len(seq_lens_list)
    max_blocks = max(1, max(math.ceil(x / block_size) for x in seq_lens_list))
    num_blocks = batch * max_blocks

    query = torch.randn(batch, num_q_heads, head_dim, dtype=dtype)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch, max_blocks)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    output = torch.empty_like(query)
    output_lse = torch.empty(num_q_heads, batch, dtype=torch.float32)
    scale = head_dim**-0.5

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

    ref_output, ref_lse = _reference_gqa_suffix_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        scale,
    )

    torch.testing.assert_close(output, ref_output, atol=2.5e-2, rtol=1.5e-2)
    torch.testing.assert_close(output_lse, ref_lse, atol=2.5e-2, rtol=1e-3)


def test_cots_gqa_suffix_attention_rejects_out_of_range_block_id() -> None:
    dtype = torch.bfloat16
    query = torch.randn(1, 28, 128, dtype=dtype)
    key_cache = torch.randn(2, 4, 16, 128, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor([[2]], dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)
    output = torch.empty_like(query)
    output_lse = torch.empty(28, 1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="out-of-range block id"):
        cots_gqa_bf16_suffix_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=128**-0.5,
            output=output,
            output_lse=output_lse,
        )


@pytest.mark.parametrize(("num_q_heads", "num_kv_heads", "head_dim"), MODEL_SHAPES)
def test_cots_gqa_scatter_suffix_kv_matches_index_assignment(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(2029)
    dtype = torch.bfloat16
    batch = 5
    block_size = 16
    num_blocks = 8

    qkv = torch.randn(batch, num_q_heads + 2 * num_kv_heads, head_dim, dtype=dtype)
    key = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]
    value = qkv[:, num_q_heads + num_kv_heads :, :]
    block_ids = torch.tensor([3, 1, 6, 0, 5], dtype=torch.long)
    block_offsets = torch.tensor([0, 7, 15, 2, 11], dtype=torch.long)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()

    ref_key_cache[block_ids, :, block_offsets, :] = key
    ref_value_cache[block_ids, :, block_offsets, :] = value
    cots_gqa_bf16_scatter_suffix_kv(
        key,
        value,
        block_ids,
        block_offsets,
        key_cache,
        value_cache,
    )

    torch.testing.assert_close(key_cache, ref_key_cache, atol=0, rtol=0)
    torch.testing.assert_close(value_cache, ref_value_cache, atol=0, rtol=0)
