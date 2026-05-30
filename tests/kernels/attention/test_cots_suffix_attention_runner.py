# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

_cots_C = pytest.importorskip("vllm._cots_C")
if not hasattr(_cots_C, "CotsSuffixAttentionTaskRunner"):
    pytest.skip(
        "CotsSuffixAttentionTaskRunner is not built; rebuild vLLM for this test",
        allow_module_level=True,
    )

from vllm._custom_ops import (  # noqa: E402
    cots_gqa_bf16_scatter_suffix_kv,
    cots_gqa_bf16_suffix_attention,
)
from vllm.v1.attention.backends.cots_hybrid_attention import (  # noqa: E402
    NativeCotsSuffixAttentionRunner,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="prepared COTS suffix runner custom ops require a CUDA anchor",
)


MODEL_SHAPES = [
    pytest.param(28, 4, 128, id="qwen2.5-7b"),
    pytest.param(32, 8, 128, id="llama3-8b"),
]


@pytest.mark.parametrize(("num_q_heads", "num_kv_heads", "head_dim"), MODEL_SHAPES)
def test_prepared_native_suffix_runner_matches_direct_kernel(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(2034)
    dtype = torch.bfloat16
    batch = 3
    block_size = 16
    seq_lens = torch.tensor([1, 17, 31], dtype=torch.int32)
    max_blocks = max(math.ceil(int(v) / block_size) for v in seq_lens.tolist())
    num_blocks = batch * max_blocks
    scale = head_dim**-0.5
    total_heads = num_q_heads + 2 * num_kv_heads

    qkv = torch.randn(batch, total_heads, head_dim, dtype=dtype)
    query = qkv[:, :num_q_heads, :]
    assert not query.is_contiguous()
    assert query.stride() == (total_heads * head_dim, head_dim, 1)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch, max_blocks)
    runner_out = torch.empty_like(query)
    runner_lse = torch.empty(num_q_heads, batch, dtype=torch.float32)
    direct_out = torch.empty_like(query)
    direct_lse = torch.empty_like(runner_lse)

    runner = NativeCotsSuffixAttentionRunner(num_tasks=1)
    try:
        assert runner.wait_kernel_sync_installed(0)
        assert runner.wait_kernel_slots(0) == (0, 0)
        runner.run_gqa_bf16_suffix_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=runner_out,
            output_lse=runner_lse,
            cuda_anchor=torch.empty(1, device="cuda"),
            task_id=0,
        )
        torch.accelerator.synchronize()
        assert runner.wait_kernel_slots(0) == (1, 1)
    finally:
        runner.close()

    cots_gqa_bf16_suffix_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        output=direct_out,
        output_lse=direct_lse,
    )

    torch.testing.assert_close(runner_out, direct_out, atol=0, rtol=0)
    torch.testing.assert_close(runner_lse, direct_lse, atol=0, rtol=0)


@pytest.mark.parametrize(("num_q_heads", "num_kv_heads", "head_dim"), MODEL_SHAPES)
def test_prepared_native_suffix_runner_scatters_qkv_before_attention(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(2035)
    dtype = torch.bfloat16
    batch = 3
    block_size = 16
    seq_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
    num_blocks = 5
    scale = head_dim**-0.5
    total_heads = num_q_heads + 2 * num_kv_heads

    qkv = torch.randn(batch, total_heads, head_dim, dtype=dtype)
    query = qkv[:, :num_q_heads, :]
    key_src = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]
    value_src = qkv[:, num_q_heads + num_kv_heads :, :]
    scatter_block_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    scatter_block_offsets = torch.tensor([0, 1, 2], dtype=torch.long)
    block_table = scatter_block_ids.to(torch.int32).reshape(batch, 1)

    runner_key_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
    )
    runner_value_cache = torch.zeros_like(runner_key_cache)
    manual_key_cache = torch.zeros_like(runner_key_cache)
    manual_value_cache = torch.zeros_like(runner_key_cache)
    runner_out = torch.empty_like(query)
    runner_lse = torch.empty(num_q_heads, batch, dtype=torch.float32)
    direct_out = torch.empty_like(query)
    direct_lse = torch.empty_like(runner_lse)

    runner = NativeCotsSuffixAttentionRunner(num_tasks=1)
    try:
        runner.run_gqa_bf16_suffix_attention(
            query=query,
            key_cache=runner_key_cache,
            value_cache=runner_value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=runner_out,
            output_lse=runner_lse,
            cuda_anchor=torch.empty(1, device="cuda"),
            task_id=0,
            scatter_block_ids=scatter_block_ids,
            scatter_block_offsets=scatter_block_offsets,
            scatter_from_qkv=True,
        )
        torch.accelerator.synchronize()
    finally:
        runner.close()

    cots_gqa_bf16_scatter_suffix_kv(
        key_src,
        value_src,
        scatter_block_ids,
        scatter_block_offsets,
        manual_key_cache,
        manual_value_cache,
    )
    cots_gqa_bf16_suffix_attention(
        query=query,
        key_cache=manual_key_cache,
        value_cache=manual_value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        output=direct_out,
        output_lse=direct_lse,
    )

    torch.testing.assert_close(runner_key_cache, manual_key_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_value_cache, manual_value_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_out, direct_out, atol=0, rtol=0)
    torch.testing.assert_close(runner_lse, direct_lse, atol=0, rtol=0)


@pytest.mark.parametrize(("num_q_heads", "num_kv_heads", "head_dim"), MODEL_SHAPES)
def test_prepared_native_suffix_runner_scatters_qkv_two_suffix_blocks(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(2042)
    dtype = torch.bfloat16
    batch = 2
    block_size = 16
    max_blocks = 2
    num_blocks = batch * max_blocks
    seq_lens = torch.tensor([17, 18], dtype=torch.int32)
    scale = head_dim**-0.5
    total_heads = num_q_heads + 2 * num_kv_heads

    qkv = torch.randn(batch, total_heads, head_dim, dtype=dtype)
    query = qkv[:, :num_q_heads, :]
    key_src = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]
    value_src = qkv[:, num_q_heads + num_kv_heads :, :]
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    scatter_block_ids = torch.tensor([1, 3], dtype=torch.long)
    scatter_block_offsets = torch.tensor([0, 1], dtype=torch.long)

    runner_key_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
    )
    runner_value_cache = torch.randn_like(runner_key_cache)
    manual_key_cache = runner_key_cache.clone()
    manual_value_cache = runner_value_cache.clone()
    runner_out = torch.empty_like(query)
    runner_lse = torch.empty(num_q_heads, batch, dtype=torch.float32)
    direct_out = torch.empty_like(query)
    direct_lse = torch.empty_like(runner_lse)

    runner = NativeCotsSuffixAttentionRunner(num_tasks=1)
    try:
        runner.run_gqa_bf16_suffix_attention(
            query=query,
            key_cache=runner_key_cache,
            value_cache=runner_value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=runner_out,
            output_lse=runner_lse,
            cuda_anchor=torch.empty(1, device="cuda"),
            task_id=0,
            scatter_block_ids=scatter_block_ids,
            scatter_block_offsets=scatter_block_offsets,
            scatter_from_qkv=True,
        )
        torch.accelerator.synchronize()
    finally:
        runner.close()

    cots_gqa_bf16_scatter_suffix_kv(
        key_src,
        value_src,
        scatter_block_ids,
        scatter_block_offsets,
        manual_key_cache,
        manual_value_cache,
    )
    cots_gqa_bf16_suffix_attention(
        query=query,
        key_cache=manual_key_cache,
        value_cache=manual_value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        output=direct_out,
        output_lse=direct_lse,
    )

    torch.testing.assert_close(runner_key_cache, manual_key_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_value_cache, manual_value_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_out, direct_out, atol=0, rtol=0)
    torch.testing.assert_close(runner_lse, direct_lse, atol=0, rtol=0)


def test_prepared_native_suffix_runner_cudagraph_replay_uses_updated_qkv() -> None:
    torch.manual_seed(2037)
    dtype = torch.bfloat16
    batch = 2
    block_size = 16
    seq_lens = torch.ones(batch, dtype=torch.int32)
    num_blocks = batch
    scale = 128**-0.5

    qkv = torch.randn(batch, 36, 128, dtype=dtype)
    query = qkv[:, :28, :]
    scatter_block_ids = torch.arange(batch, dtype=torch.long)
    scatter_block_offsets = torch.zeros(batch, dtype=torch.long)
    block_table = scatter_block_ids.to(torch.int32).reshape(batch, 1)
    runner_key_cache = torch.zeros(num_blocks, 4, block_size, 128, dtype=dtype)
    runner_value_cache = torch.zeros_like(runner_key_cache)
    manual_key_cache = torch.zeros_like(runner_key_cache)
    manual_value_cache = torch.zeros_like(runner_key_cache)
    runner_out = torch.empty_like(query)
    runner_lse = torch.empty(28, batch, dtype=torch.float32)
    direct_out = torch.empty_like(query)
    direct_lse = torch.empty_like(runner_lse)
    anchor = torch.empty(1, device="cuda")

    runner = NativeCotsSuffixAttentionRunner(num_tasks=1)
    try:
        runner.run_gqa_bf16_suffix_attention(
            query=query,
            key_cache=runner_key_cache,
            value_cache=runner_value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=runner_out,
            output_lse=runner_lse,
            cuda_anchor=anchor,
            task_id=0,
            scatter_block_ids=scatter_block_ids,
            scatter_block_offsets=scatter_block_offsets,
            scatter_from_qkv=True,
        )
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.run_gqa_bf16_suffix_attention(
                query=query,
                key_cache=runner_key_cache,
                value_cache=runner_value_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                scale=scale,
                output=runner_out,
                output_lse=runner_lse,
                cuda_anchor=anchor,
                task_id=0,
                scatter_block_ids=scatter_block_ids,
                scatter_block_offsets=scatter_block_offsets,
                scatter_from_qkv=True,
            )
        torch.accelerator.synchronize()

        qkv.copy_(torch.randn_like(qkv))
        runner_key_cache.zero_()
        runner_value_cache.zero_()
        graph.replay()
        torch.accelerator.synchronize()
    finally:
        runner.close()

    cots_gqa_bf16_scatter_suffix_kv(
        qkv[:, 28:32, :],
        qkv[:, 32:36, :],
        scatter_block_ids,
        scatter_block_offsets,
        manual_key_cache,
        manual_value_cache,
    )
    cots_gqa_bf16_suffix_attention(
        query=query,
        key_cache=manual_key_cache,
        value_cache=manual_value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        output=direct_out,
        output_lse=direct_lse,
    )

    torch.testing.assert_close(runner_key_cache, manual_key_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_value_cache, manual_value_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_out, direct_out, atol=0, rtol=0)
    torch.testing.assert_close(runner_lse, direct_lse, atol=0, rtol=0)


def test_prepared_native_suffix_runner_cudagraph_replay_uses_updated_separate_kv() -> (
    None
):
    torch.manual_seed(2038)
    dtype = torch.bfloat16
    batch = 2
    block_size = 16
    seq_lens = torch.ones(batch, dtype=torch.int32)
    num_blocks = batch
    scale = 128**-0.5

    query = torch.randn(batch, 28, 128, dtype=dtype)
    key_src = torch.randn(batch, 4, 128, dtype=dtype)
    value_src = torch.randn_like(key_src)
    scatter_block_ids = torch.arange(batch, dtype=torch.long)
    scatter_block_offsets = torch.zeros(batch, dtype=torch.long)
    block_table = scatter_block_ids.to(torch.int32).reshape(batch, 1)
    runner_key_cache = torch.zeros(num_blocks, 4, block_size, 128, dtype=dtype)
    runner_value_cache = torch.zeros_like(runner_key_cache)
    manual_key_cache = torch.zeros_like(runner_key_cache)
    manual_value_cache = torch.zeros_like(runner_key_cache)
    runner_out = torch.empty_like(query)
    runner_lse = torch.empty(28, batch, dtype=torch.float32)
    direct_out = torch.empty_like(query)
    direct_lse = torch.empty_like(runner_lse)
    anchor = torch.empty(1, device="cuda")

    runner = NativeCotsSuffixAttentionRunner(num_tasks=1)
    try:
        runner.run_gqa_bf16_suffix_attention(
            query=query,
            key_cache=runner_key_cache,
            value_cache=runner_value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=runner_out,
            output_lse=runner_lse,
            cuda_anchor=anchor,
            task_id=0,
            scatter_block_ids=scatter_block_ids,
            scatter_block_offsets=scatter_block_offsets,
            scatter_key_cpu=key_src,
            scatter_value_cpu=value_src,
            scatter_from_separate_kv=True,
        )
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.run_gqa_bf16_suffix_attention(
                query=query,
                key_cache=runner_key_cache,
                value_cache=runner_value_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                scale=scale,
                output=runner_out,
                output_lse=runner_lse,
                cuda_anchor=anchor,
                task_id=0,
                scatter_block_ids=scatter_block_ids,
                scatter_block_offsets=scatter_block_offsets,
                scatter_key_cpu=key_src,
                scatter_value_cpu=value_src,
                scatter_from_separate_kv=True,
            )
        torch.accelerator.synchronize()

        key1 = torch.randn_like(key_src)
        value1 = torch.randn_like(value_src)
        query1 = torch.randn_like(query)
        key_src.copy_(key1)
        value_src.copy_(value1)
        query.copy_(query1)
        scatter_block_offsets.fill_(0)
        seq_lens.fill_(1)
        runner_key_cache.zero_()
        runner_value_cache.zero_()
        graph.replay()
        torch.accelerator.synchronize()

        manual_key_cache.zero_()
        manual_value_cache.zero_()
        cots_gqa_bf16_scatter_suffix_kv(
            key1,
            value1,
            scatter_block_ids,
            scatter_block_offsets,
            manual_key_cache,
            manual_value_cache,
        )
        cots_gqa_bf16_suffix_attention(
            query1,
            manual_key_cache,
            manual_value_cache,
            block_table,
            seq_lens,
            scale,
            direct_out,
            direct_lse,
        )
        torch.testing.assert_close(runner_key_cache, manual_key_cache, atol=0, rtol=0)
        torch.testing.assert_close(
            runner_value_cache, manual_value_cache, atol=0, rtol=0
        )
        torch.testing.assert_close(runner_out, direct_out, atol=0, rtol=0)
        torch.testing.assert_close(runner_lse, direct_lse, atol=0, rtol=0)

        key2 = torch.randn_like(key_src)
        value2 = torch.randn_like(value_src)
        query2 = torch.randn_like(query)
        key_src.copy_(key2)
        value_src.copy_(value2)
        query.copy_(query2)
        scatter_block_offsets.fill_(1)
        seq_lens.fill_(2)
        graph.replay()
        torch.accelerator.synchronize()
    finally:
        runner.close()

    cots_gqa_bf16_scatter_suffix_kv(
        key2,
        value2,
        scatter_block_ids,
        scatter_block_offsets,
        manual_key_cache,
        manual_value_cache,
    )
    cots_gqa_bf16_suffix_attention(
        query2,
        manual_key_cache,
        manual_value_cache,
        block_table,
        seq_lens,
        scale,
        direct_out,
        direct_lse,
    )

    torch.testing.assert_close(runner_key_cache, manual_key_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_value_cache, manual_value_cache, atol=0, rtol=0)
    torch.testing.assert_close(runner_out, direct_out, atol=0, rtol=0)
    torch.testing.assert_close(runner_lse, direct_lse, atol=0, rtol=0)


def test_prepared_native_suffix_runner_cudagraph_replay_uses_live_counts() -> None:
    torch.manual_seed(2039)
    dtype = torch.bfloat16
    batch = 4
    live = 2
    block_size = 16
    seq_lens = torch.ones(batch, dtype=torch.int32)
    num_blocks = batch
    scale = 128**-0.5

    query = torch.randn(batch, 28, 128, dtype=dtype)
    key_src = torch.randn(batch, 4, 128, dtype=dtype)
    value_src = torch.randn_like(key_src)
    scatter_block_ids = torch.arange(batch, dtype=torch.long)
    scatter_block_offsets = torch.zeros(batch, dtype=torch.long)
    block_table = scatter_block_ids.to(torch.int32).reshape(batch, 1)
    runner_key_cache = torch.zeros(num_blocks, 4, block_size, 128, dtype=dtype)
    runner_value_cache = torch.zeros_like(runner_key_cache)
    manual_key_cache = torch.zeros_like(runner_key_cache)
    manual_value_cache = torch.zeros_like(runner_key_cache)
    runner_out = torch.empty_like(query)
    runner_lse = torch.empty(28, batch, dtype=torch.float32)
    direct_out = torch.empty(live, 28, 128, dtype=dtype)
    direct_lse = torch.empty(28, live, dtype=torch.float32)
    anchor = torch.empty(1, device="cuda")

    runner = NativeCotsSuffixAttentionRunner(num_tasks=1)
    try:
        runner.run_gqa_bf16_suffix_attention(
            query=query,
            key_cache=runner_key_cache,
            value_cache=runner_value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            output=runner_out,
            output_lse=runner_lse,
            cuda_anchor=anchor,
            task_id=0,
            scatter_block_ids=scatter_block_ids,
            scatter_block_offsets=scatter_block_offsets,
            scatter_key_cpu=key_src,
            scatter_value_cpu=value_src,
            scatter_from_separate_kv=True,
        )
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.run_gqa_bf16_suffix_attention(
                query=query,
                key_cache=runner_key_cache,
                value_cache=runner_value_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                scale=scale,
                output=runner_out,
                output_lse=runner_lse,
                cuda_anchor=anchor,
                task_id=0,
                scatter_block_ids=scatter_block_ids,
                scatter_block_offsets=scatter_block_offsets,
                scatter_key_cpu=key_src,
                scatter_value_cpu=value_src,
                scatter_from_separate_kv=True,
            )
        torch.accelerator.synchronize()

        query1 = torch.randn_like(query)
        key1 = torch.randn_like(key_src)
        value1 = torch.randn_like(value_src)
        query.copy_(query1)
        key_src.copy_(key1)
        value_src.copy_(value1)
        runner_key_cache.zero_()
        runner_value_cache.zero_()
        runner_out.fill_(42)
        runner_lse.fill_(7)
        runner.set_runtime_counts(live, live)
        graph.replay()
        torch.accelerator.synchronize()

        cots_gqa_bf16_scatter_suffix_kv(
            key1[:live],
            value1[:live],
            scatter_block_ids[:live],
            scatter_block_offsets[:live],
            manual_key_cache,
            manual_value_cache,
        )
        cots_gqa_bf16_suffix_attention(
            query1[:live],
            manual_key_cache,
            manual_value_cache,
            block_table[:live],
            seq_lens[:live],
            scale,
            direct_out,
            direct_lse,
        )

        torch.testing.assert_close(runner_key_cache, manual_key_cache, atol=0, rtol=0)
        torch.testing.assert_close(
            runner_value_cache, manual_value_cache, atol=0, rtol=0
        )
        torch.testing.assert_close(runner_out[:live], direct_out, atol=0, rtol=0)
        torch.testing.assert_close(runner_lse[:, :live], direct_lse, atol=0, rtol=0)
        torch.testing.assert_close(
            runner_out[live:],
            torch.zeros_like(runner_out[live:]),
            atol=0,
            rtol=0,
        )
        assert torch.isneginf(runner_lse[:, live:]).all()

        runner_key_cache.fill_(1)
        runner_value_cache.fill_(1)
        runner_out.fill_(42)
        runner_lse.fill_(7)
        runner.set_runtime_counts(0, 0)
        graph.replay()
        torch.accelerator.synchronize()

        torch.testing.assert_close(
            runner_key_cache, torch.ones_like(runner_key_cache), atol=0, rtol=0
        )
        torch.testing.assert_close(
            runner_value_cache, torch.ones_like(runner_value_cache), atol=0, rtol=0
        )
        torch.testing.assert_close(
            runner_out, torch.zeros_like(runner_out), atol=0, rtol=0
        )
        assert torch.isneginf(runner_lse).all()
    finally:
        runner.close()
