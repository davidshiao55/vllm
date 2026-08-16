# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

hybrid_c = pytest.importorskip("vllm._hybrid_C")


@pytest.fixture(scope="module")
def runner():
    return hybrid_c.HybridWeightTaskRunner()


def _bf16_randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape, dtype=torch.float32) * scale).to(torch.bfloat16)


def test_hybrid_bf16_kernel_isa_is_reported() -> None:
    assert hybrid_c.bf16_kernel_isa() in {"avx2", "avx512"}


@pytest.mark.parametrize("m,k,n", [(1, 31, 17), (2, 32, 64), (5, 33, 67)])
def test_hybrid_bf16_gemms_match_reference(runner, m: int, k: int, n: int):
    torch.manual_seed(20260815 + m)
    x = _bf16_randn(m, k, scale=0.25)
    w_natural = _bf16_randn(n, k, scale=0.25)
    w_transposed = w_natural.t().contiguous()
    natural = torch.empty(m, n, dtype=torch.bfloat16)
    transposed = torch.empty_like(natural)

    runner.run_bf16_gemm_natural_inline(x, w_natural, natural)
    runner.run_bf16_gemm_transposed_inline(x, w_transposed, transposed)
    expected = (x.float() @ w_natural.float().t()).to(torch.bfloat16)

    torch.testing.assert_close(natural, expected, rtol=0.04, atol=0.04)
    torch.testing.assert_close(transposed, expected, rtol=0.04, atol=0.04)
    torch.testing.assert_close(natural, transposed, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("m,h,i,o", [(1, 31, 19, 17), (2, 32, 64, 64), (5, 33, 67, 65)])
def test_hybrid_bf16_mlp_matches_reference(runner, m: int, h: int, i: int, o: int):
    torch.manual_seed(20260831 + m)
    x = _bf16_randn(m, h, scale=0.2)
    gate = _bf16_randn(i, h, scale=1.0 / math.sqrt(h))
    up = _bf16_randn(i, h, scale=1.0 / math.sqrt(h))
    down = _bf16_randn(i, o, scale=1.0 / math.sqrt(i))
    output = torch.empty(m, o, dtype=torch.bfloat16)

    runner.run_bf16_mlp_inline(x, gate, up, down, output)
    gate_ref = (x.float() @ gate.float().t()).to(torch.bfloat16).float()
    up_ref = (x.float() @ up.float().t()).to(torch.bfloat16).float()
    z_ref = (torch.nn.functional.silu(gate_ref) * up_ref).to(torch.bfloat16)
    expected = (z_ref.float() @ down.float()).to(torch.bfloat16)

    torch.testing.assert_close(output, expected, rtol=0.04, atol=0.04)
