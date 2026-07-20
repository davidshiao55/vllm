# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.offloader.hybrid_offloader import (
    _hidden_states_arg_position,
    _resolve_hidden_states_arg,
)
from vllm.model_executor.offloader.hybrid_operators import (
    _assert_prefetch_slot_ready,
)
from vllm.model_executor.offloader.hybrid_storage import HybridLinearHandle


class _QwenStyleDecoderLayer(torch.nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return hidden_states, residual


class _UnnamedHiddenStatesLayer(torch.nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        return activations


def test_hybrid_prefetch_anchor_resolves_qwen_hidden_states() -> None:
    layer = _QwenStyleDecoderLayer()
    positions = torch.arange(4)
    hidden_states = torch.randn(4, 8)
    residual = torch.randn(4, 8)

    position = _hidden_states_arg_position(layer.forward)
    anchor = _resolve_hidden_states_arg(
        (positions, hidden_states, residual), {}, position, layer
    )

    assert position == 1
    assert anchor is hidden_states
    assert anchor is not positions


def test_hybrid_prefetch_anchor_requires_named_hidden_states() -> None:
    layer = _UnnamedHiddenStatesLayer()

    with pytest.raises(ValueError, match="expose a hidden_states argument"):
        _hidden_states_arg_position(layer.forward)


def _stale_prefetch_handle() -> HybridLinearHandle:
    handle = HybridLinearHandle(
        role="mlp_down",
        linear=torch.nn.Linear(2, 2, bias=False),
        qualified_name="test.row",
        in_dim=2,
        out_dim=2,
        n_cpu=1,
        cpu_indices=torch.tensor([0]),
        gpu_indices=torch.tensor([1]),
        dtype=torch.bfloat16,
    )
    handle.slot_idx = 0
    handle.prefetch_owner_in_slot = [None]
    handle.prefetch_available_rows_in_slot = [0]
    return handle


def test_hybrid_prefetch_slot_check_rejects_stale_owner_in_eager() -> None:
    handle = _stale_prefetch_handle()

    with pytest.raises(AssertionError, match="slot owner mismatch"):
        _assert_prefetch_slot_ready(handle, 1, underfilled_name="slot")


def test_hybrid_prefetch_slot_check_allows_dynamo_trace() -> None:
    handle = _stale_prefetch_handle()

    def fn(x: torch.Tensor) -> torch.Tensor:
        _assert_prefetch_slot_ready(handle, 1, underfilled_name="slot")
        return x + 1

    out = torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

    assert torch.equal(out, torch.tensor([2.0]))
