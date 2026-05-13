# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.offloader.cots_operators import (
    _assert_prefetch_slot_ready,
)
from vllm.model_executor.offloader.cots_storage import CotsLinearHandle


def _stale_prefetch_handle() -> CotsLinearHandle:
    handle = CotsLinearHandle(
        kind="row",
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


def test_cots_prefetch_slot_check_rejects_stale_owner_in_eager() -> None:
    handle = _stale_prefetch_handle()

    with pytest.raises(AssertionError, match="slot owner mismatch"):
        _assert_prefetch_slot_ready(handle, 1, underfilled_name="slot")


def test_cots_prefetch_slot_check_allows_dynamo_trace() -> None:
    handle = _stale_prefetch_handle()

    def fn(x: torch.Tensor) -> torch.Tensor:
        _assert_prefetch_slot_ready(handle, 1, underfilled_name="slot")
        return x + 1

    out = torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

    assert torch.equal(out, torch.tensor([2.0]))
