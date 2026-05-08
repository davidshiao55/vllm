# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom op for the thesis deferred native-prefetch backend."""

from __future__ import annotations

import torch

from vllm.model_executor.offloader.base import get_offloader
from vllm.utils.torch_utils import direct_register_custom_op


def _start_deferred_prefetch_impl(input_tensor: torch.Tensor) -> None:
    """Start the graph-safe deferred wrap-around prefetch, if enabled.

    The first-decoder hook calls this op instead of reaching into the
    active offloader object directly. That keeps Dynamo/AOT from guard-
    serializing the offloader object graph, which can contain unpicklable
    torch.cuda.Event instances.
    """
    get_offloader()._start_deferred_prefetch()


def _start_deferred_prefetch_fake(input_tensor: torch.Tensor) -> None:
    """Fake implementation for torch.compile tracing."""
    return


def register_prefetch_defer_offloader_ops() -> None:
    """Register custom ops for the deferred prefetch offloader."""
    direct_register_custom_op(
        op_name="start_deferred_prefetch",
        op_func=_start_deferred_prefetch_impl,
        mutates_args=["input_tensor"],
        fake_impl=_start_deferred_prefetch_fake,
    )


register_prefetch_defer_offloader_ops()
