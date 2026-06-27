# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared COTS CPU-runtime dispatch boundary for vLLM V1 workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.forward_context import BatchDescriptor
from vllm.model_executor.offloader import ForwardDispatchInfo, get_offloader


class CotsRuntime:
    """Coordinate per-forward dispatch state for COTS weight offload."""

    def on_dispatch(
        self,
        *,
        batch_descriptor: BatchDescriptor,
        num_tokens_unpadded: int,
        trace_context: Mapping[str, Any] | None = None,
    ) -> ForwardDispatchInfo:
        """Publish COTS weight-offload per-forward OOG state."""

        dispatch_info = ForwardDispatchInfo(
            batch_descriptor=batch_descriptor,
            num_tokens_unpadded=int(num_tokens_unpadded),
            trace_context=trace_context,
        )
        get_offloader().on_dispatch(dispatch_info)
        return dispatch_info
