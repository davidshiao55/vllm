# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared COTS CPU-runtime dispatch boundary for vLLM V1 workers.

Phase 1 weight offload and Phase 2 hybrid KV both need per-forward
out-of-graph state before model execution. This coordinator keeps that
runner-side boundary unified while preserving feature ownership: the weight
offloader remains responsible for parameter compute/prefetch, and the hybrid KV
store remains responsible for suffix attention live/scatter counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from vllm.forward_context import BatchDescriptor
from vllm.model_executor.offloader import ForwardDispatchInfo, get_offloader

if TYPE_CHECKING:
    from vllm.v1.worker.cots_hybrid_kv import CotsHybridKVStore


class CotsRuntime:
    """Coordinate per-forward dispatch state for active COTS runtimes."""

    def __init__(self) -> None:
        self.hybrid_kv: CotsHybridKVStore | None = None

    def set_hybrid_kv(self, hybrid_kv: CotsHybridKVStore | None) -> None:
        self.hybrid_kv = hybrid_kv

    def on_dispatch(
        self,
        *,
        batch_descriptor: BatchDescriptor,
        num_tokens_unpadded: int,
        positions_cpu: Sequence[int] | None = None,
        positions_have_suffix: bool | None = None,
        trace_context: Mapping[str, Any] | None = None,
    ) -> ForwardDispatchInfo:
        """Publish all COTS per-forward OOG state.

        Weight offload gets the canonical `ForwardDispatchInfo` through the
        existing offloader singleton, including the NoopOffloader case. Hybrid
        KV receives the same dispatch object plus KV-specific token positions
        used to derive live suffix rows.
        """

        dispatch_info = ForwardDispatchInfo(
            batch_descriptor=batch_descriptor,
            num_tokens_unpadded=int(num_tokens_unpadded),
            trace_context=trace_context,
        )
        get_offloader().on_dispatch(dispatch_info)
        if self.hybrid_kv is not None:
            has_suffix = positions_have_suffix
            if has_suffix is None and positions_cpu is not None:
                has_suffix = self.hybrid_kv.positions_have_suffix(
                    positions_cpu, num_tokens_unpadded
                )
            if has_suffix is False and self.hybrid_kv.live_counts_are_zero():
                return dispatch_info
            self.hybrid_kv.on_dispatch(
                dispatch_info,
                positions_cpu=positions_cpu,
            )
        return dispatch_info
