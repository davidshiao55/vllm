# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public facade for the Hybrid thesis offloader."""

from __future__ import annotations

from vllm.model_executor.offloader.hybrid_offloader import HybridOffloader
from vllm.model_executor.offloader.hybrid_operators import (
    HybridOutputSplitLinearOp,
    HybridQKVOp,
    HybridSwiGLUMLPOp,
    HybridWOOp,
    _RaiseOnDirectCall,
)
from vllm.model_executor.offloader.hybrid_runners import (
    NativeHybridWeightRunner,
    NativeWeightSlabSpec,
    _NativeWeightSlabSpecLinear,
    _NativeWeightSlabSpecMlp,
)
from vllm.model_executor.offloader.hybrid_storage import (
    INPUT_SPLIT_AXIS,
    MLP_DOWN_ROLE,
    MLP_GATE_UP_ROLE,
    OUTPUT_SPLIT_AXIS,
    QKV_ROLE,
    WO_ROLE,
    HybridLinearHandle,
    HybridPrefetchBufferPool,
    WeightPrefetchStreamer,
)
from vllm.model_executor.offloader.hybrid_utils import (
    _complement,
    _has_pinned_host_storage,
    _uva_copy_trusted_host_into_gpu,
    uva_copy_into_gpu,
)

__all__ = [
    "HybridOffloader",
    "HybridLinearHandle",
    "HybridPrefetchBufferPool",
    "WeightPrefetchStreamer",
    "OUTPUT_SPLIT_AXIS",
    "INPUT_SPLIT_AXIS",
    "QKV_ROLE",
    "MLP_GATE_UP_ROLE",
    "MLP_DOWN_ROLE",
    "WO_ROLE",
    "NativeHybridWeightRunner",
    "NativeWeightSlabSpec",
    "HybridOutputSplitLinearOp",
    "HybridQKVOp",
    "HybridWOOp",
    "HybridSwiGLUMLPOp",
    "uva_copy_into_gpu",
    "_RaiseOnDirectCall",
    "_NativeWeightSlabSpecLinear",
    "_NativeWeightSlabSpecMlp",
    "_complement",
    "_has_pinned_host_storage",
    "_uva_copy_trusted_host_into_gpu",
]
