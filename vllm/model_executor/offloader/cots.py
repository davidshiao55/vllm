# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public facade for the COTS thesis offloader."""

from __future__ import annotations

from vllm.model_executor.offloader.cots_offloader import CotsOffloader
from vllm.model_executor.offloader.cots_operators import (
    CotsOutputSplitLinearOp,
    CotsQKVOp,
    CotsSwiGLUMLPOp,
    CotsWOOp,
    _RaiseOnDirectCall,
    _scatter_col_outputs_three_way,
)
from vllm.model_executor.offloader.cots_runners import (
    NativeCotsWeightRunner,
    NativeWeightSlabSpec,
    _NativeWeightSlabSpecLinear,
    _NativeWeightSlabSpecMlp,
)
from vllm.model_executor.offloader.cots_storage import (
    INPUT_SPLIT_AXIS,
    MLP_DOWN_ROLE,
    MLP_GATE_UP_ROLE,
    OUTPUT_SPLIT_AXIS,
    QKV_ROLE,
    WO_ROLE,
    CotsLinearHandle,
    CotsPrefetchBufferPool,
    WeightPrefetchStreamer,
)
from vllm.model_executor.offloader.cots_utils import (
    _complement,
    _has_pinned_host_storage,
    _qkv_kv_biased_counts,
    _qkv_kv_biased_indices,
    _uva_copy_trusted_host_into_gpu,
    uva_copy_into_gpu,
)

__all__ = [
    "CotsOffloader",
    "CotsLinearHandle",
    "CotsPrefetchBufferPool",
    "WeightPrefetchStreamer",
    "OUTPUT_SPLIT_AXIS",
    "INPUT_SPLIT_AXIS",
    "QKV_ROLE",
    "MLP_GATE_UP_ROLE",
    "MLP_DOWN_ROLE",
    "WO_ROLE",
    "NativeCotsWeightRunner",
    "NativeWeightSlabSpec",
    "CotsOutputSplitLinearOp",
    "CotsQKVOp",
    "CotsWOOp",
    "CotsSwiGLUMLPOp",
    "uva_copy_into_gpu",
    "_RaiseOnDirectCall",
    "_NativeWeightSlabSpecLinear",
    "_NativeWeightSlabSpecMlp",
    "_scatter_col_outputs_three_way",
    "_complement",
    "_has_pinned_host_storage",
    "_qkv_kv_biased_counts",
    "_qkv_kv_biased_indices",
    "_uva_copy_trusted_host_into_gpu",
]
