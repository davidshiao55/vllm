# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public facade for the COTS thesis offloader."""

from __future__ import annotations

from vllm.model_executor.offloader.cots_offloader import CotsOffloader
from vllm.model_executor.offloader.cots_operators import (
    CotsQKVOp,
    CotsSwiGLUMLPOp,
    _RaiseOnDirectCall,
    _scatter_col_outputs_three_way,
)
from vllm.model_executor.offloader.cots_runners import (
    NativeCotsWeightRunner,
    NativeWeightSlabSpec,
    PyCotsWeightCallback,
    PythonCotsWeightRunner,
    _cpu_dryrun_noop,
    _cpu_gemm_into_after_event,
    _make_runner,
    _NativeWeightSlabSpecMlp,
    _NativeWeightSlabSpecQkv,
)
from vllm.model_executor.offloader.cots_storage import (
    CotsLinearHandle,
    CotsPrefetchBufferPool,
    WeightPrefetchStreamer,
)
from vllm.model_executor.offloader.cots_utils import (
    _complement,
    _get_executor,
    _has_pinned_host_storage,
    _qkv_kv_biased_counts,
    _qkv_kv_biased_indices,
    _set_os_thread_name,
    _uva_copy_trusted_host_into_gpu,
    uva_copy_into_gpu,
)

__all__ = [
    "CotsOffloader",
    "CotsLinearHandle",
    "CotsPrefetchBufferPool",
    "WeightPrefetchStreamer",
    "PythonCotsWeightRunner",
    "NativeCotsWeightRunner",
    "NativeWeightSlabSpec",
    "PyCotsWeightCallback",
    "CotsQKVOp",
    "CotsSwiGLUMLPOp",
    "uva_copy_into_gpu",
    "_RaiseOnDirectCall",
    "_NativeWeightSlabSpecQkv",
    "_NativeWeightSlabSpecMlp",
    "_cpu_dryrun_noop",
    "_cpu_gemm_into_after_event",
    "_make_runner",
    "_scatter_col_outputs_three_way",
    "_complement",
    "_get_executor",
    "_has_pinned_host_storage",
    "_qkv_kv_biased_counts",
    "_qkv_kv_biased_indices",
    "_set_os_thread_name",
    "_uva_copy_trusted_host_into_gpu",
]
