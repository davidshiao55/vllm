# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public COTS offloader lifecycle and model patching logic."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from cots.snap import gqa_num_cpu_groups

# Register prefetch custom ops; COTS reuses wait_prefetch/start_prefetch.
import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.config.cots import (
    cots_weight_module_for_name,
    normalize_cots_weight_modules,
)
from vllm.config.offload import (
    CotsDispatchTableEntry,
    cots_dispatch_kv_group_pair,
    cots_dispatch_weight_pair,
    normalize_cots_dispatch_table_entry,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.offloader.base import BaseOffloader, ForwardDispatchInfo
from vllm.model_executor.offloader.cots_operators import (
    CotsQKVOp,
    CotsSwiGLUMLPOp,
    CotsWOInputSplitOp,
    CotsWOOp,
    _RaiseOnDirectCall,
)
from vllm.model_executor.offloader.cots_runners import (
    NativeCotsWeightRunner,
    NativeWeightSlabSpec,
    PyCotsWeightCallback,
    PythonCotsWeightRunner,
    _make_runner,
    _NativeWeightSlabSpecInputSplitLinear,
    _NativeWeightSlabSpecLinear,
    _NativeWeightSlabSpecMlp,
)
from vllm.model_executor.offloader.cots_storage import (
    DEFAULT_QKVO_HEAD_DIM,
    INPUT_SPLIT_AXIS,
    MLP_DOWN_ROLE,
    MLP_GATE_UP_ROLE,
    QKV_ROLE,
    WO_INPUT_ROLE,
    WO_QKVO_GRANULARITY_MULTIPLIER,
    WO_ROLE,
    CotsLinearHandle,
    CotsLinearRole,
    KVPrefetchBufferPool,
    KVPrefetchStreamer,
    WeightPrefetchBufferPool,
    WeightPrefetchStreamer,
)
from vllm.utils.cots_diag import COUNTERS_ENABLED as _COTS_COUNTERS_ENABLED
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import CotsOffloadConfig
    from vllm.forward_context import BatchDescriptor

logger = init_logger(__name__)


def _cots_py_timer_start() -> int:
    return time.perf_counter_ns() if _COTS_COUNTERS_ENABLED else 0


def _cots_py_timing(name: str, start_ns: int) -> None:
    if start_ns == 0:
        return
    from vllm.model_executor.offloader import cots_ops

    cots_ops.add_python_timing(name, time.perf_counter_ns() - start_ns)


def _cots_py_counter(name: str, value: int = 1) -> None:
    if not _COTS_COUNTERS_ENABLED:
        return
    from vllm.model_executor.offloader import cots_ops

    cots_ops.add_python_counter(name, int(value))


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _head_split_phase(num_tokens: int) -> str:
    return "decode" if int(num_tokens) <= 128 else "prefill"


LINEAR_OP_KIND_BY_ROLE = {
    QKV_ROLE: "qkv",
    WO_ROLE: "wo",
    WO_INPUT_ROLE: "wo",
}


@dataclass(frozen=True)
class CotsDispatchEntry:
    """Normalized runtime dispatch row for one COTS bucket."""

    f_cpu_compute: float
    f_prefetch_compute: float
    cpu_kv_compute_groups: int | None = None
    kv_prefetch_groups: int | None = None

    @property
    def weight_pair(self) -> tuple[float, float]:
        return self.f_cpu_compute, self.f_prefetch_compute

    @property
    def kv_group_pair(self) -> tuple[int, int] | None:
        if self.cpu_kv_compute_groups is None or self.kv_prefetch_groups is None:
            return None
        return self.cpu_kv_compute_groups, self.kv_prefetch_groups


@dataclass
class CotsHeadSplitQKVSidecar:
    """CPU-resident routed Q/K/V for one layer's head-split attention call."""

    storage_key: int
    num_tokens: int
    num_groups: int
    cpu_attention_groups: int
    cpu_weight_groups: int
    cpu_compute_kv_heads: int
    prefetch_kv_heads: int
    q_heads_per_kv: int
    head_dim: int
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    rope_applied: bool = False


@dataclass
class CotsHeadSplitAttentionOutputSidecar:
    """CPU-resident attention output for CPU-owned GQA groups."""

    storage_key: int
    num_tokens: int
    num_groups: int
    cpu_attention_groups: int
    cpu_weight_groups: int
    cpu_compute_kv_heads: int
    prefetch_kv_heads: int
    q_heads_per_kv: int
    head_dim: int
    output: torch.Tensor


class CotsOffloader(BaseOffloader):
    """Collaborative CPU-GPU weight offloader (thesis Phase 1a).

    Three-pass per layer in `wrap_modules`:
      1. Build & install handles for each offloadable Linear.
      2. Install operator adapters: `CotsQKVOp` per QKV linear,
         `CotsSwiGLUMLPOp` per recognized MLP block (replaces parent.forward,
         installs `_RaiseOnDirectCall` guards on the MLP linears), and
         `CotsWOOp` per opt-in WO linear.
      3. Reject orphan MLP gate/up/down handles loudly: MergedCol/Row offload
         requires an MLP block parent.
    """

    def __init__(
        self,
        config: CotsOffloadConfig,
        dispatch_table_factory: Callable[
            [Sequence[int]], dict[int, CotsDispatchTableEntry]
        ]
        | None = None,
    ):
        self.config = config
        self.f_cpu_store = float(config.f_cpu_store)
        self.f_prefetch = float(config.f_prefetch)
        self.kv_mode = str(getattr(config, "kv_mode", "prefix_suffix"))
        self.kv_head_prefetch_enabled = bool(
            getattr(config, "head_split_kv_prefetch_enabled", False)
        )
        self.weight_modules = frozenset(
            normalize_cots_weight_modules(getattr(config, "weight_modules", None))
        )
        self.dry_run = bool(config.dry_run)
        # Optional injection point for the Planner. None → uniform fill
        # from config in `post_init`.
        self._dispatch_table_factory = dispatch_table_factory
        if not (0.0 <= self.f_cpu_store <= 1.0):
            raise ValueError(f"f_cpu_store must be in [0, 1], got {self.f_cpu_store}")
        if self.f_prefetch > self.f_cpu_store:
            raise ValueError(
                f"f_prefetch ({self.f_prefetch}) must be <= "
                f"f_cpu_store ({self.f_cpu_store})"
            )
        # Only the Python runner uses process-wide PyTorch thread count. The
        # native runner carries per-bucket thread policy on each C++ slab, and
        # `f_cpu_store=0` should remain a no-side-effect control.
        if self.f_cpu_store > 0.0 and config.cpu_runner == "python":
            torch.set_num_threads(int(config.cpu_num_threads))

        # Populated in wrap_modules. _handles is the master list of all
        # offloaded linears (in insertion order); _fused_ops tracks installed
        # MLP-block ops (one per recognized parent), and _wo_ops tracks
        # opt-in output-split WO adapters.
        self._handles: list[CotsLinearHandle] = []
        self._fused_ops: list[CotsSwiGLUMLPOp] = []
        self._wo_ops: list[CotsWOOp | CotsWOInputSplitOp] = []

        # Per-layer tracking for prefetch hook installation. `_layer_modules[i]`
        # is the i-th offloaded decoder layer; `_layer_handles[i]` is its
        # handle list. Indices align with the streamer's per-layer events.
        self._layer_modules: list[nn.Module] = []
        self._layer_handles: list[list[CotsLinearHandle]] = []

        # Shared activation I/O buffers — allocated at the end of
        # wrap_modules (so vLLM's DeviceMemoryProfiler sees them in
        # `model_memory_usage`). Flat 1D so per-forward views are always
        # contiguous regardless of which sub-module is active.
        self._x_pinned: torch.Tensor | None = None
        self._y_pinned: torch.Tensor | None = None
        self._y_gpu: torch.Tensor | None = None

        # Dispatch table populated in wrap_modules (Phase 1b: needed before
        # the prefetch buffer pool is sized).
        self._dispatch_entries: dict[int, CotsDispatchEntry] = {}
        self._dispatch_table: dict[int, tuple[float, float]] = {}
        # CUDA graph capture buckets and Planner dispatch buckets are related
        # but distinct. Graph buckets describe replay shapes. Dispatch buckets
        # describe the COTS route selected from BatchDescriptor.num_tokens and
        # must exist even in eager mode, where CUDA graph buckets are empty.
        self._graph_capture_buckets: tuple[int, ...] = ()
        self._dispatch_buckets: tuple[int, ...] = ()
        self._route_signature_by_bucket: dict[int, int] = {}
        self._max_num_tokens: int = 0
        self._eager_fallback_entry: tuple[float, float] = (0.0, 0.0)
        self._has_cpu_compute_work: bool = False

        # Prefetch infrastructure — allocated in wrap_modules when the
        # dispatch table reserves any prefetch capacity. Active buckets may
        # still have zero prefetched rows after runtime snapping.
        self._prefetch_buffer_pool: WeightPrefetchBufferPool | None = None
        self._streamer: WeightPrefetchStreamer | None = None
        self._kv_prefetch_buffer_pool: KVPrefetchBufferPool | None = None
        self._kv_prefetch_streamer: KVPrefetchStreamer | None = None
        self._prefetch_hooks_installed: bool = False

        # One offloader-owned runner is shared across all operator call sites.
        # The no-offload path leaves it unset to avoid starting a worker thread.
        self._runner: PythonCotsWeightRunner | NativeCotsWeightRunner | None = None
        if self.f_cpu_store > 0.0:
            self._runner = _make_runner(config)

        # Active bucket is set out-of-graph by `on_dispatch` before operators
        # run. This stays valid even when prefetch is disabled.
        self._current_bucket: int | None = None
        # Two distinct dummy CUDA anchors for the cots_sync_then_uva
        # custom op's mutates_args=["y_gpu", "gpu_anchor_a",
        # "gpu_anchor_b"]. Operators pass these when out_perm/out_pref
        # are absent so the two anchor slots never alias (aliasing
        # confuses torch.compile / functionalization). Allocated in
        # `_allocate_activation_buffers` because that runs inside
        # vLLM's DeviceMemoryProfiler accounting window
        # (phase1a_findings.md §1.5).
        self._dummy_gpu_anchor_a: torch.Tensor | None = None
        self._dummy_gpu_anchor_b: torch.Tensor | None = None
        self._head_split_qkv_sidecars: dict[int, CotsHeadSplitQKVSidecar] = {}
        self._head_split_attention_outputs: dict[
            int, CotsHeadSplitAttentionOutputSidecar
        ] = {}
        self._head_split_q_work: torch.Tensor | None = None
        self._head_split_k_work: torch.Tensor | None = None
        self._head_split_v_work: torch.Tensor | None = None
        self._head_split_qkv_work_signature: (
            tuple[
                int,
                int,
                int,
                torch.dtype,
            ]
            | None
        ) = None
        self._head_split_qkv_work_capacity: int = 0
        self._head_split_positions_cpu: torch.Tensor | None = None
        self._live_num_tokens: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle: wrap_modules → (weight loading) → post_init.
    # ------------------------------------------------------------------
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Walk decoder layers lazily; install handles + operators.

        Lazy iteration mirrors `prefetch.py:175`: each layer's empty GPU
        tensors are dereferenced as soon as we replace its offloaded params
        with GPU-slice tensors, so peak GPU during construction stays
        bounded by ~one layer's worth.
        """
        self._check_tensor_parallel_size_one()
        if self.f_cpu_store > 0.0:
            self._check_pin_memory_available()

        # Resolved early so per-bucket geometry can be built before the
        # prefetch buffer pool is sized.
        self._resolve_bucket_sets()

        modules: list[nn.Module] = []
        layer_idx = 0
        for layer in modules_generator:
            modules.append(layer)
            layer_handles: list[CotsLinearHandle] = []
            if self.f_cpu_store > 0.0:
                layer_handles = self._build_handles(layer)
                if layer_handles:
                    for h in layer_handles:
                        h.layer_idx = layer_idx
                    self._install_qkv_ops(layer_handles)
                    self._install_mlp_ops(layer, layer_handles)
                    self._install_wo_ops(layer_handles)
                    self._check_no_orphan_mlp_handles(layer_handles)
            if layer_handles or self.kv_head_prefetch_enabled:
                self._layer_modules.append(layer)
                self._layer_handles.append(layer_handles)
                layer_idx += 1

        if self.f_cpu_store == 0.0:
            if self.kv_head_prefetch_enabled:
                self._build_dispatch_table()
                self._validate_head_split_kv_dispatch_geometry()
                self._install_kv_prefetch_machinery()
                logger.info_once(
                    "CotsOffloader: f_cpu_store=0, installed experimental "
                    "head-split KV prefetch lifecycle only.",
                    scope="local",
                )
            else:
                logger.info_once(
                    "CotsOffloader: f_cpu_store=0, no offloading.", scope="local"
                )
            return modules

        # Phase 1b: build dispatch table, populate per-handle dispatch
        # geometry, and allocate option-A prefetch capacity when the table
        # may need it. A bucket with raw f_prefetch=0 still routes all stored
        # rows to CPU compute, but the pool can exist for planner accounting.
        self._build_dispatch_table()
        for h in self._handles:
            h.apply_prefetch_split_per_bucket(self._dispatch_table)
        self._validate_head_split_kv_dispatch_geometry()
        self._has_cpu_compute_work = self._compute_has_cpu_compute_work()
        self._build_route_signatures()

        # Allocate shared CPU-compute buffers only when some bucket actually
        # leaves rows on the CPU-compute path. Pure-prefetch configurations
        # skip these pinned/GPU scratch buffers entirely.
        if self._has_cpu_compute_work:
            self._allocate_activation_buffers()
        # Install based on option-A reserved capacity, not the config fallback
        # knob. A Planner-emitted table can request prefetch even when config
        # f_prefetch == 0, and zero-prefetch tables still reserve full-store
        # slots so runtime accounting matches planner GPU buffer accounting.
        if self.kv_head_prefetch_enabled:
            self._install_kv_prefetch_machinery()
        if any(h.max_n_prefetch > 0 for h in self._handles):
            self._install_prefetch_machinery()
        logger.debug(
            "CotsOffloader: wrapped %d linear modules, %d fused MLP blocks, "
            "and %d WO ops (modules=%s, kv_mode=%s, f_cpu_store=%.4f, "
            "f_prefetch=%.4f, cpu_num_threads=%d, dry_run=%s).",
            len(self._handles),
            len(self._fused_ops),
            len(self._wo_ops),
            sorted(self.weight_modules),
            self.kv_mode,
            self.f_cpu_store,
            self.f_prefetch,
            self.config.cpu_num_threads,
            self.dry_run,
        )
        return modules

    # --- Pass 1: build handles ---

    def _build_handles(self, layer: nn.Module) -> list[CotsLinearHandle]:
        """Discover offloadable linears in `layer`, build & install handles."""
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
            UnquantizedLinearMethod,
        )

        layer_handles: list[CotsLinearHandle] = []
        qkvo_head_dim = self._qkvo_head_dim_for_layer(layer, QKVParallelLinear)
        gqa_shape = (
            self._gqa_shape_for_layer(layer, QKVParallelLinear)
            if self.kv_mode == "head_split"
            else None
        )
        for qualified_name, child in layer.named_modules():
            module = self._module_for_qualified_name(qualified_name)
            if module is None:
                continue
            if not isinstance(child.quant_method, UnquantizedLinearMethod):
                raise RuntimeError(
                    f"CotsOffloader only supports unquantized "
                    f"linear layers, got {type(child.quant_method).__name__} "
                    f"on {qualified_name}."
                )
            self._check_dtype_is_bfloat16(child, qualified_name)
            if module == "qkv" and isinstance(child, QKVParallelLinear):
                if self.kv_mode == "head_split":
                    handle = CotsLinearHandle.for_qkv_gqa_groups(
                        child,
                        qualified_name,
                        head_dim=int(child.head_size),
                        f_cpu_store=self.f_cpu_store,
                    )
                else:
                    handle = CotsLinearHandle.for_qkv(
                        child,
                        qualified_name,
                        head_dim=int(child.head_size),
                        f_cpu_store=self.f_cpu_store,
                    )
            elif module == "mlp" and isinstance(child, MergedColumnParallelLinear):
                handle = CotsLinearHandle.for_mlp_gate_up(
                    child,
                    qualified_name,
                    f_cpu_store=self.f_cpu_store,
                )
            elif module == "mlp" and isinstance(child, RowParallelLinear):
                handle = CotsLinearHandle.for_mlp_down(
                    child,
                    qualified_name,
                    f_cpu_store=self.f_cpu_store,
                )
            elif module == "wo" and isinstance(child, RowParallelLinear):
                if self.kv_mode == "head_split":
                    assert gqa_shape is not None
                    num_q_heads, num_kv_heads, head_dim = gqa_shape
                    handle = CotsLinearHandle.for_wo_gqa_input(
                        child,
                        qualified_name,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        f_cpu_store=self.f_cpu_store,
                    )
                else:
                    handle = CotsLinearHandle.for_wo(
                        child,
                        qualified_name,
                        f_cpu_store=self.f_cpu_store,
                        qkvo_head_dim=qkvo_head_dim,
                    )
            else:
                raise RuntimeError(
                    f"CotsOffloader: {qualified_name} matched enabled COTS "
                    f"module {module!r} but has unsupported linear type "
                    f"(got {type(child).__name__})"
                )
            if handle is None:
                continue  # f rounded to 0 cols for this module
            handle.install(child.weight.data.device)
            self._handles.append(handle)
            layer_handles.append(handle)
        return layer_handles

    def _module_for_qualified_name(self, qualified_name: str) -> str | None:
        """Return the enabled semantic COTS module for a linear name."""
        return cots_weight_module_for_name(self.weight_modules, qualified_name)

    @staticmethod
    def _qkvo_head_dim_for_layer(
        layer: nn.Module,
        qkv_cls: type[nn.Module],
    ) -> int:
        """Use the layer's WQKV head size as the QKVO snap base.

        WO has no semantic heads, but its dense output split uses the same
        `2 * head_dim` K/V-pair quantum as WQKV so QKVO starts on one grid.
        Synthetic WO-only tests may not include a QKV module, so they fall back
        to the common Qwen/Llama 128-channel head grid.
        """
        for _, child in layer.named_modules():
            if isinstance(child, qkv_cls):
                return int(child.head_size)
        return DEFAULT_QKVO_HEAD_DIM

    @staticmethod
    def _gqa_shape_for_layer(
        layer: nn.Module,
        qkv_cls: type[nn.Module],
    ) -> tuple[int, int, int]:
        for _, child in layer.named_modules():
            if isinstance(child, qkv_cls):
                parts = child.output_partition_sizes
                assert len(parts) == 3, f"QKV expected 3 partitions, got {parts}"
                q_part, k_part, v_part = parts
                head_dim = int(child.head_size)
                assert k_part == v_part, (
                    f"QKV expected k_part == v_part, got k={k_part}, v={v_part}"
                )
                return q_part // head_dim, k_part // head_dim, head_dim
        raise RuntimeError(
            "CotsOffloader head_split mode requires a QKVParallelLinear in "
            "each offloaded attention layer so WO can align to GQA groups."
        )

    # --- Pass 2a: QKV operator install ---

    def _install_qkv_ops(self, handles: list[CotsLinearHandle]) -> None:
        # Operators share the offloader-owned runner constructed in __init__.
        assert self._runner is not None, (
            "_install_qkv_ops called with f_cpu_store=0 — runner not constructed"
        )
        for h in handles:
            if h.role != QKV_ROLE:
                continue
            h.linear.quant_method = CotsQKVOp(
                handle=h,
                runner=self._runner,
                offloader=self,
                original_quant_method=h.linear.quant_method,
            )

    # --- Pass 2a-2: WO operator install ---

    def _install_wo_ops(self, handles: list[CotsLinearHandle]) -> None:
        assert self._runner is not None, (
            "_install_wo_ops called with f_cpu_store=0 — runner not constructed"
        )
        for h in handles:
            if h.role not in (WO_ROLE, WO_INPUT_ROLE):
                continue
            if h.role == WO_INPUT_ROLE:
                h.linear.quant_method = CotsWOInputSplitOp(
                    handle=h,
                    runner=self._runner,
                    offloader=self,
                    original_quant_method=h.linear.quant_method,
                )
            else:
                h.linear.quant_method = CotsWOOp(
                    handle=h,
                    runner=self._runner,
                    offloader=self,
                    original_quant_method=h.linear.quant_method,
                )
            self._wo_ops.append(h.linear.quant_method)

    # --- Pass 2b: MLP block operator install ---

    def _install_mlp_ops(
        self, layer: nn.Module, layer_handles: list[CotsLinearHandle]
    ) -> None:
        """Recognize Qwen2MLP-style parents and install `CotsSwiGLUMLPOp` on
        their `forward`. Strict checks: SiluAndMul-only act_fn, no biases,
        no skip_bias_add. Reject mismatches loudly.
        """
        for parent_name, parent in layer.named_modules():
            gu = getattr(parent, "gate_up_proj", None)
            dp = getattr(parent, "down_proj", None)
            af = getattr(parent, "act_fn", None)
            if gu is None or dp is None or af is None:
                continue
            gu_h = getattr(gu, "_cots_handle", None)
            dp_h = getattr(dp, "_cots_handle", None)
            if gu_h is None or dp_h is None:
                continue
            qualified_name = parent_name or "<root>"

            # Strict checks: the fused CPU path is hard-coded to silu*up and
            # ignores biases / skip_bias_add. Reject loudly.
            if not isinstance(af, SiluAndMul):
                raise RuntimeError(
                    f"cots: {qualified_name}.act_fn is "
                    f"{type(af).__name__}, expected SiluAndMul (the fused "
                    f"CPU path is hard-coded to silu(gate)*up)."
                )
            if (
                getattr(gu, "bias", None) is not None
                or getattr(dp, "bias", None) is not None
            ):
                raise RuntimeError(
                    f"cots: {qualified_name} MLP has bias on gate_up or down; "
                    f"the fused path doesn't handle MLP biases."
                )
            if getattr(gu, "skip_bias_add", False) or getattr(
                dp, "skip_bias_add", False
            ):
                raise RuntimeError(
                    f"cots: {qualified_name} MLP has skip_bias_add=True; not supported."
                )

            # Installer refactor: shared offloader runner (see
            # _install_qkv_ops above for rationale).
            assert self._runner is not None, (
                "_install_mlp_ops called with f_cpu_store=0 — runner not constructed"
            )
            mlp_op = CotsSwiGLUMLPOp(
                gate_up_layer=gu,
                down_layer=dp,
                gate_up_handle=gu_h,
                down_handle=dp_h,
                act_fn=af,
                runner=self._runner,
                offloader=self,
                qualified_name=qualified_name,
            )
            parent.forward = mlp_op  # type: ignore[method-assign]
            gu.quant_method = _RaiseOnDirectCall(
                qualified_name=f"{qualified_name}.gate_up_proj",
                original=gu.quant_method,
            )
            dp.quant_method = _RaiseOnDirectCall(
                qualified_name=f"{qualified_name}.down_proj",
                original=dp.quant_method,
            )
            gu_h.in_block = True
            dp_h.in_block = True
            self._fused_ops.append(mlp_op)
            assert gu_h in layer_handles and dp_h in layer_handles

    # --- Pass 3: orphan check ---

    @staticmethod
    def _check_tensor_parallel_size_one() -> None:
        """Current COTS contract: TP=1 only. The loader closures assert full
        unsharded `loaded_weight` shapes (no per-rank narrow); native vLLM
        loaders narrow by TP rank before copying. Cleanly fail at wrap time
        rather than mismatch in a loader closure later.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if tp_size != 1:
            raise RuntimeError(
                f"CotsOffloader requires tensor_parallel_size=1; "
                f"got tp_size={tp_size}. Multi-rank TP is out of scope for "
                f"the current implementation (loader closures assume full "
                f"unsharded weights)."
            )

    @staticmethod
    def _check_pin_memory_available() -> None:
        """`uva_copy_into_gpu` requires pinned host memory. Fail at wrap
        time rather than at first forward.
        """
        if not is_pin_memory_available():
            raise RuntimeError(
                "CotsOffloader requires pinned host memory; "
                "is_pin_memory_available() returned False. Pinned memory is "
                "unavailable on this host (e.g., container cgroup limits)."
            )

    @staticmethod
    def _check_dtype_is_bfloat16(linear: nn.Module, qualified_name: str) -> None:
        """Current COTS contract: BF16-only (`offload.py` `cpu_dtype`).
        oneDNN BF16 is the fast CPU path measured in Phase 0; FP16/FP32
        is not part of the production path.
        """
        if linear.weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"CotsOffloader requires bfloat16; "
                f"{qualified_name} has dtype={linear.weight.dtype}. "
                f"Launch with --dtype bfloat16."
            )

    @staticmethod
    def _check_no_orphan_mlp_handles(handles: list[CotsLinearHandle]) -> None:
        """Every MLP split handle must be in a fused MLP block."""
        for h in handles:
            if h.role in (MLP_GATE_UP_ROLE, MLP_DOWN_ROLE) and not h.in_block:
                raise RuntimeError(
                    f"cots: {h.qualified_name} is offloaded but not "
                    f"part of a recognized MLP block (gate_up_proj + act_fn "
                    f"+ down_proj structure). Add the structural parent or "
                    f"exclude this linear from offload."
                )

    # --- Phase 1b: dispatch table + prefetch machinery installation ---

    @staticmethod
    def _normalize_dispatch_entry(
        bucket: int, entry: CotsDispatchTableEntry
    ) -> CotsDispatchEntry:
        normalized = normalize_cots_dispatch_table_entry(
            entry, label="cots.dispatch_table", bucket=int(bucket)
        )
        f_cpu_compute, f_prefetch_compute = cots_dispatch_weight_pair(normalized)
        kv_groups = cots_dispatch_kv_group_pair(normalized)
        if kv_groups is None:
            return CotsDispatchEntry(
                f_cpu_compute=f_cpu_compute,
                f_prefetch_compute=f_prefetch_compute,
            )
        return CotsDispatchEntry(
            f_cpu_compute=f_cpu_compute,
            f_prefetch_compute=f_prefetch_compute,
            cpu_kv_compute_groups=int(kv_groups[0]),
            kv_prefetch_groups=int(kv_groups[1]),
        )

    def _build_dispatch_table(self) -> None:
        """Construct the uniform Planner dispatch table."""
        if self._dispatch_table_factory is not None:
            raw_table = self._dispatch_table_factory(self._dispatch_buckets)
        else:
            pair = (self.f_cpu_store - self.f_prefetch, self.f_prefetch)
            raw_table = {b: pair for b in self._dispatch_buckets}
        self._dispatch_entries = {
            int(bucket): self._normalize_dispatch_entry(int(bucket), entry)
            for bucket, entry in raw_table.items()
        }
        self._dispatch_table = {
            bucket: entry.weight_pair
            for bucket, entry in self._dispatch_entries.items()
        }
        if self.kv_head_prefetch_enabled:
            missing_kv_rows = [
                int(bucket)
                for bucket in self._dispatch_buckets
                if self._dispatch_entries.get(int(bucket)) is None
                or self._dispatch_entries[int(bucket)].kv_group_pair is None
            ]
            if missing_kv_rows:
                raise ValueError(
                    "COTS 3-way head-split KV requires compact dispatch rows "
                    "with (cpu_kv_compute_groups, kv_prefetch_groups); "
                    f"missing buckets: {missing_kv_rows}"
                )
        self._validate_graph_capture_dispatch_coverage()

    def _head_split_num_kv_heads_for_validation(self) -> int:
        _, num_kv_heads, _ = self._head_split_qkv_shape_for_validation()
        return int(num_kv_heads)

    def _head_split_qkv_shape_for_validation(self) -> tuple[int, int, int]:
        num_q_heads: int | None = None
        num_kv_heads: int | None = None
        head_dim: int | None = None
        for h in self._handles:
            if h.role not in (QKV_ROLE, WO_INPUT_ROLE):
                continue
            handle_num_q_heads_attr = getattr(h, "num_q_heads", None)
            handle_num_kv_heads_attr = getattr(h, "num_kv_heads", None)
            handle_head_dim_attr = getattr(h, "head_dim", None)
            if handle_num_kv_heads_attr is None:
                continue
            if handle_num_q_heads_attr is None or handle_head_dim_attr is None:
                continue
            handle_num_q_heads = int(handle_num_q_heads_attr)
            handle_num_kv_heads = int(handle_num_kv_heads_attr)
            handle_head_dim = int(handle_head_dim_attr)
            if num_kv_heads is None:
                num_q_heads = handle_num_q_heads
                num_kv_heads = handle_num_kv_heads
                head_dim = handle_head_dim
            elif (
                handle_num_q_heads != num_q_heads
                or handle_num_kv_heads != num_kv_heads
                or handle_head_dim != head_dim
            ):
                raise ValueError(
                    "COTS 3-way head-split KV validation found inconsistent "
                    "attention handle GQA geometry: "
                    f"({handle_num_q_heads}, {handle_num_kv_heads}, "
                    f"{handle_head_dim}) != "
                    f"({num_q_heads}, {num_kv_heads}, {head_dim})"
                )
        if (
            num_q_heads is not None
            and num_kv_heads is not None
            and head_dim is not None
        ):
            return int(num_q_heads), int(num_kv_heads), int(head_dim)

        if not self._layer_modules:
            raise ValueError(
                "COTS 3-way head-split KV validation requires an attention "
                "layer so CPU-owned KV group count can be checked."
            )
        from vllm.model_executor.layers.linear import QKVParallelLinear

        return self._gqa_shape_for_layer(self._layer_modules[0], QKVParallelLinear)

    def _head_split_kv_prefetch_dtype_for_validation(self) -> torch.dtype:
        for h in self._handles:
            if h.role == QKV_ROLE:
                return h.dtype
        if not self._layer_modules:
            raise ValueError(
                "COTS 3-way head-split KV prefetch requires an attention "
                "layer so KV dtype can be checked."
            )
        from vllm.model_executor.layers.linear import QKVParallelLinear

        for _, child in self._layer_modules[0].named_modules():
            if isinstance(child, QKVParallelLinear):
                return child.weight.dtype
        raise RuntimeError(
            "CotsOffloader head_split mode requires a QKVParallelLinear in "
            "each offloaded attention layer so KV prefetch dtype can be inferred."
        )

    def _head_split_kv_prefetch_block_size(self) -> int:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        block_size = int(vllm_config.cache_config.block_size)
        if block_size <= 0:
            raise ValueError(
                "COTS 3-way head-split KV prefetch requires a positive "
                f"cache block_size, got {block_size}"
            )
        return block_size

    def _head_split_kv_prefetch_block_table_capacity(
        self, block_size: int
    ) -> tuple[int, int]:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        model_config = vllm_config.model_config
        if model_config is None:
            max_model_len = int(vllm_config.scheduler_config.max_num_batched_tokens)
        else:
            max_model_len = int(model_config.max_model_len)
        max_blocks_per_req = (max_model_len + int(block_size) - 1) // int(block_size)
        if max_num_reqs <= 0 or max_blocks_per_req <= 0:
            raise ValueError(
                "COTS 3-way head-split KV prefetch requires positive "
                "compact block-table capacity: "
                f"max_num_reqs={max_num_reqs}, "
                f"max_blocks_per_req={max_blocks_per_req}"
            )
        return max_num_reqs, max_blocks_per_req

    @staticmethod
    def _head_split_cpu_weight_groups(h: CotsLinearHandle, bucket: int) -> int | None:
        if h.role == QKV_ROLE and h.qkv_cpu_layout == "gqa_group":
            group_size = int(h.gqa_qkv_group_size)
        elif h.role == WO_INPUT_ROLE:
            group_size = int(h.gqa_q_group_size)
        else:
            return None
        n_cpu = int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu))
        if group_size <= 0 or n_cpu % group_size != 0:
            raise ValueError(
                "COTS 3-way head-split KV validation found non-GQA-aligned "
                f"attention weight split for {h.qualified_name}: "
                f"n_cpu={n_cpu}, group_size={group_size}"
            )
        return n_cpu // group_size

    def _validate_head_split_role_groups(
        self,
        *,
        bucket: int,
        expected_cpu_groups: int,
        role: CotsLinearRole,
        label: str,
    ) -> None:
        seen = False
        for h in self._handles:
            if h.role != role:
                continue
            groups = self._head_split_cpu_weight_groups(h, bucket)
            if groups is None:
                continue
            seen = True
            if int(groups) != int(expected_cpu_groups):
                raise ValueError(
                    "COTS 3-way head-split KV dispatch row disagrees with "
                    f"snapped {label} weight geometry for bucket {bucket}: "
                    f"dispatch C={expected_cpu_groups}, {label} groups={groups}, "
                    f"module={h.qualified_name}"
                )
        if not seen and int(expected_cpu_groups) != 0:
            raise ValueError(
                "COTS 3-way head-split KV dispatch row requests CPU "
                f"attention groups for bucket {bucket}, but no snapped "
                f"{label} CPU-compute groups exist: C={expected_cpu_groups}"
            )

    def _validate_head_split_kv_dispatch_geometry(self) -> None:
        if not self.kv_head_prefetch_enabled:
            return
        num_kv_heads = self._head_split_num_kv_heads_for_validation()
        cpu_kv_groups = gqa_num_cpu_groups(
            float(getattr(self.config, "f_cpu_kv_store", 0.0)),
            num_kv_heads=num_kv_heads,
        )
        for bucket in self._dispatch_buckets:
            entry = self._dispatch_entries.get(int(bucket))
            if entry is None or entry.kv_group_pair is None:
                raise ValueError(
                    "COTS 3-way head-split KV requires compact dispatch rows "
                    "with (cpu_kv_compute_groups, kv_prefetch_groups); "
                    f"missing bucket: {bucket}"
                )
            cpu_compute, prefetch = entry.kv_group_pair
            if int(cpu_compute) + int(prefetch) != int(cpu_kv_groups):
                raise ValueError(
                    "COTS 3-way head-split KV dispatch row must cover exactly "
                    "the CPU-owned KV groups: "
                    f"bucket={bucket}, C={cpu_compute}, P={prefetch}, "
                    f"A={cpu_kv_groups}"
                )
            self._validate_head_split_role_groups(
                bucket=int(bucket),
                expected_cpu_groups=int(cpu_compute),
                role=QKV_ROLE,
                label="WQKV",
            )
            self._validate_head_split_role_groups(
                bucket=int(bucket),
                expected_cpu_groups=int(cpu_compute),
                role=WO_INPUT_ROLE,
                label="WO",
            )

    def head_split_kv_group_plan_by_bucket(self) -> dict[int, tuple[int, int]]:
        """Return Planner-owned 3-way KV group rows keyed by dispatch bucket."""
        if not self.kv_head_prefetch_enabled:
            return {}
        plan: dict[int, tuple[int, int]] = {}
        for bucket in self._dispatch_buckets:
            entry = self._dispatch_entries.get(int(bucket))
            if entry is None or entry.kv_group_pair is None:
                raise RuntimeError(
                    "COTS 3-way head-split KV plan requested before compact "
                    f"dispatch row exists for bucket {bucket}."
                )
            cpu_compute, prefetch = entry.kv_group_pair
            plan[int(bucket)] = (int(cpu_compute), int(prefetch))
        return plan

    def publish_head_split_kv_prefetch_descriptor(
        self, layer_idx: int, descriptor: object | None
    ) -> None:
        if self._kv_prefetch_streamer is None:
            return
        self._kv_prefetch_streamer.publish_descriptor(layer_idx, descriptor)

    def head_split_kv_prefetch_attention_inputs(
        self, descriptor: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._kv_prefetch_streamer is None:
            raise RuntimeError(
                "COTS head-split KV prefetch attention requested without "
                "a KV prefetch streamer"
            )
        layer_idx = int(cast(Any, descriptor).layer_idx)
        return self._kv_prefetch_streamer.attention_inputs(
            layer_idx=layer_idx,
            descriptor=descriptor,
        )

    def patch_head_split_kv_prefetch_current(
        self,
        *,
        descriptor: object | None,
        key: torch.Tensor,
        value: torch.Tensor,
        cpu_slot_mapping: torch.Tensor,
        gpu_kv_heads: int,
        num_actual_tokens: int,
    ) -> None:
        if descriptor is None or self._kv_prefetch_streamer is None:
            return
        layer_idx = int(cast(Any, descriptor).layer_idx)
        self._kv_prefetch_streamer.patch_current_kv(
            layer_idx=layer_idx,
            descriptor=descriptor,
            key=key,
            value=value,
            cpu_slot_mapping=cpu_slot_mapping,
            gpu_kv_heads=int(gpu_kv_heads),
            num_actual_tokens=int(num_actual_tokens),
        )

    def wait_head_split_kv_prefetch_writeback(self, layer_idx: int) -> bool:
        if self._kv_prefetch_streamer is None:
            return False
        return self._kv_prefetch_streamer.wait_for_layer_writeback(int(layer_idx))

    def _install_prefetch_machinery(self) -> None:
        """Allocate prefetch buffers and install layer-level prefetch hooks."""
        device = torch.device("cuda")
        self._prefetch_buffer_pool = WeightPrefetchBufferPool(self._handles, device)
        for h in self._handles:
            if h.layer_idx >= 0:
                h.slot_idx = h.layer_idx % WeightPrefetchBufferPool.K

        n_layers = len(self._layer_modules)
        self._streamer = WeightPrefetchStreamer(
            n_layers=n_layers,
            dry_run=self.dry_run,
        )
        self._streamer.buffer_pool = self._prefetch_buffer_pool

        self._ensure_prefetch_hooks_installed()

    def _install_kv_prefetch_machinery(self) -> None:
        """Install the gated no-op lifecycle for 3-way head-split KV prefetch."""
        if self._kv_prefetch_buffer_pool is None:
            _, _, head_dim = self._head_split_qkv_shape_for_validation()
            dtype = self._head_split_kv_prefetch_dtype_for_validation()
            if dtype != torch.bfloat16:
                raise ValueError(
                    "COTS 3-way head-split KV prefetch currently supports only "
                    f"BF16 KV cache, got dtype={dtype}."
                )
            self.configure_kv_prefetch_buffer(
                max_active_blocks=int(self.config.kv_prefetch_max_active_blocks),
                block_size=self._head_split_kv_prefetch_block_size(),
                head_dim=int(head_dim),
                dtype=dtype,
                device=torch.device("cuda"),
            )
        if self._kv_prefetch_streamer is None:
            self._kv_prefetch_streamer = KVPrefetchStreamer(
                n_layers=len(self._layer_modules),
                buffer_pool=self._kv_prefetch_buffer_pool,
            )
        self._ensure_prefetch_hooks_installed()

    def configure_kv_prefetch_buffer(
        self,
        *,
        max_active_blocks: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate the compact 3-way head-split KV prefetch workspace.

        Weight routing owns the dispatch table, so this method derives the
        maximum prefetched GQA group count ``P`` from normalized compact
        dispatch rows. The offloader allocates this during model wrapping so
        vLLM's memory profiler accounts for the reserved prefetch workspace
        before sizing the main KV cache.
        """
        if not self.kv_head_prefetch_enabled:
            return
        if max_active_blocks <= 0:
            raise ValueError(
                "COTS 3-way head-split KV prefetch requires "
                f"max_active_blocks > 0, got {max_active_blocks}"
            )
        if block_size <= 0 or head_dim <= 0:
            raise ValueError(
                "COTS 3-way head-split KV prefetch requires positive "
                f"block_size/head_dim, got block_size={block_size}, "
                f"head_dim={head_dim}"
            )
        if not self._dispatch_entries:
            raise RuntimeError(
                "COTS 3-way head-split KV prefetch buffer cannot be "
                "configured before the compact dispatch table is built."
            )
        max_num_reqs, max_blocks_per_req = (
            self._head_split_kv_prefetch_block_table_capacity(block_size)
        )

        max_prefetch_kv_heads = 0
        for bucket in self._dispatch_buckets:
            entry = self._dispatch_entries.get(int(bucket))
            if entry is None or entry.kv_group_pair is None:
                raise RuntimeError(
                    "COTS 3-way head-split KV prefetch buffer requires compact "
                    f"dispatch rows for every bucket; missing bucket {bucket}."
                )
            _, prefetch = entry.kv_group_pair
            max_prefetch_kv_heads = max(max_prefetch_kv_heads, int(prefetch))

        self._kv_prefetch_buffer_pool = KVPrefetchBufferPool(
            n_layers=len(self._layer_modules),
            max_active_blocks=int(max_active_blocks),
            max_num_reqs=max_num_reqs,
            max_blocks_per_req=max_blocks_per_req,
            block_size=int(block_size),
            max_prefetch_kv_heads=max_prefetch_kv_heads,
            head_dim=int(head_dim),
            dtype=dtype,
            device=device,
        )
        if self._kv_prefetch_streamer is not None:
            self._kv_prefetch_streamer.buffer_pool = self._kv_prefetch_buffer_pool

    def _ensure_prefetch_hooks_installed(self) -> None:
        if self._prefetch_hooks_installed:
            return
        for i, layer in enumerate(self._layer_modules):
            self._hook_layer_forward(i, layer)
        self._prefetch_hooks_installed = True

    # --- Runner install (closures / slab specs) -----

    @staticmethod
    def _make_output_split_python_callback(
        w_cpu: torch.Tensor,
    ) -> PyCotsWeightCallback:
        """Build a closure that captures a per-(layer, bucket) output-split
        weight slice. Module-level helper so closures don't accidentally
        capture `self` (would leak the offloader through the registry)."""

        def cb(
            event: torch.cuda.Event,
            x_pinned: torch.Tensor,
            y_pinned: torch.Tensor,
        ) -> None:
            event.synchronize()
            y_pinned.copy_(F.linear(x_pinned, w_cpu))

        return cb

    @staticmethod
    def _make_input_split_python_callback(
        w_cpu: torch.Tensor,
    ) -> PyCotsWeightCallback:
        """Build a closure for row-parallel standalone WO.

        The operator passes a compact activation slice whose columns already
        match ``w_cpu``'s rows, so the CPU work is one ``x @ w`` matmul.
        """

        def cb(
            event: torch.cuda.Event,
            x_pinned: torch.Tensor,
            y_pinned: torch.Tensor,
        ) -> None:
            event.synchronize()
            y_pinned.copy_(torch.matmul(x_pinned, w_cpu))

        return cb

    @staticmethod
    def _make_mlp_python_callback(
        w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
    ) -> PyCotsWeightCallback:
        """Closure for the fused MLP block (gate + up + silu*up + down).

        `w_down` is the row-major `(K, N)` CPU-compute slice, so it is passed
        directly to `torch.matmul`. Gate/up use the PyTorch-natural `(out, in)`
        layout and stay on `F.linear`.
        """

        def cb(
            event: torch.cuda.Event,
            x_pinned: torch.Tensor,
            y_pinned: torch.Tensor,
        ) -> None:
            event.synchronize()
            gate_out = F.linear(x_pinned, w_gate)
            up_out = F.linear(x_pinned, w_up)
            z = F.silu(gate_out) * up_out
            y_pinned.copy_(torch.matmul(z, w_down))

        return cb

    def _build_python_callbacks(
        self,
    ) -> dict[tuple[int, int, str], PyCotsWeightCallback]:
        """Build the (layer_idx, bucket, op_kind) -> closure table that
        PythonCotsWeightRunner consults at submit time. Skip slabs where there
        is no CPU work (n_cpu_compute == 0).
        """
        callbacks: dict[tuple[int, int, str], PyCotsWeightCallback] = {}
        for h in self._handles:
            if h.role not in (QKV_ROLE, WO_ROLE, WO_INPUT_ROLE):
                continue
            assert h.w_cpu is not None
            op_kind = LINEAR_OP_KIND_BY_ROLE[h.role]
            for bucket in self._dispatch_buckets:
                n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu = h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)
                if n_cpu == 0:
                    continue
                w_view = h.w_cpu.narrow(0, n_pref, n_cpu)
                if h.role == WO_INPUT_ROLE:
                    callbacks[(h.layer_idx, bucket, op_kind)] = (
                        self._make_input_split_python_callback(w_view)
                    )
                else:
                    callbacks[(h.layer_idx, bucket, op_kind)] = (
                        self._make_output_split_python_callback(w_view)
                    )
        for fop in self._fused_ops:
            gu_h = fop._gate_up
            dn_h = fop._down
            assert gu_h.w_cpu is not None
            assert dn_h.w_cpu is not None
            n_cpu_per_half_total = gu_h.n_cpu // 2
            for bucket in self._dispatch_buckets:
                gu_n_pref = gu_h.n_prefetch_by_bucket.get(bucket, 0)
                dn_n_pref = dn_h.n_prefetch_by_bucket.get(bucket, 0)
                dn_n_cpu = dn_h.n_cpu_compute_by_bucket.get(bucket, dn_h.n_cpu)
                if dn_n_cpu == 0:
                    continue
                n_pref_per_half = gu_n_pref // 2
                w_gate_view = gu_h.w_cpu[n_pref_per_half:n_cpu_per_half_total, :]
                w_up_view = gu_h.w_cpu[
                    n_cpu_per_half_total + n_pref_per_half : 2 * n_cpu_per_half_total,
                    :,
                ]
                w_down_view = dn_h.w_cpu.narrow(0, dn_n_pref, dn_n_cpu)
                callbacks[(gu_h.layer_idx, bucket, "mlp_block")] = (
                    self._make_mlp_python_callback(w_gate_view, w_up_view, w_down_view)
                )
        return callbacks

    def _n_threads_for(self, bucket: int) -> int:
        """Resolve the CPU GEMM thread count for a given bucket.

        When `config.cpu_num_threads_by_bucket` is set, missing buckets fall
        back to the scalar `cpu_num_threads`. The Planner may specify only
        buckets it has profile data for.

        Validation lives in `_validate_thread_policy` because
        `_dispatch_buckets` is not known until `_resolve_bucket_sets` runs in
        `wrap_modules`.
        """
        per_bucket = getattr(self.config, "cpu_num_threads_by_bucket", None)
        if per_bucket is None:
            return int(self.config.cpu_num_threads)
        return int(per_bucket.get(bucket, self.config.cpu_num_threads))

    def _validate_thread_policy(self) -> None:
        """Reject `cpu_num_threads_by_bucket` keys that aren't dispatch
        buckets — would silently fall back to scalar and the Planner's intent
        would be lost."""
        per_bucket = getattr(self.config, "cpu_num_threads_by_bucket", None)
        if per_bucket is None:
            return
        unknown = set(per_bucket.keys()) - set(self._dispatch_buckets)
        if unknown:
            raise ValueError(
                f"cots: cpu_num_threads_by_bucket has keys "
                f"{sorted(unknown)} that are not in COTS dispatch buckets "
                f"({self._dispatch_buckets}). Per-bucket thread policy must "
                f"only reference dispatch buckets."
            )
        for b, n in per_bucket.items():
            if n < 1:
                raise ValueError(
                    f"cots: cpu_num_threads_by_bucket[{b}] = {n}, must be >= 1"
                )

    def _native_routing_uniform_across_buckets(self) -> bool:
        """Whether compile-visible operator geometry is bucket-invariant.

        The dispatch bucket still selects native task ids at runtime, while the
        route signature selects the compiled graph variant for Python-visible
        geometry (`n_prefetch`, `n_cpu_compute`, scatter indices, GPU branch
        shape). This helper remains useful as a diagnostic and for tests that
        distinguish uniform and nonuniform routing.
        """
        if not self._dispatch_buckets:
            return True
        for h in self._handles:
            pref_values = {
                int(h.n_prefetch_by_bucket.get(bucket, 0))
                for bucket in self._dispatch_buckets
            }
            cpu_values = {
                int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu))
                for bucket in self._dispatch_buckets
            }
            if len(pref_values) > 1 or len(cpu_values) > 1:
                return False
        return True

    def _compute_has_cpu_compute_work(self) -> bool:
        """Whether any dispatch bucket leaves rows for CPU GEMM."""
        for h in self._handles:
            for bucket in self._dispatch_buckets:
                if int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)) > 0:
                    return True
        return False

    def _build_route_signatures(self) -> None:
        """Assign stable ids to compile-visible COTS route geometries.

        The dispatch bucket selects the Planner row and native slabs. The
        route signature captures only Python-visible geometry that can change
        the traced graph: CPU/prefetch slice sizes and therefore branch/scatter
        structure. Buckets with identical geometry share a signature so
        torch.compile does not build redundant variants.
        """
        signature_for_geometry: dict[tuple[tuple[str, int, int], ...], int] = {}
        route_signature_by_bucket: dict[int, int] = {}
        for bucket in self._dispatch_buckets:
            geometry = tuple(
                (
                    h.role,
                    int(h.n_prefetch_by_bucket.get(bucket, 0)),
                    int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)),
                )
                for h in self._handles
            )
            signature = signature_for_geometry.get(geometry)
            if signature is None:
                signature = len(signature_for_geometry) + 1
                signature_for_geometry[geometry] = signature
            route_signature_by_bucket[int(bucket)] = int(signature)
        self._route_signature_by_bucket = route_signature_by_bucket

    def _route_signature_for_bucket(self, bucket: int) -> int:
        if not self._route_signature_by_bucket and self._handles:
            self._build_route_signatures()
        return int(self._route_signature_by_bucket.get(int(bucket), int(bucket)))

    def _build_native_slab_specs(self) -> list[NativeWeightSlabSpec]:
        """Build the per-(layer, bucket, op_kind) slab specs that
        NativeCotsWeightRunner.install populates into the C++ slab pool. All
        weight pointers are POST-narrow `data_ptr()`s; the down-proj
        slabs additionally carry strides reflecting the source tensor.
        Each slab's `n_threads` is per-bucket via `_n_threads_for` so
        the C++ worker dispatcher's cache-guarded `at::set_num_threads`
        picks up the per-`BatchDescriptor` policy.
        """
        assert self._x_pinned is not None
        assert self._y_pinned is not None
        x_pinned_ptr = int(self._x_pinned.data_ptr())
        y_pinned_ptr = int(self._y_pinned.data_ptr())
        specs: list[NativeWeightSlabSpec] = []
        for h in self._handles:
            if h.role not in (QKV_ROLE, WO_ROLE, WO_INPUT_ROLE):
                continue
            assert h.w_cpu is not None
            op_kind = LINEAR_OP_KIND_BY_ROLE[h.role]
            for bucket in self._dispatch_buckets:
                n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu = h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)
                if n_cpu == 0:
                    continue
                w_view = h.w_cpu.narrow(0, n_pref, n_cpu)
                if h.role == WO_INPUT_ROLE:
                    specs.append(
                        _NativeWeightSlabSpecInputSplitLinear(
                            op_descriptor=(h.layer_idx, bucket, op_kind),
                            n_threads=self._n_threads_for(bucket),
                            x_pinned_ptr=x_pinned_ptr,
                            in_dim=int(n_cpu),
                            x_col_offset=int(
                                h.cpu_compute_input_start_by_bucket[bucket]
                            ),
                            y_pinned_ptr=y_pinned_ptr,
                            cpu_out_dim=int(h.out_dim),
                            w_cpu_ptr=int(w_view.data_ptr()),
                            w_cpu_rows=int(w_view.shape[0]),
                            w_cpu_cols=int(w_view.shape[1]),
                        )
                    )
                else:
                    specs.append(
                        _NativeWeightSlabSpecLinear(
                            op_descriptor=(h.layer_idx, bucket, op_kind),
                            n_threads=self._n_threads_for(bucket),
                            x_pinned_ptr=x_pinned_ptr,
                            in_dim=int(h.in_dim),
                            y_pinned_ptr=y_pinned_ptr,
                            cpu_out_dim=int(n_cpu),
                            w_cpu_ptr=int(w_view.data_ptr()),
                            w_cpu_rows=int(w_view.shape[0]),
                        )
                    )
        for fop in self._fused_ops:
            gu_h = fop._gate_up
            dn_h = fop._down
            assert gu_h.w_cpu is not None
            assert dn_h.w_cpu is not None
            n_cpu_per_half_total = gu_h.n_cpu // 2
            for bucket in self._dispatch_buckets:
                gu_n_pref = gu_h.n_prefetch_by_bucket.get(bucket, 0)
                dn_n_pref = dn_h.n_prefetch_by_bucket.get(bucket, 0)
                dn_n_cpu = dn_h.n_cpu_compute_by_bucket.get(bucket, dn_h.n_cpu)
                if dn_n_cpu == 0:
                    continue
                n_pref_per_half = gu_n_pref // 2
                w_gate_view = gu_h.w_cpu[n_pref_per_half:n_cpu_per_half_total, :]
                w_up_view = gu_h.w_cpu[
                    n_cpu_per_half_total + n_pref_per_half : 2 * n_cpu_per_half_total,
                    :,
                ]
                w_down_view = dn_h.w_cpu.narrow(0, dn_n_pref, dn_n_cpu)
                specs.append(
                    _NativeWeightSlabSpecMlp(
                        op_descriptor=(gu_h.layer_idx, bucket, "mlp_block"),
                        n_threads=self._n_threads_for(bucket),
                        x_pinned_ptr=x_pinned_ptr,
                        in_dim=int(fop._in_dim),
                        y_pinned_ptr=y_pinned_ptr,
                        cpu_out_dim=int(fop._out_dim),
                        w_gate_ptr=int(w_gate_view.data_ptr()),
                        w_gate_rows=int(w_gate_view.shape[0]),
                        w_up_ptr=int(w_up_view.data_ptr()),
                        w_up_rows=int(w_up_view.shape[0]),
                        w_down_ptr=int(w_down_view.data_ptr()),
                        w_down_rows=int(w_down_view.shape[0]),
                        w_down_cols=int(w_down_view.shape[1]),
                    )
                )
        return specs

    def _install_runner(self) -> None:
        """Hand the per-bucket work table to the runner. Called from
        `post_init` after weights have loaded (closures / slab pointers
        are stable post-install regardless of when they were taken,
        but post_init is the natural ordering point)."""
        if self._runner is None or not self._handles or not self._has_cpu_compute_work:
            return
        # Validate per-bucket thread policy keys before building specs so a
        # Planner-mistyped bucket fails loudly at install.
        self._validate_thread_policy()
        if isinstance(self._runner, PythonCotsWeightRunner):
            callbacks = self._build_python_callbacks()
            self._runner.install(callbacks)
        elif isinstance(self._runner, NativeCotsWeightRunner):
            slab_specs = self._build_native_slab_specs()
            # `max_num_tokens` gates the C++ worker's submit-side and
            # run-side bounds checks against the pinned x/y buffers.
            self._runner.install(
                slab_specs=slab_specs,
                max_num_tokens=int(self._max_num_tokens),
            )
            # Optional worker-thread CPU affinity. None / empty list means
            # "no opinion" and leaves the kernel default unchanged.
            cpu_affinity = getattr(self.config, "cpu_worker_affinity", None)
            if cpu_affinity:
                from vllm.model_executor.offloader import cots_ops

                mask = 0
                for cpu_id in cpu_affinity:
                    if not (0 <= int(cpu_id) < 64):
                        raise ValueError(
                            f"cots: cpu_worker_affinity contains cpu_id "
                            f"{cpu_id}; must be in [0, 64)"
                        )
                    mask |= 1 << int(cpu_id)
                cots_ops.set_worker_affinity(self._runner._runner_id, mask)

    def _cots_snap_payload(
        self,
        *,
        cpu_weight_bytes: int,
        gpu_output_scratch_bytes: int,
        gpu_prefetch_pool_bytes: int,
    ) -> dict[str, object]:
        """Structured runtime realization for profiler/planner handoff."""

        gpu_buffer_bytes = int(gpu_output_scratch_bytes + gpu_prefetch_pool_bytes)
        storage_key = f"{float(self.f_cpu_store):.12g}"
        payload: dict[str, object] = {
            "schema_version": 1,
            "snap_model": "cots_snap_v1",
            "kv_mode": self.kv_mode,
            "wo_qkvo_granularity_multiplier": WO_QKVO_GRANULARITY_MULTIPLIER,
            "storage_by_store_fraction": {
                storage_key: {
                    "cpu_weight_bytes": int(cpu_weight_bytes),
                    "gpu_buffer_bytes": gpu_buffer_bytes,
                    "gpu_output_scratch_bytes": int(gpu_output_scratch_bytes),
                    "gpu_prefetch_pool_bytes": int(gpu_prefetch_pool_bytes),
                }
            },
            "modules": sorted(self.weight_modules),
            "linears": len(self._handles),
            "mlp_blocks": len(self._fused_ops),
            "wo_ops": len(self._wo_ops),
            "graph_buckets": list(self._graph_capture_buckets),
            "dispatch_buckets": list(self._dispatch_buckets),
        }
        if self._handles:
            payload["dtype"] = str(self._handles[0].dtype).replace("torch.", "")

        role_names = {
            QKV_ROLE: "qkv",
            MLP_GATE_UP_ROLE: "mlp_gate_up",
            MLP_DOWN_ROLE: "mlp_down",
            WO_ROLE: "wo",
            WO_INPUT_ROLE: "wo",
        }
        dispatch_by_bucket: dict[str, object] = {}
        for bucket in self._dispatch_buckets:
            by_role: dict[str, dict[str, int]] = {}
            total_prefetch = 0
            total_cpu_compute = 0
            for h in self._handles:
                role = role_names[h.role]
                n_prefetch = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu_compute = h.n_cpu_compute_by_bucket.get(bucket, 0)
                other_dim = h.in_dim if h.split_axis != INPUT_SPLIT_AXIS else h.out_dim
                element_size = h.dtype.itemsize
                prefetch_bytes = int(n_prefetch * other_dim * element_size)
                cpu_compute_bytes = int(n_cpu_compute * other_dim * element_size)
                row = by_role.setdefault(
                    role,
                    {
                        "prefetch_weight_bytes": 0,
                        "cpu_compute_weight_bytes": 0,
                    },
                )
                row["prefetch_weight_bytes"] += prefetch_bytes
                row["cpu_compute_weight_bytes"] += cpu_compute_bytes
                total_prefetch += prefetch_bytes
                total_cpu_compute += cpu_compute_bytes
            f_cpu, f_prefetch = self._dispatch_table.get(bucket, (0.0, 0.0))
            dispatch_by_bucket[str(bucket)] = {
                "requested_f_cpu_compute": round(float(f_cpu), 12),
                "requested_f_prefetch_compute": round(float(f_prefetch), 12),
                "prefetch_weight_bytes": total_prefetch,
                "cpu_compute_weight_bytes": total_cpu_compute,
                "by_role": by_role,
            }
        payload["dispatch_by_bucket"] = dispatch_by_bucket
        return payload

    def _hook_layer_forward(self, index: int, layer: nn.Module) -> None:
        """Wrap the decoder layer's `forward` with pre-compute scheduling.

        For layer i: wait for layer i's prefetched weights, start prefetch
        for layer i+1, then run layer i. Experimental KV prefetch reuses this
        same custom-op lifecycle when enabled. With K=2 slot rotation, i reads
        slot i%2 while i+1 writes slot (i+1)%2, so H2D overlaps with layer i
        compute without a wraparound special case.
        """
        original_forward = layer.forward
        n_layers = len(self._layer_modules)

        layer_has_weight_prefetch = any(
            h.max_n_prefetch > 0 for h in self._layer_handles[index]
        )
        layer_has_kv_prefetch = (
            self._kv_prefetch_streamer is not None
            and self._kv_prefetch_streamer.has_layer_prefetch(index)
        )
        layer_has_prefetch = layer_has_weight_prefetch or layer_has_kv_prefetch
        next_idx = (index + 1) % n_layers if n_layers > 0 else 0
        if n_layers > 1 and next_idx != index:
            next_has_weight_prefetch = any(
                h.max_n_prefetch > 0 for h in self._layer_handles[next_idx]
            )
            next_has_kv_prefetch = (
                self._kv_prefetch_streamer is not None
                and self._kv_prefetch_streamer.has_layer_prefetch(next_idx)
            )
            next_has_prefetch = next_has_weight_prefetch or next_has_kv_prefetch
        else:
            next_has_prefetch = False

        def forward(*args, **kwargs):
            layer.forward = original_forward
            anchor = args[0] if args else next(iter(kwargs.values()))
            if layer_has_prefetch:
                torch.ops.vllm.wait_prefetch(anchor, index)
            if next_has_prefetch:
                torch.ops.vllm.start_prefetch(anchor, next_idx)
            output = original_forward(*args, **kwargs)
            layer.forward = forward
            return output

        layer.forward = forward

    def _dispatch_bucket_for(self, num_tokens: int) -> int:
        """Round-up lookup on `_dispatch_buckets`.

        Returns the Planner/COTS dispatch bucket key that should select routing
        geometry and native runner slabs. Out-of-range returns the largest
        dispatch bucket.

        This used to call `bisect.bisect_left`, but the C builtin was not
        Dynamo-friendly when bucket repair lived in a traced pre-hook. Keeping
        the simple linear scan avoids reintroducing that constraint, and this
        runs once per forward boundary rather than per GEMM.
        """
        for bucket in self._dispatch_buckets:
            if num_tokens <= bucket:
                return bucket
        return self._dispatch_buckets[-1]

    def _dispatch_bucket_from_descriptor(self, batch_descriptor) -> int:
        bucket = getattr(batch_descriptor, "cots_dispatch_bucket", None)
        if bucket is None:
            return self._dispatch_bucket_for(int(batch_descriptor.num_tokens))
        bucket = int(bucket)
        if bucket not in self._dispatch_buckets:
            raise RuntimeError(
                "CotsOffloader: BatchDescriptor carries unknown "
                f"cots_dispatch_bucket={bucket}; known dispatch buckets are "
                f"{self._dispatch_buckets}."
            )
        return bucket

    def decorate_batch_descriptor(
        self, batch_descriptor: BatchDescriptor
    ) -> BatchDescriptor:
        if not self._dispatch_buckets:
            self._resolve_bucket_sets()
        bucket = self._dispatch_bucket_for(int(batch_descriptor.num_tokens))
        signature = self._route_signature_for_bucket(bucket)
        if (
            batch_descriptor.cots_dispatch_bucket == bucket
            and batch_descriptor.cots_route_signature == signature
        ):
            return batch_descriptor
        return replace(
            batch_descriptor,
            cots_dispatch_bucket=bucket,
            cots_route_signature=signature,
        )

    def _prepare_before_forward_bucket(self, num_tokens: int, bucket: int) -> None:
        self._current_bucket = int(bucket)
        if self._kv_prefetch_streamer is not None:
            self._kv_prefetch_streamer.current_bucket = int(bucket)
            self._kv_prefetch_streamer.prepare_for_forward_bucket(0)
        if self._streamer is None:
            return
        self._streamer.current_bucket = int(bucket)
        if self._layer_handles:
            self._streamer.prepare_for_forward_bucket(0, self._layer_handles[0])

    # --- BaseOffloader lifecycle delegation ---

    def prepare_before_forward(self, num_tokens: int) -> None:
        """Repair active-bucket state before a forward starts.

        Always sets `_current_bucket` (plan §design-decision 11) so the
        operator slab/closure lookup has a valid bucket regardless of
        whether prefetch is active. Layer-0 slot repair and streamer
        bucket mirroring run only when the option-A streamer exists.
        Steady-state next-layer prefetches are
        emitted inside each layer wrapper so FULL CUDA graph capture
        records them as graph nodes rather than relying on replay-time
        Python state.

        Kept free of pybind calls so it can be used from graph-boundary helper
        paths. The C++ runtime token row cap is pushed separately by
        `on_dispatch`, outside captured graphs.
        """
        self._prepare_before_forward_bucket(
            num_tokens, self._dispatch_bucket_for(num_tokens)
        )

    def _operator_bucket(self, x_rows: int) -> int:
        """Return the operator routing bucket for the active forward.

        Native COTS must see the explicit OOG dispatch boundary before
        any operator runs; falling back to `x.shape[0]` is the bug this
        path exists to remove. The fallback remains only for Python-runner
        direct tests and eager kill-switch use.
        """
        if self._current_bucket is not None:
            return int(self._current_bucket)
        if isinstance(self._runner, NativeCotsWeightRunner):
            raise RuntimeError(
                "CotsOffloader operator ran before dispatch state was "
                "published. GPUModelRunner._publish_forward_dispatch "
                "must call CotsOffloader.on_dispatch before native COTS "
                "operators execute or capture."
            )
        return self._dispatch_bucket_for(x_rows)

    @property
    def head_split_activation_routing_enabled(self) -> bool:
        return self.kv_mode == "head_split"

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().data_ptr())

    def _clear_head_split_sidecars(self) -> None:
        self._head_split_qkv_sidecars.clear()
        self._head_split_attention_outputs.clear()

    def set_head_split_cpu_positions(
        self,
        positions_cpu: Sequence[int] | torch.Tensor | None,
        num_tokens: int,
    ) -> None:
        """Publish CPU token positions for CPU-side head-split RoPE.

        The worker already derives positions on CPU before filling vLLM's GPU
        position tensor. Reusing that source avoids a per-layer GPU->CPU copy
        in the CPU RoPE path.
        """

        if not self.head_split_activation_routing_enabled or positions_cpu is None:
            self._head_split_positions_cpu = None
            return
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            self._head_split_positions_cpu = None
            return

        if isinstance(positions_cpu, torch.Tensor):
            positions = positions_cpu[:num_tokens].flatten()
            if positions.device.type != "cpu":
                raise RuntimeError(
                    "COTS head-split CPU positions must be published from CPU; "
                    f"got {positions.device}"
                )
            if positions.dtype != torch.long:
                positions = positions.to(dtype=torch.long)
        else:
            positions = torch.as_tensor(
                positions_cpu[:num_tokens],
                dtype=torch.long,
                device="cpu",
            ).flatten()
        if int(positions.numel()) < num_tokens:
            raise RuntimeError(
                "COTS head-split CPU positions are shorter than the active "
                f"forward: got={int(positions.numel())}, needed={num_tokens}"
            )
        self._head_split_positions_cpu = positions

    def _head_split_qkv_work_views(
        self,
        *,
        num_tokens: int,
        cpu_attention_groups: int,
        q_heads_per_kv: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reusable pinned CPU Q/K/V work views for head-split attention."""

        signature = (cpu_attention_groups, q_heads_per_kv, head_dim, dtype)
        capacity = max(int(num_tokens), int(self._max_num_tokens), 1)
        if (
            self._head_split_q_work is None
            or self._head_split_k_work is None
            or self._head_split_v_work is None
            or self._head_split_qkv_work_signature != signature
            or self._head_split_qkv_work_capacity < capacity
        ):
            self._head_split_qkv_work_signature = signature
            self._head_split_qkv_work_capacity = capacity
            self._head_split_q_work = torch.empty(
                (capacity, cpu_attention_groups * q_heads_per_kv, head_dim),
                dtype=dtype,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            self._head_split_k_work = torch.empty(
                (capacity, cpu_attention_groups, head_dim),
                dtype=dtype,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            self._head_split_v_work = torch.empty_like(self._head_split_k_work)

        assert self._head_split_q_work is not None
        assert self._head_split_k_work is not None
        assert self._head_split_v_work is not None
        return (
            self._head_split_q_work[:num_tokens],
            self._head_split_k_work[:num_tokens],
            self._head_split_v_work[:num_tokens],
        )

    def _head_split_live_num_tokens(
        self,
        *,
        num_tokens_padded: int,
        out_cpu: torch.Tensor | None,
    ) -> int:
        """Return the live row count for CPU-side routed activations."""

        num_tokens_padded = int(num_tokens_padded)
        if num_tokens_padded < 0:
            raise RuntimeError(
                "COTS head-split QKV route got a negative padded token count: "
                f"{num_tokens_padded}"
            )
        live = self._live_num_tokens
        if live is None and out_cpu is not None:
            live = int(out_cpu.shape[0])
        if live is None:
            live = num_tokens_padded
        live = int(live)
        if live < 0 or live > num_tokens_padded:
            raise RuntimeError(
                "COTS head-split QKV route live token count is outside the "
                f"padded activation shape: live={live}, padded={num_tokens_padded}"
            )
        if out_cpu is not None:
            # Native CPU slabs are dispatch-bucket capacity sized. Profile
            # dummy runs may have many more model rows than the largest COTS
            # dispatch bucket, so the pinned output row count is the
            # authoritative row count for CPU-produced routed activations.
            live = min(live, int(out_cpu.shape[0]))
        return live

    def _qkv_group_route(
        self,
        h: CotsLinearHandle,
        bucket: int,
    ) -> tuple[int, int, int, int, int, int, int]:
        if (
            h.role != QKV_ROLE
            or h.qkv_cpu_layout != "gqa_group"
            or h.num_kv_heads is None
            or h.head_dim is None
        ):
            raise RuntimeError(
                "COTS head-split activation routing requires GQA-group QKV "
                f"handle, got {h.qualified_name}"
            )
        n_cpu = int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu))
        qkv_group = int(h.gqa_qkv_group_size)
        if n_cpu % qkv_group != 0:
            raise RuntimeError(
                f"COTS head-split QKV CPU rows are not group-aligned: "
                f"n_cpu={n_cpu}, group={qkv_group}"
            )
        num_groups = int(h.num_kv_heads)
        cpu_attention_groups = gqa_num_cpu_groups(
            float(getattr(self.config, "f_cpu_kv_store", 0.0)),
            num_kv_heads=num_groups,
        )
        if not (0 <= cpu_attention_groups < num_groups):
            raise RuntimeError(
                "COTS head-split activation routing requires at least one "
                "GPU attention group"
            )
        cpu_weight_groups = n_cpu // qkv_group
        cpu_compute_kv_heads, prefetch_kv_heads = self._lookup_head_split_kv_dispatch(
            bucket=bucket,
            cpu_attention_groups=cpu_attention_groups,
            cpu_weight_groups=cpu_weight_groups,
        )
        return (
            num_groups,
            cpu_attention_groups,
            cpu_weight_groups,
            cpu_compute_kv_heads,
            prefetch_kv_heads,
            int(h.gqa_q_group_size),
            int(h.head_dim),
        )

    def _lookup_head_split_kv_dispatch(
        self,
        *,
        bucket: int,
        cpu_attention_groups: int,
        cpu_weight_groups: int,
    ) -> tuple[int, int]:
        """Return the planner-emitted 3-way KV dispatch row for one bucket.

        ``cpu_attention_groups`` is the static CPU-owned KV count ``A``. The
        returned pair is ``(C, P)`` where ``C`` is the CPU-compute attention
        island and ``P`` is the CPU-owned KV prefix that the 3-way path
        prefetches to GPU attention. vLLM validates this row but does not
        derive policy when 3-way KV prefetch is enabled.
        """

        if not self.kv_head_prefetch_enabled:
            return int(cpu_attention_groups), 0
        entry = self._dispatch_entries.get(int(bucket))
        if entry is None or entry.kv_group_pair is None:
            raise RuntimeError(
                "COTS 3-way head-split KV routing requires a compact dispatch "
                f"row for bucket {bucket} with "
                "(cpu_kv_compute_groups, kv_prefetch_groups)."
            )
        cpu_compute, prefetch = entry.kv_group_pair
        if cpu_compute < 0 or prefetch < 0:
            raise RuntimeError(
                "COTS head-split KV dispatch row must be non-negative: "
                f"C={cpu_compute}, P={prefetch}"
            )
        if cpu_compute + prefetch != int(cpu_attention_groups):
            raise RuntimeError(
                "COTS head-split KV dispatch row must cover all CPU-owned "
                f"KV groups: C={cpu_compute}, P={prefetch}, "
                f"A={cpu_attention_groups}"
            )
        if int(cpu_compute) != int(cpu_weight_groups):
            raise RuntimeError(
                "COTS head-split KV dispatch row must match snapped WQKV "
                f"CPU-compute groups: C={cpu_compute}, "
                f"WQKV={cpu_weight_groups}, bucket={bucket}"
            )
        return int(cpu_compute), int(prefetch)

    def route_head_split_qkv_output(
        self,
        *,
        handle: CotsLinearHandle,
        bucket: int,
        num_tokens: int,
        reference: torch.Tensor,
        out_perm: torch.Tensor | None,
        out_pref: torch.Tensor | None,
        out_cpu: torch.Tensor | None,
        pref_idx: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build GPU QKV for GPU-owned groups and CPU sidecar for CPU groups."""

        timer_start = _cots_py_timer_start()
        h = handle
        assert h.gpu_indices_cuda is not None
        assert h.q_size is not None and h.kv_size is not None
        (
            num_groups,
            cpu_attention_groups,
            cpu_weight_groups,
            cpu_compute_kv_heads,
            prefetch_kv_heads,
            q_group,
            head_dim,
        ) = self._qkv_group_route(h, bucket)
        q_heads_per_kv = q_group // head_dim
        qkv_group = q_group + 2 * head_dim
        q_size = int(h.q_size)
        kv_size = int(h.kv_size)
        cpu_compute_start = num_groups - cpu_weight_groups
        cpu_sidecar_groups = cpu_compute_kv_heads
        cpu_sidecar_start = num_groups - cpu_sidecar_groups
        if self.kv_head_prefetch_enabled and cpu_sidecar_groups != cpu_weight_groups:
            raise RuntimeError(
                "COTS 3-way head-split QKV route requires WQKV CPU-compute "
                "groups to match the compact CPU-attention sidecar: "
                f"WQKV={cpu_weight_groups}, C={cpu_sidecar_groups}, "
                f"P={prefetch_kv_heads}, bucket={bucket}"
            )

        num_tokens_padded = int(num_tokens)
        num_tokens_live = self._head_split_live_num_tokens(
            num_tokens_padded=num_tokens_padded,
            out_cpu=out_cpu,
        )

        out = torch.empty(
            (num_tokens_padded, h.out_dim),
            dtype=reference.dtype,
            device=reference.device,
        )
        if out_perm is not None:
            out.index_copy_(1, h.gpu_indices_cuda, out_perm)
        if out_pref is not None:
            out.index_copy_(1, pref_idx, out_pref)
        if bias is not None:
            out = out + bias

        q_cpu, k_cpu, v_cpu = self._head_split_qkv_work_views(
            num_tokens=num_tokens_live,
            cpu_attention_groups=cpu_sidecar_groups,
            q_heads_per_kv=q_heads_per_kv,
            head_dim=head_dim,
            dtype=h.dtype,
        )

        def cpu_group_with_bias(global_group: int) -> torch.Tensor:
            if out_cpu is None:
                raise RuntimeError(
                    "CPU-produced QKV group requested without CPU output"
                )
            local_group = global_group - cpu_compute_start
            group_start = local_group * qkv_group
            group_end = (local_group + 1) * qkv_group
            group = out_cpu[:num_tokens_live, group_start:group_end]
            group = group.contiguous()
            if bias is None:
                return group
            q_start = global_group * q_group
            k_start = q_size + global_group * head_dim
            v_start = q_size + kv_size + global_group * head_dim
            bias_group = torch.cat(
                [
                    bias[q_start : q_start + q_group],
                    bias[k_start : k_start + head_dim],
                    bias[v_start : v_start + head_dim],
                ]
            ).to(device="cpu")
            return group + bias_group

        def copy_group_to_cpu(
            global_group: int,
            local_attention: int,
        ) -> None:
            q_dst = q_cpu[
                :,
                local_attention * q_heads_per_kv : (local_attention + 1)
                * q_heads_per_kv,
                :,
            ]
            k_dst = k_cpu[:, local_attention : local_attention + 1, :]
            v_dst = v_cpu[:, local_attention : local_attention + 1, :]
            if global_group >= cpu_compute_start and cpu_weight_groups > 0:
                group = cpu_group_with_bias(global_group)
                q_view = group[:, :q_group].view(
                    num_tokens_live, q_heads_per_kv, head_dim
                )
                q_dst.copy_(q_view)
                k_dst.copy_(
                    group[:, q_group : q_group + head_dim].view(
                        num_tokens_live, 1, head_dim
                    )
                )
                v_dst.copy_(
                    group[:, q_group + head_dim : qkv_group].view(
                        num_tokens_live, 1, head_dim
                    )
                )
                return

            q_start = global_group * q_group
            k_start = q_size + global_group * head_dim
            v_start = q_size + kv_size + global_group * head_dim
            q_src = (
                out[:num_tokens_live, q_start : q_start + q_group]
                .detach()
                .to(device="cpu")
            )
            k_src = (
                out[:num_tokens_live, k_start : k_start + head_dim]
                .detach()
                .to(device="cpu")
            )
            v_src = (
                out[:num_tokens_live, v_start : v_start + head_dim]
                .detach()
                .to(device="cpu")
            )
            q_dst.copy_(q_src.view(num_tokens_live, q_heads_per_kv, head_dim))
            k_dst.copy_(k_src.view(num_tokens_live, 1, head_dim))
            v_dst.copy_(v_src.view(num_tokens_live, 1, head_dim))

        # CPU-computed groups that do not run CPU attention are the activation
        # mismatch. Insert only those groups into the canonical GPU QKV tensor.
        if not self.kv_head_prefetch_enabled and cpu_weight_groups > cpu_sidecar_groups:
            for global_group in range(cpu_compute_start, cpu_sidecar_start):
                group = cpu_group_with_bias(global_group).to(
                    device=reference.device, non_blocking=True
                )
                q_start = global_group * q_group
                k_start = q_size + global_group * head_dim
                v_start = q_size + kv_size + global_group * head_dim
                out[:num_tokens_live, q_start : q_start + q_group].copy_(
                    group[:, :q_group]
                )
                out[:num_tokens_live, k_start : k_start + head_dim].copy_(
                    group[:, q_group : q_group + head_dim]
                )
                out[:num_tokens_live, v_start : v_start + head_dim].copy_(
                    group[:, q_group + head_dim : qkv_group]
                )
        qkv_h2d_groups = max(0, cpu_weight_groups - cpu_sidecar_groups)

        for local_attention, global_group in enumerate(
            range(cpu_sidecar_start, num_groups)
        ):
            copy_group_to_cpu(global_group, local_attention)
        qkv_d2h_groups = max(0, cpu_sidecar_groups - cpu_weight_groups)

        storage_key = self._storage_key(out)
        self._head_split_qkv_sidecars[storage_key] = CotsHeadSplitQKVSidecar(
            storage_key=storage_key,
            num_tokens=num_tokens_live,
            num_groups=num_groups,
            cpu_attention_groups=cpu_sidecar_groups,
            cpu_weight_groups=cpu_weight_groups,
            cpu_compute_kv_heads=cpu_compute_kv_heads,
            prefetch_kv_heads=prefetch_kv_heads,
            q_heads_per_kv=q_heads_per_kv,
            head_dim=head_dim,
            query=q_cpu,
            key=k_cpu,
            value=v_cpu,
            rope_applied=cpu_sidecar_groups == 0,
        )
        if _COTS_COUNTERS_ENABLED:
            phase = _head_split_phase(num_tokens_live)
            element_bytes = _dtype_nbytes(h.dtype)
            qkv_d2h_elements = qkv_d2h_groups * qkv_group
            _cots_py_counter("head_split_qkv_route_tokens", num_tokens_live)
            _cots_py_counter(
                f"head_split_qkv_route_{phase}_tokens",
                num_tokens_live,
            )
            _cots_py_counter("head_split_qkv_route_layers")
            _cots_py_counter(f"head_split_qkv_route_{phase}_layers")
            _cots_py_counter(
                "head_split_qkv_route_cpu_attention_groups",
                cpu_sidecar_groups,
            )
            _cots_py_counter(
                "head_split_qkv_route_cpu_weight_groups",
                cpu_weight_groups,
            )
            _cots_py_counter(
                "head_split_qkv_route_d2h_bytes",
                num_tokens_live * qkv_d2h_elements * element_bytes,
            )
            _cots_py_counter(
                f"head_split_qkv_route_{phase}_d2h_bytes",
                num_tokens_live * qkv_d2h_elements * element_bytes,
            )
            _cots_py_counter(
                "head_split_qkv_route_h2d_bytes",
                num_tokens_live * qkv_h2d_groups * qkv_group * element_bytes,
            )
            _cots_py_counter(
                f"head_split_qkv_route_{phase}_h2d_bytes",
                num_tokens_live * qkv_h2d_groups * qkv_group * element_bytes,
            )
            _cots_py_timing(f"head_split_qkv_route_{phase}", timer_start)
        _cots_py_timing("head_split_qkv_route", timer_start)
        return out

    def maybe_apply_head_split_cpu_rope(
        self,
        *,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> None:
        if key is None or not self.head_split_activation_routing_enabled:
            return
        sidecar = self._head_split_qkv_sidecars.get(self._storage_key(query))
        if sidecar is None or sidecar.rope_applied:
            return
        if sidecar.cpu_attention_groups == 0:
            sidecar.rope_applied = True
            return
        timer_start = _cots_py_timer_start()
        if head_size != sidecar.head_dim:
            raise RuntimeError(
                f"COTS head-split sidecar head_dim={sidecar.head_dim} but "
                f"RoPE head_size={head_size}"
            )

        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if cos_sin_cache.device.type != "cpu":
            raise RuntimeError(
                "COTS head-split CPU RoPE requires a CPU cos/sin cache; "
                f"got {cos_sin_cache.device}"
            )
        positions_cpu = self._head_split_positions_cpu
        if positions_cpu is None:
            if positions.device.type != "cpu":
                raise RuntimeError(
                    "COTS head-split CPU RoPE is missing CPU positions. "
                    "GPU positions are not copied back in the head-split path."
                )
            positions_cpu = positions.flatten().detach().to(dtype=torch.long)
        positions_cpu = positions_cpu[: sidecar.num_tokens]
        if int(positions_cpu.numel()) < sidecar.num_tokens:
            raise RuntimeError(
                "COTS head-split CPU RoPE positions are shorter than the "
                f"sidecar: got={int(positions_cpu.numel())}, "
                f"needed={sidecar.num_tokens}"
            )
        cos_sin = cos_sin_cache.to(dtype=sidecar.query.dtype)
        cos_sin = cos_sin.index_select(0, positions_cpu)
        cos, sin = cos_sin.chunk(2, dim=-1)

        q_shape = sidecar.query.shape
        q = sidecar.query.view(q_shape[0], -1, head_size)
        q_rot = ApplyRotaryEmb.forward_static(
            q[..., :rotary_dim],
            cos,
            sin,
            is_neox_style,
            enable_fp32_compute=True,
        )
        q[..., :rotary_dim].copy_(q_rot)

        k_shape = sidecar.key.shape
        k = sidecar.key.view(k_shape[0], -1, head_size)
        k_rot = ApplyRotaryEmb.forward_static(
            k[..., :rotary_dim],
            cos,
            sin,
            is_neox_style,
            enable_fp32_compute=True,
        )
        k[..., :rotary_dim].copy_(k_rot)
        sidecar.rope_applied = True
        if _COTS_COUNTERS_ENABLED:
            phase = _head_split_phase(sidecar.num_tokens)
            _cots_py_counter("head_split_cpu_rope_tokens", sidecar.num_tokens)
            _cots_py_counter(
                f"head_split_cpu_rope_{phase}_tokens",
                sidecar.num_tokens,
            )
            _cots_py_counter("head_split_cpu_rope_layers")
            _cots_py_counter(f"head_split_cpu_rope_{phase}_layers")
            _cots_py_counter(
                "head_split_cpu_rope_groups",
                sidecar.cpu_attention_groups,
            )
            _cots_py_timing(f"head_split_cpu_rope_{phase}", timer_start)
        _cots_py_timing("head_split_cpu_rope", timer_start)

    def head_split_cpu_rope_needed(self, query: torch.Tensor) -> bool:
        sidecar = self._head_split_qkv_sidecars.get(self._storage_key(query))
        return bool(
            sidecar is not None
            and not sidecar.rope_applied
            and sidecar.cpu_attention_groups > 0
        )

    def lookup_head_split_qkv_sidecar(
        self, tensor: torch.Tensor
    ) -> CotsHeadSplitQKVSidecar | None:
        return self._head_split_qkv_sidecars.get(self._storage_key(tensor))

    def register_head_split_attention_output(
        self,
        *,
        output: torch.Tensor,
        qkv_sidecar: CotsHeadSplitQKVSidecar,
        output_cpu: torch.Tensor,
    ) -> None:
        num_tokens = int(output_cpu.shape[0])
        num_groups = qkv_sidecar.num_groups
        q_heads_per_kv = qkv_sidecar.q_heads_per_kv
        if int(output_cpu.shape[1]) % q_heads_per_kv != 0:
            raise RuntimeError(
                "COTS head-split CPU attention output is not GQA aligned: "
                f"heads={output_cpu.shape[1]}, q_heads_per_kv={q_heads_per_kv}"
            )
        cpu_attention_groups = int(output_cpu.shape[1]) // q_heads_per_kv
        cpu_weight_groups = qkv_sidecar.cpu_weight_groups
        cpu_attention_start = num_groups - cpu_attention_groups
        timer_start = _cots_py_timer_start()
        h2d_groups = max(0, cpu_attention_groups - cpu_weight_groups)

        # C < A mismatch: CPU attention produced groups that GPU/PREFETCH WO
        # will consume. Copy only those leading CPU-attention groups to GPU.
        if cpu_weight_groups < cpu_attention_groups:
            for local_group in range(cpu_attention_groups - cpu_weight_groups):
                global_group = cpu_attention_start + local_group
                q_start = global_group * q_heads_per_kv
                q_end = q_start + q_heads_per_kv
                src_start = local_group * q_heads_per_kv
                src_end = src_start + q_heads_per_kv
                output[:num_tokens, q_start:q_end, :].copy_(
                    output_cpu[:, src_start:src_end, :].to(
                        device=output.device, non_blocking=True
                    )
                )

        storage_key = self._storage_key(output)
        self._head_split_attention_outputs[storage_key] = (
            CotsHeadSplitAttentionOutputSidecar(
                storage_key=storage_key,
                num_tokens=num_tokens,
                num_groups=num_groups,
                cpu_attention_groups=cpu_attention_groups,
                cpu_weight_groups=cpu_weight_groups,
                cpu_compute_kv_heads=qkv_sidecar.cpu_compute_kv_heads,
                prefetch_kv_heads=qkv_sidecar.prefetch_kv_heads,
                q_heads_per_kv=q_heads_per_kv,
                head_dim=qkv_sidecar.head_dim,
                output=output_cpu,
            )
        )
        self._head_split_qkv_sidecars.pop(qkv_sidecar.storage_key, None)
        if _COTS_COUNTERS_ENABLED:
            phase = _head_split_phase(num_tokens)
            element_bytes = _dtype_nbytes(output_cpu.dtype)
            _cots_py_counter("head_split_attention_output_route_tokens", num_tokens)
            _cots_py_counter(
                f"head_split_attention_output_route_{phase}_tokens",
                num_tokens,
            )
            _cots_py_counter("head_split_attention_output_route_layers")
            _cots_py_counter(f"head_split_attention_output_route_{phase}_layers")
            _cots_py_counter(
                "head_split_attention_output_route_h2d_bytes",
                num_tokens
                * h2d_groups
                * q_heads_per_kv
                * qkv_sidecar.head_dim
                * element_bytes,
            )
            _cots_py_counter(
                f"head_split_attention_output_route_{phase}_h2d_bytes",
                num_tokens
                * h2d_groups
                * q_heads_per_kv
                * qkv_sidecar.head_dim
                * element_bytes,
            )
            _cots_py_timing(
                f"head_split_attention_output_route_{phase}",
                timer_start,
            )
        _cots_py_timing("head_split_attention_output_route", timer_start)

    def register_head_split_gpu_attention_output(
        self,
        *,
        output: torch.Tensor,
        query: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Register a full-GPU attention result for A=0 head-split routing.

        When CPU KV is disabled, all attention heads live on GPU (A=0). The
        head-split QKV route can still have CPU-produced weight groups (C>0),
        so WO needs the same sidecar handoff as the CPU-attention path in
        order to gather those groups into the native pinned input.
        """

        if not self.head_split_activation_routing_enabled:
            return
        qkv_sidecar = self.lookup_head_split_qkv_sidecar(query)
        if qkv_sidecar is None:
            return
        if qkv_sidecar.cpu_attention_groups != 0:
            return

        empty_cpu_output = torch.empty(
            (int(num_tokens), 0, int(qkv_sidecar.head_dim)),
            dtype=output.dtype,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        storage_key = self._storage_key(output)
        self._head_split_attention_outputs[storage_key] = (
            CotsHeadSplitAttentionOutputSidecar(
                storage_key=storage_key,
                num_tokens=int(num_tokens),
                num_groups=qkv_sidecar.num_groups,
                cpu_attention_groups=0,
                cpu_weight_groups=qkv_sidecar.cpu_weight_groups,
                cpu_compute_kv_heads=qkv_sidecar.cpu_compute_kv_heads,
                prefetch_kv_heads=qkv_sidecar.prefetch_kv_heads,
                q_heads_per_kv=qkv_sidecar.q_heads_per_kv,
                head_dim=qkv_sidecar.head_dim,
                output=empty_cpu_output,
            )
        )
        self._head_split_qkv_sidecars.pop(qkv_sidecar.storage_key, None)

    def lookup_head_split_attention_output(
        self, tensor: torch.Tensor
    ) -> CotsHeadSplitAttentionOutputSidecar | None:
        return self._head_split_attention_outputs.get(self._storage_key(tensor))

    def build_head_split_wo_cpu_input(
        self,
        *,
        x: torch.Tensor,
        handle: CotsLinearHandle,
        bucket: int,
        num_tokens: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if handle.role != WO_INPUT_ROLE:
            return None
        sidecar = self.lookup_head_split_attention_output(x)
        if sidecar is None:
            return None
        timer_start = _cots_py_timer_start()
        q_group = int(handle.gqa_q_group_size)
        n_cpu = int(handle.n_cpu_compute_by_bucket.get(bucket, handle.n_cpu))
        if n_cpu <= 0:
            return None
        if n_cpu % q_group != 0:
            raise RuntimeError(
                f"COTS WO head-split CPU rows are not group-aligned: "
                f"n_cpu={n_cpu}, group={q_group}"
            )
        cpu_weight_groups = n_cpu // q_group
        if cpu_weight_groups != sidecar.cpu_weight_groups:
            raise RuntimeError(
                "COTS WQKV and WO CPU-compute group counts diverged for the "
                "active bucket: "
                f"qkv={sidecar.cpu_weight_groups}, wo={cpu_weight_groups}"
            )
        num_groups = sidecar.num_groups
        cpu_attention_groups = sidecar.cpu_attention_groups
        cpu_attention_start = num_groups - cpu_attention_groups
        cpu_compute_start = num_groups - cpu_weight_groups
        q_heads_per_kv = sidecar.q_heads_per_kv
        d2h_groups = max(0, cpu_weight_groups - cpu_attention_groups)

        if out is None:
            cpu_input = torch.empty(
                (num_tokens, n_cpu),
                dtype=x.dtype,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
        else:
            if tuple(out.shape) != (num_tokens, n_cpu):
                raise RuntimeError(
                    "COTS routed WO pinned input has wrong shape: "
                    f"got={tuple(out.shape)}, expected={(num_tokens, n_cpu)}"
                )
            if out.device.type != "cpu":
                raise RuntimeError(
                    "COTS routed WO input destination must be a CPU tensor, "
                    f"got {out.device}"
                )
            cpu_input = out
        for local_cpu, global_group in enumerate(range(cpu_compute_start, num_groups)):
            dst = cpu_input[:, local_cpu * q_group : (local_cpu + 1) * q_group]
            if global_group >= cpu_attention_start:
                local_attention = global_group - cpu_attention_start
                src = sidecar.output[
                    :num_tokens,
                    local_attention * q_heads_per_kv : (local_attention + 1)
                    * q_heads_per_kv,
                    :,
                ]
                dst.copy_(src.reshape(num_tokens, q_group))
            else:
                src = x[
                    :num_tokens,
                    global_group * q_group : (global_group + 1) * q_group,
                ].detach()
                dst.copy_(src, non_blocking=True)
        self._head_split_attention_outputs.pop(sidecar.storage_key, None)
        if _COTS_COUNTERS_ENABLED:
            phase = _head_split_phase(num_tokens)
            element_bytes = _dtype_nbytes(x.dtype)
            _cots_py_counter("head_split_wo_gather_tokens", num_tokens)
            _cots_py_counter(f"head_split_wo_gather_{phase}_tokens", num_tokens)
            _cots_py_counter("head_split_wo_gather_layers")
            _cots_py_counter(f"head_split_wo_gather_{phase}_layers")
            _cots_py_counter(
                "head_split_wo_gather_d2h_bytes",
                num_tokens * d2h_groups * q_group * element_bytes,
            )
            _cots_py_counter(
                f"head_split_wo_gather_{phase}_d2h_bytes",
                num_tokens * d2h_groups * q_group * element_bytes,
            )
            _cots_py_timing(f"head_split_wo_gather_{phase}", timer_start)
        _cots_py_timing("head_split_wo_gather", timer_start)
        return cpu_input

    def set_live_num_tokens(self, live_num_tokens: int) -> None:
        """Push the live unpadded token count to the C++ worker.

        CUDA graph buckets and native slabs are capacity-sized. This
        live-row cap lets the CPU worker skip padded rows inside the
        selected bucket.

        No-op when not using the native runner (PythonCotsWeightRunner is
        eager-only and reads the live count directly off
        `slab.num_tokens.store` at submit time). Also no-op for
        `live_num_tokens <= 0` (sentinel).
        """
        self._live_num_tokens = int(live_num_tokens)
        if not self._has_cpu_compute_work:
            return
        if not isinstance(self._runner, NativeCotsWeightRunner):
            return
        if int(live_num_tokens) <= 0:
            return
        from vllm.model_executor.offloader import cots_ops

        cots_ops.set_live_num_tokens(self._runner._runner_id, int(live_num_tokens))

    def _log_dispatch_trace(
        self,
        info: ForwardDispatchInfo,
        *,
        num_tokens_padded: int,
        num_tokens_unpadded: int,
        active_bucket: int,
    ) -> None:
        if (
            info.trace_context is None
            or os.environ.get("VLLM_COTS_DISPATCH_TRACE", "0") != "1"
        ):
            return

        f_cpu_compute, f_prefetch_compute = self._dispatch_table.get(
            active_bucket, (0.0, 0.0)
        )
        descriptor_bucket = getattr(info.batch_descriptor, "cots_dispatch_bucket", None)
        payload = {
            **dict(info.trace_context),
            "event": "cots_dispatch_trace",
            "pid": os.getpid(),
            "num_tokens_padded": int(num_tokens_padded),
            "num_tokens_unpadded": int(num_tokens_unpadded),
            "cots_dispatch_bucket": int(active_bucket),
            "dispatch_bucket_source": (
                "descriptor" if descriptor_bucket is not None else "num_tokens"
            ),
            "descriptor_dispatch_bucket": (
                None if descriptor_bucket is None else int(descriptor_bucket)
            ),
            "cots_route_signature": getattr(
                info.batch_descriptor, "cots_route_signature", None
            ),
            "f_cpu_store": float(self.f_cpu_store),
            "f_cpu_compute": float(f_cpu_compute),
            "f_prefetch_compute": float(f_prefetch_compute),
            "has_cpu_compute_work": bool(self._has_cpu_compute_work),
            "has_prefetch_buffer": bool(self._prefetch_buffer_pool is not None),
        }
        logger.info("COTS_DISPATCH_TRACE %s", json.dumps(payload, sort_keys=True))

    def on_dispatch(self, info: ForwardDispatchInfo) -> None:
        """OOG per-forward entry. Owns ALL pre-forward state setup that
        was previously split between the in-graph pre-hook (bucket +
        slot repair) and the OOG live-token update. Single boundary for
        both eager and FULL CUDA Graph paths (replay-time too, not just
        capture-time).

        Order matters:
        1. `prepare_before_forward(num_tokens_padded)` — sets
            `_current_bucket`, mirrors streamer's bucket, runs layer-0
            slot repair (issues H2D on copy_stream). H2D is OOG so it
            isn't captured; each replay gets fresh repair.
        2. `set_active_dispatch(bucket, live)` — publishes the
            authoritative vLLM dispatch bucket to native custom ops.
            They resolve `(layer_idx, bucket, op_kind) -> task_id`
            during eager execution / CUDA Graph capture; task_id is
            not a compile-visible scalar anymore.
        3. `sync_prev_onload()` — drains copy_stream into compute
            stream so the forward sees the filled slot.
        4. `set_live_num_tokens(num_tokens_unpadded)` — live-row cap
            pushed to the C++ worker. This is independent from task
            selection.
        """
        num_tokens_padded = int(info.batch_descriptor.num_tokens)
        num_tokens_unpadded = int(info.num_tokens_unpadded)
        active_bucket = self._dispatch_bucket_from_descriptor(info.batch_descriptor)
        self._live_num_tokens = num_tokens_unpadded
        self._clear_head_split_sidecars()
        self.set_head_split_cpu_positions(
            getattr(info, "positions_cpu", None),
            num_tokens_unpadded,
        )
        self._log_dispatch_trace(
            info,
            num_tokens_padded=num_tokens_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            active_bucket=active_bucket,
        )
        self._prepare_before_forward_bucket(num_tokens_padded, active_bucket)
        if self._has_cpu_compute_work and isinstance(
            self._runner, NativeCotsWeightRunner
        ):
            self._runner.set_active_dispatch(active_bucket, num_tokens_unpadded)
        self.sync_prev_onload()
        # CPU work scales with the semantic batch size, not bucket
        # capacity. Task selection is handled by the active dispatch
        # state above.
        self.set_live_num_tokens(num_tokens_unpadded)

    def post_cudagraph_capture(self) -> None:
        """Optionally reset COTS counters after bucket graphs are captured."""

        if (
            os.environ.get("VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE", "0")
            != "1"
        ):
            return
        from vllm.model_executor.offloader import cots_ops

        cots_ops.reset_all_counters()
        logger.info("COTS reset_all_counters() fired post-cudagraph-capture")

    def shutdown(self) -> None:
        """Drain and release the shared CPU runner at worker shutdown."""
        if self._runner is None:
            return
        if os.environ.get("VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN", "0") == "1":
            from vllm.model_executor.offloader import cots_ops

            if (
                torch.cuda.is_available()
                and torch.cuda.is_initialized()
                and isinstance(self._runner, NativeCotsWeightRunner)
            ):
                torch.cuda.current_stream().synchronize()
                cots_ops.sync_blocking(self._runner._runner_id)
            counters = cots_ops.get_all_counters()
            logger.info(
                "[CotsOffloader] counters: %s",
                json.dumps(counters, sort_keys=True),
            )
        self._runner.close()
        self._runner = None

    def _start_prefetch(self, layer_idx: int) -> None:
        if self._streamer is not None:
            self._streamer.start(layer_idx, self._layer_handles[layer_idx])
        if self._kv_prefetch_streamer is not None:
            self._kv_prefetch_streamer.start(layer_idx)

    def _wait_for_layer(self, layer_idx: int) -> None:
        if self._streamer is not None:
            self._streamer.wait(layer_idx)
        if self._kv_prefetch_streamer is not None:
            self._kv_prefetch_streamer.wait(layer_idx)

    def sync_prev_onload(self) -> None:
        if self._streamer is not None:
            self._streamer.sync_prev_onload()
        if self._kv_prefetch_streamer is not None:
            self._kv_prefetch_streamer.sync_prev_onload()

    def join_after_forward(self) -> None:
        if self._streamer is not None:
            self._streamer.join_after_forward()
        if self._kv_prefetch_streamer is not None:
            self._kv_prefetch_streamer.join_after_forward()

    # --- Graph/dispatch bucket resolution ---

    def _resolve_bucket_sets(self) -> None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self._max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        capture_sizes = list(
            vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
        self._graph_capture_buckets = tuple(sorted(set(capture_sizes)))

        configured = self._configured_dispatch_buckets()
        dispatch_sizes = configured or self._default_dispatch_buckets(vllm_config)
        if not dispatch_sizes:
            dispatch_sizes = [self._max_num_tokens]
        # Tuple (not list) so Dynamo treats the bucket container as constant if
        # a direct-test Python fallback ever reaches `_dispatch_bucket_for` while
        # tracing. Normal native forwards publish the active bucket OOG.
        self._dispatch_buckets = tuple(sorted(set(int(b) for b in dispatch_sizes)))

    def _configured_dispatch_buckets(self) -> list[int]:
        """Return explicit Planner bucket keys, if any were provided."""
        keys: set[int] = set()
        if self.config.dispatch_table is not None:
            keys.update(int(bucket) for bucket in self.config.dispatch_table)
        return sorted(keys)

    def _default_dispatch_buckets(self, vllm_config) -> list[int]:
        """Default Planner bucket grid for COTS dispatch.

        In graph mode this starts from vLLM's actual CUDA graph capture sizes.
        In eager mode vLLM intentionally clears those sizes, so COTS rebuilds
        the same would-have-been decode grid. Both modes add a small set of
        larger fallback buckets for non-captured prefill/mixed forwards.
        """
        if self._graph_capture_buckets:
            buckets = list(self._graph_capture_buckets)
        else:
            buckets = self._would_have_been_graph_buckets(vllm_config)
        buckets.extend(self._large_dispatch_fallback_buckets(self._max_num_tokens))
        return sorted(set(buckets))

    @staticmethod
    def _would_have_been_graph_buckets(vllm_config) -> list[int]:
        scheduler_config = vllm_config.scheduler_config
        compilation_config = vllm_config.compilation_config
        max_num_tokens = int(scheduler_config.max_num_batched_tokens)
        max_capture_size = compilation_config.max_cudagraph_capture_size
        if max_capture_size is None or int(max_capture_size) <= 0:
            decode_query_len = 1
            speculative_config = getattr(vllm_config, "speculative_config", None)
            if speculative_config and speculative_config.num_speculative_tokens:
                decode_query_len += int(speculative_config.num_speculative_tokens)
            max_capture_size = min(
                int(scheduler_config.max_num_seqs) * decode_query_len * 2,
                512,
            )
        max_capture_size = min(max_num_tokens, int(max_capture_size))
        if max_capture_size < 1:
            return []

        performance_mode = getattr(vllm_config, "performance_mode", None)
        if performance_mode == "interactivity":
            interactivity_max = min(max_capture_size, 32)
            buckets = list(range(1, interactivity_max + 1))
        else:
            buckets = [i for i in (1, 2, 4) if i <= max_capture_size]
        if max_capture_size >= 8:
            buckets.extend(range(8, min(max_capture_size + 1, 256), 8))
        if max_capture_size >= 256:
            buckets.extend(range(256, max_capture_size + 1, 16))
        return sorted(set(buckets))

    @staticmethod
    def _large_dispatch_fallback_buckets(max_num_tokens: int) -> list[int]:
        candidates = (768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
        buckets = [b for b in candidates if b <= max_num_tokens]
        if max_num_tokens > 0:
            buckets.append(int(max_num_tokens))
        return buckets

    def _validate_graph_capture_dispatch_coverage(self) -> None:
        missing = sorted(set(self._graph_capture_buckets) - set(self._dispatch_buckets))
        if missing:
            raise ValueError(
                "CotsOffloader: COTS dispatch buckets are missing CUDA graph "
                f"capture buckets: {missing}. Captured graph buckets must map "
                "1:1 to dispatch rows so graph replay cannot use a different "
                "routing policy from capture."
            )

    # --- Activation buffer allocation ---

    def _allocate_activation_buffers(self) -> None:
        """Allocate `_x_pinned`, `_y_pinned`, `_y_gpu` sized to the worst-case
        across registered handles. Flat 1D backing storage.
        """
        if not self._handles:
            return
        device = torch.device("cuda")
        max_in_dim = max(h.cpu_in_dim for h in self._handles)
        max_cpu_out_dim = max(h.cpu_out_dim for h in self._handles)
        dtype = self._handles[0].dtype
        x_capacity = self._max_num_tokens * max_in_dim
        y_capacity = self._max_num_tokens * max_cpu_out_dim
        self._x_pinned = torch.empty(
            x_capacity,
            dtype=dtype,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self._y_pinned = torch.empty(
            y_capacity,
            dtype=dtype,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self._y_gpu = torch.empty(y_capacity, dtype=dtype, device=device)

        # Two distinct dummy CUDA tensors for cots_sync_then_uva's anchor
        # mutates_args. Operators pass these when a GPU GEMM output is absent,
        # so torch.compile / functionalization sees distinct mutation slots.
        # Allocate here because __init__ can predate CUDA device setup.
        self._dummy_gpu_anchor_a = torch.empty(1, dtype=dtype, device=device)
        self._dummy_gpu_anchor_b = torch.empty(1, dtype=dtype, device=device)

    # --- post_init: bookkeeping only ---

    def post_init(self) -> None:
        """Verify enforce_eager (conditional on runner) and finalize
        bookkeeping. The dispatch table and per-bucket geometry are
        built in `wrap_modules` before the prefetch buffer pool is sized inside
        the DeviceMemoryProfiler context."""
        if not self._handles:
            return
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        # The `enforce_eager` requirement is conditional on runner type:
        #   * cpu_runner='native': the C++ `cudaLaunchHostFunc`
        #     substrate IS graph-capturable (CUDA Graph host-function
        #     nodes, supported since CUDA 11.1).
        #   * cpu_runner='python': the `ThreadPoolExecutor.submit` /
        #     `future.result()` substrate is NOT graph-capturable —
        #     capturing it would silently produce wrong results, not
        #     just a slower one. Hard fail until the user either
        #     enables enforce_eager or switches to the native runner.
        if self._has_cpu_compute_work and not vllm_config.model_config.enforce_eager:
            cpu_runner = getattr(self.config, "cpu_runner", "python")
            if cpu_runner != "native":
                raise RuntimeError(
                    "CotsOffloader: cpu_runner='python' requires "
                    "enforce_eager=True — Python runner uses "
                    "ThreadPoolExecutor + future.result() which is NOT "
                    "graph-capturable. Either set enforce_eager=True or "
                    "switch to cpu_runner='native'."
                )

        # Native weight offload is incompatible with vLLM microbatching/
        # ubatching (DBO or ubatch_size > 1). The live-token cap
        # (`GPUModelRunner._publish_forward_dispatch` →
        # `BaseOffloader.set_live_num_tokens`)
        # currently sets ONE global value per scheduler batch. Under
        # ubatching, a COTS operator runs on a per-ubatch slice but
        # sees the cap as the FULL batch token count, which can
        # over-compute (the worker would read past the per-ubatch
        # x_pinned slice into stale data). Hard-fail until per-ubatch live
        # counts are plumbed.
        cpu_runner = getattr(self.config, "cpu_runner", "python")
        if (
            self._has_cpu_compute_work
            and cpu_runner == "native"
            and vllm_config.parallel_config.use_ubatching
        ):
            raise RuntimeError(
                "CotsOffloader: cpu_runner='native' is currently "
                "incompatible with vLLM microbatching/ubatching "
                "(`enable_dbo` or `ubatch_size > 1`). The live-token "
                "cap sets one global live_num_tokens value per scheduler "
                "batch; under ubatching a per-ubatch slice would "
                "over-compute against the full-batch cap. "
                "Either disable ubatching or use cpu_runner='python' "
                "with enforce_eager=True. Tracking under "
                "phase1c_findings.md."
            )

        # Wait-kernel-sync safety gates for the Phase 1 weight runner.
        # Hard-fail at post_init when the captured weight-sync mode is set
        # to wait-kernel but a precondition is missing. We check
        # BEFORE _install_runner so misconfigurations surface before
        # any C++ slab allocation. The host_callback path is unchanged
        # by these gates and remains available for diagnostics.
        weight_capture_sync_mode = self.config.weight_capture_sync_mode
        wait_kernel_enabled = weight_capture_sync_mode == "wait_kernel"
        if self._has_cpu_compute_work and wait_kernel_enabled:
            # Gate 1: native runner only. Python runner has no slabs /
            # no host-mapped done_slot / no worker thread to publish.
            if cpu_runner != "native":
                raise RuntimeError(
                    "CotsOffloader: weight_capture_sync_mode="
                    f"{weight_capture_sync_mode!r} requires cpu_runner='native' "
                    f"(got {cpu_runner!r}). The Python runner has no "
                    "host-mapped done_slot mechanism; the wait kernel is "
                    "meaningful only on the native weight substrate. Set "
                    "weight_capture_sync_mode='host_callback' or switch to "
                    "cpu_runner='native'."
                )
            # Gate 2: graph-capture mode required. Eager mode launches
            # and syncs each iteration, so the wait kernel adds
            # round-trip cost without removing any captured sync_cb
            # node — net negative.
            if vllm_config.model_config.enforce_eager:
                raise RuntimeError(
                    "CotsOffloader: weight_capture_sync_mode="
                    f"{weight_capture_sync_mode!r} requires enforce_eager=False "
                    "(graph-capture mode). The wait kernel replaces the "
                    "captured weight sync_cb host_fn node; under "
                    "enforce_eager=True there is no captured node to replace. "
                    "Set weight_capture_sync_mode='host_callback' or "
                    "enforce_eager=False."
                )
            # Gate 3: CUDA available (defensive — native runner already
            # requires CUDA, but a clearer error here pinpoints the
            # wait-kernel sync as the configuration that needs the GPU).
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CotsOffloader: weight_capture_sync_mode="
                    f"{weight_capture_sync_mode!r} requires CUDA to be "
                    "available; the wait kernel runs on the GPU."
                )
            # Gate 4: _cots_C extension built (defensive — already
            # required by NativeCotsWeightRunner, but with the wait kernel
            # we want a tight blame line if the build was partial).
            try:
                from vllm import _cots_C  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "CotsOffloader: weight_capture_sync_mode="
                    f"{weight_capture_sync_mode!r} requires the `vllm._cots_C` "
                    "extension. Rebuild vLLM with CUDA support "
                    "(/TTC/scripts/rebuild-vllm.sh). Underlying "
                    f"ImportError: {e}"
                ) from e

        self._eager_fallback_entry = self._dispatch_table[self._dispatch_buckets[-1]]

        dispatch_policy = {
            int(bucket): (round(float(f_cpu), 6), round(float(f_prefetch), 6))
            for bucket, (f_cpu, f_prefetch) in sorted(self._dispatch_table.items())
        }
        logger.info(
            "[CotsOffloader] dispatch policy: f_cpu_store=%.6f, dispatch_table=%s",
            self.f_cpu_store,
            dispatch_policy,
        )

        # Install the runner's per-(layer, bucket, op_kind) work table after
        # the underlying CPU and pinned storages are stable.
        self._install_runner()

        # Install wait-kernel sync host-mapped pinned slots once
        # the slab pool exists. The installer walks every slab in the
        # pool and allocates the req/done slot pair. After this returns, the
        # captured graph's sync host-callback nodes are replaced with wait
        # kernel launches at every cots_sync_then_uva call site.
        if (
            self._has_cpu_compute_work
            and wait_kernel_enabled
            and isinstance(self._runner, NativeCotsWeightRunner)
        ):
            self._runner.install_wait_kernel_sync()

        # Layer 0 is filled lazily at the first forward boundary by
        # `prepare_before_forward`. This keeps col prefetch slots
        # active-bucket-adjacent (`[gate_active | up_active]`) instead of
        # post-init max-filled (`[gate_max | up_max]`), so the MLP prefetch
        # path can use one fused [gate|up] GEMM even when f_prefetch <
        # f_cpu_store.

        total_offloaded = sum(
            h.w_cpu.numel() * h.w_cpu.element_size()
            for h in self._handles
            if h.w_cpu is not None
        )
        x_pinned_bytes = (
            0
            if self._x_pinned is None
            else self._x_pinned.numel() * self._x_pinned.element_size()
        )
        y_pinned_bytes = (
            0
            if self._y_pinned is None
            else self._y_pinned.numel() * self._y_pinned.element_size()
        )
        y_gpu_bytes = (
            0
            if self._y_gpu is None
            else self._y_gpu.numel() * self._y_gpu.element_size()
        )

        # Prefetch summary. Option-A accounting can reserve this pool even
        # when a particular bucket has zero active prefetched rows.
        if self._prefetch_buffer_pool is not None:
            prefetch_bytes = self._prefetch_buffer_pool.total_bytes
        else:
            prefetch_bytes = 0

        logger.info(
            "[CotsOffloader] ready: runner=%s, sync=%s, modules=%s, "
            "wo_qkvo_granularity_multiplier=%d, "
            "linears=%d, mlp_blocks=%d, wo_ops=%d, weights_saved=%.4f GB, "
            "buffers=%.4f GB "
            "pinned_in + %.4f GB pinned_out + %.4f GB gpu_uva, "
            "prefetch_pool=%.4f GB, graph_buckets=%s, dispatch_buckets=%s",
            cpu_runner,
            weight_capture_sync_mode,
            sorted(self.weight_modules),
            WO_QKVO_GRANULARITY_MULTIPLIER,
            len(self._handles),
            len(self._fused_ops),
            len(self._wo_ops),
            total_offloaded / 1e9,
            x_pinned_bytes / 1e9,
            y_pinned_bytes / 1e9,
            y_gpu_bytes / 1e9,
            prefetch_bytes / 1e9,
            self._graph_capture_buckets,
            self._dispatch_buckets,
        )
        logger.info(
            "[CotsOffloader] cots_snap: %s",
            json.dumps(
                self._cots_snap_payload(
                    cpu_weight_bytes=total_offloaded,
                    gpu_output_scratch_bytes=y_gpu_bytes,
                    gpu_prefetch_pool_bytes=prefetch_bytes,
                ),
                sort_keys=True,
            ),
        )

        # Effective routing breakdown — actual bytes routed through each
        # path, accounting for head-aligned snapping and per-role geometry.
        # Reported at the largest dispatch bucket (worst case for prefetch
        # buffer sizing).
        bucket = self._dispatch_buckets[-1]
        elem = self._handles[0].dtype.itemsize
        per_role_pref = {"qkv": 0, "mlp_col": 0, "mlp_row": 0, "wo": 0}
        per_role_cpu = {"qkv": 0, "mlp_col": 0, "mlp_row": 0, "wo": 0}
        for h in self._handles:
            n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
            n_cpu = h.n_cpu_compute_by_bucket.get(bucket, 0)
            other_dim = h.in_dim if h.split_axis != INPUT_SPLIT_AXIS else h.out_dim
            key = {
                QKV_ROLE: "qkv",
                MLP_GATE_UP_ROLE: "mlp_col",
                MLP_DOWN_ROLE: "mlp_row",
                WO_ROLE: "wo",
                WO_INPUT_ROLE: "wo",
            }[h.role]
            per_role_pref[key] += n_pref * other_dim * elem
            per_role_cpu[key] += n_cpu * other_dim * elem
        total_pref = sum(per_role_pref.values())
        total_cpu = sum(per_role_cpu.values())
        logger.debug(
            "[CotsOffloader] Effective routing @ bucket=%d:\n"
            "  qkv:     prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  mlp_col: prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  mlp_row: prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  wo:      prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  total: prefetched=%.4f GiB, cpu-computed=%.4f GiB",
            bucket,
            per_role_pref["qkv"] / 1024**3,
            per_role_cpu["qkv"] / 1024**3,
            per_role_pref["mlp_col"] / 1024**3,
            per_role_cpu["mlp_col"] / 1024**3,
            per_role_pref["mlp_row"] / 1024**3,
            per_role_cpu["mlp_row"] / 1024**3,
            per_role_pref["wo"] / 1024**3,
            per_role_cpu["wo"] / 1024**3,
            total_pref / 1024**3,
            total_cpu / 1024**3,
        )

    # --- Runtime: dispatch lookup ---

    def lookup_dispatch(self, num_tokens: int) -> tuple[float, float]:
        """Round `num_tokens` up to the nearest COTS dispatch bucket."""
        if num_tokens > self._dispatch_buckets[-1]:
            return self._eager_fallback_entry
        return self._dispatch_table[self._dispatch_bucket_for(num_tokens)]
