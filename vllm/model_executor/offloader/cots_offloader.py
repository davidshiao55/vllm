# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public COTS offloader lifecycle and model patching logic."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from cots.snap import COTS_SNAP_MODEL, qkvo_output_granularity

# Register prefetch custom ops. COTS uses generic wait_prefetch plus
# COTS-specific guarded start/consumer ops for hidden prefetch-slot aliases.
import vllm.model_executor.offloader.cots_ops  # noqa: F401
import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.config.cots import (
    cots_weight_module_for_name,
    normalize_cots_weight_modules,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.offloader.base import BaseOffloader, ForwardDispatchInfo
from vllm.model_executor.offloader.cots_operators import (
    CotsQKVOp,
    CotsSwiGLUMLPOp,
    CotsWOOp,
    _RaiseOnDirectCall,
)
from vllm.model_executor.offloader.cots_runners import (
    NativeCotsWeightRunner,
    NativeWeightSlabSpec,
    _NativeWeightSlabSpecLinear,
    _NativeWeightSlabSpecMlp,
)
from vllm.model_executor.offloader.cots_storage import (
    DEFAULT_QKVO_HEAD_DIM,
    INPUT_SPLIT_AXIS,
    MLP_DOWN_ROLE,
    MLP_GATE_UP_ROLE,
    QKV_ROLE,
    WO_ROLE,
    CotsLinearHandle,
    CotsPrefetchBufferPool,
    WeightPrefetchStreamer,
)
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import CotsOffloadConfig

logger = init_logger(__name__)

LINEAR_OP_KIND_BY_ROLE = {
    QKV_ROLE: "qkv",
    WO_ROLE: "wo",
}


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
            [Sequence[int]], dict[int, tuple[float, float]]
        ]
        | None = None,
    ):
        self.config = config
        self.f_cpu_store = float(config.f_cpu_store)
        self.f_prefetch = float(config.f_prefetch)
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
        # Populated in wrap_modules. _handles is the master list of all
        # offloaded linears (in insertion order); _fused_ops tracks installed
        # MLP-block ops (one per recognized parent), and _wo_ops tracks
        # opt-in output-split WO adapters.
        self._handles: list[CotsLinearHandle] = []
        self._fused_ops: list[CotsSwiGLUMLPOp] = []
        self._wo_ops: list[CotsWOOp] = []

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
        self._dispatch_table: dict[int, tuple[float, float]] = {}
        # CUDA graph capture buckets and Planner dispatch buckets are related
        # but distinct. Graph buckets describe replay shapes. Dispatch buckets
        # describe the COTS route selected from BatchDescriptor.num_tokens and
        # must exist even in eager mode, where CUDA graph buckets are empty.
        self._graph_capture_buckets: tuple[int, ...] = ()
        self._dispatch_buckets: tuple[int, ...] = ()
        self._max_num_tokens: int = 0
        self._eager_fallback_entry: tuple[float, float] = (0.0, 0.0)
        self._has_cpu_compute_work: bool = False

        # Prefetch infrastructure — allocated in wrap_modules when the
        # dispatch table reserves any prefetch capacity. Active buckets may
        # still have zero prefetched rows after runtime snapping.
        self._prefetch_buffer_pool: CotsPrefetchBufferPool | None = None
        self._streamer: WeightPrefetchStreamer | None = None

        # One offloader-owned runner is shared across all operator call sites.
        # The no-offload path leaves it unset to avoid starting a worker thread.
        self._runner: NativeCotsWeightRunner | None = None
        if self.f_cpu_store > 0.0:
            self._runner = NativeCotsWeightRunner(dry_run=bool(config.dry_run))

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
        # Distinct fallback guards for COTS prefetch custom-op schemas. Real
        # prefetch handles pass per-slot guards; these fill unused guard slots.
        self._dummy_prefetch_slot_guards: tuple[torch.Tensor, ...] = ()

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
            if self.f_cpu_store == 0.0:
                continue
            layer_handles = self._build_handles(layer)
            if not layer_handles:
                continue
            for h in layer_handles:
                h.layer_idx = layer_idx
            self._install_qkv_ops(layer_handles)
            self._install_mlp_ops(layer, layer_handles)
            self._install_wo_ops(layer_handles)
            self._check_no_orphan_mlp_handles(layer_handles)
            self._layer_modules.append(layer)
            self._layer_handles.append(layer_handles)
            layer_idx += 1

        if self.f_cpu_store == 0.0:
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
        self._has_cpu_compute_work = self._compute_has_cpu_compute_work()
        if self._handles:
            self._allocate_prefetch_slot_guard_dummies()

        # Allocate shared CPU-compute buffers only when some bucket actually
        # leaves rows on the CPU-compute path. Pure-prefetch configurations
        # skip these pinned/GPU scratch buffers entirely.
        if self._has_cpu_compute_work:
            self._allocate_activation_buffers()
        # Install based on option-A reserved capacity, not the config fallback
        # knob. A Planner-emitted table can request prefetch even when config
        # f_prefetch == 0, and zero-prefetch tables still reserve full-store
        # slots so runtime accounting matches planner GPU buffer accounting.
        if any(h.max_n_prefetch > 0 for h in self._handles):
            self._install_prefetch_machinery()
        logger.debug(
            "CotsOffloader: wrapped %d linear modules, %d fused MLP blocks, "
            "and %d WO ops (modules=%s, f_cpu_store=%.4f, f_prefetch=%.4f, "
            "cpu_num_threads=%d, dry_run=%s).",
            len(self._handles),
            len(self._fused_ops),
            len(self._wo_ops),
            sorted(self.weight_modules),
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
            if h.role != WO_ROLE:
                continue
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

    def _build_dispatch_table(self) -> None:
        """Construct the uniform Planner dispatch table."""
        if self._dispatch_table_factory is not None:
            self._dispatch_table = self._dispatch_table_factory(self._dispatch_buckets)
        else:
            pair = (self.f_cpu_store - self.f_prefetch, self.f_prefetch)
            self._dispatch_table = {b: pair for b in self._dispatch_buckets}
        self._validate_graph_capture_dispatch_coverage()

    def _install_prefetch_machinery(self) -> None:
        """Allocate prefetch buffers and install layer-level prefetch hooks."""
        device = torch.device("cuda")
        self._prefetch_buffer_pool = CotsPrefetchBufferPool(self._handles, device)
        for h in self._handles:
            if h.layer_idx >= 0:
                h.slot_idx = h.layer_idx % CotsPrefetchBufferPool.K

        n_layers = len(self._layer_modules)
        self._streamer = WeightPrefetchStreamer(
            n_layers=n_layers,
            dry_run=self.dry_run,
        )
        self._streamer.buffer_pool = self._prefetch_buffer_pool

        for i, layer in enumerate(self._layer_modules):
            self._hook_layer_forward(i, layer)

    # --- Runner install (closures / slab specs) -----

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

    def _compute_has_cpu_compute_work(self) -> bool:
        """Whether any dispatch bucket leaves rows for CPU GEMM."""
        for h in self._handles:
            for bucket in self._dispatch_buckets:
                if int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)) > 0:
                    return True
        return False

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
            if h.role not in (QKV_ROLE, WO_ROLE):
                continue
            assert h.w_cpu is not None
            op_kind = LINEAR_OP_KIND_BY_ROLE[h.role]
            for bucket in self._dispatch_buckets:
                n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu = h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)
                if n_cpu == 0:
                    continue
                w_view = h.w_cpu.narrow(0, n_pref, n_cpu)
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

    def _register_runner_routes(self) -> None:
        """Publish existing handle objects for opaque COTS operator routing."""
        if self._runner is None or not self._handles:
            return
        from vllm.model_executor.offloader import cots_ops

        for h in self._handles:
            if h.role not in (QKV_ROLE, WO_ROLE):
                continue
            cots_ops.register_weight_route(
                self._runner._runner_id,
                layer_idx=h.layer_idx,
                op_kind=LINEAR_OP_KIND_BY_ROLE[h.role],
                route=h,
            )
        for fop in self._fused_ops:
            cots_ops.register_weight_route(
                self._runner._runner_id,
                layer_idx=fop._gate_up.layer_idx,
                op_kind="mlp_block",
                route=fop,
            )

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
            "snap_model": COTS_SNAP_MODEL,
            "qkvo_granularity_channels": self._qkvo_granularity_channels(),
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

    def _qkvo_granularity_channels(self) -> int:
        for handle in self._handles:
            if handle.role in (QKV_ROLE, WO_ROLE):
                return qkvo_output_granularity(handle.qkvo_head_dim)
        return qkvo_output_granularity(DEFAULT_QKVO_HEAD_DIM)

    def _hook_layer_forward(self, index: int, layer: nn.Module) -> None:
        """Wrap the decoder layer's `forward` with pre-compute scheduling.

        For layer i: wait for layer i's prefetched weights, start prefetch
        for layer i+1, then run layer i. With K=2 slot rotation, i reads
        slot i%2 while i+1 writes slot (i+1)%2, so the H2D overlaps with
        layer i compute without a wraparound special case.
        """
        original_forward = layer.forward
        n_layers = len(self._layer_modules)

        layer_has_prefetch = any(
            h.max_n_prefetch > 0 for h in self._layer_handles[index]
        )
        next_idx = (index + 1) % n_layers if n_layers > 0 else 0
        next_has_prefetch = (
            n_layers > 1
            and next_idx != index
            and any(h.max_n_prefetch > 0 for h in self._layer_handles[next_idx])
        )
        next_prefetch_guards = (
            self._prefetch_slot_guards_for_layer(next_idx) if next_has_prefetch else ()
        )

        def forward(*args, **kwargs):
            layer.forward = original_forward
            anchor = args[0] if args else next(iter(kwargs.values()))
            if layer_has_prefetch:
                torch.ops.vllm.wait_prefetch(anchor, index)
            if next_has_prefetch:
                torch.ops.vllm.cots_start_prefetch(
                    anchor, *next_prefetch_guards, next_idx
                )
            output = original_forward(*args, **kwargs)
            layer.forward = forward
            return output

        layer.forward = forward

    def _prefetch_slot_guards_for_layer(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        guards: list[torch.Tensor] = []
        seen: set[int] = set()
        for h in self._layer_handles[layer_idx]:
            if h.max_n_prefetch <= 0 or not h.prefetch_slot_guards:
                continue
            guard = h.prefetch_slot_guards[h.slot_idx]
            key = id(guard)
            if key in seen:
                continue
            guards.append(guard)
            seen.add(key)
        if len(guards) > 4:
            raise RuntimeError(
                "COTS guarded prefetch supports at most four distinct slot "
                f"guards per layer, got {len(guards)} for layer {layer_idx}"
            )
        if len(self._dummy_prefetch_slot_guards) < 4:
            raise RuntimeError("COTS prefetch guard dummies were not allocated")
        guards.extend(self._dummy_prefetch_slot_guards[len(guards) : 4])
        return guards[0], guards[1], guards[2], guards[3]

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
        return self._dispatch_bucket_for(int(batch_descriptor.num_tokens))

    def _prepare_before_forward_bucket(self, num_tokens: int, bucket: int) -> None:
        self._current_bucket = int(bucket)
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

    def set_live_num_tokens(self, live_num_tokens: int) -> None:
        """Push the live unpadded token count to the C++ worker.

        CUDA graph buckets and native slabs are capacity-sized. This
        live-row cap lets the CPU worker skip padded rows inside the
        selected bucket.

        No-op when there is no active CPU work or when
        `live_num_tokens <= 0` (sentinel).
        """
        if not self._has_cpu_compute_work:
            return
        assert self._runner is not None
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
        payload = {
            **dict(info.trace_context),
            "event": "cots_dispatch_trace",
            "pid": os.getpid(),
            "num_tokens_padded": int(num_tokens_padded),
            "num_tokens_unpadded": int(num_tokens_unpadded),
            "cots_dispatch_bucket": int(active_bucket),
            "dispatch_bucket_source": "num_tokens",
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
            authoritative vLLM dispatch bucket to COTS custom ops.
            CPU ops use it to resolve `(layer_idx, bucket, op_kind) -> task_id`;
            prefetch/scatter ops use it to resolve existing handle route tables.
        3. `sync_prev_onload()` — drains copy_stream into compute
            stream so the forward sees the filled slot.
        4. `set_live_num_tokens(num_tokens_unpadded)` — live-row cap
            pushed to the C++ worker. This is independent from task
            selection.
        """
        num_tokens_padded = int(info.batch_descriptor.num_tokens)
        num_tokens_unpadded = int(info.num_tokens_unpadded)
        active_bucket = self._dispatch_bucket_from_descriptor(info.batch_descriptor)
        self._log_dispatch_trace(
            info,
            num_tokens_padded=num_tokens_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            active_bucket=active_bucket,
        )
        self._prepare_before_forward_bucket(num_tokens_padded, active_bucket)
        if self._runner is not None:
            assert self._runner is not None
            self._runner.set_active_dispatch(active_bucket, num_tokens_unpadded)
        self.sync_prev_onload()
        # CPU work scales with the semantic batch size, not bucket
        # capacity. Task selection is handled by the active dispatch
        # state above.
        self.set_live_num_tokens(num_tokens_unpadded)

    def shutdown(self) -> None:
        """Drain and release the shared CPU runner at worker shutdown."""
        if self._runner is None:
            return
        if os.environ.get("VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN", "0") == "1":
            from vllm.model_executor.offloader import cots_ops

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                assert self._runner is not None
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

    def _wait_for_layer(self, layer_idx: int) -> None:
        if self._streamer is not None:
            self._streamer.wait(layer_idx)

    def sync_prev_onload(self) -> None:
        if self._streamer is not None:
            self._streamer.sync_prev_onload()

    def join_after_forward(self) -> None:
        if self._streamer is not None:
            self._streamer.join_after_forward()

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

    def _allocate_prefetch_slot_guard_dummies(self) -> None:
        if self._dummy_prefetch_slot_guards:
            return
        device = torch.device("cuda")
        self._dummy_prefetch_slot_guards = tuple(
            torch.empty((), dtype=torch.int32, device=device) for _ in range(4)
        )

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
        """Finalize native COTS weight bookkeeping.

        The dispatch table and per-bucket geometry are built in
        `wrap_modules` before the prefetch buffer pool is sized inside the
        DeviceMemoryProfiler context.
        """
        if not self._handles:
            return
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
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
        if self._has_cpu_compute_work and vllm_config.parallel_config.use_ubatching:
            raise RuntimeError(
                "CotsOffloader is currently incompatible with vLLM "
                "microbatching/ubatching "
                "(`enable_dbo` or `ubatch_size > 1`). The live-token "
                "cap sets one global live_num_tokens value per scheduler "
                "batch; under ubatching a per-ubatch slice would "
                "over-compute against the full-batch cap. Disable "
                "ubatching until per-ubatch live counts are plumbed."
            )

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
        self._register_runner_routes()
        self._install_runner()

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

        qkvo_granularity_channels = self._qkvo_granularity_channels()
        logger.info(
            "[CotsOffloader] ready: runner=%s, sync=%s, modules=%s, "
            "qkvo_granularity_channels=%d, "
            "linears=%d, mlp_blocks=%d, wo_ops=%d, weights_saved=%.4f GB, "
            "buffers=%.4f GB "
            "pinned_in + %.4f GB pinned_out + %.4f GB gpu_uva, "
            "prefetch_pool=%.4f GB, graph_buckets=%s, dispatch_buckets=%s",
            "native",
            "host_callback",
            sorted(self.weight_modules),
            qkvo_granularity_channels,
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
