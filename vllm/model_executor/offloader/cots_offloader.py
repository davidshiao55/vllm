# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public COTS offloader lifecycle and model patching logic."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

# Register prefetch custom ops; COTS reuses wait_prefetch/start_prefetch.
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
    PyCotsWeightCallback,
    PythonCotsWeightRunner,
    _make_runner,
    _NativeWeightSlabSpecLinear,
    _NativeWeightSlabSpecMlp,
)
from vllm.model_executor.offloader.cots_storage import (
    DEFAULT_OUTPUT_CHANNEL_GRANULARITY,
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
        self.kv_biased = bool(config.kv_biased)
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
        self._capture_buckets: tuple[int, ...] = ()
        self._max_num_tokens: int = 0
        self._eager_fallback_entry: tuple[float, float] = (0.0, 0.0)
        self._has_cpu_compute_work: bool = False

        # Prefetch infrastructure — allocated in wrap_modules iff
        # `f_prefetch > 0`. Phase 1a behavior (`f_prefetch == 0`) leaves
        # both at None and skips hook installation.
        self._prefetch_buffer_pool: CotsPrefetchBufferPool | None = None
        self._streamer: WeightPrefetchStreamer | None = None

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
        self._resolve_capture_buckets()

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

        # Phase 1b: build dispatch table, populate per-handle prefetch
        # geometry, and (if f_prefetch > 0) allocate the streamer + buffer
        # pool and install layer-level prefetch hooks. Phase 1a (f_prefetch=0)
        # leaves all of this no-op'd.
        self._build_dispatch_table()
        for h in self._handles:
            h.apply_prefetch_split_per_bucket(self._dispatch_table)
        self._has_cpu_compute_work = self._compute_has_cpu_compute_work()

        # Allocate shared CPU-compute buffers only when some bucket actually
        # leaves rows on the CPU-compute path. Pure-prefetch configurations
        # skip these pinned/GPU scratch buffers entirely.
        if self._has_cpu_compute_work:
            self._allocate_activation_buffers()
        # Install based on effective dispatch table, not config knob — a
        # Planner-emitted table can request prefetch even when config
        # f_prefetch == 0.
        if any(h.max_n_prefetch > 0 for h in self._handles):
            self._install_prefetch_machinery()
        logger.debug(
            "CotsOffloader: wrapped %d linear modules, %d fused MLP blocks, "
            "and %d WO ops (modules=%s, f_cpu_store=%.4f, f_prefetch=%.4f, "
            "kv_biased=%s, cpu_num_threads=%d, dry_run=%s).",
            len(self._handles),
            len(self._fused_ops),
            len(self._wo_ops),
            sorted(self.weight_modules),
            self.f_cpu_store,
            self.f_prefetch,
            self.kv_biased,
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
        output_granularity = self._dense_output_granularity_for_layer(
            layer, QKVParallelLinear
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
                handle = CotsLinearHandle.for_qkv(
                    child,
                    qualified_name,
                    head_dim=int(child.head_size),
                    kv_biased=self.kv_biased,
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
                    output_granularity=output_granularity,
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
    def _dense_output_granularity_for_layer(
        layer: nn.Module,
        qkv_cls: type[nn.Module],
    ) -> int:
        """Use the layer's WQKV head size as dense output-split alignment.

        WO has no semantic heads, but aligning its output-channel split to the
        same grid as WQKV keeps the storage and prefetch policy consistent.
        Synthetic WO-only tests may not include a QKV module, so they fall back
        to the common Qwen/Llama 128-channel head grid.
        """
        for _, child in layer.named_modules():
            if isinstance(child, qkv_cls):
                return int(child.head_size)
        return DEFAULT_OUTPUT_CHANNEL_GRANULARITY

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
        """Construct `_dispatch_table`. Uses the injected factory if present,
        otherwise uniform fill from config (`f_cpu_store - f_prefetch`,
        `f_prefetch`)."""
        if self._dispatch_table_factory is not None:
            self._dispatch_table = self._dispatch_table_factory(self._capture_buckets)
        else:
            pair = (self.f_cpu_store - self.f_prefetch, self.f_prefetch)
            self._dispatch_table = {b: pair for b in self._capture_buckets}

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
            if h.role not in (QKV_ROLE, WO_ROLE):
                continue
            assert h.w_cpu is not None
            op_kind = LINEAR_OP_KIND_BY_ROLE[h.role]
            for bucket in self._capture_buckets:
                n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu = h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)
                if n_cpu == 0:
                    continue
                w_view = h.w_cpu.narrow(0, n_pref, n_cpu)
                callbacks[(h.layer_idx, bucket, op_kind)] = (
                    self._make_output_split_python_callback(w_view)
                )
        for fop in self._fused_ops:
            gu_h = fop._gate_up
            dn_h = fop._down
            assert gu_h.w_cpu is not None
            assert dn_h.w_cpu is not None
            n_cpu_per_half_total = gu_h.n_cpu // 2
            for bucket in self._capture_buckets:
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
        `_capture_buckets` is not known until `_resolve_capture_buckets`
        runs in `wrap_modules`.
        """
        per_bucket = getattr(self.config, "cpu_num_threads_by_bucket", None)
        if per_bucket is None:
            return int(self.config.cpu_num_threads)
        return int(per_bucket.get(bucket, self.config.cpu_num_threads))

    def _validate_thread_policy(self) -> None:
        """Reject `cpu_num_threads_by_bucket` keys that aren't captured
        buckets — would silently fall back to scalar and the Planner's
        intent would be lost."""
        per_bucket = getattr(self.config, "cpu_num_threads_by_bucket", None)
        if per_bucket is None:
            return
        unknown = set(per_bucket.keys()) - set(self._capture_buckets)
        if unknown:
            raise ValueError(
                f"cots: cpu_num_threads_by_bucket has keys "
                f"{sorted(unknown)} that are not in cudagraph_capture_sizes "
                f"({self._capture_buckets}). Per-bucket thread policy must "
                f"only reference captured buckets."
            )
        for b, n in per_bucket.items():
            if n < 1:
                raise ValueError(
                    f"cots: cpu_num_threads_by_bucket[{b}] = {n}, must be >= 1"
                )

    def _native_routing_uniform_across_buckets(self) -> bool:
        """Whether compile-visible operator geometry is bucket-invariant.

        Native custom ops now resolve task_id from OOG dispatch state, but
        Python-side routing geometry (`n_prefetch`, `n_cpu_compute`, scatter
        indices, GPU branch shape) is still selected in the compiled forward.
        FULL CUDA Graph mode is therefore only structurally sound for
        uniform routing until that geometry is also moved behind an OOG or
        per-capture boundary.
        """
        if not self._capture_buckets:
            return True
        for h in self._handles:
            pref_values = {
                int(h.n_prefetch_by_bucket.get(bucket, 0))
                for bucket in self._capture_buckets
            }
            cpu_values = {
                int(h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu))
                for bucket in self._capture_buckets
            }
            if len(pref_values) > 1 or len(cpu_values) > 1:
                return False
        return True

    def _compute_has_cpu_compute_work(self) -> bool:
        """Whether any captured bucket leaves rows for CPU GEMM."""
        for h in self._handles:
            for bucket in self._capture_buckets:
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
            for bucket in self._capture_buckets:
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
            for bucket in self._capture_buckets:
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

    def _bucket_for(self, num_tokens: int) -> int:
        """Round-up lookup on `_capture_buckets`. Returns the bucket key
        (matches `lookup_dispatch`'s rounding semantics). Out-of-range returns
        the largest captured bucket.

        This used to call `bisect.bisect_left`, but the C builtin was not
        Dynamo-friendly when bucket repair lived in a traced pre-hook. Keeping
        the simple linear scan avoids reintroducing that constraint, and this
        runs once per forward boundary rather than per GEMM.
        """
        for bucket in self._capture_buckets:
            if num_tokens <= bucket:
                return bucket
        return self._capture_buckets[-1]

    # --- BaseOffloader lifecycle delegation ---

    def prepare_before_forward(self, num_tokens: int) -> None:
        """Repair active-bucket state before a forward starts.

        Always sets `_current_bucket` (plan §design-decision 11) so the
        operator slab/closure lookup has a valid bucket regardless of
        whether prefetch is active. Layer-0 slot repair and streamer
        bucket mirroring run only when the streamer exists
        (`f_prefetch > 0`). Steady-state next-layer prefetches are
        emitted inside each layer wrapper so FULL CUDA graph capture
        records them as graph nodes rather than relying on replay-time
        Python state.

        Kept free of pybind calls so it can be used from graph-boundary helper
        paths. The C++ runtime token row cap is pushed separately by
        `on_dispatch`, outside captured graphs.
        """
        self._current_bucket = self._bucket_for(num_tokens)
        if self._streamer is None:
            return
        self._streamer.set_current_bucket(num_tokens, self._bucket_for)
        if self._layer_handles:
            self._streamer.prepare_for_forward_bucket(0, self._layer_handles[0])

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
        return self._bucket_for(x_rows)

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
        if not self._has_cpu_compute_work:
            return
        if not isinstance(self._runner, NativeCotsWeightRunner):
            return
        if int(live_num_tokens) <= 0:
            return
        from vllm.model_executor.offloader import cots_ops

        cots_ops.set_live_num_tokens(self._runner._runner_id, int(live_num_tokens))

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
        self.prepare_before_forward(num_tokens_padded)
        active_bucket = self._current_bucket
        if active_bucket is None:
            active_bucket = self._bucket_for(num_tokens_padded)
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

    # --- Capture-bucket resolution ---

    def _resolve_capture_buckets(self) -> None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self._max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        capture_sizes = list(
            vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
        if not capture_sizes:
            capture_sizes = [self._max_num_tokens]
        # Tuple (not list) so Dynamo treats `_capture_buckets` as a constant
        # container during graph capture.
        self._capture_buckets = tuple(sorted(set(capture_sizes)))

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

        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        has_full_cudagraphs = (
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )
        if (
            cpu_runner == "native"
            and not vllm_config.model_config.enforce_eager
            and has_full_cudagraphs
            and not self._native_routing_uniform_across_buckets()
        ):
            raise RuntimeError(
                "CotsOffloader: native FULL CUDA Graph mode requires uniform "
                "COTS routing geometry across capture buckets. The native "
                "custom ops now resolve slab task_id from OOG dispatch state, "
                "but Python-side routing geometry (n_prefetch/n_cpu_compute "
                "and scatter shape) is still compile-visible. Use "
                "enforce_eager=True, disable FULL cudagraphs, or use a "
                "uniform dispatch table until routing geometry is moved "
                "behind the same structural dispatch boundary."
            )

        self._eager_fallback_entry = self._dispatch_table[self._capture_buckets[-1]]

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
        x_pinned_gb = (
            0.0
            if self._x_pinned is None
            else self._x_pinned.numel() * self._x_pinned.element_size() / 1e9
        )
        y_pinned_gb = (
            0.0
            if self._y_pinned is None
            else self._y_pinned.numel() * self._y_pinned.element_size() / 1e9
        )
        y_gpu_gb = (
            0.0
            if self._y_gpu is None
            else self._y_gpu.numel() * self._y_gpu.element_size() / 1e9
        )

        # Prefetch summary (zero / disabled when f_prefetch == 0).
        if self._prefetch_buffer_pool is not None:
            prefetch_gb = self._prefetch_buffer_pool.total_bytes / 1e9
        else:
            prefetch_gb = 0.0

        logger.info(
            "[CotsOffloader] ready: runner=%s, sync=%s, modules=%s, "
            "linears=%d, mlp_blocks=%d, wo_ops=%d, weights_saved=%.4f GB, "
            "buffers=%.4f GB "
            "pinned_in + %.4f GB pinned_out + %.4f GB gpu_uva, "
            "prefetch_pool=%.4f GB, buckets=%s",
            cpu_runner,
            weight_capture_sync_mode,
            sorted(self.weight_modules),
            len(self._handles),
            len(self._fused_ops),
            len(self._wo_ops),
            total_offloaded / 1e9,
            x_pinned_gb,
            y_pinned_gb,
            y_gpu_gb,
            prefetch_gb,
            self._capture_buckets,
        )

        # Effective routing breakdown — actual bytes routed through each
        # path, accounting for head-aligned snapping and per-role geometry.
        # Reported at the largest capture bucket (worst case for prefetch
        # buffer sizing).
        bucket = self._capture_buckets[-1]
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
        """Round `num_tokens` up to the nearest capture bucket."""
        if num_tokens > self._capture_buckets[-1]:
            return self._eager_fallback_entry
        return self._dispatch_table[self._bucket_for(num_tokens)]
