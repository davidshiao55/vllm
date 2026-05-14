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
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.offloader.base import BaseOffloader, ForwardDispatchInfo
from vllm.model_executor.offloader.cots_operators import (
    CotsQKVOp,
    CotsSwiGLUMLPOp,
    _RaiseOnDirectCall,
)
from vllm.model_executor.offloader.cots_runners import (
    NativeCotsRunner,
    NativeSlabSpec,
    PyCotsCallback,
    PythonCotsRunner,
    _make_runner,
    _NativeSlabSpecMlp,
    _NativeSlabSpecQkv,
)
from vllm.model_executor.offloader.cots_storage import (
    CotsLinearHandle,
    CotsPrefetchBufferPool,
    WeightPrefetchStreamer,
)
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import CotsOffloadConfig

logger = init_logger(__name__)


class CotsOffloader(BaseOffloader):
    """Collaborative CPU-GPU weight offloader (thesis Phase 1a).

    Three-pass per layer in `wrap_modules`:
      1. Build & install handles for each offloadable Linear.
      2. Install operator adapters: `CotsQKVOp` per QKV linear,
         `CotsSwiGLUMLPOp` per recognized MLP block (replaces parent.forward,
         installs `_RaiseOnDirectCall` guards on the MLP linears).
      3. Reject orphan col/row handles loudly: MergedCol/Row offload
         requires an MLP block parent.
    """

    # `o_proj` intentionally absent — WO is GPU-resident in Phase 1/2 per
    # `weight_offload_design.md §WO Split Axis Decision`.
    _OFFLOAD_SUFFIXES = ("qkv_proj", "gate_up_proj", "down_proj")

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
        # Only touch the process-wide PyTorch thread count when we actually
        # offload — `--cots-f-cpu-store=0` should be a clean control with
        # no side effects on CPU thread behavior. See `phase1a_findings.md
        # §1.14`'s dryrun-vs-none baseline.
        #
        # §1c.21 review fix: gate this to PythonCotsRunner. The native
        # runner's thread policy lives inside the C++ worker via the
        # per-slab `n_threads` field (§1c.8) — applying a global
        # `torch.set_num_threads` for the native path leaks oneDNN /
        # main-thread settings into Inductor's compiled forward and was
        # implicated as a possible contributor to the native+capture orch
        # regression. Python runner still uses the global because its
        # callbacks run on `at::linear` from `_get_executor()`'s thread
        # pool, which inherits `set_num_threads` semantics.
        if self.f_cpu_store > 0.0 and config.cpu_runner == "python":
            torch.set_num_threads(int(config.cpu_num_threads))

        # Populated in wrap_modules. _handles is the master list of all
        # offloaded linears (in insertion order); _fused_ops tracks installed
        # MLP-block ops (one per recognized parent).
        self._handles: list[CotsLinearHandle] = []
        self._fused_ops: list[CotsSwiGLUMLPOp] = []

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

        # Phase 1c: one offloader-owned runner shared across all operator
        # call sites (Stage 2 installer refactor — replaces the Phase
        # 1a/1b pattern of fresh `CpuTaskRunner()` per op). The factory
        # selects PythonCotsRunner / NativeCotsRunner from
        # `config.cpu_runner`. See `_make_runner` and the runner classes
        # above. Only constructed when COTS is actually offloading
        # (`f_cpu_store > 0`); the no-offload path leaves it None to
        # avoid spinning up a worker thread for a no-op session.
        # Stage 3 dropped the Stage-2 native rejection — operators now
        # flip to the uniform facade and either runner runs end-to-end.
        self._runner: PythonCotsRunner | NativeCotsRunner | None = None
        if self.f_cpu_store > 0.0:
            self._runner = _make_runner(config)

        # Phase 1c: active-bucket lives on the offloader, not the
        # streamer (plan §design-decision 11). `prepare_before_forward`
        # always sets `_current_bucket` regardless of streamer presence
        # so the operator slab/closure lookup never sees `bucket=None`
        # at `f_prefetch=0`. The first-decoder pre-hook is installed
        # unconditionally when COTS is active (see
        # `_install_bucket_prehook`), independent of prefetch.
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
            self._check_no_orphan_col_row(layer_handles)
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
        # NOTE: the in-graph pre-hook used to set `_current_bucket` via
        # `anchor.shape[0]` (the persistent input buffer under FULL
        # CUDA Graph capture). Under fullgraph trace it clobbered the
        # OOG bucket set by the model runner and saturated the
        # bucket key to the largest captured size. The replacement is
        # `on_dispatch`, called OOG per-forward from the active
        # model-runner path and from the older cudagraph utility path.
        # `prepare_before_forward` is kept as the offloader-local
        # bucket/slot repair primitive — only the pre-hook registration
        # is removed. See bucket-key-fix discussion.
        # self._install_bucket_prehook()  # DISABLED — see comment above

        logger.debug(
            "CotsOffloader: wrapped %d linear modules and %d fused MLP blocks "
            "(f_cpu_store=%.4f, f_prefetch=%.4f, kv_biased=%s, "
            "cpu_num_threads=%d, dry_run=%s).",
            len(self._handles),
            len(self._fused_ops),
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
        for qualified_name, child in layer.named_modules():
            if not any(qualified_name.endswith(s) for s in self._OFFLOAD_SUFFIXES):
                continue
            if not isinstance(child.quant_method, UnquantizedLinearMethod):
                raise RuntimeError(
                    f"CotsOffloader only supports unquantized "
                    f"linear layers, got {type(child.quant_method).__name__} "
                    f"on {qualified_name}."
                )
            self._check_dtype_is_bfloat16(child, qualified_name)
            if isinstance(child, QKVParallelLinear):
                handle = CotsLinearHandle.for_qkv(
                    child,
                    qualified_name,
                    head_dim=int(child.head_size),
                    kv_biased=self.kv_biased,
                    f_cpu_store=self.f_cpu_store,
                )
            elif isinstance(child, MergedColumnParallelLinear):
                handle = CotsLinearHandle.for_col(
                    child,
                    qualified_name,
                    f_cpu_store=self.f_cpu_store,
                )
            elif isinstance(child, RowParallelLinear):
                handle = CotsLinearHandle.for_row(
                    child,
                    qualified_name,
                    f_cpu_store=self.f_cpu_store,
                )
            else:
                raise RuntimeError(
                    f"CotsOffloader: {qualified_name} matched offload suffix "
                    f"but is not Merged/QKV/Row ParallelLinear "
                    f"(got {type(child).__name__})"
                )
            if handle is None:
                continue  # f rounded to 0 cols for this module
            handle.install(child.weight.data.device)
            self._handles.append(handle)
            layer_handles.append(handle)
        return layer_handles

    # --- Pass 2a: QKV operator install ---

    def _install_qkv_ops(self, handles: list[CotsLinearHandle]) -> None:
        # Phase 1c installer refactor: operators share the offloader's
        # single runner (constructed once in __init__). Phase 1a/1b
        # constructed a fresh `CpuTaskRunner` per op; that pattern is
        # incompatible with Stage 3's per-offloader slab pool +
        # runner_id design.
        assert self._runner is not None, (
            "_install_qkv_ops called with f_cpu_store=0 — runner not constructed"
        )
        for h in handles:
            if h.kind != "qkv":
                continue
            h.linear.quant_method = CotsQKVOp(
                handle=h,
                runner=self._runner,
                offloader=self,
                original_quant_method=h.linear.quant_method,
            )

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
    def _check_no_orphan_col_row(handles: list[CotsLinearHandle]) -> None:
        """Every col/row handle must be in a fused MLP block."""
        for h in handles:
            if h.kind in ("col", "row") and not h.in_block:
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
        """Phase 1b: allocate buffer pool, stamp slot indices, allocate
        streamer, install layer-level forward hooks, register first-decoder
        pre-hook for bucket caching + layer-0 repair."""
        device = torch.device("cuda")
        self._prefetch_buffer_pool = CotsPrefetchBufferPool(self._handles, device)
        for h in self._handles:
            if h.layer_idx >= 0:
                h.slot_idx = h.layer_idx % CotsPrefetchBufferPool.K

        # Stage 7-C: dropped the `w_row_prefetch_src_t` duplicate
        # allocation. The unified transposed-storage `w_cpu` (allocated
        # in `CotsLinearHandle.install()` for row-handle as
        # `(n_cpu, out_dim)`) IS the contiguous prefetch source — no
        # second buffer needed. Saves ~max_n_prefetch * out_dim * 2
        # bytes pinned per row-handle.

        n_layers = len(self._layer_modules)
        self._streamer = WeightPrefetchStreamer(
            n_layers=n_layers,
            dry_run=self.dry_run,
        )
        self._streamer.buffer_pool = self._prefetch_buffer_pool

        for i, layer in enumerate(self._layer_modules):
            self._hook_layer_forward(i, layer)

    # --- Phase 1c Stage 3: runner install (closures / slab specs) -----

    @staticmethod
    def _make_qkv_python_callback(w_cpu: torch.Tensor) -> PyCotsCallback:
        """Build a closure that captures the per-(layer, bucket) QKV
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
    ) -> PyCotsCallback:
        """Closure for the fused MLP block (gate + up + silu*up + down).
        Encapsulates the fused MLP block (gate + up + silu*up + down)
        with weights captured at install time so the operator-side call
        only carries x_pinned/y_pinned views.

        Stage 7-C: `w_down` is now Stage 7-C's transposed-storage view
        `(dn_n_cpu, out_dim) = (K, N)` row-major contig — passed to
        `torch.matmul` directly (the GEMM is `y = z @ w_down`, no
        F.linear-style implicit `.T`). gate / up stay on F.linear
        because they use the PyTorch-natural `(out, in)` layout.
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
    ) -> dict[tuple[int, int, str], PyCotsCallback]:
        """Build the (layer_idx, bucket, op_kind) -> closure table that
        PythonCotsRunner consults at submit time. Skip slabs where there
        is no CPU work (n_cpu_compute == 0).
        """
        callbacks: dict[tuple[int, int, str], PyCotsCallback] = {}
        for h in self._handles:
            if h.kind != "qkv":
                continue
            assert h.w_cpu is not None
            for bucket in self._capture_buckets:
                n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu = h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)
                if n_cpu == 0:
                    continue
                w_view = h.w_cpu.narrow(0, n_pref, n_cpu)
                callbacks[(h.layer_idx, bucket, "qkv")] = (
                    self._make_qkv_python_callback(w_view)
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
                # Stage 7-C: `dn_h.w_cpu` is now (n_cpu, out_dim) row-major.
                # The CPU-compute slice is rows `[dn_n_pref, dn_n_pref+dn_n_cpu)`
                # → contiguous `(dn_n_cpu, out_dim)` = `(K, N)` view for the
                # transposed-storage kernel path.
                w_down_view = dn_h.w_cpu.narrow(0, dn_n_pref, dn_n_cpu)
                callbacks[(gu_h.layer_idx, bucket, "mlp_block")] = (
                    self._make_mlp_python_callback(w_gate_view, w_up_view, w_down_view)
                )
        return callbacks

    def _n_threads_for(self, bucket: int) -> int:
        """Resolve the CPU GEMM thread count for a given bucket.

        Phase 1c Stage 4: when `config.cpu_num_threads_by_bucket` is
        set, look up the bucket; missing keys fall back to the scalar
        `cpu_num_threads` (the Planner is allowed to specify only the
        buckets it has profile data for; uncovered buckets get the
        scalar default rather than failing). When the dict is None
        (default), every bucket uses the scalar `cpu_num_threads` —
        Phase 1a/1b behavior, no per-bucket variation.

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

    def _build_native_slab_specs(self) -> list[NativeSlabSpec]:
        """Build the per-(layer, bucket, op_kind) slab specs that
        NativeCotsRunner.install populates into the C++ slab pool. All
        weight pointers are POST-narrow `data_ptr()`s; the down-proj
        slabs additionally carry strides reflecting the source tensor.
        Each slab's `n_threads` is per-bucket via `_n_threads_for` so
        the C++ worker dispatcher's cache-guarded `at::set_num_threads`
        picks up Stage 4's per-`BatchDescriptor` policy.
        """
        assert self._x_pinned is not None
        assert self._y_pinned is not None
        x_pinned_ptr = int(self._x_pinned.data_ptr())
        y_pinned_ptr = int(self._y_pinned.data_ptr())
        specs: list[NativeSlabSpec] = []
        for h in self._handles:
            if h.kind != "qkv":
                continue
            assert h.w_cpu is not None
            for bucket in self._capture_buckets:
                n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
                n_cpu = h.n_cpu_compute_by_bucket.get(bucket, h.n_cpu)
                if n_cpu == 0:
                    continue
                w_view = h.w_cpu.narrow(0, n_pref, n_cpu)
                specs.append(
                    _NativeSlabSpecQkv(
                        op_descriptor=(h.layer_idx, bucket, "qkv"),
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
                inter_per_half = n_cpu_per_half_total - n_pref_per_half
                w_gate_view = gu_h.w_cpu[n_pref_per_half:n_cpu_per_half_total, :]
                w_up_view = gu_h.w_cpu[
                    n_cpu_per_half_total + n_pref_per_half : 2 * n_cpu_per_half_total,
                    :,
                ]
                # Stage 7-C: `dn_h.w_cpu` is (n_cpu_total, out_dim);
                # narrow on dim 0 yields a contiguous (dn_n_cpu, out_dim)
                # = (K, N) view feedable to the transposed-storage kernel.
                #   w_down_rows = K (= dn_n_cpu)
                #   w_down_cols = N (= out_dim)
                w_down_view = dn_h.w_cpu.narrow(0, dn_n_pref, dn_n_cpu)
                specs.append(
                    _NativeSlabSpecMlp(
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
                        intermediate_per_half=int(inter_per_half),
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
        # Stage 4: validate per-bucket thread policy keys before
        # building specs so a Planner-mistyped bucket fails loudly at
        # install instead of silently falling back to scalar.
        self._validate_thread_policy()
        if isinstance(self._runner, PythonCotsRunner):
            callbacks = self._build_python_callbacks()
            self._runner.install(callbacks)
        elif isinstance(self._runner, NativeCotsRunner):
            slab_specs = self._build_native_slab_specs()
            # Stage 7-C: `max_num_tokens` gates the C++ worker's
            # submit-side / run-side bounds checks against the pinned
            # x/y buffers. The prior `scratch_max_intermediate_per_half`
            # parameter is gone — the silu(gate)*up intermediate is
            # now a fresh contig tensor allocated per call.
            self._runner.install(
                slab_specs=slab_specs,
                max_num_tokens=int(self._max_num_tokens),
            )
            # Stage 4: optional worker-thread CPU affinity. The C++
            # `set_worker_affinity` packs cpu IDs into a uint64 bitmask,
            # intersects with the process's `sched_getaffinity` mask,
            # and warns-and-skips on empty intersection. None / empty
            # list means "no opinion" — leave the kernel default.
            # §1c.19: routed through the cots_ops registry helper so
            # the offloader doesn't need to dereference
            # `runner._infer` (which no longer exists on the runner).
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

    def _install_bucket_prehook(self) -> None:
        """Phase 1c (Stage 3): install the first-decoder pre-hook
        UNCONDITIONALLY whenever COTS has handles, not gated on
        prefetch. The pre-hook calls `prepare_before_forward` which now
        always sets `_current_bucket` (plan §design-decision 11). Under
        `f_prefetch=0` (no streamer) this is the only path that sets
        `_current_bucket` under eager mode — without it, the operator's
        slab lookup would see `bucket=None`. Kept only as historical
        kill-switch scaffolding; production dispatch uses `on_dispatch`
        outside the captured graph.
        """
        if not self._layer_modules:
            return
        self._layer_modules[0].register_forward_pre_hook(
            self._first_decoder_pre_hook, with_kwargs=True
        )

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

    def _first_decoder_pre_hook(self, module, args, kwargs):
        """Fires at the start of every model forward. Sets the active
        bucket and repairs layer 0's slot if it's underfilled relative to
        the active bucket.

        Invariant: this hook is traced into the captured graph under
        `torch.compile(fullgraph=True)`, so the call chain
        (`prepare_before_forward` → `_bucket_for`) must stay
        Dynamo-traceable. `_bucket_for` uses a linear scan over the
        capture buckets (treated as a constant by Dynamo and unrolled
        at trace time) rather than `bisect.bisect_left` (a C builtin
        Dynamo cannot trace). Resolved as §1c.18 — see
        `phase1c_findings.md` for the history.

        Layer 0 is the only slot consumed before the current forward can
        issue a prefetch for it; all other layers are prefetched by their
        predecessor's pre-compute hook. The same repair is called at the
        Phase 1c FULL CUDA graph boundary.
        """
        del module
        anchor = args[0] if args else next(iter(kwargs.values()))
        self.prepare_before_forward(anchor.shape[0])

    def _bucket_for(self, num_tokens: int) -> int:
        """Round-up lookup on `_capture_buckets`. Returns the bucket key
        (matches `lookup_dispatch`'s rounding semantics, `planner_design.md
        §4.5`). Out-of-range returns the largest captured bucket.

        Implementation note: this used to call `bisect.bisect_left`, but
        `_bisect.bisect_left` is a C builtin Dynamo can't trace. Under
        `torch.compile(fullgraph=True)` the model's forward pre-hooks
        (which include this offloader's first-decoder pre-hook) are
        traced into the captured graph, so this lookup must be
        Dynamo-friendly. A linear scan over the bucket tuple is treated
        as a constant by Dynamo and unrolled at trace time. N is the
        number of capture buckets (typically O(10)), and this runs once
        per forward boundary — not per-GEMM — so the O(N) vs O(log N)
        difference is unobservable. See `phase1c_findings.md §1c.18`.
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

        Kept Dynamo-clean (no pybind calls) because vLLM's
        `_first_decoder_pre_hook` is registered on the model and gets
        traced into the captured graph. The C++-side runtime token
        row cap is pushed via the separate `set_live_num_tokens`
        method, called OUT OF GRAPH by `on_dispatch`.
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
        if isinstance(self._runner, NativeCotsRunner):
            raise RuntimeError(
                "CotsOffloader operator ran before dispatch state was "
                "published. GPUModelRunner._publish_offloader_dispatch "
                "must call CotsOffloader.on_dispatch before native COTS "
                "operators execute or capture."
            )
        return self._bucket_for(x_rows)

    def set_live_num_tokens(self, live_num_tokens: int) -> None:
        """Push the live unpadded token count to the C++ worker.

        CUDA graph buckets and native slabs are capacity-sized. This
        live-row cap lets the CPU worker skip padded rows inside the
        selected bucket.

        No-op when not using the native runner (PythonCotsRunner is
        eager-only and reads the live count directly off
        `slab.num_tokens.store` at submit time). Also no-op for
        `live_num_tokens <= 0` (sentinel).
        """
        if not self._has_cpu_compute_work:
            return
        if not isinstance(self._runner, NativeCotsRunner):
            return
        if int(live_num_tokens) <= 0:
            return
        from vllm.model_executor.offloader import cots_ops

        cots_ops.set_live_num_tokens(self._runner._runner_id, int(live_num_tokens))

    def set_runtime_num_tokens(self, actual_num_tokens: int) -> None:
        """Legacy alias for `set_live_num_tokens`."""
        self.set_live_num_tokens(actual_num_tokens)

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
        if self._has_cpu_compute_work and isinstance(self._runner, NativeCotsRunner):
            self._runner.set_active_dispatch(active_bucket, num_tokens_unpadded)
        self.sync_prev_onload()
        # CPU work scales with the semantic batch size, not bucket
        # capacity. Task selection is handled by the active dispatch
        # state above.
        self.set_live_num_tokens(num_tokens_unpadded)

    def post_cudagraph_capture(self) -> None:
        """§1c.22: env-gated counter reset after all bucket graphs are
        captured. Set `VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1`
        when running the byte-accounting bench so per-generate
        diagnostics isolate replay-time activity from capture/warmup.
        Default off — counters accumulate across capture + replay
        (which is fine for most diagnostics, including the §1c.21
        live-token confirmation)."""

        if (
            os.environ.get("VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE", "0")
            != "1"
        ):
            return
        from vllm.model_executor.offloader import cots_ops

        cots_ops.reset_all_counters()
        logger.info("[cots §1c.22] reset_all_counters() fired post-cudagraph-capture")

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
        # Tuple (not list) so Dynamo treats `_capture_buckets` as a
        # constant container during graph capture; the linear scan in
        # `_bucket_for` then unrolls cleanly. See §1c.18.
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

        # Phase 1c: two distinct dummy CUDA tensors for the
        # cots_sync_then_uva op's gpu_anchor_a / gpu_anchor_b mutates_args.
        # Operators pass these when the corresponding GPU GEMM (out_perm
        # / out_pref / out_gpu) didn't run for this slab — never aliased,
        # so torch.compile / functionalization sees two distinct
        # mutation slots. Allocated here (inside the
        # DeviceMemoryProfiler accounting window per phase1a §1.5),
        # NOT in __init__ which can predate CUDA device setup.
        self._dummy_gpu_anchor_a = torch.empty(1, dtype=dtype, device=device)
        self._dummy_gpu_anchor_b = torch.empty(1, dtype=dtype, device=device)

    # --- post_init: bookkeeping only ---

    def post_init(self) -> None:
        """Verify enforce_eager (conditional on runner) and finalize
        bookkeeping. The dispatch table and per-bucket geometry are
        built in `wrap_modules` (Phase 1b — they must exist before the
        prefetch buffer pool is sized inside the DeviceMemoryProfiler
        context)."""
        if not self._handles:
            return
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        # Phase 1c: the `enforce_eager` requirement is CONDITIONAL on
        # runner type:
        #   * cpu_runner='native': the C++ `cudaLaunchHostFunc`
        #     substrate IS graph-capturable (CUDA Graph host-function
        #     nodes, supported since CUDA 11.1). `enforce_eager=False`
        #     is SUPPORTED but is NOT the Phase 2 production
        #     recommendation — §1c.32 / §1c.33 nsys attribution showed
        #     native eager beats native+capture+wait-kernel on the
        #     measured Qwen2.5-7B workload grid. Captured-graph mode is
        #     preserved as an opt-in research path.
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

        # §1c.21 review-fix: native runner is incompatible with vLLM
        # microbatching/ubatching (DBO or ubatch_size > 1). The
        # live-token cap
        # (`GPUModelRunner._publish_offloader_dispatch` →
        # `BaseOffloader.set_live_num_tokens`)
        # currently sets ONE global value per scheduler batch. Under
        # ubatching, a COTS operator runs on a per-ubatch slice but
        # sees the cap as the FULL batch token count, which can
        # over-compute (the worker would read past the per-ubatch
        # x_pinned slice into stale data). Hard-fail at construction
        # with a clear message until §1c.23 plumbs per-ubatch live
        # counts.
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
                "phase1c_findings.md §1c.23."
            )

        # Wait-kernel-sync safety gates. Hard-fail at post_init when the
        # captured-sync mode is
        # set to wait-kernel but a precondition is missing. We check
        # BEFORE _install_runner so misconfigurations surface before
        # any C++ slab allocation. The host_callback path is unchanged
        # by these gates and remains available for diagnostics.
        capture_sync_mode = getattr(
            self.config, "cots_capture_sync_mode", "host_callback"
        )
        wait_kernel_enabled = capture_sync_mode == "wait_kernel"
        if self._has_cpu_compute_work and wait_kernel_enabled:
            # Gate 1: native runner only. Python runner has no slabs /
            # no host-mapped done_slot / no worker thread to publish.
            if cpu_runner != "native":
                raise RuntimeError(
                    f"CotsOffloader: cots_capture_sync_mode={capture_sync_mode!r} "
                    f"requires cpu_runner='native' (got {cpu_runner!r}). "
                    "The Python runner has no host-mapped done_slot "
                    "mechanism; the wait kernel is meaningful only on the "
                    "native substrate. Set cots_capture_sync_mode='host_callback' "
                    "or switch to cpu_runner='native'."
                )
            # Gate 2: graph-capture mode required. Eager mode launches
            # and syncs each iteration, so the wait kernel adds
            # round-trip cost without removing any captured sync_cb
            # node — net negative.
            if vllm_config.model_config.enforce_eager:
                raise RuntimeError(
                    f"CotsOffloader: cots_capture_sync_mode={capture_sync_mode!r} "
                    "requires enforce_eager=False (graph-capture mode). "
                    "The wait kernel replaces the captured sync_cb host_fn "
                    "node; under enforce_eager=True there is no captured "
                    "node to replace. Set cots_capture_sync_mode='host_callback' "
                    "or enforce_eager=False."
                )
            # Gate 3: CUDA available (defensive — native runner already
            # requires CUDA, but a clearer error here pinpoints the
            # wait-kernel sync as the configuration that needs the GPU).
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CotsOffloader: cots_capture_sync_mode={capture_sync_mode!r} "
                    "requires CUDA to be available; the wait kernel runs "
                    "on the GPU."
                )
            # Gate 4: _cots_C extension built (defensive — already
            # required by NativeCotsRunner, but with the wait kernel
            # we want a tight blame line if the build was partial).
            try:
                from vllm import _cots_C  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    f"CotsOffloader: cots_capture_sync_mode={capture_sync_mode!r} "
                    "requires the `vllm._cots_C` extension. Rebuild vLLM "
                    "with CUDA support (./rebuild_vllm.sh). Underlying "
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

        # Phase 1c Stage 3: install the runner's per-(layer, bucket,
        # op_kind) work table. Python runner gets a closures dict
        # capturing weight views; native runner populates the C++ slab
        # pool. Both can be done at post_init: the underlying
        # `w_cpu`/`_x_pinned`/`_y_pinned` storages are stable
        # post-allocation regardless of when their views are taken.
        self._install_runner()

        # Install wait-kernel sync host-mapped pinned slots once
        # the slab pool exists. The installer walks every slab in the
        # pool and allocates the req/done slot pair (and the diag
        # counter cells lazily, only when VLLM_COTS_DIAG=1 — see
        # commit 1 review-fix). After this returns, the captured
        # graph's sync_cb host_fn nodes are replaced with wait kernel
        # launches at every cots_sync_then_uva call site.
        if (
            self._has_cpu_compute_work
            and wait_kernel_enabled
            and isinstance(self._runner, NativeCotsRunner)
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
            "[CotsOffloader] ready: runner=%s, sync=%s, linears=%d, "
            "mlp_blocks=%d, weights_saved=%.4f GB, buffers=%.4f GB "
            "pinned_in + %.4f GB pinned_out + %.4f GB gpu_uva, "
            "prefetch_pool=%.4f GB, buckets=%s",
            cpu_runner,
            capture_sync_mode,
            len(self._handles),
            len(self._fused_ops),
            total_offloaded / 1e9,
            x_pinned_gb,
            y_pinned_gb,
            y_gpu_gb,
            prefetch_gb,
            self._capture_buckets,
        )

        # Effective routing breakdown — actual bytes routed through each
        # path, accounting for head-aligned snapping and per-kind geometry.
        # Reported at the largest capture bucket (worst case for prefetch
        # buffer sizing).
        bucket = self._capture_buckets[-1]
        elem = self._handles[0].dtype.itemsize
        per_kind_pref = {"qkv": 0, "col": 0, "row": 0}
        per_kind_cpu = {"qkv": 0, "col": 0, "row": 0}
        for h in self._handles:
            n_pref = h.n_prefetch_by_bucket.get(bucket, 0)
            n_cpu = h.n_cpu_compute_by_bucket.get(bucket, 0)
            other_dim = h.in_dim if h.kind != "row" else h.out_dim
            per_kind_pref[h.kind] += n_pref * other_dim * elem
            per_kind_cpu[h.kind] += n_cpu * other_dim * elem
        total_pref = sum(per_kind_pref.values())
        total_cpu = sum(per_kind_cpu.values())
        logger.debug(
            "[CotsOffloader] Effective routing @ bucket=%d:\n"
            "  qkv:  prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  col:  prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  row:  prefetched=%.4f GiB, cpu-computed=%.4f GiB\n"
            "  total: prefetched=%.4f GiB, cpu-computed=%.4f GiB",
            bucket,
            per_kind_pref["qkv"] / 1024**3,
            per_kind_cpu["qkv"] / 1024**3,
            per_kind_pref["col"] / 1024**3,
            per_kind_cpu["col"] / 1024**3,
            per_kind_pref["row"] / 1024**3,
            per_kind_cpu["row"] / 1024**3,
            total_pref / 1024**3,
            total_cpu / 1024**3,
        )

    # --- Runtime: dispatch lookup ---

    def lookup_dispatch(self, num_tokens: int) -> tuple[float, float]:
        """Per `planner_design.md §4.5`: round `num_tokens` UP to the nearest
        capture bucket; out-of-range falls back to the largest bucket's entry.
        Shares `_bucket_for`'s rounding semantics so dispatch and bucket
        state always agree (and so neither path uses `bisect_left` —
        see §1c.18).
        """
        if num_tokens > self._capture_buckets[-1]:
            return self._eager_fallback_entry
        return self._dispatch_table[self._bucket_for(num_tokens)]
