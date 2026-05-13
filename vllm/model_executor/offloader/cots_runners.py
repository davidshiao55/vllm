# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime runners and native slab specs for COTS."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.model_executor.offloader.cots_utils import (
    _get_executor,
    uva_copy_into_gpu,
)

if TYPE_CHECKING:
    from vllm.config import CotsOffloadConfig


class NativeSlabSpec:
    """Builder record for one C++ TaskSlab. The offloader builds a list
    of these at `post_init` (one per (layer_idx, bucket, op_kind) with
    n_cpu_compute > 0); `NativeCotsRunner.install` walks the list and
    calls the right `populate_slab_*` per record.

    All weight pointers passed to populate_slab_* must be POST-narrow
    `data_ptr()`s — `at::from_blob` has no storage-offset parameter, so
    the offset must be baked into the pointer. The strided variant
    (down-proj column slice) additionally carries `(stride_row,
    stride_col)` from the source tensor's `.stride()`.

    `dry_run=True` overrides the op_kind to dryrun_noop on every slab so
    the host-callback round-trip is exercised end-to-end without real
    CPU GEMM (used by the Stage 2 substrate gate, mirrors
    PythonCotsRunner's dry_run path).
    """

    op_descriptor: tuple[int, int, str]

    def __init__(self, op_descriptor: tuple[int, int, str]) -> None:
        self.op_descriptor = op_descriptor

    def populate(
        self,
        infer: object,
        task_id: int,
        *,
        dry_run: bool,
    ) -> None:
        raise NotImplementedError


class _NativeSlabSpecQkv(NativeSlabSpec):
    def __init__(
        self,
        op_descriptor: tuple[int, int, str],
        *,
        n_threads: int,
        x_pinned_ptr: int,
        in_dim: int,
        y_pinned_ptr: int,
        cpu_out_dim: int,
        w_cpu_ptr: int,
        w_cpu_rows: int,
    ) -> None:
        super().__init__(op_descriptor)
        self.n_threads = n_threads
        self.x_pinned_ptr = x_pinned_ptr
        self.in_dim = in_dim
        self.y_pinned_ptr = y_pinned_ptr
        self.cpu_out_dim = cpu_out_dim
        self.w_cpu_ptr = w_cpu_ptr
        self.w_cpu_rows = w_cpu_rows

    def populate(self, infer: object, task_id: int, *, dry_run: bool) -> None:
        # §1c.22 review-fix: bucket_capacity_tokens is the descriptor
        # bucket (op_descriptor[1]). Stable for the slab's lifetime;
        # what replay-time byte counters use to attribute bucket-sized
        # captured copies. Distinct from `slab.num_tokens` which is
        # mutable submit-time/captured state.
        bucket_capacity_tokens = int(self.op_descriptor[1])
        if dry_run:
            # §1c.20: dryrun must pass x_pinned_ptr + in_dim (for the
            # D2H in submit_on_stream) AND y_pinned_ptr + cpu_out_dim
            # (for y_pinned_view in sync) so both captured-graph
            # paths resolve at task_id even though the worker skips
            # the GEMM.
            infer.populate_slab_dryrun(  # type: ignore[attr-defined]
                task_id=task_id,
                bucket_capacity_tokens=bucket_capacity_tokens,
                x_pinned_ptr=self.x_pinned_ptr,
                in_dim=self.in_dim,
                y_pinned_ptr=self.y_pinned_ptr,
                cpu_out_dim=self.cpu_out_dim,
            )
            return
        infer.populate_slab_qkv(  # type: ignore[attr-defined]
            task_id=task_id,
            n_threads=self.n_threads,
            bucket_capacity_tokens=bucket_capacity_tokens,
            x_pinned_ptr=self.x_pinned_ptr,
            in_dim=self.in_dim,
            y_pinned_ptr=self.y_pinned_ptr,
            cpu_out_dim=self.cpu_out_dim,
            w_cpu_ptr=self.w_cpu_ptr,
            w_cpu_rows=self.w_cpu_rows,
        )


class _NativeSlabSpecMlp(NativeSlabSpec):
    def __init__(
        self,
        op_descriptor: tuple[int, int, str],
        *,
        n_threads: int,
        x_pinned_ptr: int,
        in_dim: int,
        y_pinned_ptr: int,
        cpu_out_dim: int,
        w_gate_ptr: int,
        w_gate_rows: int,
        w_up_ptr: int,
        w_up_rows: int,
        w_down_ptr: int,
        w_down_rows: int,
        w_down_cols: int,
        intermediate_per_half: int,
    ) -> None:
        super().__init__(op_descriptor)
        self.n_threads = n_threads
        self.x_pinned_ptr = x_pinned_ptr
        self.in_dim = in_dim
        self.y_pinned_ptr = y_pinned_ptr
        self.cpu_out_dim = cpu_out_dim
        self.w_gate_ptr = w_gate_ptr
        self.w_gate_rows = w_gate_rows
        self.w_up_ptr = w_up_ptr
        self.w_up_rows = w_up_rows
        self.w_down_ptr = w_down_ptr
        self.w_down_rows = w_down_rows
        self.w_down_cols = w_down_cols
        self.intermediate_per_half = intermediate_per_half

    def populate(self, infer: object, task_id: int, *, dry_run: bool) -> None:
        # §1c.22 review-fix: see _NativeSlabSpecQkv.populate.
        bucket_capacity_tokens = int(self.op_descriptor[1])
        if dry_run:
            # §1c.20: dryrun must pass x_pinned_ptr + in_dim (for the
            # D2H in submit_on_stream) AND y_pinned_ptr + cpu_out_dim
            # (for y_pinned_view in sync) so both captured-graph
            # paths resolve at task_id even though the worker skips
            # the GEMM.
            infer.populate_slab_dryrun(  # type: ignore[attr-defined]
                task_id=task_id,
                bucket_capacity_tokens=bucket_capacity_tokens,
                x_pinned_ptr=self.x_pinned_ptr,
                in_dim=self.in_dim,
                y_pinned_ptr=self.y_pinned_ptr,
                cpu_out_dim=self.cpu_out_dim,
            )
            return
        infer.populate_slab_mlp(  # type: ignore[attr-defined]
            task_id=task_id,
            n_threads=self.n_threads,
            bucket_capacity_tokens=bucket_capacity_tokens,
            x_pinned_ptr=self.x_pinned_ptr,
            in_dim=self.in_dim,
            y_pinned_ptr=self.y_pinned_ptr,
            cpu_out_dim=self.cpu_out_dim,
            w_gate_ptr=self.w_gate_ptr,
            w_gate_rows=self.w_gate_rows,
            w_up_ptr=self.w_up_ptr,
            w_up_rows=self.w_up_rows,
            w_down_ptr=self.w_down_ptr,
            w_down_rows=self.w_down_rows,
            w_down_cols=self.w_down_cols,
            intermediate_per_half=self.intermediate_per_half,
        )


PyCotsCallback = Callable[[torch.cuda.Event, torch.Tensor, torch.Tensor], None]


class PythonCotsRunner:
    """Phase 1a/1b-shaped CPU task runner — the kill-switch path under
    `CotsOffloadConfig.cpu_runner = "python"`. Same execution semantics
    as the original `CpuTaskRunner` (single-worker `ThreadPoolExecutor`
    + `future.result()`), but the operator-facing API is now the Phase
    1c uniform facade so operators can use `submit_with_d2h(x, x_pinned,
    y_pinned, op_descriptor)` without branching on runner type.

    The per-(layer, bucket, op_kind) weight views are bound at install
    time into closures stored in `_callbacks`; the operator just hands
    over an op_descriptor and the runner looks up the right closure.

    Eager-only: `ThreadPoolExecutor.submit` is not graph-capturable, so
    selecting this runner with `enforce_eager=False` is a hard error
    (Stage 5 enforces).
    """

    kind = "python"

    def __init__(self, dry_run: bool = False) -> None:
        self._future: Future | None = None
        # §1c.20 facade: y_pinned is stashed at submit time so
        # wait_and_uva doesn't need to take it as a parameter.
        # NativeCotsRunner reaches y_pinned via the slab pointer; for
        # facade symmetry the eager Python runner uses this stashed
        # reference, so no Python-side caller has to pass y_pinned
        # through to wait_and_uva.
        self._pending_y_pinned: torch.Tensor | None = None
        self._dry_run = dry_run
        self._callbacks: dict[tuple[int, int, str], PyCotsCallback] = {}
        self._installed = False

    def install(
        self,
        callbacks: dict[tuple[int, int, str], PyCotsCallback],
    ) -> None:
        """Register the per-(layer, bucket, op_kind) work closures.

        The offloader builds these at `post_init` from the dispatch table
        + handle weight views; one closure per slab-equivalent. Under
        `dry_run=True` the offloader installs noop closures (only the
        D2H event sync runs, no real CPU GEMM) — the runner's
        submit/wait shape is identical either way so timing-sensitive
        diagnostics can A/B by toggling `dry_run`.

        §1c.19: this method intentionally does NOT take a
        `bucket_for_fallback` callable. Operators are required to
        resolve `op_descriptor[1]` to a non-None int before calling
        `submit_with_d2h` (matches the NativeCotsRunner contract so
        operators don't branch on runner type).
        """
        if self._installed:
            raise RuntimeError(
                "PythonCotsRunner.install() called twice on the same instance"
            )
        self._callbacks = dict(callbacks)
        self._installed = True

    def submit_with_d2h(
        self,
        x_gpu: torch.Tensor,
        x_pinned: torch.Tensor,
        y_pinned: torch.Tensor,
        op_descriptor: tuple[int, int, str],
    ) -> None:
        """Phase 1c uniform facade. D2H copies `x_gpu` into `x_pinned`,
        records an event ordering the H2D, and submits the work closure
        that fills `y_pinned`.

        §1c.20: this runner is eager-only (graph capture is hard-fail
        under `cpu_runner='python'`); Inductor isn't involved, so the
        facade keeps `x_pinned` / `y_pinned` as Python parameters
        without any of the captured-graph materialization concerns
        that drove NativeCotsRunner to the slab-pointer-first design.

        §1c.19: `op_descriptor[1]` MUST be a resolved int. Operators
        run `offloader._bucket_for(num_tokens)` themselves before
        invoking the runner — the runner facade carries no
        offloader-bound state.
        """
        layer_idx, bucket, op_kind = op_descriptor
        assert isinstance(bucket, int), (
            "PythonCotsRunner.submit_with_d2h: op_descriptor[1] must be "
            "a resolved int bucket. See phase1c_findings.md §1c.19."
        )
        x_pinned.copy_(x_gpu, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        if self._dry_run:
            cb: PyCotsCallback = _cpu_dryrun_noop
        else:
            cb = self._callbacks[(layer_idx, bucket, op_kind)]
        # §1c.20: stash y_pinned so wait_and_uva doesn't need it as a
        # parameter (matches the NativeCotsRunner facade where
        # y_pinned isn't part of the wait_and_uva signature).
        self._pending_y_pinned = y_pinned
        self._future = _get_executor().submit(cb, event, x_pinned, y_pinned)

    def wait_and_uva(
        self,
        y_gpu: torch.Tensor,
        gpu_anchor_a: torch.Tensor,
        gpu_anchor_b: torch.Tensor,
        submit_anchor: torch.Tensor,
        op_descriptor: tuple[int, int, str],
    ) -> None:
        """Drain the worker and copy the pinned result into the GPU
        buffer via the SM-issued UVA kernel. `gpu_anchor_a` /
        `gpu_anchor_b` / `submit_anchor` / `op_descriptor` are accepted
        for API symmetry with NativeCotsRunner
        — they're the barrier-installing anchors that pin the
        `cots_sync_then_uva` custom op AFTER each independent GPU GEMM
        under graph capture. Under PythonCotsRunner we're eager-only
        (no graph capture), so the anchors are unused; but accepting
        them lets operators call both runners with one signature.
        """
        del gpu_anchor_a, gpu_anchor_b, submit_anchor, op_descriptor
        assert self._future is not None, "submit_with_d2h() not called"
        assert self._pending_y_pinned is not None, (
            "submit_with_d2h() did not stash y_pinned"
        )
        y_pinned = self._pending_y_pinned
        try:
            self._future.result()  # re-raises worker exceptions
        finally:
            self._future = None
            self._pending_y_pinned = None
        uva_copy_into_gpu(y_pinned, y_gpu)

    def close(self) -> None:
        """Drain any pending task. Idempotent; safe to call from teardown."""
        if self._future is not None:
            with contextlib.suppress(Exception):
                self._future.result()
            self._future = None


# Backwards-compat alias for any external code that imported the old name.
CpuTaskRunner = PythonCotsRunner


class NativeCotsRunner:
    """Phase 1c production runner. Wraps the C++ `CotsCpuInfer` via the
    `vllm._cots_C` extension; dispatches CPU work through
    `cudaLaunchHostFunc` so the forward pass is graph-capturable.

    Operator-facing API carries only stable call-site identity:
    `(layer_idx, op_kind)`. Active bucket and live-token state are
    published out of graph by `CotsOffloader.on_dispatch`, then resolved
    inside the `vllm.cots_submit_gemm` / `vllm.cots_sync_then_uva`
    custom-op impls at eager execution / CUDA Graph capture time. This
    keeps native task selection out of compile-visible scalar arguments
    while preserving the barrier-installing `mutates_args` declarations
    on `x_gpu` and `gpu_anchor_a/_b`.

    Multi-engine safety: each instance allocates a `runner_id` and
    the underlying `CotsCpuInfer` pybind handle is owned by the
    `cots_ops._COTS_INFER` strong-ref registry (§1c.19 split — was a
    `_COTS_RUNNERS` weak map prior). The runner facade itself only
    holds the integer id + picklable state, so PyTorch's AOT compile
    guard cache can serialize it. Two offloaders (FastTTS gen + ver)
    coexist with independent slab pools. `close()` drains the worker
    then unregisters.

    Two row counts are first-class in native COTS:

      * `slab.num_tokens` — the **dispatched graph bucket** selected by
        vLLM's `BatchDescriptor.num_tokens`. It is pushed out of graph
        before the forward and resolved by the custom-op impl during
        capture/execution; it is not derived from `int(x.shape[0])`.
      * `live_num_tokens` — the **live rows to compute**, set OUT
        OF GRAPH by `GPUModelRunner._publish_offloader_dispatch`
        BEFORE every scheduler, dummy/profile, warmup, and CUDA Graph
        capture forward. Always `live_num_tokens <= slab.num_tokens`. The
        worker's `at::linear` shapes, scratch slicing, and y_pinned
        write region all key off this.

    For B=1 decode under FULL capture: bucket might be 256 but
    runtime is 1. The worker computes 1 row of GEMM even though the
    captured graph fired at the bucket size. Captured D2H copies
    bucket-sized bytes (PCIe waste tracked under
    `worker_input_live_bytes` vs `d2h_replay_bucket_bytes`); §1c.22 follow-up
    if material.
    """

    kind = "native"

    def __init__(self, dry_run: bool = False) -> None:
        # Lazy import: _cots_C is built only on CUDA. Users on CPU-only
        # / ROCm builds shouldn't hit ImportError just by importing this
        # module — the runner type is constructed only when the offload
        # config selects `cpu_runner='native'`. Any reference we hold
        # to the pybind handle is on the cots_ops registry, NOT on this
        # runner's `__dict__` (§1c.19): if the handle were stored on
        # `self`, Dynamo's guard serialization would walk it and try to
        # pickle a `CotsCpuInfer`, which is unpicklable.
        try:
            from vllm import _cots_C
        except ImportError as e:
            raise RuntimeError(
                "NativeCotsRunner requires the `vllm._cots_C` extension, "
                "which builds only on CUDA targets. Either select "
                "`cpu_runner='python'` or rebuild vLLM with CUDA support."
            ) from e
        from vllm.model_executor.offloader import cots_ops

        # Hand the freshly-constructed handle to the registry; the
        # registry's strong reference is now the SOLE owner. The local
        # variable goes out of scope at the end of __init__, so nothing
        # in `self.__dict__` references it.
        self._runner_id: int = cots_ops._register_infer(_cots_C.CotsCpuInfer())
        self._dry_run: bool = bool(dry_run)
        # Stage 3 will populate this from CotsOffloader._build_slab_table.
        # Format: {(layer_idx, bucket, op_kind): task_id}.
        self._task_id_for: dict[tuple[int, int, str], int] = {}
        self._installed: bool = False
        # §1c.19 ownership flag. Multiple `NativeCotsRunner` instances
        # can share the same `_runner_id` after a pickle round-trip
        # (PyTorch's AOT guard cache pickles + unpickles the runner as
        # part of guard serialization). Only the ORIGINAL constructor
        # owns the registry entry — the unpickled copy is non-owning.
        # `close()` and `__del__` no-op for non-owners so GC of a
        # guard-cache copy can't unregister the live runner's infer.
        # See `__getstate__` / `__setstate__` below for the pickle
        # path that flips this to False.
        self._owns_infer_registry_entry: bool = True

    def install(
        self,
        slab_specs: list[NativeSlabSpec],
        max_num_tokens: int,
    ) -> None:
        """Allocate the C++ slab pool, populate slabs, and build the
        op_descriptor -> task_id map. Called once at offloader
        `post_init` after the dispatch table + handle weight views are
        known.

        `slab_specs` is a list (ordering = task_id) of NativeSlabSpec
        records — one per (layer_idx, bucket, op_kind) where there is
        actual CPU work (n_cpu_compute > 0). Under `dry_run=True` the
        offloader passes specs with op_kind='dryrun_noop' so the
        runtime path exercises the full host-callback round-trip but
        skips real GEMM (mirrors Phase 1a/1b's PythonCotsRunner dry_run
        diagnostic, see phase1a_findings.md §1.14).

        §1c.35: native operators do not pass descriptor buckets at
        runtime. They pass stable call-site identity, and cots_ops
        resolves the bucket/task from OOG dispatch state plus this
        install-time map.
        """
        from vllm.model_executor.offloader import cots_ops

        if self._installed:
            raise RuntimeError(
                "NativeCotsRunner.install() called twice on the same instance"
            )
        n_slabs = len(slab_specs)
        cots_ops.install_infer(
            self._runner_id,
            n_slabs=n_slabs,
            max_num_tokens=max_num_tokens,
        )
        for tid, spec in enumerate(slab_specs):
            self._task_id_for[spec.op_descriptor] = tid
            cots_ops.populate_slab_via_spec(
                self._runner_id, spec, tid, dry_run=self._dry_run
            )
        cots_ops.register_task_id_map(self._runner_id, self._task_id_for)
        self._installed = True
        # Cache n_slabs so wait-kernel installation does not have to
        # re-walk task_id_for. Set last so any populate-time error
        # leaves the cache empty.
        self._n_slabs: int = n_slabs

    def install_wait_kernel_sync(self) -> None:
        """§1c.29 commit 2: install host-mapped pinned req/done
        slots for every slab in the pool. Must be called AFTER
        `install()` and only when the offloader's feature flag
        `cots_capture_sync_mode="wait_kernel"`.

        After this returns, every slab's `wait_kernel_sync_installed=True`, the
        captured `dispatch_cb` writes `req_slot=seq` per dispatch,
        the worker writes `done_slot=seq` (in a finally-style
        block) per dispatch, and `cots_sync_then_uva` routes
        through the wait kernel via `sync_or_wait_on_stream`.
        """
        from vllm.model_executor.offloader import cots_ops

        if not self._installed:
            raise RuntimeError(
                "NativeCotsRunner.install_wait_kernel_sync() called before install(); "
                "wait-kernel sync needs the slab pool to exist first"
            )
        cots_ops.install_wait_kernel_sync_for_all_tasks(
            self._runner_id,
            self._n_slabs,
        )

    def set_active_dispatch(self, bucket: int, live_num_tokens: int) -> None:
        """Publish OOG dispatch state for native custom-op resolution."""
        from vllm.model_executor.offloader import cots_ops

        cots_ops.set_active_dispatch_state(
            self._runner_id,
            bucket=int(bucket),
            live_num_tokens=int(live_num_tokens),
        )

    def submit_with_d2h(
        self,
        x_gpu: torch.Tensor,
        layer_idx: int,
        op_kind: str,
    ) -> None:
        """Phase 1c native facade. Routes through
        `torch.ops.vllm.cots_submit_gemm` whose impl bundles a
        `cudaMemcpyAsync(D2H)` from `x_gpu` to `slab.x_pinned_ptr`
        with the host-callback enqueue — both on the current CUDA
        stream. The op's `mutates_args=["x_gpu"]` pins this op BEFORE
        every GPU GEMM that reads x_gpu and provides the
        (submit → sync) ordering edge consumed by
        `cots_sync_then_uva`'s `submit_anchor` read.

        §1c.20: x_pinned and y_pinned are NOT parameters here — even
        passing them as Python args would let Dynamo/Inductor trace
        the slice/view chain into the captured graph and materialize
        the CPU views via blocking GPU→CPU copies that CUDA Graph
        capture rejects. The slab pointers (populated at install
        time) are the source of truth for both directions.

        §1c.35: bucket/task selection is intentionally absent from
        this signature. `CotsOffloader.on_dispatch` publishes the
        active `BatchDescriptor.num_tokens` OOG; the custom-op impl
        resolves `(layer_idx, active_bucket, op_kind)` to the C++
        slab task_id at execution/capture time.
        """
        from vllm.model_executor.offloader import cots_ops

        torch.ops.vllm.cots_submit_gemm(
            x_gpu,
            self._runner_id,
            int(layer_idx),
            cots_ops.op_kind_code(op_kind),
        )

    def wait_and_uva(
        self,
        y_gpu: torch.Tensor,
        gpu_anchor_a: torch.Tensor,
        gpu_anchor_b: torch.Tensor,
        submit_anchor: torch.Tensor,
        layer_idx: int,
        op_kind: str,
    ) -> None:
        """Routes through `torch.ops.vllm.cots_sync_then_uva` so the
        cudaLaunchHostFunc-based stream sync + the Triton UVA copy
        bundle into one graph-recorded entry.

        Barrier roles:
        - `gpu_anchor_a` / `gpu_anchor_b` are mutated, pinning sync
          AFTER each independent GPU GEMM (`out_perm`, `out_pref`).
          Operators pass two distinct CUDA tensors, never aliased.
        - `submit_anchor` (§1c.20) is read-only — the same `x_gpu`
          that `cots_submit_gemm` mutated. Reading it pins sync
          AFTER submit.

        §1c.20: `y_pinned` is intentionally NOT a parameter — not on
        the custom op AND not on this facade. Even passing y_pinned
        as a Python parameter (and immediately discarding it) was
        enough to make Inductor trace the `_y_pinned[:N].view(...)`
        slice/view chain into the captured graph and materialize it
        via a GPU intermediate + blocking GPU→CPU copy that CUDA
        Graph capture rejects. The impl reaches the worker's pinned
        output via `CotsCpuInfer.y_pinned_view(task_id, bucket)`.
        Task id and bucket are resolved from OOG dispatch state inside
        the custom-op impl, not passed as compile-visible scalars.
        """
        from vllm.model_executor.offloader import cots_ops

        torch.ops.vllm.cots_sync_then_uva(
            y_gpu,
            gpu_anchor_a,
            gpu_anchor_b,
            submit_anchor,
            self._runner_id,
            int(layer_idx),
            cots_ops.op_kind_code(op_kind),
        )

    def __getstate__(self) -> dict:
        """Pickle hook — used by PyTorch's AOT compile guard cache when
        it serializes the captured forward's closure values
        (§1c.19/§1c.19.1). The unpickled facade points at the SAME
        `_runner_id` as the original; if it ran the same `__del__` /
        `close()` path, GC of a guard-cache copy could yank the live
        registry entry while the original is still serving requests.
        Mark the pickled state non-owning so the unpickled copy's
        `close()` and `__del__` no-op."""
        state = self.__dict__.copy()
        state["_owns_infer_registry_entry"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Defensive: in case a state dict from a future pickle didn't
        # carry the flag, default to non-owning.
        self.__dict__.setdefault("_owns_infer_registry_entry", False)

    def close(self) -> None:
        """Drain any in-flight worker task and drop the registry entry.
        Idempotent; safe to call from teardown. Both the drain and the
        unregister go through cots_ops helpers so this runner facade
        never has to dereference the pybind handle directly (see
        §1c.19).

        No-ops for non-owning copies (`_owns_infer_registry_entry` is
        False after `__setstate__`). The original constructor is the
        sole owner; only IT may drain or unregister.
        """
        if not getattr(self, "_owns_infer_registry_entry", False):
            return
        from vllm.model_executor.offloader import cots_ops

        try:
            cots_ops.sync_blocking(self._runner_id)
        finally:
            cots_ops._unregister_infer(self._runner_id)
            self._owns_infer_registry_entry = False

    def __del__(self) -> None:
        # Best-effort registry cleanup if the user forgot to call close().
        # Note: don't raise from __del__ — the GC log is unhelpful.
        #
        # No-op for non-owning copies (e.g., AOT guard-cache unpickled
        # facades) — they share `_runner_id` with the original but must
        # NOT unregister it on GC. The original's `__del__` / `close()`
        # is the sole path that may drop the entry.
        #
        # FORWARD RISK (review finding #3, Stage 2 sign-off): the
        # owning path here only unregisters; it does NOT drain the CUDA
        # stream of any in-flight `cudaLaunchHostFunc` callbacks. Stage
        # 4 / 5 follow-up should add a BaseOffloader-level shutdown
        # hook that drains the compute stream and closes the runner
        # before slabs are freed, OR a best-effort
        # `torch.cuda.current_stream().synchronize()` here (also
        # dangerous from a finalizer if CUDA is torn down).
        #
        # `try/except: pass` rather than `contextlib.suppress` because at
        # interpreter shutdown `contextlib` itself can be None;
        # try/except is the only finalizer-safe form. Lazy-import the
        # registry module too — interpreter shutdown can already have
        # cleared it.
        try:  # noqa: SIM105
            if getattr(self, "_owns_infer_registry_entry", False):
                from vllm.model_executor.offloader import cots_ops

                cots_ops._unregister_infer(self._runner_id)
                self._owns_infer_registry_entry = False
        except Exception:  # noqa: BLE001
            pass


def _make_runner(config: CotsOffloadConfig) -> PythonCotsRunner | NativeCotsRunner:
    """Construct the offloader's single runner per `config.cpu_runner`.

    One runner per offloader (not per operator) — see plan §design-decision
    4 "One offloader-owned runner shared across all operators". This
    moves the Phase 1a/1b pattern of fresh `CpuTaskRunner()` per op
    (cots.py:1752 and :1807 in the legacy code) onto a single shared
    instance, which is the structural prerequisite for the native
    runner's per-offloader slab pool + runner_id.
    """
    # CotsOffloadConfig.cpu_runner now defaults to "native" (Stage 5
    # production path). The fallback here intentionally stays "python"
    # for legacy config shims that lack the field entirely — those
    # have never been exercised under the native runner's slab/install
    # path, so silently routing them through native would surface
    # untested code paths at the operator call site. New config
    # objects always carry the `cpu_runner` field via pydantic so the
    # fallback only fires for hand-rolled stub configs in tests.
    cpu_runner = getattr(config, "cpu_runner", "python")
    dry_run = bool(getattr(config, "dry_run", False))
    if cpu_runner == "python":
        return PythonCotsRunner(dry_run=dry_run)
    if cpu_runner == "native":
        return NativeCotsRunner(dry_run=dry_run)
    raise ValueError(
        f"Unknown cpu_runner={cpu_runner!r}; expected 'native' or 'python'"
    )


# ---------------------------------------------------------------------------
# Worker-thread bodies: pure CPU work (event sync + GEMMs / SwiGLU).
# Standalone module-level functions so Phase 1c's cudaLaunchHostFunc can
# bind them as host-function userData callbacks without method-binding.
# ---------------------------------------------------------------------------
def _cpu_gemm_into_after_event(
    d2h_event: torch.cuda.Event,
    x_pinned: torch.Tensor,
    w_cpu: torch.Tensor,
    y_pinned: torch.Tensor,
) -> None:
    """Generic CPU GEMM: D2H wait → BF16 matmul → write to pinned out."""
    d2h_event.synchronize()
    y_pinned.copy_(F.linear(x_pinned, w_cpu))


def _cpu_dryrun_noop(
    d2h_event: torch.cuda.Event,
    *_args,
    **_kwargs,
) -> None:
    """Diagnostic: sync the D2H event but skip the GEMM. Token output is
    garbage; only host bookkeeping cost is measured. See `phase1a_findings.md
    §1.14`."""
    d2h_event.synchronize()


# ---------------------------------------------------------------------------
# Operator layer
#
# Operators encapsulate the forward semantics for each block of the model
# we offload. They consume `CotsLinearHandle`s for storage and a
# `CpuTaskRunner` for execution.
# ---------------------------------------------------------------------------
