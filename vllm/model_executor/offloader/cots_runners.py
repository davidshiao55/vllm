# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime runners and native slab specs for COTS."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.offloader.cots_utils import (
    _get_executor,
    uva_copy_into_gpu,
)

if TYPE_CHECKING:
    from vllm.config import CotsOffloadConfig


class NativeWeightSlabSpec:
    """Builder record for one C++ TaskSlab. The offloader builds a list
    of these at `post_init` (one per (layer_idx, bucket, op_kind) with
    n_cpu_compute > 0); `NativeCotsWeightRunner.install` walks the list and
    calls the right `populate_slab_*` per record.

    All weight pointers passed to populate_slab_* must be POST-narrow
    `data_ptr()`s — `at::from_blob` has no storage-offset parameter, so
    the offset must be baked into the pointer. The strided variant
    (down-proj column slice) additionally carries `(stride_row,
    stride_col)` from the source tensor's `.stride()`.

    `dry_run=True` overrides the op_kind to dryrun_noop on every slab so
    the host-callback round-trip is exercised end-to-end without real
    CPU GEMM.
    """

    op_descriptor: tuple[int, int, str]

    def __init__(self, op_descriptor: tuple[int, int, str]) -> None:
        self.op_descriptor = op_descriptor

    def populate(
        self,
        runner_handle: object,
        task_id: int,
        *,
        dry_run: bool,
    ) -> None:
        raise NotImplementedError


class _NativeWeightSlabSpecLinear(NativeWeightSlabSpec):
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

    def populate(self, runner_handle: object, task_id: int, *, dry_run: bool) -> None:
        # Stable descriptor bucket for replay-time byte accounting. Distinct
        # from `slab.num_tokens`, which is mutable submit-time state.
        bucket_capacity_tokens = int(self.op_descriptor[1])
        if dry_run:
            # Dry-run slabs still need pinned input/output metadata so submit
            # and sync resolve by task_id even though the worker skips the GEMM.
            runner_handle.populate_slab_dryrun(  # type: ignore[attr-defined]
                task_id=task_id,
                bucket_capacity_tokens=bucket_capacity_tokens,
                x_pinned_ptr=self.x_pinned_ptr,
                in_dim=self.in_dim,
                y_pinned_ptr=self.y_pinned_ptr,
                cpu_out_dim=self.cpu_out_dim,
            )
            return
        runner_handle.populate_slab_qkv(  # type: ignore[attr-defined]
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


class _NativeWeightSlabSpecInputSplitLinear(NativeWeightSlabSpec):
    def __init__(
        self,
        op_descriptor: tuple[int, int, str],
        *,
        n_threads: int,
        x_pinned_ptr: int,
        in_dim: int,
        x_col_offset: int,
        y_pinned_ptr: int,
        cpu_out_dim: int,
        w_cpu_ptr: int,
        w_cpu_rows: int,
        w_cpu_cols: int,
    ) -> None:
        super().__init__(op_descriptor)
        self.n_threads = n_threads
        self.x_pinned_ptr = x_pinned_ptr
        self.in_dim = in_dim
        self.x_col_offset = x_col_offset
        self.y_pinned_ptr = y_pinned_ptr
        self.cpu_out_dim = cpu_out_dim
        self.w_cpu_ptr = w_cpu_ptr
        self.w_cpu_rows = w_cpu_rows
        self.w_cpu_cols = w_cpu_cols

    def populate(self, runner_handle: object, task_id: int, *, dry_run: bool) -> None:
        bucket_capacity_tokens = int(self.op_descriptor[1])
        if dry_run:
            runner_handle.populate_slab_dryrun(  # type: ignore[attr-defined]
                task_id=task_id,
                bucket_capacity_tokens=bucket_capacity_tokens,
                x_pinned_ptr=self.x_pinned_ptr,
                in_dim=self.in_dim,
                y_pinned_ptr=self.y_pinned_ptr,
                cpu_out_dim=self.cpu_out_dim,
            )
            return
        runner_handle.populate_slab_wo_input(  # type: ignore[attr-defined]
            task_id=task_id,
            n_threads=self.n_threads,
            bucket_capacity_tokens=bucket_capacity_tokens,
            x_pinned_ptr=self.x_pinned_ptr,
            in_dim=self.in_dim,
            x_col_offset=self.x_col_offset,
            y_pinned_ptr=self.y_pinned_ptr,
            cpu_out_dim=self.cpu_out_dim,
            w_cpu_ptr=self.w_cpu_ptr,
            w_cpu_rows=self.w_cpu_rows,
            w_cpu_cols=self.w_cpu_cols,
        )


class _NativeWeightSlabSpecMlp(NativeWeightSlabSpec):
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

    def populate(self, runner_handle: object, task_id: int, *, dry_run: bool) -> None:
        bucket_capacity_tokens = int(self.op_descriptor[1])
        if dry_run:
            # Dry-run slabs still need pinned input/output metadata so submit
            # and sync resolve by task_id even though the worker skips the GEMM.
            runner_handle.populate_slab_dryrun(  # type: ignore[attr-defined]
                task_id=task_id,
                bucket_capacity_tokens=bucket_capacity_tokens,
                x_pinned_ptr=self.x_pinned_ptr,
                in_dim=self.in_dim,
                y_pinned_ptr=self.y_pinned_ptr,
                cpu_out_dim=self.cpu_out_dim,
            )
            return
        runner_handle.populate_slab_mlp(  # type: ignore[attr-defined]
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
        )


PyCotsWeightCallback = Callable[[torch.cuda.Event, torch.Tensor, torch.Tensor], None]


class PythonCotsWeightRunner:
    """Eager Python CPU task runner — the kill-switch path under
    `CotsOffloadConfig.cpu_runner = "python"`. Same execution semantics
    as the original prototype: single-worker `ThreadPoolExecutor` plus
    `future.result()`.

    The per-(layer, bucket, op_kind) weight views are bound at install
    time into closures stored in `_callbacks`; the operator just hands
    over an op_descriptor and the runner looks up the right closure.

    Eager-only: `ThreadPoolExecutor.submit` is not graph-capturable, so
    selecting this runner with `enforce_eager=False` is a hard error
    at engine launch.
    """

    kind = "python"

    def __init__(self, dry_run: bool = False) -> None:
        self._future: Future | None = None
        # Stash y_pinned at submit time so wait_and_uva matches the native
        # facade, where y_pinned is reached through the slab pointer.
        self._pending_y_pinned: torch.Tensor | None = None
        self._dry_run = dry_run
        self._callbacks: dict[tuple[int, int, str], PyCotsWeightCallback] = {}
        self._installed = False

    def install(
        self,
        callbacks: dict[tuple[int, int, str], PyCotsWeightCallback],
    ) -> None:
        """Register the per-(layer, bucket, op_kind) work closures.

        The offloader builds these at `post_init` from the dispatch table
        + handle weight views; one closure per slab-equivalent. Under
        `dry_run=True` the offloader installs noop closures (only the
        D2H event sync runs, no real CPU GEMM) — the runner's
        submit/wait shape is identical either way so timing-sensitive
        diagnostics can A/B by toggling `dry_run`.

        Operators are required to resolve `op_descriptor[1]` to a non-None int
        before calling `submit_with_d2h`, matching the native runner contract.
        """
        if self._installed:
            raise RuntimeError(
                "PythonCotsWeightRunner.install() called twice on the same instance"
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
        """Copy `x_gpu` into `x_pinned` and submit the CPU work closure.

        This runner is eager-only, so it can keep `x_pinned` and `y_pinned` as
        Python parameters. The native runner uses slab pointers to avoid tracing
        CPU views into captured graphs.
        """
        layer_idx, bucket, op_kind = op_descriptor
        assert isinstance(bucket, int), (
            "PythonCotsWeightRunner.submit_with_d2h: op_descriptor[1] must be "
            "a resolved int bucket."
        )
        x_pinned.copy_(x_gpu, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        if self._dry_run:
            cb: PyCotsWeightCallback = _cpu_dryrun_noop
        else:
            cb = self._callbacks[(layer_idx, bucket, op_kind)]
        # Stash y_pinned so wait_and_uva matches the native runner signature.
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
        for API symmetry with NativeCotsWeightRunner
        — they're the barrier-installing anchors that pin the
        `cots_sync_then_uva` custom op AFTER each independent GPU GEMM
        under graph capture. Under PythonCotsWeightRunner we're eager-only
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


class NativeCotsWeightRunner:
    """Production COTS weight runner. Wraps the C++ `CotsWeightTaskRunner` via the
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
    the underlying `CotsWeightTaskRunner` pybind handle is owned by the
    `cots_ops._COTS_WEIGHT_RUNNERS` strong-ref registry. The runner facade
    itself only holds the integer id + picklable state, so PyTorch's AOT compile
    guard cache can serialize it. Two offloaders coexist with independent slab
    pools. `close()` drains the worker then unregisters.

    Two row counts are first-class in native COTS:

      * `slab.num_tokens` — the **dispatched graph bucket** selected by
        vLLM's `BatchDescriptor.num_tokens`. It is pushed out of graph
        before the forward and resolved by the custom-op impl during
        capture/execution; it is not derived from `int(x.shape[0])`.
      * `live_num_tokens` — the **live rows to compute**, set OUT
        OF GRAPH by `GPUModelRunner._publish_forward_dispatch`
        BEFORE every scheduler, dummy/profile, warmup, and CUDA Graph
        capture forward. Always `live_num_tokens <= slab.num_tokens`. The
        worker's `at::linear` shapes, scratch slicing, and y_pinned
        write region all key off this.

    For B=1 decode under FULL capture: bucket might be 256 but
    runtime is 1. The worker computes 1 row of GEMM even though the
    captured graph fired at the bucket size. Captured D2H copies
    bucket-sized bytes (PCIe waste tracked under
    `worker_input_live_bytes` vs `d2h_replay_bucket_bytes`).
    """

    kind = "native"

    def __init__(self, dry_run: bool = False) -> None:
        # Lazy import: _cots_C is built only on CUDA. Users on CPU-only
        # / ROCm builds shouldn't hit ImportError just by importing this
        # module — the runner type is constructed only when the offload
        # config selects `cpu_runner='native'`. Any reference we hold
        # to the pybind handle is on the cots_ops registry, NOT on this
        # runner's `__dict__`: if the handle were stored on `self`, Dynamo's
        # guard serialization would try to pickle a `CotsWeightTaskRunner`,
        # which is unpicklable.
        try:
            from vllm import _cots_C
        except ImportError as e:
            raise RuntimeError(
                "NativeCotsWeightRunner requires the `vllm._cots_C` extension, "
                "which builds only on CUDA targets. Either select "
                "`cpu_runner='python'` or rebuild vLLM with CUDA support."
            ) from e
        from vllm.model_executor.offloader import cots_ops

        # Hand the freshly-constructed handle to the registry; the
        # registry's strong reference is now the SOLE owner. The local
        # variable goes out of scope at the end of __init__, so nothing
        # in `self.__dict__` references it.
        self._runner_id: int = cots_ops.register_weight_runner(
            _cots_C.CotsWeightTaskRunner()
        )
        self._dry_run: bool = bool(dry_run)
        # Format: {(layer_idx, bucket, op_kind): task_id}.
        self._task_id_for: dict[tuple[int, int, str], int] = {}
        self._installed: bool = False
        # Ownership flag for AOT guard-cache pickle round-trips. Only the
        # original constructor owns the registry entry; unpickled copies are
        # non-owning so their GC cannot unregister the live runner handle.
        self._owns_runner_registry_entry: bool = True

    def install(
        self,
        slab_specs: list[NativeWeightSlabSpec],
        max_num_tokens: int,
    ) -> None:
        """Allocate the C++ slab pool, populate slabs, and build the
        op_descriptor -> task_id map. Called once at offloader
        `post_init` after the dispatch table + handle weight views are
        known.

        `slab_specs` is a list (ordering = task_id) of NativeWeightSlabSpec
        records — one per (layer_idx, bucket, op_kind) where there is
        actual CPU work (n_cpu_compute > 0). Under `dry_run=True` the
        offloader passes specs with op_kind='dryrun_noop' so the
        runtime path exercises the full host-callback round-trip but
        skips real GEMM.

        Native operators do not pass descriptor buckets at runtime. They pass
        stable call-site identity, and cots_ops resolves the bucket/task from
        OOG dispatch state plus this install-time map.
        """
        from vllm.model_executor.offloader import cots_ops

        if self._installed:
            raise RuntimeError(
                "NativeCotsWeightRunner.install() called twice on the same instance"
            )
        n_slabs = len(slab_specs)
        cots_ops.install_weight_runner(
            self._runner_id,
            n_slabs=n_slabs,
            max_num_tokens=max_num_tokens,
        )
        for tid, spec in enumerate(slab_specs):
            self._task_id_for[spec.op_descriptor] = tid
            cots_ops.populate_slab_via_spec(
                self._runner_id, spec, tid, dry_run=self._dry_run
            )
        cots_ops.register_weight_task_id_map(self._runner_id, self._task_id_for)
        self._installed = True
        # Cache n_slabs so wait-kernel installation does not have to
        # re-walk task_id_for. Set last so any populate-time error
        # leaves the cache empty.
        self._n_slabs: int = n_slabs

    def install_wait_kernel_sync(self) -> None:
        """Install host-mapped pinned req/done slots for every slab.

        Must be called after `install()` and only when the offloader's feature
        flag `weight_capture_sync_mode="wait_kernel"`.

        After this returns, every slab's `wait_kernel_sync_installed=True`, the
        captured `dispatch_cb` writes `req_slot=seq` per dispatch,
        the worker writes `done_slot=seq` (in a finally-style
        block) per dispatch, and `cots_sync_then_uva` routes
        through the wait kernel via `sync_or_wait_on_stream`.
        """
        from vllm.model_executor.offloader import cots_ops

        if not self._installed:
            raise RuntimeError(
                "NativeCotsWeightRunner.install_wait_kernel_sync() called "
                "before install(); wait-kernel sync needs the slab pool to "
                "exist first"
            )
        cots_ops.install_wait_kernel_sync_for_all_tasks(
            self._runner_id,
            self._n_slabs,
        )

    def set_active_dispatch(self, bucket: int, live_num_tokens: int) -> None:
        """Publish OOG dispatch state for native custom-op resolution."""
        from vllm.model_executor.offloader import cots_ops

        cots_ops.set_active_weight_dispatch_state(
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
        """Submit native CPU GEMM work on the current CUDA stream.

        `x_pinned` and `y_pinned` are intentionally absent from the signature:
        slab pointers populated at install time are the source of truth for
        both directions. Bucket/task selection is also resolved from OOG
        dispatch state inside the custom-op impl.
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
        - `submit_anchor` is read-only — the same `x_gpu` that
          `cots_submit_gemm` mutated. Reading it pins sync AFTER submit.

        `y_pinned` is intentionally absent from the custom op and this facade;
        the impl reaches the worker's pinned output through the resolved slab.
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
        """Pickle hook used by PyTorch's AOT compile guard cache.

        The unpickled facade points at the same `_runner_id` as the original.
        Mark it non-owning so GC of a guard-cache copy cannot unregister the
        live runner.
        """
        state = self.__dict__.copy()
        state["_owns_runner_registry_entry"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Defensive: in case a state dict from a future pickle didn't
        # carry the flag, default to non-owning.
        self.__dict__.setdefault("_owns_runner_registry_entry", False)

    def close(self) -> None:
        """Drain any in-flight worker task and drop the registry entry.
        Idempotent; safe to call from teardown. Both the drain and the
        unregister go through cots_ops helpers so this runner facade
        never has to dereference the pybind handle directly.

        No-ops for non-owning copies (`_owns_runner_registry_entry` is
        False after `__setstate__`). The original constructor is the
        sole owner; only IT may drain or unregister.
        """
        if not getattr(self, "_owns_runner_registry_entry", False):
            return
        from vllm.model_executor.offloader import cots_ops

        try:
            # The CPU task queue can be idle while stream-ordered COTS custom
            # ops are still unwinding. Drain CUDA first so callbacks/UVA glue
            # cannot race with dropping the pybind runner handle from the registry.
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.current_stream().synchronize()
            cots_ops.sync_blocking(self._runner_id)
        finally:
            cots_ops.unregister_weight_runner(self._runner_id)
            self._owns_runner_registry_entry = False

    def __del__(self) -> None:
        # Best-effort registry cleanup if the user forgot to call close().
        # Note: don't raise from __del__ — the GC log is unhelpful.
        #
        # No-op for non-owning copies (e.g., AOT guard-cache unpickled
        # facades) — they share `_runner_id` with the original but must
        # NOT unregister it on GC. The original's `__del__` / `close()`
        # is the sole path that may drop the entry.
        # `try/except: pass` rather than `contextlib.suppress` because at
        # interpreter shutdown `contextlib` itself can be None;
        # try/except is the only finalizer-safe form. Lazy-import the
        # registry module too — interpreter shutdown can already have
        # cleared it.
        try:  # noqa: SIM105
            if getattr(self, "_owns_runner_registry_entry", False):
                from vllm.model_executor.offloader import cots_ops

                cots_ops.unregister_weight_runner(self._runner_id)
                self._owns_runner_registry_entry = False
        except Exception:  # noqa: BLE001
            pass


def _make_runner(
    config: CotsOffloadConfig,
) -> PythonCotsWeightRunner | NativeCotsWeightRunner:
    """Construct the offloader's single runner per `config.cpu_runner`."""
    cpu_runner = config.cpu_runner
    dry_run = bool(config.dry_run)
    if cpu_runner == "python":
        return PythonCotsWeightRunner(dry_run=dry_run)
    if cpu_runner == "native":
        return NativeCotsWeightRunner(dry_run=dry_run)
    raise ValueError(
        f"Unknown cpu_runner={cpu_runner!r}; expected 'native' or 'python'"
    )


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
# PythonCotsWeightRunner for execution.
# ---------------------------------------------------------------------------
