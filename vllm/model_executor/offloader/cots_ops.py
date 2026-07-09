# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for the COTS native weight task runner.

Mirrors `prefetch_ops.py`'s pattern: registers COTS ops via
`direct_register_custom_op` (which lands them under `torch.ops.vllm.*`).
The CPU submit/sync ops use `mutates_args` to declare barrier-installing
dependencies so torch.compile / CUDA graph capture preserve the overlap
ordering between submit, GPU GEMMs, sync, and UVA copy.

  * vllm.cots_submit_gemm(x_gpu, runner_id, layer_idx, op_kind_code)
      mutates_args=["x_gpu"]
      x_gpu mutated → pins submit BEFORE every GPU GEMM that reads
      x_gpu, AND provides the (submit → sync) edge consumed by
      cots_sync_then_uva's `submit_anchor` read. The C++ impl
      resolves the current dispatch bucket from OOG runner state,
      resolves the slab task_id from (layer_idx, bucket, op_kind),
      then bundles `cudaMemcpyAsync(D2H)` from x_gpu to
      slab.x_pinned_ptr with the host-callback enqueue.

  * vllm.cots_sync_then_uva(y_gpu, gpu_anchor_a, gpu_anchor_b,
                            submit_anchor, runner_id, layer_idx,
                            op_kind_code)
      mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]
      `gpu_anchor_a/_b` pin sync AFTER each independent GPU GEMM.
      `submit_anchor` (read-only, == x_gpu) pins sync AFTER submit.
      The impl reaches the slab's pinned output via
      `CotsWeightTaskRunner.y_pinned_view(task_id, bucket)`.

These ops accept ONLY CUDA tensors and scalar ids — no CPU tensor
arguments. Inductor's functionalization on captured graphs
materializes any CPU view it sees (in the worst case via a GPU
intermediate + blocking GPU→CPU copy that CUDA Graph capture
rejects with cudaErrorStreamCaptureUnsupported), so the design
keeps pinned-buffer addresses entirely on the C++ side.

The registry stores the `CotsWeightTaskRunner` pybind handle directly by
runner_id, not a `NativeCotsWeightRunner` instance. The compile-visible runner
is a thin facade with only pickleable state; the unpicklable C++ handle lives
here.
"""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.cots_diag import (
    COUNTERS_ENABLED as _COTS_COUNTERS_ENABLED,
)
from vllm.utils.cots_diag import (
    NVTX_ENABLED as _COTS_NVTX_ENABLED,
)
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    # Type-only import; avoids forcing _cots_C at module load on
    # non-CUDA builds.
    from vllm import _cots_C  # noqa: F401

# Module-private runner registry. Strong refs (NOT weak) — the registry
# IS the storage for the `CotsWeightTaskRunner` instance. NativeCotsWeightRunner's
# `__del__` / `close()` is the only thing that removes entries; if a
# runner is GC'd without close() the __del__ unregisters there.
_COTS_WEIGHT_RUNNERS: dict[int, Any] = {}
_COTS_WEIGHT_ACTIVE_DISPATCH: dict[int, tuple[int, int]] = {}
_COTS_WEIGHT_TASK_ID_FOR: dict[int, dict[tuple[int, int, str], int]] = {}
_COTS_WEIGHT_ROUTE_FOR: dict[int, dict[tuple[int, str], Any]] = {}
_NEXT_WEIGHT_RUNNER_ID = itertools.count(1)
_COTS_PY_COUNTERS: dict[str, int] = {}
_COTS_PY_COUNTER_LOCK = threading.Lock()

_OP_KIND_TO_CODE: dict[str, int] = {
    "qkv": 1,
    "mlp_block": 2,
    "wo": 3,
}
_OP_KIND_BY_CODE: dict[int, str] = {v: k for k, v in _OP_KIND_TO_CODE.items()}


def op_kind_code(op_kind: str) -> int:
    """Encode a stable op kind for the native custom-op boundary."""
    try:
        return _OP_KIND_TO_CODE[op_kind]
    except KeyError as e:
        raise ValueError(f"unknown COTS op_kind {op_kind!r}") from e


def register_weight_runner(runner: Any) -> int:
    """Register a `CotsWeightTaskRunner` instance and return a fresh runner_id.
    The registry takes ownership of the strong reference; the caller
    should retain only the runner_id."""
    rid = next(_NEXT_WEIGHT_RUNNER_ID)
    _COTS_WEIGHT_RUNNERS[rid] = runner
    return rid


def unregister_weight_runner(runner_id: int) -> None:
    """Drop the registry entry for a runner. Idempotent."""
    _COTS_WEIGHT_RUNNERS.pop(runner_id, None)
    _COTS_WEIGHT_ACTIVE_DISPATCH.pop(runner_id, None)
    _COTS_WEIGHT_TASK_ID_FOR.pop(runner_id, None)
    _COTS_WEIGHT_ROUTE_FOR.pop(runner_id, None)


def register_weight_task_id_map(
    runner_id: int,
    task_id_for: dict[tuple[int, int, str], int],
) -> None:
    """Publish the install-time slab map used by custom op dispatch."""
    _COTS_WEIGHT_TASK_ID_FOR[int(runner_id)] = dict(task_id_for)


def register_weight_route(
    runner_id: int,
    *,
    layer_idx: int,
    op_kind: str,
    route: Any,
) -> None:
    """Publish the existing COTS handle/fused-op route for custom ops.

    The route object is intentionally the existing Python object owned by the
    offloader. It keeps per-bucket fields such as n_prefetch_by_bucket,
    n_cpu_compute_by_bucket, and scatter indices behind an opaque operator
    boundary without duplicating the routing table in C++.
    """

    _COTS_WEIGHT_ROUTE_FOR.setdefault(int(runner_id), {})[
        (int(layer_idx), str(op_kind))
    ] = route


def set_active_weight_dispatch_state(
    runner_id: int,
    *,
    bucket: int,
    live_num_tokens: int,
) -> None:
    """Publish the authoritative per-forward COTS dispatch state.

    Called outside the compiled forward by `CotsOffloader.on_dispatch`.
    Native custom ops read this state at eager execution / CUDA Graph
    capture time and resolve slab task_ids from it. This keeps bucket
    and task selection out of compile-visible scalar arguments.
    """
    bucket = int(bucket)
    live_num_tokens = int(live_num_tokens)
    if bucket <= 0:
        raise ValueError(f"active COTS dispatch bucket must be > 0, got {bucket}")
    if live_num_tokens < 0:
        raise ValueError(
            f"active COTS dispatch live_num_tokens must be >= 0, got {live_num_tokens}"
        )
    _COTS_WEIGHT_ACTIVE_DISPATCH[int(runner_id)] = (bucket, live_num_tokens)


def _active_dispatch_state(runner_id: int, op_name: str) -> tuple[int, int]:
    runner_id = int(runner_id)
    state = _COTS_WEIGHT_ACTIVE_DISPATCH.get(runner_id)
    if state is None:
        raise RuntimeError(
            f"{op_name}: no active COTS dispatch state for runner_id={runner_id}. "
            "CotsOffloader.on_dispatch must publish the BatchDescriptor bucket "
            "before native custom ops execute or capture."
        )
    bucket, live_num_tokens = state
    return int(bucket), int(live_num_tokens)


def _op_kind_from_code(op_kind_code: int, op_name: str, runner_id: int) -> str:
    try:
        return _OP_KIND_BY_CODE[int(op_kind_code)]
    except KeyError as e:
        raise RuntimeError(
            f"{op_name}: unknown op_kind_code={op_kind_code} for runner_id={runner_id}"
        ) from e


def _lookup_route(
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    op_name: str,
) -> tuple[Any | None, str]:
    runner_id = int(runner_id)
    op_kind = _op_kind_from_code(op_kind_code, op_name, runner_id)
    route_for = _COTS_WEIGHT_ROUTE_FOR.get(runner_id)
    if route_for is None:
        return None, op_kind
    return route_for.get((int(layer_idx), op_kind)), op_kind


def _route_cpu_rows(route: Any, op_kind: str, bucket: int) -> int:
    if op_kind == "mlp_block":
        down = route._down
        return int(down.n_cpu_compute_by_bucket.get(bucket, down.n_cpu))
    return int(route.n_cpu_compute_by_bucket.get(bucket, route.n_cpu))


def _route_has_zero_cpu_work(
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    op_name: str,
) -> bool:
    route, op_kind = _lookup_route(runner_id, layer_idx, op_kind_code, op_name)
    if route is None:
        # Older direct tests may install only task ids. Preserve that path.
        return False
    bucket, _ = _active_dispatch_state(runner_id, op_name)
    return _route_cpu_rows(route, op_kind, bucket) <= 0


def _resolve_task_for_dispatch(
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    op_name: str,
) -> tuple[int, int, int]:
    """Resolve active dispatch state to a concrete C++ slab task."""
    runner_id = int(runner_id)
    bucket, live_num_tokens = _active_dispatch_state(runner_id, op_name)
    op_kind = _op_kind_from_code(op_kind_code, op_name, runner_id)
    task_id_for = _COTS_WEIGHT_TASK_ID_FOR.get(runner_id)
    if task_id_for is None:
        raise RuntimeError(
            f"{op_name}: runner_id={runner_id} has no task_id map; "
            "NativeCotsWeightRunner.install() must complete before dispatch."
        )
    key = (int(layer_idx), int(bucket), op_kind)
    task_id = task_id_for.get(key)
    if task_id is None:
        available = sorted(
            b
            for layer, b, kind in task_id_for
            if layer == int(layer_idx) and kind == op_kind
        )
        raise RuntimeError(
            f"{op_name}: no native COTS slab for key={key}; available buckets "
            f"for layer={int(layer_idx)}, op_kind={op_kind!r}: {available}. "
            "The resolved task bucket must come from the OOG dispatch state."
        )
    return int(task_id), int(bucket), int(live_num_tokens)


def _bounded_transfer_rows(bucket: int, tensor_rows: int, op_name: str) -> int:
    """Rows to copy for the active slab without reading past the tensor.

    The active dispatch bucket selects the native slab/task. In FULL
    graph replay, the CUDA tensor may be a larger persistent buffer, so
    the bucket limits transfer work. In eager/profile dummy paths, the
    slab can be the max-capacity fallback while the tensor is smaller,
    so the concrete tensor rows limit transfer work.
    """
    rows = min(int(bucket), int(tensor_rows))
    if rows <= 0:
        raise AssertionError(
            f"{op_name}: transfer row count must be > 0, got "
            f"bucket={bucket}, tensor_rows={tensor_rows}"
        )
    return rows


def lookup_weight_runner(runner_id: int, op_name: str) -> Any:
    """Resolve runner_id → `CotsWeightTaskRunner` instance. Raises a clear
    error if the runner was already torn down."""
    runner = _COTS_WEIGHT_RUNNERS.get(runner_id)
    if runner is None:
        raise RuntimeError(
            f"{op_name}: runner_id={runner_id} not in registry "
            f"(known ids: {list(_COTS_WEIGHT_RUNNERS.keys())}). The owning "
            f"NativeCotsWeightRunner was likely torn down before its "
            f"in-flight ops drained."
        )
    return runner


# Offloader-side install/teardown helpers. These all run OUTSIDE the
# compiled forward path, so they can dereference the pybind handle
# freely. They exist so the runner facade itself never has to hold
# the handle on its `__dict__`.


def install_weight_runner(
    runner_id: int,
    n_slabs: int,
    max_num_tokens: int,
) -> None:
    """Allocate the C++ slab pool. Called once at offloader post_init."""
    runner = lookup_weight_runner(runner_id, "install_weight_runner")
    runner.install(
        n_slabs=int(n_slabs),
        max_num_tokens=int(max_num_tokens),
    )


def populate_slab_via_spec(
    runner_id: int,
    spec: Any,
    task_id: int,
    *,
    dry_run: bool,
) -> None:
    """Populate slot `task_id` via the spec's `populate(runner, ...)`
    method. The spec carries the per-op pointer + stride layout (QKV
    vs MLP vs dryrun); the helper just hands it the resolved runner handle."""
    runner = lookup_weight_runner(runner_id, "populate_slab_via_spec")
    spec.populate(runner, task_id, dry_run=dry_run)


def install_wait_kernel_sync_for_all_tasks(
    runner_id: int,
    n_slabs: int,
) -> None:
    """Install wait-kernel sync slots for every slab in the pool.

    This is a low-level diagnostic hook. The production offloader uses
    host-callback sync and does not install wait-kernel slots.
    """
    runner = lookup_weight_runner(runner_id, "install_wait_kernel_sync_for_all_tasks")
    for tid in range(int(n_slabs)):
        runner.install_wait_kernel_sync_for_task(int(tid))


def set_worker_affinity(runner_id: int, mask: int) -> None:
    """Pin the worker thread to a CPU set (uint64 bitmask). One-shot
    call from `CotsOffloader.post_init` after install."""
    runner = lookup_weight_runner(runner_id, "set_worker_affinity")
    runner.set_worker_affinity(int(mask))


def reset_all_counters() -> None:
    """Zero every registered CotsWeightTaskRunner counter."""
    import contextlib

    with _COTS_PY_COUNTER_LOCK:
        _COTS_PY_COUNTERS.clear()
    for runner in _COTS_WEIGHT_RUNNERS.values():
        # Best-effort — a stale runner shouldn't break the reset
        # for the rest.
        with contextlib.suppress(Exception):
            runner.reset_counters()


def get_all_counters() -> dict[int, dict[str, int]]:
    """Return diagnostic counters for every registered weight runner."""
    out: dict[int, dict[str, int]] = {}
    with _COTS_PY_COUNTER_LOCK:
        if _COTS_PY_COUNTERS:
            out[-1] = dict(_COTS_PY_COUNTERS)
    for runner_id, runner in _COTS_WEIGHT_RUNNERS.items():
        counters = runner.get_counters()
        out[int(runner_id)] = {str(k): int(v) for k, v in counters.items()}
    return out


def add_python_counter(name: str, value: int = 1) -> None:
    """Add an env-gated Python-side diagnostic counter."""

    if not _COTS_COUNTERS_ENABLED:
        return
    with _COTS_PY_COUNTER_LOCK:
        _COTS_PY_COUNTERS[name] = _COTS_PY_COUNTERS.get(name, 0) + int(value)


def add_python_timing(name: str, elapsed_ns: int, *, count: int = 1) -> None:
    """Record a Python-side timing sample in nanoseconds."""

    if not _COTS_COUNTERS_ENABLED:
        return
    add_python_counter(f"{name}_total_ns", int(elapsed_ns))
    add_python_counter(f"{name}_count", int(count))


def set_live_num_tokens(runner_id: int, n: int) -> None:
    """Publish the live-token row cap to the native COTS worker.

    `n` is the number of semantically live rows in the active bucket.
    Slab capacity, graph capture, and buffer sizing stay bucket-based;
    the worker reads this value on the next host-callback fire and
    avoids CPU GEMM work for padded rows.
    """
    runner = _COTS_WEIGHT_RUNNERS.get(runner_id)
    if runner is None:
        # Best-effort: a stale runner_id call here shouldn't crash —
        # just skip. The next custom-op call will surface the missing
        # registry entry with a clearer error.
        return
    runner.set_live_num_tokens(int(n))


def sync_blocking(runner_id: int) -> None:
    """Drain any in-flight worker task synchronously. Called from
    `NativeCotsWeightRunner.close()`."""
    runner = _COTS_WEIGHT_RUNNERS.get(runner_id)
    if runner is None:
        # Already torn down — nothing to drain.
        return
    runner.sync_blocking()


# --- vllm.cots_submit_gemm -------------------------------------------------


def _cots_submit_gemm_impl(
    x_gpu: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
) -> None:
    """Real impl: dispatched to the per-runner pybind handle.

    Bundles the x_gpu -> slab.x_pinned_ptr D2H copy with the host-callback
    enqueue, all on the current CUDA stream. CPU pinned buffers are not Python
    arguments because Inductor materializes CPU tensors visible in captured
    graphs. Both custom ops are CUDA tensors plus scalar ids only.

    Moving the D2H from Python `copy_(non_blocking=True)` to a raw
    `cudaMemcpyAsync` loses Python's automatic shape/stride/dtype
    handling, so we validate `x_gpu` here BEFORE handing the raw
    pointer to C++. The slab task is selected by the OOG dispatch
    bucket. The transfer row count is bounded by the concrete tensor
    rows as well, because eager/profile dummy runs can route through a
    max-capacity slab while executing a smaller tensor.
    """
    if not x_gpu.is_cuda:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu must be on CUDA, got {x_gpu.device}"
        )
    if x_gpu.dtype != torch.bfloat16:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu must be bfloat16 (matches slab dtype), "
            f"got {x_gpu.dtype}"
        )
    if x_gpu.dim() != 2:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu must be 2D (num_tokens, in_dim); "
            f"got shape {tuple(x_gpu.shape)}"
        )
    if _route_has_zero_cpu_work(runner_id, layer_idx, op_kind_code, "cots_submit_gemm"):
        return
    task_id, bucket, _live_num_tokens = _resolve_task_for_dispatch(
        runner_id, layer_idx, op_kind_code, "cots_submit_gemm"
    )
    num_transfer_rows = _bounded_transfer_rows(
        bucket, int(x_gpu.shape[0]), "cots_submit_gemm"
    )
    # `stride(1) == 1` is the production contract — feature-dim
    # contiguous, possibly row-strided. The C++ side handles the
    # row-strided case via cudaMemcpy2DAsync; transposed/exotic
    # layouts (stride(1) != 1) would need a separate copy plan and
    # are rejected. Real Qwen2-style hidden_states tensors satisfy
    # this even when they come from padded / sliced bases.
    if x_gpu.stride(1) != 1:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu.stride(1)={x_gpu.stride(1)} (must be 1; "
            f"no transposed-stride layouts in production decode). For "
            f"row-strided inputs (stride(0) > shape[1]) the C++ D2H uses "
            f"cudaMemcpy2DAsync."
        )
    runner = lookup_weight_runner(runner_id, "cots_submit_gemm")
    stream = torch.cuda.current_stream().cuda_stream
    # Diagnostic-only NVTX scope for attributing the Python-side dispatch
    # boundary separately from the C++ submit body.
    if _COTS_NVTX_ENABLED:
        torch.cuda.nvtx.range_push("cots:py_submit_gemm")
    try:
        # Pass shape/stride so the C++ D2H can dispatch the right
        # cudaMemcpy* variant — see CotsWeightTaskRunner::submit_on_stream for
        # the 1D-vs-2D branch.
        runner.submit_on_stream(
            task_id,
            num_transfer_rows,
            x_gpu.data_ptr(),
            x_gpu.shape[1],
            x_gpu.stride(0),
            x_gpu.stride(1),
            stream,
        )
    finally:
        if _COTS_NVTX_ENABLED:
            torch.cuda.nvtx.range_pop()


def _cots_submit_gemm_fake(
    x_gpu: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
) -> None:
    """torch.compile tracing: no side effects, just a barrier."""
    return


# --- vllm.cots_sync_then_uva -----------------------------------------------


def _cots_sync_then_uva_impl(
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    submit_anchor: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
) -> None:
    """Real impl: schedule the sync host callback then run the UVA copy.

    `gpu_anchor_a` / `gpu_anchor_b` are CUDA tensors that the GPU
    GEMMs produced; mutating them pins this op AFTER both independent
    GEMMs (out_perm, out_pref). `submit_anchor` is `x_gpu` from the
    matching `cots_submit_gemm`; reading it pins this op AFTER submit.

    `y_pinned` is intentionally not a parameter. The slab pointer the worker
    wrote to is the source of truth; we reach it through the C++-side
    `y_pinned_view(task_id, num_transfer_rows)` helper. The trust boundary is
    install-time: the
    slab pointer came from `_y_pinned`, allocated `pin_memory=True` and
    validated there.
    """
    if _route_has_zero_cpu_work(
        runner_id, layer_idx, op_kind_code, "cots_sync_then_uva"
    ):
        return
    task_id, bucket, _live_num_tokens = _resolve_task_for_dispatch(
        runner_id, layer_idx, op_kind_code, "cots_sync_then_uva"
    )
    num_transfer_rows = _bounded_transfer_rows(
        bucket, int(submit_anchor.shape[0]), "cots_sync_then_uva"
    )
    runner = lookup_weight_runner(runner_id, "cots_sync_then_uva")
    stream = torch.cuda.current_stream().cuda_stream
    if _COTS_NVTX_ENABLED:
        torch.cuda.nvtx.range_push("cots:py_sync_then_uva")
    try:
        # C++ side branches per-slab on `wait_kernel_sync_installed`.
        # Production slabs use the host-callback SyncCallback node. Low-level
        # diagnostic tests can install wait-kernel slots so this call waits on
        # the worker-published `done_slot=seq` from a tiny GPU kernel.
        runner.sync_or_wait_on_stream(task_id, stream)
        # Build the CPU view over the slab pointer locally — never escapes
        # back to Python in a way Inductor would see.
        y_pinned = runner.y_pinned_view(task_id, num_transfer_rows)
        runner.note_uva_request(num_transfer_rows, y_pinned.shape[1])
        # Lazy import to avoid a top-level circular import (cots.py imports
        # this module via cots_ops and we'd loop on `from .cots import ...`).
        from vllm.model_executor.offloader.cots import (
            _uva_copy_trusted_host_into_gpu,
        )

        if _COTS_NVTX_ENABLED:
            torch.cuda.nvtx.range_push("cots:py_uva_copy")
        try:
            if y_gpu.dim() == 1:
                dst_gpu = y_gpu.narrow(0, 0, y_pinned.numel())
                src_pinned = y_pinned.reshape(-1)
            else:
                dst_gpu = y_gpu
                src_pinned = y_pinned
            _uva_copy_trusted_host_into_gpu(src_pinned, dst_gpu)
        finally:
            if _COTS_NVTX_ENABLED:
                torch.cuda.nvtx.range_pop()
    finally:
        if _COTS_NVTX_ENABLED:
            torch.cuda.nvtx.range_pop()


def _cots_sync_then_uva_fake(
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    submit_anchor: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
) -> None:
    return


def _require_route(
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    op_name: str,
) -> tuple[Any, str, int]:
    route, op_kind = _lookup_route(runner_id, layer_idx, op_kind_code, op_name)
    if route is None:
        raise RuntimeError(
            f"{op_name}: runner_id={int(runner_id)} has no registered route for "
            f"layer_idx={int(layer_idx)}, op_kind={op_kind!r}"
        )
    bucket, _ = _active_dispatch_state(runner_id, op_name)
    return route, op_kind, bucket


def _check_prefetch_slot_ready(handle: Any, required_rows: int, op_name: str) -> None:
    required_rows = int(required_rows)
    if required_rows <= 0:
        return
    if not handle.w_prefetch_slots:
        raise RuntimeError(
            f"{op_name}: active bucket needs {required_rows} prefetched rows for "
            f"{handle.qualified_name}, but no prefetch slots are installed"
        )
    owner = handle.prefetch_owner_in_slot[handle.slot_idx]
    have = int(handle.prefetch_available_rows_in_slot[handle.slot_idx])
    if owner is not handle:
        owner_desc = (
            "None"
            if owner is None
            else (
                f"{getattr(owner, 'qualified_name', '<unknown>')} "
                f"layer={getattr(owner, 'layer_idx', '?')} "
                f"role={getattr(owner, 'role', '?')}"
            )
        )
        raise RuntimeError(
            f"{op_name}: slot owner mismatch on {handle.qualified_name} "
            f"layer={getattr(handle, 'layer_idx', '?')} "
            f"role={getattr(handle, 'role', '?')} slot {handle.slot_idx} "
            f"(owner={owner_desc})"
        )
    if have < required_rows:
        raise RuntimeError(
            f"{op_name}: prefetch slot underfilled on {handle.qualified_name}: "
            f"have {have}, need {required_rows}"
        )


def _mark_prefetch_slot_consumed(handle: Any, required_rows: int) -> None:
    if int(required_rows) > 0 and handle.prefetch_consumed_in_slot:
        handle.prefetch_consumed_in_slot[handle.slot_idx] = True


def _cots_prefetch_linear_impl(
    x_gpu: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    max_n_prefetch: int,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return x_gpu.new_empty((x_gpu.shape[0], int(max_n_prefetch)))
    handle, op_kind, bucket = _require_route(
        runner_id, layer_idx, op_kind_code, "cots_prefetch_linear"
    )
    if op_kind == "mlp_block":
        raise RuntimeError("cots_prefetch_linear does not handle mlp_block routes")
    n_pref = int(handle.n_prefetch_by_bucket.get(bucket, 0))
    max_n_prefetch = int(max_n_prefetch)
    out = x_gpu.new_empty((x_gpu.shape[0], max_n_prefetch))
    if n_pref <= 0:
        return out
    _check_prefetch_slot_ready(handle, n_pref, "cots_prefetch_linear")
    slot_view = handle.w_prefetch_slots[handle.slot_idx].narrow(0, 0, n_pref)
    active = torch.nn.functional.linear(x_gpu, slot_view, None)
    _mark_prefetch_slot_consumed(handle, n_pref)
    if n_pref == max_n_prefetch:
        return active
    out.narrow(1, 0, n_pref).copy_(active)
    return out


def _cots_prefetch_linear_fake(
    x_gpu: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    max_n_prefetch: int,
    enabled: bool,
) -> torch.Tensor:
    del runner_id, layer_idx, op_kind_code, enabled
    return x_gpu.new_empty((x_gpu.shape[0], int(max_n_prefetch)))


def _cots_scatter_col_outputs_impl(
    out_perm: torch.Tensor,
    out_pref: torch.Tensor,
    out_cpu_flat: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    out_dim: int,
    enable_prefetch: bool,
    enable_cpu: bool,
) -> torch.Tensor:
    route, op_kind = _lookup_route(
        runner_id, layer_idx, op_kind_code, "cots_scatter_col_outputs"
    )
    if route is None:
        raise RuntimeError(
            f"cots_scatter_col_outputs: runner_id={int(runner_id)} has no "
            f"registered route for layer_idx={int(layer_idx)}, op_kind={op_kind!r}"
        )
    handle = route
    if op_kind == "mlp_block":
        raise RuntimeError("cots_scatter_col_outputs does not handle mlp_block routes")
    bucket = None
    if enable_prefetch or enable_cpu:
        bucket, _ = _active_dispatch_state(runner_id, "cots_scatter_col_outputs")
    assert handle.gpu_indices_cuda is not None
    out = torch.empty(
        (out_perm.shape[0], int(out_dim)),
        dtype=out_perm.dtype,
        device=out_perm.device,
    )
    if out_perm.shape[1] > 0:
        out.index_copy_(1, handle.gpu_indices_cuda, out_perm)
    if enable_prefetch:
        assert bucket is not None
        n_pref = int(handle.n_prefetch_by_bucket.get(bucket, 0))
        if n_pref > 0:
            pref_idx = handle.prefetch_indices_cuda_by_bucket[bucket]
            pref_src = out_pref.narrow(1, 0, n_pref)
            out.index_copy_(1, pref_idx, pref_src)
    if enable_cpu:
        assert bucket is not None
        n_cpu = int(handle.n_cpu_compute_by_bucket.get(bucket, handle.n_cpu))
        if n_cpu > 0:
            cpu_idx = handle.cpu_compute_indices_cuda_by_bucket[bucket]
            rows = int(out_perm.shape[0])
            cpu_src = out_cpu_flat.narrow(0, 0, rows * n_cpu).view(rows, n_cpu)
            out.index_copy_(1, cpu_idx, cpu_src)
    return out


def _cots_scatter_col_outputs_fake(
    out_perm: torch.Tensor,
    out_pref: torch.Tensor,
    out_cpu_flat: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    out_dim: int,
    enable_prefetch: bool,
    enable_cpu: bool,
) -> torch.Tensor:
    del out_pref, out_cpu_flat, runner_id, layer_idx, op_kind_code
    del enable_prefetch, enable_cpu
    return out_perm.new_empty((out_perm.shape[0], int(out_dim)))


def _cots_mlp_prefetch_add_impl(
    x_gpu: torch.Tensor,
    out_base: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    has_base_gpu: bool,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return out_base
    fused_op, op_kind, bucket = _require_route(
        runner_id, layer_idx, op_kind_code, "cots_mlp_prefetch_add"
    )
    if op_kind != "mlp_block":
        raise RuntimeError("cots_mlp_prefetch_add requires an mlp_block route")
    gu_h = fused_op._gate_up
    dn_h = fused_op._down
    gu_n_pref = int(gu_h.n_prefetch_by_bucket.get(bucket, 0))
    dn_n_pref = int(dn_h.n_prefetch_by_bucket.get(bucket, 0))
    if gu_n_pref <= 0 or dn_n_pref <= 0:
        return out_base
    _check_prefetch_slot_ready(gu_h, gu_n_pref // 2, "cots_mlp_prefetch_add")
    _check_prefetch_slot_ready(dn_h, dn_n_pref, "cots_mlp_prefetch_add")
    gu_slot = gu_h.w_prefetch_slots[gu_h.slot_idx]
    dn_slot = dn_h.w_prefetch_slots[dn_h.slot_idx]
    pref_mlp1 = torch.nn.functional.linear(x_gpu, gu_slot.narrow(0, 0, gu_n_pref), None)
    pref_silu = fused_op._act_fn(pref_mlp1)
    partial = pref_silu.matmul(dn_slot.narrow(0, 0, dn_n_pref))
    _mark_prefetch_slot_consumed(gu_h, gu_n_pref // 2)
    _mark_prefetch_slot_consumed(dn_h, dn_n_pref)
    if has_base_gpu:
        return out_base + partial
    return partial


def _cots_mlp_prefetch_add_fake(
    x_gpu: torch.Tensor,
    out_base: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    has_base_gpu: bool,
    enabled: bool,
) -> torch.Tensor:
    del x_gpu, runner_id, layer_idx, op_kind_code, has_base_gpu, enabled
    return torch.empty_like(out_base)


def _cots_mlp_merge_cpu_impl(
    out_gpu: torch.Tensor,
    out_cpu_flat: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    has_base_gpu: bool,
    enable_prefetch: bool,
    enable_cpu: bool,
) -> torch.Tensor:
    if not enable_cpu:
        return out_gpu
    fused_op, op_kind, bucket = _require_route(
        runner_id, layer_idx, op_kind_code, "cots_mlp_merge_cpu"
    )
    if op_kind != "mlp_block":
        raise RuntimeError("cots_mlp_merge_cpu requires an mlp_block route")
    dn_h = fused_op._down
    n_cpu = int(dn_h.n_cpu_compute_by_bucket.get(bucket, dn_h.n_cpu))
    if n_cpu <= 0:
        return out_gpu
    rows = int(out_gpu.shape[0])
    out_dim = int(fused_op._out_dim)
    cpu_src = out_cpu_flat.narrow(0, 0, rows * out_dim).view(rows, out_dim)
    gu_n_pref = int(fused_op._gate_up.n_prefetch_by_bucket.get(bucket, 0))
    has_prefetch_gpu = bool(enable_prefetch and gu_n_pref > 0)
    if has_base_gpu or has_prefetch_gpu:
        return out_gpu + cpu_src
    return cpu_src.clone()


def _cots_mlp_merge_cpu_fake(
    out_gpu: torch.Tensor,
    out_cpu_flat: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    has_base_gpu: bool,
    enable_prefetch: bool,
    enable_cpu: bool,
) -> torch.Tensor:
    del out_cpu_flat, runner_id, layer_idx, op_kind_code
    del has_base_gpu, enable_prefetch, enable_cpu
    return torch.empty_like(out_gpu)


# --- registration ----------------------------------------------------------


def register_cots_offloader_ops() -> None:
    """Register the COTS custom ops. Idempotent at import time."""
    direct_register_custom_op(
        op_name="cots_submit_gemm",
        op_func=_cots_submit_gemm_impl,
        # x_gpu is the CUDA dispatch anchor and ordering pin: mutating it forces
        # every subsequent GPU GEMM that reads x_gpu (F.linear
        # permanent / prefetched) to be ordered after submit, AND
        # `cots_sync_then_uva` reads x_gpu as `submit_anchor` to
        # stay ordered after submit. NEITHER `x_pinned` NOR
        # `y_pinned` appears in the op signature; both pinned
        # buffers are reached via slab pointers in C++.
        mutates_args=["x_gpu"],
        fake_impl=_cots_submit_gemm_fake,
    )
    direct_register_custom_op(
        op_name="cots_sync_then_uva",
        op_func=_cots_sync_then_uva_impl,
        # y_gpu is the CUDA dispatch anchor + downstream-scatter dep.
        # gpu_anchor_a / gpu_anchor_b pin sync AFTER each independent
        # GPU GEMM (in QKV, out_perm and out_pref are independent
        # F.linears; one anchor only pins one of them). The runner
        # passes two DISTINCT dummy CUDA tensors when an anchor is
        # absent — never alias.
        mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"],
        fake_impl=_cots_sync_then_uva_fake,
    )
    direct_register_custom_op(
        op_name="cots_prefetch_linear",
        op_func=_cots_prefetch_linear_impl,
        fake_impl=_cots_prefetch_linear_fake,
    )
    direct_register_custom_op(
        op_name="cots_scatter_col_outputs",
        op_func=_cots_scatter_col_outputs_impl,
        fake_impl=_cots_scatter_col_outputs_fake,
    )
    direct_register_custom_op(
        op_name="cots_mlp_prefetch_add",
        op_func=_cots_mlp_prefetch_add_impl,
        fake_impl=_cots_mlp_prefetch_add_fake,
    )
    direct_register_custom_op(
        op_name="cots_mlp_merge_cpu",
        op_func=_cots_mlp_merge_cpu_impl,
        fake_impl=_cots_mlp_merge_cpu_fake,
    )


# Register at module import time so `torch.ops.vllm.cots_*` exist as
# soon as cots.py imports this module.
register_cots_offloader_ops()
