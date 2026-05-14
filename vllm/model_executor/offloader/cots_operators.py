# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QKV, MLP, scatter, and UVA-facing operators for COTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.offloader.cots_runners import (
    NativeCotsRunner,
    PythonCotsRunner,
)
from vllm.model_executor.offloader.cots_storage import CotsLinearHandle

if TYPE_CHECKING:
    from vllm.model_executor.offloader.cots_offloader import CotsOffloader


def _assert_prefetch_slot_ready(
    h: CotsLinearHandle,
    required_rows: int,
    *,
    underfilled_name: str,
) -> None:
    """Validate eager/runtime prefetch slot metadata.

    During Dynamo tracing, ``start_prefetch``/``wait_prefetch`` use their
    fake implementations and therefore do not update Python owner metadata.
    The real custom ops still run when the compiled/captured graph executes,
    so trace-time metadata assertions would reject a valid graph before the
    actual prefetch path has a chance to run.
    """
    if torch.compiler.is_compiling():
        return
    assert h.prefetch_owner_in_slot[h.slot_idx] is h, (
        f"slot owner mismatch on {h.qualified_name} slot {h.slot_idx}"
    )
    assert h.prefetch_available_rows_in_slot[h.slot_idx] >= required_rows, (
        f"{underfilled_name} underfilled on {h.qualified_name}: have "
        f"{h.prefetch_available_rows_in_slot[h.slot_idx]}, need {required_rows}"
    )


class CotsQKVOp:
    """Patched `quant_method.apply` for QKVParallelLinear.

    GPU computes its slice; CPU computes its slice via the runner; outputs
    are scattered through `cpu_indices_cuda` / `gpu_indices_cuda` to restore
    the canonical `[Q | K | V]` column ordering.
    """

    def __init__(
        self,
        handle: CotsLinearHandle,
        runner: PythonCotsRunner | NativeCotsRunner,
        offloader: CotsOffloader,
        original_quant_method,
    ):
        assert handle.kind == "qkv"
        self._handle = handle
        self._runner = runner
        self._offloader = offloader
        self._original = original_quant_method

    def __getattr__(self, name):
        return getattr(self._original, name)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        offloader = self._offloader
        dry_run = offloader.dry_run

        h = self._handle
        assert h.w_cpu is not None
        # NOTE: under torch.compile + BACKED dynamic shapes (vLLM's
        # FULL_AND_PIECEWISE mode), `x.shape[0]` resolves at trace
        # time to the SymInt hint = max_num_batched_tokens — NOT the
        # live token count. So `num_tokens` here is really the
        # activation buffer's row capacity (max-sized), not the
        # number of live tokens. Used only for buffer-view shape
        # alignment with x (y_dst / x_in / y_out / scatter `out` all
        # carry the same row dim as x for index_copy_ consistency).
        # Python-side routing geometry uses `b`; native slab/task
        # selection uses the active dispatch state published OOG.
        # Worker row count uses the live-token cap inside the active
        # dispatch bucket.
        num_tokens = x.shape[0]
        # Phase 1c Stage 3: bucket lives on the offloader (set
        # unconditionally by `on_dispatch`), NOT on the streamer. The
        # operator must resolve `b` BEFORE reading per-bucket dicts so
        # that y_pinned's shape (sized to `n_cpu_compute_by_bucket[b]`)
        # agrees with the runner's install closure / slab (also keyed
        # on `b`). If prefetch is inactive (`max_n_prefetch == 0`)
        # every bucket has n_pref=0 and n_cpu_compute=h.n_cpu, so we
        # can use h.n_cpu directly. Python-runner tests that bypass
        # on_dispatch may still fall back to `_bucket_for(x_rows)`;
        # native forwards require the explicit OOG dispatch boundary.
        # §1c.19: resolve bucket to a non-None int up-front. The runner
        # facade does NOT carry a `bucket_for_fallback` callable anymore
        # (that bound method dragged the offloader into Dynamo's guard
        # graph); operators are responsible for handing the runner a
        # fully-resolved descriptor.
        b = offloader._operator_bucket(num_tokens)
        if h.max_n_prefetch == 0:
            n_pref = 0
            n_cpu = h.n_cpu
            cpu_idx = h.cpu_indices_cuda
            pref_idx = cpu_idx  # unused when n_pref == 0
        else:
            n_pref = h.n_prefetch_by_bucket[b]
            n_cpu = h.n_cpu_compute_by_bucket[b]
            pref_idx = h.prefetch_indices_cuda_by_bucket[b]
            cpu_idx = h.cpu_compute_indices_cuda_by_bucket[b]
            if n_pref > 0 and not dry_run:
                _assert_prefetch_slot_ready(
                    h,
                    n_pref,
                    underfilled_name="slot",
                )

        # CPU compute path skipped when n_cpu_compute == 0 (pure-prefetch).
        y_dst: torch.Tensor | None = None
        if n_cpu > 0 and not dry_run:
            assert self._runner is not None
            assert offloader._y_gpu is not None
            desc = (h.layer_idx, b, "qkv")
            # §1c.20: BRANCH before constructing CPU pinned views.
            # The native captured path has a different compiler
            # contract — Inductor materializes any CPU view it sees,
            # so we must NOT compute x_in / y_out at all when running
            # under the native runner. Only y_dst (a GPU view) is
            # built unconditionally; native reads its pinned input
            # via `cudaMemcpyAsync` from x_gpu.data_ptr() inside the
            # C++ side and reaches the pinned output via
            # `y_pinned_view(task_id, bucket)` on the slab
            # pointer. Python (eager kill-switch) keeps the original
            # x_in/y_out flow because it isn't traced by Inductor.
            y_dst = offloader._y_gpu[: num_tokens * n_cpu].view(num_tokens, n_cpu)
            if isinstance(self._runner, NativeCotsRunner):
                self._runner.submit_with_d2h(x, h.layer_idx, "qkv")
            else:
                assert offloader._x_pinned is not None
                assert offloader._y_pinned is not None
                x_in = offloader._x_pinned[: num_tokens * h.in_dim].view(
                    num_tokens, h.in_dim
                )
                y_out = offloader._y_pinned[: num_tokens * n_cpu].view(
                    num_tokens, n_cpu
                )
                self._runner.submit_with_d2h(x, x_in, y_out, desc)

        # GPU permanent slice. Skipped at f_cpu_store=1.0: F.linear on
        # weight (0, in_dim) returns (B, 0) which crashes downstream
        # custom CUDA ops (SiluAndMul) that can't handle zero-size.
        out_perm: torch.Tensor | None = None
        if layer.weight.shape[0] > 0:
            out_perm = F.linear(x, layer.weight, None)

        # GPU prefetched slice — runs concurrently on the same compute stream
        # after `wait_prefetch` (issued by the layer-forward hook) has joined
        # the copy stream's H2D.
        out_pref: torch.Tensor | None = None
        if n_pref > 0 and h.w_prefetch_slots and not dry_run:
            slot_view = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n_pref)
            out_pref = F.linear(x, slot_view, None)

        if n_cpu > 0 and not dry_run:
            assert y_dst is not None
            assert offloader._dummy_gpu_anchor_a is not None
            assert offloader._dummy_gpu_anchor_b is not None
            # Two-anchor schema (plan §design-decision 6): pin sync_then_uva
            # AFTER each independent GPU GEMM. out_perm and out_pref come
            # from independent F.linear calls; mutating only one would let
            # torch.compile reorder the other across sync. Distinct dummy
            # CUDA anchors fill in when a GPU GEMM didn't run for this slab
            # — never aliased.
            gpu_a = out_perm if out_perm is not None else offloader._dummy_gpu_anchor_a
            gpu_b = out_pref if out_pref is not None else offloader._dummy_gpu_anchor_b
            # §1c.20: y_pinned is intentionally not a wait_and_uva
            # parameter — native uses the slab pointer via
            # y_pinned_view; python stashes it on submit. The
            # captured-graph custom op sees only CUDA tensors +
            # scalars.
            if isinstance(self._runner, NativeCotsRunner):
                self._runner.wait_and_uva(y_dst, gpu_a, gpu_b, x, h.layer_idx, "qkv")
            else:
                self._runner.wait_and_uva(y_dst, gpu_a, gpu_b, x, desc)

        if out_perm is None and out_pref is None and y_dst is None:
            # Dry-run/full-offload corner: all active offloaded work is
            # intentionally skipped, but downstream layers still need the
            # canonical QKV shape. Values are diagnostic garbage by design.
            out = x.new_empty((num_tokens, h.out_dim))
        else:
            out = _scatter_col_outputs_three_way(
                out_perm, out_pref, y_dst, pref_idx, cpu_idx, h, num_tokens
            )
        if bias is not None:
            out = out + bias
        return out


class CotsSwiGLUMLPOp:
    """Block-level operator for fused MLP1 + SwiGLU + MLP2. Installed by
    replacing the parent module's `forward` (e.g., `Qwen2MLP.forward`).

    GPU runs the canonical MLP block on its weight slices; CPU runs the
    fused MLP1 → SwiGLU → MLP2 on its weight slices via the runner. CPU
    keeps its intermediate locally — single UVA return per block, not three
    (matched-index invariant, `weight_offload_design.md`).
    """

    def __init__(
        self,
        gate_up_layer: nn.Module,
        down_layer: nn.Module,
        gate_up_handle: CotsLinearHandle,
        down_handle: CotsLinearHandle,
        act_fn: nn.Module,
        runner: PythonCotsRunner | NativeCotsRunner,
        offloader: CotsOffloader,
        qualified_name: str,
    ):
        assert gate_up_handle.kind == "col"
        assert down_handle.kind == "row"
        # Matched-index invariant.
        assert gate_up_handle.n_cpu_per_half == down_handle.n_cpu, (
            f"MLP block matched-index violated at {qualified_name}: "
            f"gate_up.n_cpu_per_half={gate_up_handle.n_cpu_per_half} != "
            f"down.n_cpu={down_handle.n_cpu}"
        )
        assert gate_up_handle.w_cpu is not None
        assert down_handle.w_cpu is not None
        self._gate_up_layer = gate_up_layer
        self._down_layer = down_layer
        self._gate_up = gate_up_handle
        self._down = down_handle
        self._act_fn = act_fn
        self._runner = runner
        self._offloader = offloader
        self._qualified_name = qualified_name
        self._n_cpu_per_half = gate_up_handle.n_cpu_per_half
        self._in_dim = gate_up_handle.in_dim
        self._out_dim = down_handle.out_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        offloader = self._offloader
        dry_run = offloader.dry_run

        gu_h = self._gate_up
        dn_h = self._down
        assert gu_h.w_cpu is not None
        assert dn_h.w_cpu is not None
        # NOTE: under torch.compile + BACKED dynamic shapes, this is
        # really the activation buffer's row capacity (= max_num_batched_tokens),
        # not the live token count. See CotsQKVOp.apply for the full
        # comment. Used only for shape-consistent buffer slicing.
        num_tokens = x.shape[0]
        # Phase 1c Stage 3: bucket from offloader, not streamer (see
        # CotsQKVOp.apply for the resolution rationale). gu and dn
        # share the active bucket by construction. §1c.19: resolve to
        # a non-None int up-front; the runner facade no longer carries
        # a fallback callable.
        b = offloader._operator_bucket(num_tokens)
        if gu_h.max_n_prefetch == 0:
            gu_n_pref = 0
            dn_n_pref = 0
            dn_n_cpu = dn_h.n_cpu
        else:
            gu_n_pref = gu_h.n_prefetch_by_bucket[b]
            dn_n_pref = dn_h.n_prefetch_by_bucket[b]
            dn_n_cpu = dn_h.n_cpu_compute_by_bucket[b]
            if gu_n_pref > 0 and not dry_run:
                gu_n_per_half = gu_n_pref // 2
                _assert_prefetch_slot_ready(
                    gu_h,
                    gu_n_per_half,
                    underfilled_name="col slot",
                )
            if dn_n_pref > 0 and not dry_run:
                _assert_prefetch_slot_ready(
                    dn_h,
                    dn_n_pref,
                    underfilled_name="row slot",
                )

        # CPU compute path — skipped entirely when n_cpu_compute == 0
        # (pure-prefetch case). Without this fast-path the runner / D2H /
        # UVA overhead leaks into the prefetch-only regime. Phase 1c
        # Stage 3: weight slicing (n_pref_per_half / n_cpu_per_half_total)
        # is now done at install time inside the runner. Native sees
        # only stable call-site identity; Python runner still uses the
        # full descriptor to select its eager callback.
        y2_gpu: torch.Tensor | None = None
        if dn_n_cpu > 0 and not dry_run:
            assert self._runner is not None
            assert offloader._y_gpu is not None
            desc = (gu_h.layer_idx, b, "mlp_block")
            # §1c.20: branch BEFORE constructing the CPU pinned views
            # — Inductor materializes any CPU view it sees in the
            # captured graph (see CotsQKVOp.apply for the rationale).
            y2_gpu = offloader._y_gpu[: num_tokens * self._out_dim].view(
                num_tokens, self._out_dim
            )
            if isinstance(self._runner, NativeCotsRunner):
                self._runner.submit_with_d2h(x, gu_h.layer_idx, "mlp_block")
            else:
                assert offloader._x_pinned is not None
                assert offloader._y_pinned is not None
                x_pinned = offloader._x_pinned[: num_tokens * self._in_dim].view(
                    num_tokens, self._in_dim
                )
                y2_pinned = offloader._y_pinned[: num_tokens * self._out_dim].view(
                    num_tokens, self._out_dim
                )
                self._runner.submit_with_d2h(x, x_pinned, y2_pinned, desc)

        # GPU permanent MLP block. Skipped at f_cpu_store=1.0: gate_up's
        # (0, in_dim) weight makes act_fn run on (B, 0) which crashes the
        # CUDA SiluAndMul custom op.
        out_gpu: torch.Tensor | None = None
        if (
            self._gate_up_layer.weight.shape[0] > 0
            and self._down_layer.weight.shape[1] > 0
        ):
            gpu_mlp1 = F.linear(x, self._gate_up_layer.weight, None)
            gpu_silu = self._act_fn(gpu_mlp1)
            out_gpu = F.linear(gpu_silu, self._down_layer.weight, None)

        # GPU prefetched MLP block — adds a row-parallel partial to out_gpu.
        # Col prefetch slots are filled active-bucket-adjacent as
        # [gate_active | up_active], so MLP1 can use one fused [gate|up] GEMM
        # even when f_prefetch < f_cpu_store. MLP2/down slot uses the unified
        # transposed storage layout: shape (dn_n_pref, out_dim).
        if (
            gu_n_pref > 0
            and gu_h.w_prefetch_slots
            and dn_h.w_prefetch_slots
            and not dry_run
        ):
            gu_slot = gu_h.w_prefetch_slots[gu_h.slot_idx]
            dn_slot = dn_h.w_prefetch_slots[dn_h.slot_idx]
            pref_mlp1 = F.linear(x, gu_slot[:gu_n_pref, :], None)
            pref_silu = self._act_fn(pref_mlp1)
            dn_pref = dn_slot[:dn_n_pref, :]
            if out_gpu is None:
                out_gpu = pref_silu.matmul(dn_pref)
            else:
                out_gpu.addmm_(pref_silu, dn_pref)

        if dn_n_cpu > 0 and not dry_run:
            assert y2_gpu is not None
            assert offloader._dummy_gpu_anchor_a is not None
            assert offloader._dummy_gpu_anchor_b is not None
            # MLP block: out_gpu carries a combined dep on both perm and
            # pref GEMMs (via in-place addmm_ above), so
            # gpu_anchor_a alone covers the GPU work; gpu_anchor_b is
            # always the dummy. In the degenerate f_cpu_store=1.0 case
            # (no GPU work at all), anchor_a falls back to the dummy too
            # — there's nothing to order after sync except the
            # downstream consumer of the returned tensor, which the
            # `y_gpu` mutate already covers.
            gpu_a = out_gpu if out_gpu is not None else offloader._dummy_gpu_anchor_a
            gpu_b = offloader._dummy_gpu_anchor_b
            # §1c.20: y_pinned (y2_pinned) is intentionally not
            # passed; native uses y_pinned_view via the slab pointer
            # and python stashes y_pinned at submit time.
            if isinstance(self._runner, NativeCotsRunner):
                self._runner.wait_and_uva(
                    y2_gpu, gpu_a, gpu_b, x, gu_h.layer_idx, "mlp_block"
                )
            else:
                self._runner.wait_and_uva(y2_gpu, gpu_a, gpu_b, x, desc)
            # When CPU is the sole contributor, clone — y2_gpu is a shared
            # activation buffer and would be clobbered by the next layer.
            out_gpu = y2_gpu.clone() if out_gpu is None else out_gpu.add_(y2_gpu)

        if out_gpu is None:
            # Dry-run/full-offload corner: CPU and prefetched contributions are
            # deliberately omitted, but the parent decoder layer still needs a
            # hidden-state-shaped tensor. Values are diagnostic garbage.
            out_gpu = x.new_empty((num_tokens, self._out_dim))
        return out_gpu


class _RaiseOnDirectCall:
    """Defensive `quant_method` wrapper for MLP linears whose parent's
    forward we replaced with `CotsSwiGLUMLPOp`. Calling the linear directly
    (`mlp.gate_up_proj(x)` instead of `mlp(x)`) would silently use the
    GPU-slice weight and produce wrong-sized output; this raises instead.
    """

    def __init__(self, qualified_name: str, original):
        self._original = original
        self._qualified_name = qualified_name

    def __getattr__(self, name):
        return getattr(self._original, name)

    def apply(self, layer, x, bias=None):
        del layer, x, bias
        raise RuntimeError(
            f"cots: {self._qualified_name} is fused into its parent MLP "
            f"block. Call the parent module's forward(x), not the linear "
            f"directly."
        )


def _scatter_col_outputs_three_way(
    out_perm: torch.Tensor | None,
    out_pref: torch.Tensor | None,
    out_cpu_on_gpu: torch.Tensor | None,
    pref_idx: torch.Tensor,
    cpu_idx: torch.Tensor,
    handle: CotsLinearHandle,
    num_tokens: int,
) -> torch.Tensor:
    """Combine GPU permanent, GPU prefetched, and CPU-on-GPU column slices
    into the canonical layer output. All three slices are optional:
      `out_perm is None`     → skipped (f_cpu_store=1.0; permanent slice empty).
      `out_pref is None`     → skipped (f_prefetch=0).
      `out_cpu_on_gpu is None` → skipped (n_cpu_compute=0)."""
    # §1c.25 note: a Python-side NVTX scope here would only fire at
    # trace time, not on captured-graph replay (NVTX range_push/pop
    # are host CPU calls, not stream ops; they don't get captured).
    # Per-replay scatter cost is best attributed via
    # `nsys stats --report cuda_gpu_kern_sum` filtering for
    # `index_copy_*_kernel` / `IndexCopy*Kernel` rather than NVTX.
    assert handle.gpu_indices_cuda is not None
    ref = next(t for t in (out_perm, out_pref, out_cpu_on_gpu) if t is not None)
    out = torch.empty((num_tokens, handle.out_dim), dtype=ref.dtype, device=ref.device)
    if out_perm is not None:
        out.index_copy_(1, handle.gpu_indices_cuda, out_perm)
    if out_pref is not None:
        out.index_copy_(1, pref_idx, out_pref)
    if out_cpu_on_gpu is not None:
        out.index_copy_(1, cpu_idx, out_cpu_on_gpu)
    return out


# ---------------------------------------------------------------------------
# Offloader: lifecycle adapter. Discovery, handle construction, op installation,
# orphan check, activation buffer allocation. No execution policy.
# ---------------------------------------------------------------------------
