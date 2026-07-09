# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-split linear, MLP, scatter, and UVA-facing operators for COTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.offloader.cots_runners import (
    NativeCotsWeightRunner,
)
from vllm.model_executor.offloader.cots_storage import (
    MLP_DOWN_ROLE,
    MLP_GATE_UP_ROLE,
    OUTPUT_SPLIT_AXIS,
    QKV_ROLE,
    WO_ROLE,
    CotsLinearHandle,
)

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


def _prefetch_slot_guard(
    h: CotsLinearHandle,
    offloader: CotsOffloader,
    dummy_idx: int,
) -> torch.Tensor:
    if h.prefetch_slot_guards:
        return h.prefetch_slot_guards[h.slot_idx]
    if len(offloader._dummy_prefetch_slot_guards) <= dummy_idx:
        raise RuntimeError("COTS prefetch guard dummies were not allocated")
    return offloader._dummy_prefetch_slot_guards[dummy_idx]


class CotsOutputSplitLinearOp:
    """Patched `quant_method.apply` for output-split linears.

    GPU computes its permanent output slice; CPU computes its slice via the
    runner; optional prefetched rows run on GPU. The three disjoint outputs are
    scattered through `cpu_indices_cuda` / `gpu_indices_cuda` to restore the
    canonical output-channel order. WQKV and WO share this execution shape; the
    handle decides which channels were stored on CPU.
    """

    def __init__(
        self,
        handle: CotsLinearHandle,
        runner: NativeCotsWeightRunner,
        offloader: CotsOffloader,
        original_quant_method,
        *,
        op_kind: str = "qkv",
        expected_role: str = QKV_ROLE,
    ):
        assert handle.role == expected_role
        assert handle.split_axis == OUTPUT_SPLIT_AXIS
        self._handle = handle
        self._runner = runner
        self._offloader = offloader
        self._original = original_quant_method
        self._op_kind = op_kind

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
        # number of live tokens. Bucket-specific routing is resolved behind
        # COTS custom-op boundaries from the active dispatch state published
        # out of graph.
        num_tokens = x.shape[0]
        enable_cpu = not dry_run and offloader._y_gpu is not None
        enable_prefetch = not dry_run
        runner_id = self._runner._runner_id
        from vllm.model_executor.offloader import cots_ops

        y_dst: torch.Tensor
        if enable_cpu:
            assert self._runner is not None
            assert offloader._y_gpu is not None
            # §1c.20: never construct CPU pinned views here. Inductor
            # materializes any CPU view it sees, so the native runner reaches
            # pinned input/output through slab pointers populated at install.
            y_dst = offloader._y_gpu
            self._runner.submit_with_d2h(x, h.layer_idx, self._op_kind)
        else:
            y_dst = x.new_empty((0,))

        # GPU permanent slice. Skipped at f_cpu_store=1.0: F.linear on
        # weight (0, in_dim) returns (B, 0) which crashes downstream
        # custom CUDA ops (SiluAndMul) that can't handle zero-size.
        if layer.weight.shape[0] > 0:
            out_perm = F.linear(x, layer.weight, None)
        else:
            out_perm = x.new_empty((num_tokens, 0))

        # GPU prefetched slice — runs concurrently on the same compute stream
        # after `wait_prefetch` (issued by the layer-forward hook) has joined
        # the copy stream's H2D.
        out_pref = torch.ops.vllm.cots_prefetch_linear(
            x,
            _prefetch_slot_guard(h, offloader, 0),
            runner_id,
            int(h.layer_idx),
            cots_ops.op_kind_code(self._op_kind),
            int(h.max_n_prefetch),
            bool(enable_prefetch),
        )

        if enable_cpu:
            assert offloader._dummy_gpu_anchor_a is not None
            assert offloader._dummy_gpu_anchor_b is not None
            # Two-anchor schema (plan §design-decision 6): pin sync_then_uva
            # AFTER each independent GPU GEMM. out_perm and out_pref come
            # from independent F.linear calls; mutating only one would let
            # torch.compile reorder the other across sync. Distinct dummy
            # CUDA anchors fill in when a GPU GEMM didn't run for this slab
            # — never aliased.
            # §1c.20: y_pinned is intentionally not a wait_and_uva
            # parameter — native uses the slab pointer via
            # y_pinned_view. The
            # captured-graph custom op sees only CUDA tensors +
            # scalars.
            gpu_a = (
                out_perm if layer.weight.shape[0] > 0 else offloader._dummy_gpu_anchor_a
            )
            gpu_b = out_pref if h.max_n_prefetch > 0 else offloader._dummy_gpu_anchor_b
            self._runner.wait_and_uva(
                y_dst, gpu_a, gpu_b, x, h.layer_idx, self._op_kind
            )

        out = torch.ops.vllm.cots_scatter_col_outputs(
            out_perm,
            out_pref,
            y_dst,
            runner_id,
            int(h.layer_idx),
            cots_ops.op_kind_code(self._op_kind),
            int(h.out_dim),
            bool(enable_prefetch),
            bool(enable_cpu),
        )
        if bias is not None:
            out = out + bias
        return out


class CotsQKVOp(CotsOutputSplitLinearOp):
    """Output-split operator for QKVParallelLinear."""

    def __init__(
        self,
        handle: CotsLinearHandle,
        runner: NativeCotsWeightRunner,
        offloader: CotsOffloader,
        original_quant_method,
    ):
        super().__init__(
            handle=handle,
            runner=runner,
            offloader=offloader,
            original_quant_method=original_quant_method,
            op_kind="qkv",
            expected_role=QKV_ROLE,
        )


class CotsWOOp(CotsOutputSplitLinearOp):
    """Patched `quant_method.apply` for WO (`o_proj`) output-column split.

    WO uses a dense output split: GPU, prefetched-GPU, and CPU slices produce
    disjoint output hidden channels which are scattered back into canonical
    hidden-state order. The split is deliberately not WQKV/KV-biased.
    """

    def __init__(
        self,
        handle: CotsLinearHandle,
        runner: NativeCotsWeightRunner,
        offloader: CotsOffloader,
        original_quant_method,
    ):
        super().__init__(
            handle=handle,
            runner=runner,
            offloader=offloader,
            original_quant_method=original_quant_method,
            op_kind="wo",
            expected_role=WO_ROLE,
        )


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
        runner: NativeCotsWeightRunner,
        offloader: CotsOffloader,
        qualified_name: str,
    ):
        assert gate_up_handle.role == MLP_GATE_UP_ROLE
        assert down_handle.role == MLP_DOWN_ROLE
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
        # not the live token count. See CotsOutputSplitLinearOp.apply for the full
        # comment. Used only for shape-consistent buffer slicing.
        num_tokens = x.shape[0]
        enable_cpu = not dry_run and offloader._y_gpu is not None
        enable_prefetch = not dry_run
        runner_id = self._runner._runner_id
        from vllm.model_executor.offloader import cots_ops

        y2_gpu: torch.Tensor
        if enable_cpu:
            assert self._runner is not None
            assert offloader._y_gpu is not None
            # §1c.20: do not construct CPU pinned views in the operator.
            # Inductor materializes any CPU view it sees in the captured graph;
            # native COTS reaches pinned buffers through install-time slabs.
            y2_gpu = offloader._y_gpu
            self._runner.submit_with_d2h(x, gu_h.layer_idx, "mlp_block")
        else:
            y2_gpu = x.new_empty((0,))

        # GPU permanent MLP block. Skipped at f_cpu_store=1.0: gate_up's
        # (0, in_dim) weight makes act_fn run on (B, 0) which crashes the
        # CUDA SiluAndMul custom op.
        has_base_gpu = (
            self._gate_up_layer.weight.shape[0] > 0
            and self._down_layer.weight.shape[1] > 0
        )
        if has_base_gpu:
            gpu_mlp1 = F.linear(x, self._gate_up_layer.weight, None)
            gpu_silu = self._act_fn(gpu_mlp1)
            out_gpu = F.linear(gpu_silu, self._down_layer.weight, None)
        else:
            out_gpu = x.new_empty((num_tokens, self._out_dim))

        # GPU prefetched MLP block — adds a row-parallel partial to out_gpu.
        # Col prefetch slots are filled active-bucket-adjacent as
        # [gate_active | up_active], so MLP1 can use one fused [gate|up] GEMM
        # even when f_prefetch < f_cpu_store. MLP2/down slot uses the unified
        # transposed storage layout: shape (dn_n_pref, out_dim).
        out_gpu = torch.ops.vllm.cots_mlp_prefetch_add(
            x,
            out_gpu,
            _prefetch_slot_guard(gu_h, offloader, 0),
            _prefetch_slot_guard(dn_h, offloader, 1),
            runner_id,
            int(gu_h.layer_idx),
            cots_ops.op_kind_code("mlp_block"),
            bool(has_base_gpu),
            bool(enable_prefetch),
        )

        if enable_cpu:
            assert offloader._dummy_gpu_anchor_a is not None
            assert offloader._dummy_gpu_anchor_b is not None
            # MLP block: out_gpu carries any permanent and prefetched GPU
            # contribution, so gpu_anchor_a alone covers GPU work; gpu_anchor_b
            # is always the dummy.
            # §1c.20: y_pinned (y2_pinned) is intentionally not passed;
            # native uses y_pinned_view via the slab pointer.
            self._runner.wait_and_uva(
                y2_gpu,
                out_gpu,
                offloader._dummy_gpu_anchor_b,
                x,
                gu_h.layer_idx,
                "mlp_block",
            )
        # When CPU is the sole contributor, clone — y2_gpu is a shared
        # activation buffer and would be clobbered by the next layer. Keep the
        # mixed GPU+CPU merge out-of-place for the same captured-graph ordering
        # reason as the earlier Python branch.
        out_gpu = torch.ops.vllm.cots_mlp_merge_cpu(
            out_gpu,
            y2_gpu,
            runner_id,
            int(gu_h.layer_idx),
            cots_ops.op_kind_code("mlp_block"),
            bool(has_base_gpu),
            bool(enable_prefetch),
            bool(enable_cpu),
        )
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


# ---------------------------------------------------------------------------
# Offloader: lifecycle adapter. Discovery, handle construction, op installation,
# orphan check, activation buffer allocation. No execution policy.
# ---------------------------------------------------------------------------
