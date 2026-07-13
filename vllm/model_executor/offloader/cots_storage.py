# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage, partition, and prefetch-buffer helpers for COTS."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from cots.snap import (
    DEFAULT_QKVO_HEAD_DIM,
)
from cots.snap import (
    snap_mlp_channels as _snap_mlp_channels,
)
from cots.snap import (
    snap_qkvo_output_channels as _snap_qkvo_output_channels,
)

from vllm.model_executor.offloader.cots_utils import (
    _complement,
)
from vllm.utils.platform_utils import is_pin_memory_available

SplitAxis = Literal["output", "input"]
CotsLinearRole = Literal["qkv", "mlp_gate_up", "mlp_down", "wo"]

OUTPUT_SPLIT_AXIS: SplitAxis = "output"
INPUT_SPLIT_AXIS: SplitAxis = "input"

QKV_ROLE: CotsLinearRole = "qkv"
MLP_GATE_UP_ROLE: CotsLinearRole = "mlp_gate_up"
MLP_DOWN_ROLE: CotsLinearRole = "mlp_down"
WO_ROLE: CotsLinearRole = "wo"

ROLE_SPLIT_AXIS: dict[CotsLinearRole, SplitAxis] = {
    QKV_ROLE: OUTPUT_SPLIT_AXIS,
    MLP_GATE_UP_ROLE: OUTPUT_SPLIT_AXIS,
    MLP_DOWN_ROLE: INPUT_SPLIT_AXIS,
    WO_ROLE: OUTPUT_SPLIT_AXIS,
}


def _qkv_dense_tail_counts(
    q_size: int,
    kv_size: int,
    n_cpu_cols: int,
) -> tuple[int, int, int]:
    """Return Q/K/V shard row counts for a dense tail over `[Q | K | V]`."""

    q_size = int(q_size)
    kv_size = int(kv_size)
    n_cpu_cols = int(n_cpu_cols)
    total = q_size + 2 * kv_size
    if not (0 <= n_cpu_cols <= total):
        raise ValueError(f"n_cpu_cols={n_cpu_cols} out of range [0, {total}]")

    n_v = min(n_cpu_cols, kv_size)
    remaining = max(0, n_cpu_cols - n_v)
    n_k = min(remaining, kv_size)
    remaining -= n_k
    n_q_tail = min(remaining, q_size)
    return n_q_tail, n_k, n_v


class CotsLinearHandle:
    """Per-Linear partition primitive: storage + load. No execution.

    Role decides module-specific splitting:
      qkv         : WQKV dense output-tail split on the QKVO grid.
      mlp_gate_up : MLP gate/up merged output split, two matched halves.
      mlp_down    : MLP down input split, matched to gate/up's intermediate rows.
      wo          : Dense output split, used by opt-in WO.

    Split axis is derived from role and decides generic storage shape:
      output : CPU weight is stored as `(n_cpu, in_dim)`.
      input  : CPU weight is stored as transposed `(n_cpu, out_dim)`.

    Construction: use the role-specific class methods (`for_qkv`,
    `for_mlp_gate_up`, `for_mlp_down`, `for_wo`) which compute snapped
    `n_cpu` and pick indices.
    """

    def __init__(
        self,
        *,
        role: CotsLinearRole,
        linear: nn.Module,
        qualified_name: str,
        in_dim: int,
        out_dim: int,
        n_cpu: int,
        cpu_indices: torch.Tensor,
        gpu_indices: torch.Tensor,
        dtype: torch.dtype,
        # QKV-only metadata:
        q_size: int | None = None,
        kv_size: int | None = None,
        head_dim: int | None = None,
        # Merged-col-only metadata:
        merged_partition_sizes: tuple[int, int] | None = None,
        # Dense-output-only metadata:
        qkvo_head_dim: int = DEFAULT_QKVO_HEAD_DIM,
    ):
        if role not in ROLE_SPLIT_AXIS:
            raise ValueError(f"unknown role: {role}")
        if cpu_indices.numel() != n_cpu:
            raise ValueError(
                f"cpu_indices.numel()={cpu_indices.numel()} != n_cpu={n_cpu}"
            )

        self.role = role
        self.split_axis = ROLE_SPLIT_AXIS[role]
        self.linear = linear
        self.qualified_name = qualified_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_cpu = n_cpu
        self.dtype = dtype

        # Index tensors. CPU copies for load-time split, CUDA copies for
        # forward-time gather/scatter.
        self.cpu_indices = cpu_indices  # on CPU
        self.gpu_indices = gpu_indices  # on CPU
        self.cpu_indices_cuda: torch.Tensor | None = None
        self.gpu_indices_cuda: torch.Tensor | None = None

        # Sub-module shapes used by operator buffer-slicing.
        if self.split_axis == INPUT_SPLIT_AXIS:
            self.cpu_in_dim = n_cpu
            self.cpu_out_dim = out_dim
        else:
            self.cpu_in_dim = in_dim
            self.cpu_out_dim = n_cpu

        # Per-shard split metadata for the loader closures.
        self.q_size = q_size
        self.kv_size = kv_size
        self.head_dim = head_dim
        self.merged_partition_sizes = merged_partition_sizes
        self.qkvo_head_dim = qkvo_head_dim
        self.n_q_tail = 0
        self.n_k = 0
        self.n_v = 0
        self.n_cpu_per_half = 0
        if role == QKV_ROLE:
            assert q_size is not None and kv_size is not None
            assert head_dim is not None
            self.n_q_tail, self.n_k, self.n_v = _qkv_dense_tail_counts(
                q_size,
                kv_size,
                n_cpu,
            )
            assert self.n_q_tail + self.n_k + self.n_v == n_cpu, (
                f"QKV count mismatch at {qualified_name}: n_cpu={n_cpu} != "
                f"sum of (n_q_tail={self.n_q_tail}, n_k={self.n_k}, "
                f"n_v={self.n_v})."
            )
        elif role == MLP_GATE_UP_ROLE:
            assert merged_partition_sizes is not None
            assert merged_partition_sizes[0] == merged_partition_sizes[1], (
                f"MergedColumnParallelLinear's gate/up output partitions "
                f"must be equal-sized; got {merged_partition_sizes}"
            )
            assert n_cpu % 2 == 0, (
                f"mlp_gate_up expects n_cpu divisible by 2; got n_cpu={n_cpu}"
            )
            self.n_cpu_per_half = n_cpu // 2

        # Storage (allocated by `install`).
        self.w_cpu: torch.Tensor | None = None
        # Membership flag set by the offloader's MLP-block recognition.
        self.in_block: bool = False
        # Decoder-layer index. Set by the offloader after `_build_handles`.
        # Phase 1b uses it for prefetch slot rotation (`layer_idx % K`).
        self.layer_idx: int = -1
        # Prefetch slot index = `layer_idx % CotsPrefetchBufferPool.K`. Set
        # by the offloader when constructing the buffer pool.
        self.slot_idx: int = -1
        # Per-bucket prefetch geometry — populated by
        # `apply_prefetch_split_per_bucket`. Phase 1b: every bucket gets
        # the same value (uniform fill); Planner: per-bucket variation
        # from the dispatch table.
        self.n_prefetch_by_bucket: dict[int, int] = {}
        self.n_cpu_compute_by_bucket: dict[int, int] = {}
        self.prefetch_indices_cuda_by_bucket: dict[int, torch.Tensor] = {}
        self.cpu_compute_indices_cuda_by_bucket: dict[int, torch.Tensor] = {}
        self.max_n_prefetch: int = 0
        # GPU prefetch buffer slot views — bound by the prefetch buffer pool
        # (Step 3). Length K=2; layer i uses slot `i % K`. Each view is
        # shape `(max_n_prefetch, in_dim)` for output-split handles or
        # `(max_n_prefetch, out_dim)` for input-split handles, matching
        # `w_cpu`'s layout.
        self.w_prefetch_slots: list[torch.Tensor] = []
        # Shape-group-shared per-slot state — buffer pool binds the SAME
        # list to every handle in a group so writes are visible across
        # all sharers of the physical slot.
        # `prefetch_owner_in_slot[k]`: handle that last filled slot k
        #   (None = empty). Operators assert owner is self before reading.
        # `prefetch_available_rows_in_slot[k]`: how many leading prefix
        #   rows of the slot are valid. Per-half row count for MLP gate/up
        #   (`gate[:a]` AND `up[:a]` valid → available_rows == a); total
        #   prefix rows for qkv/wo/mlp_down. 0 = empty.
        self.prefetch_owner_in_slot: list[CotsLinearHandle | None] = []
        self.prefetch_available_rows_in_slot: list[int] = []
        # Stage 7-C: row-handle `w_cpu` is stored in transposed
        # `(n_cpu, out_dim)` layout (see install()); its first
        # `n_pref` rows are the contiguous prefetch source. No
        # separate duplicate buffer.

    # ------------------------------------------------------------------
    # Construction helpers — compute indices, snap n_cpu, build handle.
    # ------------------------------------------------------------------
    @staticmethod
    def _dense_output_tail_handle(
        *,
        role: CotsLinearRole,
        linear: nn.Module,
        qualified_name: str,
        f_cpu_store: float,
        qkvo_head_dim: int,
        q_size: int | None = None,
        kv_size: int | None = None,
    ) -> CotsLinearHandle | None:
        out_dim, in_dim = tuple(linear.weight.shape)
        n_cpu = _snap_qkvo_output_channels(
            f_cpu_store * out_dim,
            out_dim=out_dim,
            head_dim=qkvo_head_dim,
        )
        if n_cpu == 0:
            return None
        cpu_indices = torch.arange(
            out_dim - n_cpu, out_dim, dtype=torch.long, device="cpu"
        )
        return CotsLinearHandle(
            role=role,
            linear=linear,
            qualified_name=qualified_name,
            in_dim=in_dim,
            out_dim=out_dim,
            n_cpu=n_cpu,
            cpu_indices=cpu_indices,
            gpu_indices=_complement(cpu_indices, out_dim),
            dtype=linear.weight.dtype,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=qkvo_head_dim if role == QKV_ROLE else None,
            qkvo_head_dim=qkvo_head_dim,
        )

    @classmethod
    def for_qkv(
        cls,
        linear: nn.Module,
        qualified_name: str,
        *,
        head_dim: int,
        f_cpu_store: float,
    ) -> CotsLinearHandle | None:
        parts = linear.output_partition_sizes
        assert len(parts) == 3, f"QKV expected 3 partitions, got {parts}"
        q_part, k_part, v_part = parts
        assert k_part == v_part, (
            f"QKV expected k_part == v_part, got k={k_part}, v={v_part}"
        )
        assert k_part % head_dim == 0, (
            f"QKV: kv_size={k_part} not a multiple of head_dim={head_dim}"
        )
        return cls._dense_output_tail_handle(
            role=QKV_ROLE,
            linear=linear,
            qualified_name=qualified_name,
            f_cpu_store=f_cpu_store,
            qkvo_head_dim=head_dim,
            q_size=q_part,
            kv_size=k_part,
        )

    @classmethod
    def for_mlp_gate_up(
        cls,
        linear: nn.Module,
        qualified_name: str,
        *,
        f_cpu_store: float,
    ) -> CotsLinearHandle | None:
        out_dim, in_dim = tuple(linear.weight.shape)
        parts = linear.output_partition_sizes
        assert len(parts) == 2 and parts[0] == parts[1], (
            f"MergedCol expected matched partitions, got {parts}"
        )
        half = parts[0]
        n_cpu_per_half = _snap_mlp_channels(f_cpu_store * half, half)
        n_cpu = 2 * n_cpu_per_half
        if n_cpu == 0:
            return None
        # LAST n_cpu_per_half rows of each half. Aligns with TP loader
        # convention (FIRST rows on rank 0 → exactly our GPU portion).
        base = torch.arange(half - n_cpu_per_half, half, dtype=torch.long, device="cpu")
        cpu_indices = torch.cat([base, base + half])
        return cls(
            role=MLP_GATE_UP_ROLE,
            linear=linear,
            qualified_name=qualified_name,
            in_dim=in_dim,
            out_dim=out_dim,
            n_cpu=n_cpu,
            cpu_indices=cpu_indices,
            gpu_indices=_complement(cpu_indices, out_dim),
            dtype=linear.weight.dtype,
            merged_partition_sizes=(parts[0], parts[1]),
        )

    @classmethod
    def for_mlp_down(
        cls,
        linear: nn.Module,
        qualified_name: str,
        *,
        f_cpu_store: float,
    ) -> CotsLinearHandle | None:
        out_dim, in_dim = tuple(linear.weight.shape)
        n_cpu = _snap_mlp_channels(f_cpu_store * in_dim, in_dim)
        if n_cpu == 0:
            return None
        # LAST n_cpu input cols. Preserves the MLP1↔MLP2 matched-index
        # invariant under uniform f_cpu_store.
        cpu_indices = torch.arange(
            in_dim - n_cpu, in_dim, dtype=torch.long, device="cpu"
        )
        return cls(
            role=MLP_DOWN_ROLE,
            linear=linear,
            qualified_name=qualified_name,
            in_dim=in_dim,
            out_dim=out_dim,
            n_cpu=n_cpu,
            cpu_indices=cpu_indices,
            gpu_indices=_complement(cpu_indices, in_dim),
            dtype=linear.weight.dtype,
        )

    @classmethod
    def for_wo(
        cls,
        linear: nn.Module,
        qualified_name: str,
        *,
        f_cpu_store: float,
        qkvo_head_dim: int = DEFAULT_QKVO_HEAD_DIM,
    ) -> CotsLinearHandle | None:
        """WO dense output-row split.

        MLP gate/up is also output-axis split, but it uses a merged two-half
        layout and matched-index invariant. WO has no Q/K/V or gate/up
        structure, so its selected output channels are just a dense tail.
        """
        return cls._dense_output_tail_handle(
            role=WO_ROLE,
            linear=linear,
            qualified_name=qualified_name,
            f_cpu_store=f_cpu_store,
            qkvo_head_dim=qkvo_head_dim,
        )

    # ------------------------------------------------------------------
    # Installation: replace param.data, allocate w_cpu, wrap weight_loader.
    # ------------------------------------------------------------------
    def install(self, device: torch.device) -> None:
        """Replace `linear.weight.data` with a GPU-slice tensor, allocate
        `w_cpu` (pinned), and wrap the linear's `weight_loader` closure.

        After this returns, `linear.weight.data` is at slice shape; the loader
        closure is responsible for splitting incoming `loaded_weight` into
        the GPU and CPU portions at load time. Tags `linear._cots_handle`.
        """
        weight_param = linear_weight = self.linear.weight
        if self.split_axis == OUTPUT_SPLIT_AXIS:
            gpu_slice_shape = (self.out_dim - self.n_cpu, self.in_dim)
            w_cpu_shape = (self.n_cpu, self.in_dim)
        else:
            gpu_slice_shape = (self.out_dim, self.in_dim - self.n_cpu)
            # Stage 7-C: row-handle (down_proj) stores CPU weight as
            # transposed `(n_cpu, out_dim)` row-major. Both the
            # row-prefetch slice and the CPU-compute slice are
            # `w_cpu.narrow(0, ...)` — contiguous, no duplicate
            # buffer, directly feedable to `bf16_gemm_transposed`
            # which expects (K, N) row-major BF16.
            w_cpu_shape = (self.n_cpu, self.out_dim)
        weight_param.data = torch.empty(
            gpu_slice_shape, dtype=self.dtype, device=device
        )
        self.w_cpu = torch.empty(
            w_cpu_shape,
            dtype=self.dtype,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self.cpu_indices_cuda = self.cpu_indices.to(device)
        self.gpu_indices_cuda = self.gpu_indices.to(device)
        # Wrap weight_loader. vLLM's load_weights uses `param.weight_loader`
        # (set on the Parameter via set_weight_attrs); update both attrs.
        wrapped = self._build_weight_loader()
        self.linear.weight_loader = wrapped
        linear_weight.weight_loader = wrapped
        # Tag the linear so operators / tests can look up the handle.
        self.linear._cots_handle = self  # type: ignore[attr-defined]

    def _build_weight_loader(self) -> Callable:
        """Return the role-specific weight_loader closure."""
        if self.role == MLP_DOWN_ROLE:
            return self._row_weight_loader
        if self.role == MLP_GATE_UP_ROLE:
            return self._merged_col_weight_loader
        if self.role == WO_ROLE:
            return self._wo_weight_loader
        if self.role == QKV_ROLE:
            return self._qkv_weight_loader
        raise ValueError(f"unknown COTS linear role: {self.role}")

    # ------------------------------------------------------------------
    # Per-bucket prefetch geometry. Populated after install. Loader closures
    # are unaffected — `w_cpu` is loaded once with all CPU rows; the
    # prefetch / CPU-compute split is a runtime view (`w_cpu.narrow(...)`).
    # ------------------------------------------------------------------
    def apply_prefetch_split_per_bucket(
        self,
        dispatch_table: dict[int, tuple[float, float]],
    ) -> None:
        """For each bucket in `dispatch_table`, compute `n_prefetch`,
        `n_cpu_compute`, and the index tensors that scatter the three GPU
        outputs (permanent + prefetched + CPU-on-GPU) into the canonical
        layer output. Must be called after `install()` (uses CUDA-side
        `cpu_indices_cuda` for index slicing).
        """
        assert self.cpu_indices_cuda is not None, "call install() first"
        device = self.cpu_indices_cuda.device

        self.n_prefetch_by_bucket.clear()
        self.n_cpu_compute_by_bucket.clear()
        self.prefetch_indices_cuda_by_bucket.clear()
        self.cpu_compute_indices_cuda_by_bucket.clear()

        for bucket, (f_cpu_compute, _) in dispatch_table.items():
            n_cpu, pref_idx, cpu_idx = self._compute_bucket_split(f_cpu_compute)
            n_pref = self.n_cpu - n_cpu
            self.n_prefetch_by_bucket[bucket] = n_pref
            self.n_cpu_compute_by_bucket[bucket] = n_cpu
            self.prefetch_indices_cuda_by_bucket[bucket] = pref_idx.to(device)
            self.cpu_compute_indices_cuda_by_bucket[bucket] = cpu_idx.to(device)

        # Planner option-A accounting reserves full CPU-stored prefetch slot
        # capacity for a placement fraction, independent of the per-bucket
        # dispatch table. Runtime copies still narrow to
        # `n_prefetch_by_bucket[b]`, so this only changes reserved capacity.
        self.max_n_prefetch = self.n_cpu if dispatch_table else 0

    def _compute_bucket_split(
        self, f_cpu_compute: float
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Split `cpu_indices` into prefetched and CPU-computed subsets.

        Returns `(n_cpu_compute, prefetch_indices_cpu,
        cpu_compute_indices_cpu)`. Indices are still on CPU; the caller moves
        them to the device. Runtime snaps the CPU-compute side down to a valid
        module quantum and assigns the remaining CPU-stored rows to prefetch.

        Layout invariants:
          qkv: `cpu_indices` is a dense output tail over `[Q | K | V]`.
            Prefetch takes the first `n_pref` indices.
          mlp_gate_up: `cpu_indices` is
            `[gate_last_n_cpu_per_half | up_last_n_cpu_per_half]`.
            Prefetch takes the FIRST `n_pref_per_half` of each half — keeps
            the matched-index invariant with MLP2's input cols.
          mlp_down: `cpu_indices` is the LAST `n_cpu` input cols. Prefetch
            takes the first `n_pref` of those.
          wo: dense output-tail split. Prefetch takes the first `n_pref` rows.

        QKV and WO snap on the shared `2 * head_dim` QKVO grid. MLP gate/up and
        down use the shared 64-channel MLP snap grid. All roles clamp CPU
        compute to the CPU-stored cap.
        Therefore below-boundary runtime remainder goes to prefetch, not CPU
        compute.
        """
        cap = self.n_cpu

        if self.role in (QKV_ROLE, WO_ROLE):
            n_cpu_compute = _snap_qkvo_output_channels(
                f_cpu_compute * self.out_dim,
                out_dim=self.out_dim,
                head_dim=self.qkvo_head_dim,
            )
            n_cpu_compute = min(n_cpu_compute, cap)
        elif self.role == MLP_GATE_UP_ROLE:
            half = self.out_dim // 2
            n_cpu_compute_per_half = min(
                _snap_mlp_channels(f_cpu_compute * half, half),
                self.n_cpu_per_half,
            )
            n_cpu_compute = 2 * n_cpu_compute_per_half
        elif self.role == MLP_DOWN_ROLE:
            n_cpu_compute = min(
                _snap_mlp_channels(f_cpu_compute * self.in_dim, self.in_dim),
                cap,
            )
        else:
            raise ValueError(f"unknown COTS linear role: {self.role}")

        # Index extraction is per role and depends on `cpu_indices`'s layout.
        if self.role == MLP_GATE_UP_ROLE:
            n_cpu_compute_per_half = n_cpu_compute // 2
            ncph = self.n_cpu_per_half
            n_pref_per_half = ncph - n_cpu_compute_per_half
            pref_idx = torch.cat(
                [
                    self.cpu_indices[:n_pref_per_half],
                    self.cpu_indices[ncph : ncph + n_pref_per_half],
                ]
            )
            cpu_idx = torch.cat(
                [
                    self.cpu_indices[n_pref_per_half:ncph],
                    self.cpu_indices[ncph + n_pref_per_half :],
                ]
            )
            return n_cpu_compute, pref_idx, cpu_idx

        # qkv / wo / mlp_down: prefetch is a contiguous prefix of cpu_indices.
        n_prefetch = self.n_cpu - n_cpu_compute
        return (
            n_cpu_compute,
            self.cpu_indices[:n_prefetch],
            self.cpu_indices[n_prefetch:],
        )

    # --- Loader closures (per role, accessing self by closure) ---

    def _row_weight_loader(self, param, loaded_weight):
        """RowParallelLinear (down_proj): single call, full
        (out_dim, in_dim) loaded_weight. GPU keeps FIRST keep_gpu input cols;
        CPU gets LAST n_cpu — stored in TRANSPOSED orientation
        `(n_cpu, out_dim)` per Stage 7-C unified storage (see install()
        docstring). One-shot transpose at load time so every per-forward
        slice (prefetch row-narrow + CPU-compute row-narrow) is
        contiguous.
        """
        assert self.w_cpu is not None
        assert loaded_weight.shape == (self.out_dim, self.in_dim), (
            f"row loader at {self.qualified_name}: expected "
            f"({self.out_dim}, {self.in_dim}), got {tuple(loaded_weight.shape)}"
        )
        keep_gpu = self.in_dim - self.n_cpu
        param.data.copy_(loaded_weight[:, :keep_gpu], non_blocking=False)
        # `loaded_weight[:, keep_gpu:]` is (out_dim, n_cpu); we want
        # (n_cpu, out_dim) for our unified transposed-storage layout.
        # `.transpose(0, 1).contiguous()` materializes the transposed
        # tensor once at load.
        self.w_cpu.copy_(
            loaded_weight[:, keep_gpu:].transpose(0, 1).contiguous(),
            non_blocking=False,
        )

    def _merged_col_weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        """MergedColumnParallelLinear (gate_up_proj): per-shard call
        (loaded_shard_id 0=gate, 1=up). Each call delivers ONE partition.
        """
        assert self.w_cpu is not None
        assert self.merged_partition_sizes is not None
        half = self.merged_partition_sizes[0]
        n_cpu_per_half = self.n_cpu_per_half
        keep_gpu = half - n_cpu_per_half
        assert loaded_weight.shape == (half, self.in_dim), (
            f"merged col loader at {self.qualified_name}: expected "
            f"({half}, {self.in_dim}), got {tuple(loaded_weight.shape)}"
        )
        gpu_view = loaded_weight[:keep_gpu, :]
        cpu_view = loaded_weight[keep_gpu:, :]
        # param.data layout: (2*keep_gpu, in_dim) = [gate_gpu | up_gpu] stacked.
        # w_cpu layout: (2*n_cpu_per_half, in_dim) = [gate_cpu | up_cpu].
        if loaded_shard_id == 0:
            param.data[:keep_gpu, :].copy_(gpu_view, non_blocking=False)
            self.w_cpu[:n_cpu_per_half, :].copy_(cpu_view, non_blocking=False)
        elif loaded_shard_id == 1:
            param.data[keep_gpu:, :].copy_(gpu_view, non_blocking=False)
            self.w_cpu[n_cpu_per_half:, :].copy_(cpu_view, non_blocking=False)
        else:
            raise ValueError(
                f"cots merged col loader: expected loaded_shard_id in {{0, 1}}, "
                f"got {loaded_shard_id!r}"
            )

    def _wo_weight_loader(self, param, loaded_weight):
        """Dense output-row split (WO): single full `(out_dim, in_dim)` load."""
        assert self.w_cpu is not None
        assert loaded_weight.shape == (self.out_dim, self.in_dim), (
            f"wo loader at {self.qualified_name}: expected "
            f"({self.out_dim}, {self.in_dim}), got {tuple(loaded_weight.shape)}"
        )
        keep_gpu = self.out_dim - self.n_cpu
        param.data.copy_(loaded_weight[:keep_gpu, :], non_blocking=False)
        self.w_cpu.copy_(loaded_weight[keep_gpu:, :], non_blocking=False)

    def _qkv_weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        """QKVParallelLinear (qkv_proj): per-shard call ('q'/'k'/'v')."""
        assert self.w_cpu is not None
        assert self.q_size is not None and self.kv_size is not None
        q_size, kv_size = self.q_size, self.kv_size
        n_q_tail, n_k, n_v = self.n_q_tail, self.n_k, self.n_v
        keep_gpu_q = q_size - n_q_tail
        keep_gpu_k = kv_size - n_k
        keep_gpu_v = kv_size - n_v
        # w_cpu row layout: [Q_tail | K_cpu | V_cpu]
        cpu_q_offset = 0
        cpu_k_offset = n_q_tail
        cpu_v_offset = n_q_tail + n_k

        if loaded_shard_id == "q":
            assert loaded_weight.shape == (q_size, self.in_dim), (
                f"qkv 'q' loader at {self.qualified_name}: expected "
                f"({q_size}, {self.in_dim}), got {tuple(loaded_weight.shape)}"
            )
            # GPU param.data layout: [Q_gpu | K_gpu | V_gpu] stacked.
            param.data[:keep_gpu_q, :].copy_(
                loaded_weight[:keep_gpu_q, :], non_blocking=False
            )
            if n_q_tail > 0:
                self.w_cpu[cpu_q_offset : cpu_q_offset + n_q_tail, :].copy_(
                    loaded_weight[keep_gpu_q:, :], non_blocking=False
                )
        elif loaded_shard_id == "k":
            assert loaded_weight.shape == (kv_size, self.in_dim)
            if keep_gpu_k > 0:
                param.data[keep_gpu_q : keep_gpu_q + keep_gpu_k, :].copy_(
                    loaded_weight[:keep_gpu_k, :], non_blocking=False
                )
            if n_k > 0:
                self.w_cpu[cpu_k_offset : cpu_k_offset + n_k, :].copy_(
                    loaded_weight[keep_gpu_k:, :], non_blocking=False
                )
        elif loaded_shard_id == "v":
            assert loaded_weight.shape == (kv_size, self.in_dim)
            v_gpu_start = keep_gpu_q + keep_gpu_k
            if keep_gpu_v > 0:
                param.data[v_gpu_start : v_gpu_start + keep_gpu_v, :].copy_(
                    loaded_weight[:keep_gpu_v, :], non_blocking=False
                )
            if n_v > 0:
                self.w_cpu[cpu_v_offset : cpu_v_offset + n_v, :].copy_(
                    loaded_weight[keep_gpu_v:, :], non_blocking=False
                )
        else:
            raise ValueError(
                f"cots qkv loader: expected loaded_shard_id in {{'q','k','v'}}, "
                f"got {loaded_shard_id!r}"
            )


# ---------------------------------------------------------------------------
# Execution layer: CotsPrefetchBufferPool
#
# Layer-ahead weight-prefetch destination. Allocates K=2 GPU slot views per
# offloaded handle so prefetch for layer i+1 can overlap with layer i's
# compute (every layer is offloaded → K=2 is the minimum slot count for any
# overlap; see `phase0_findings.md §0.10.1d` and the Phase 1b plan).
# Slot shape mirrors `w_cpu`'s row-major storage per split axis:
#   output split : (max_n_prefetch, in_dim)   — prefetch dim 0
#   input split  : (max_n_prefetch, out_dim)  — prefetch dim 0
# MLP gate/up slots are filled in active-bucket-adjacent layout
# `[gate_active | up_active]`.
# Sized to the full CPU-stored slice (`max_n_prefetch == n_cpu`);
# per-forward H2D narrows to the active bucket's `n_prefetch_by_bucket[b]`.
# ---------------------------------------------------------------------------
class CotsPrefetchBufferPool:
    """K=2 slot rotation. K slots PER UNIQUE shape, SHARED across layers.

    Mirrors `prefetch.py`'s `StaticBufferPool`: at G=1 (every layer
    offloaded) all 28 qkv handles share the same K=2 qkv-shape slots,
    rotated by `layer_idx % K`. Layer i and layer i+2 read/write the
    same slot 0; the streamer's fork-event ordering ensures layer i+2's
    H2D writes slot 0 only after layer i's GEMMs read it.

    Pool size = K × Σ_unique_shape (slot_numel × dtype_bytes), NOT
    K × Σ_handle. At Qwen2.5-7B with 28 offloaded layers and 3 unique
    shapes per layer, this is 28× smaller than per-handle allocation
    (which over-counts by N_layers).

    Allocated inside `wrap_modules` (DeviceMemoryProfiler invariant —
    `phase1a_findings.md §1.5`).
    """

    K = 2  # slot count, hard-coded — see Phase 1b plan §"Top-Level Decisions"

    def __init__(
        self,
        handles: list[CotsLinearHandle],
        device: torch.device,
    ):
        self.total_bytes = 0
        self._buffer: torch.Tensor | None = None

        # Group handles by (role, slot_shape). Within a group, all handles
        # share K slots, rotated at runtime by `handle.slot_idx`.
        groups: dict[tuple, list[CotsLinearHandle]] = {}
        for h in handles:
            if h.max_n_prefetch == 0:
                h.w_prefetch_slots = []
                h.prefetch_owner_in_slot = []
                h.prefetch_available_rows_in_slot = []
                continue
            if h.split_axis == INPUT_SPLIT_AXIS:
                # Input-split slot is (max_n_prefetch, out_dim) — matches the
                # unified transposed-storage `w_cpu` layout
                # (n_cpu, out_dim) so H2D `narrow(0, ...)` on both
                # ends is contiguous.
                slot_shape = (h.max_n_prefetch, h.out_dim)
            else:
                slot_shape = (h.max_n_prefetch, h.in_dim)
            groups.setdefault((h.role, slot_shape), []).append(h)

        if not groups:
            return

        dtype = next(iter(groups.values()))[0].dtype
        total_numel = sum(self.K * shape[0] * shape[1] for (_, shape) in groups)
        self._buffer = torch.empty(total_numel, dtype=dtype, device=device)
        self.total_bytes = self._buffer.numel() * self._buffer.element_size()

        offset = 0
        for (_, slot_shape), group_handles in groups.items():
            slot_numel = slot_shape[0] * slot_shape[1]
            shared_slots: list[torch.Tensor] = []
            for _ in range(self.K):
                view = self._buffer[offset : offset + slot_numel].view(*slot_shape)
                shared_slots.append(view)
                offset += slot_numel
            # All handles in this shape group share the SAME K slots —
            # rotation happens at the handle level via `slot_idx`. The
            # owner / available-rows lists are also shared (same Python
            # list object bound to every handle), so start() writes from
            # one handle are visible to all sharers of the physical slot.
            shared_owners: list[CotsLinearHandle | None] = [None] * self.K
            shared_avail: list[int] = [0] * self.K
            for h in group_handles:
                h.w_prefetch_slots = shared_slots
                h.prefetch_owner_in_slot = shared_owners
                h.prefetch_available_rows_in_slot = shared_avail


# ---------------------------------------------------------------------------
# Execution layer: WeightPrefetchStreamer
#
# Layer-ahead H2D streamer. Owns the copy stream, per-layer copy-done events,
# and the slot-rotation policy. `CotsOffloader`'s four `BaseOffloader`
# lifecycle methods delegate to this class. No model knowledge — operates on
# opaque handles.
# Phase 1c does not touch this class; cudaLaunchHostFunc is the runner's
# concern.
# ---------------------------------------------------------------------------
class WeightPrefetchStreamer:
    """K=2 layer-ahead weight-prefetch streamer.

    Methods are 1:1 ports of `prefetch.py`'s lifecycle (copy stream, fork
    event, per-layer copy-done event, capture-vs-eager wait branching) but
    operate on `CotsLinearHandle` directly — no module forward hooks or
    `_ModuleOffloader` indirection. Layer ordering and hooks are owned by
    `CotsOffloader`.
    """

    def __init__(
        self,
        n_layers: int,
        dry_run: bool = False,
    ):
        self.copy_stream = torch.cuda.Stream()
        self._copy_done_events: list[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(n_layers)
        ]
        self._event_valid_for_eager: list[bool] = [False] * n_layers
        self._prefetch_in_capture: list[bool] = [False] * n_layers
        # Public COTS dry-run is a control-plane diagnostic. The streamer
        # skips H2D copies while preserving event ordering and slot metadata;
        # operators skip the prefetched-slice GPU compute contribution.
        self._dry_run = dry_run
        # Cached at the start of every model forward by the first-decoder
        # pre-hook. Read by `start` to size the bucket-specific H2D.
        self.current_bucket: int = 0
        # Owned externally; offloader sets after constructing the pool.
        self.buffer_pool: CotsPrefetchBufferPool | None = None

    def set_current_bucket(
        self, num_tokens: int, bucket_for: Callable[[int], int]
    ) -> None:
        self.current_bucket = bucket_for(num_tokens)

    def start(self, layer_idx: int, handles: list[CotsLinearHandle]) -> None:
        """Issue H2D for this layer's handles using `current_bucket`. One
        memcpy per non-zero handle on `copy_stream`; single event records
        at layer end so `wait(layer_idx)` is one event-sync."""
        b = self.current_bucket
        if not any(h.n_prefetch_by_bucket.get(b, 0) > 0 for h in handles):
            # No-op for this bucket. Clear stale wait state from a previous
            # bucket where this layer DID prefetch — otherwise wait(layer_idx)
            # would sync on a stale done-event under per-bucket Planner
            # strategies.
            self._event_valid_for_eager[layer_idx] = False
            self._prefetch_in_capture[layer_idx] = False
            return

        in_capture = torch.cuda.is_current_stream_capturing()
        self._prefetch_in_capture[layer_idx] = in_capture

        # Fork compute → copy_stream so the H2D is ordered after the
        # producing compute (e.g., last layer's output, mutates_args contract
        # on `start_prefetch`). Mirrors `prefetch.py:577-581`.
        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(self.copy_stream):
            for h in handles:
                n = h.n_prefetch_by_bucket.get(b, 0)
                if n == 0:
                    continue
                # dry_run: keep all bookkeeping (slot tracking, events,
                # fork_event ordering) but skip the actual H2D.
                if not self._dry_run:
                    if h.split_axis == INPUT_SPLIT_AXIS:
                        # Stage 7-C: row-handle `w_cpu` is now stored in
                        # transposed `(n_cpu, out_dim)` layout. The
                        # prefetch source is `w_cpu.narrow(0, 0, n)` —
                        # a contiguous `(n, out_dim)` view, no separate
                        # duplicate buffer. Slot is `(max_n_prefetch,
                        # out_dim)`; both ends narrow on dim 0,
                        # contiguous H2D as Phase 1b's fix required.
                        assert h.w_cpu is not None
                        src = h.w_cpu.narrow(0, 0, n)
                        dst = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n)
                        dst.copy_(src, non_blocking=True)
                    elif h.role == MLP_GATE_UP_ROLE:
                        # MergedCol w_cpu layout is [gate_full | up_full].
                        # Prefetch takes the first n_per_half rows of EACH
                        # half and writes them active-adjacent as
                        # [gate_active | up_active]. This keeps the common
                        # MLP prefetch path to one [gate|up] GEMM even when
                        # f_prefetch < f_cpu_store.
                        assert h.w_cpu is not None
                        n_per_half = n // 2
                        n_cpu_per_half_total = h.n_cpu // 2
                        slot = h.w_prefetch_slots[h.slot_idx]
                        slot[:n_per_half, :].copy_(
                            h.w_cpu[:n_per_half, :], non_blocking=True
                        )
                        slot[n_per_half:n, :].copy_(
                            h.w_cpu[
                                n_cpu_per_half_total : n_cpu_per_half_total
                                + n_per_half,
                                :,
                            ],
                            non_blocking=True,
                        )
                    else:
                        # qkv/wo prefetch takes a contiguous prefix.
                        assert h.w_cpu is not None
                        src = h.w_cpu.narrow(0, 0, n)
                        dst = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n)
                        dst.copy_(src, non_blocking=True)
                # Owner / available_rows on the shape-group-shared
                # metadata. Owner = this handle: lets the operator
                # assert it's reading its own weights, not another
                # layer's that overwrote the shared physical slot via
                # K=2 rotation. available_rows tracks how many leading
                # prefix rows are valid (per-half for MLP gate/up, total for
                # qkv/wo/mlp_down).
                h.prefetch_owner_in_slot[h.slot_idx] = h
                if h.role == MLP_GATE_UP_ROLE:
                    h.prefetch_available_rows_in_slot[h.slot_idx] = n // 2
                else:
                    h.prefetch_available_rows_in_slot[h.slot_idx] = n

        self._copy_done_events[layer_idx].record(self.copy_stream)
        self._event_valid_for_eager[layer_idx] = not in_capture

    def prepare_for_forward_bucket(
        self, layer_idx: int, handles: list[CotsLinearHandle]
    ) -> None:
        """Idempotent boundary repair for `layer_idx` at `current_bucket`.
        Repairs the layer-0 slot relative to the active bucket. For qkv, wo,
        and mlp_down, larger previously-filled prefixes can be reused. For
        mlp_gate_up, the slot is active-adjacent `[gate_active | up_active]`,
        so any active-size change rewrites the full active prefix. Owner
        mismatch is a hard error, except an empty slot (`owner is None`) is
        filled on demand.

        Phase 1c precondition: bucket dispatch is decided by the active
        bucket (capture-time constant), not by slot state. This method
        ensures the slot has enough valid rows for the captured operator
        to read `slot[:n_pref, :]` safely. Runs OUTSIDE captured graphs
        (in the pre-hook for eager; at the graph boundary for capture).
        """
        b = self.current_bucket
        in_capture = torch.cuda.is_current_stream_capturing()
        issued = False
        for h in handles:
            if h.max_n_prefetch == 0:
                continue
            n_pref = h.n_prefetch_by_bucket.get(b, 0)
            required = n_pref // 2 if h.role == MLP_GATE_UP_ROLE else n_pref
            if required == 0:
                continue
            avail = h.prefetch_available_rows_in_slot[h.slot_idx]
            owner = h.prefetch_owner_in_slot[h.slot_idx]
            if owner is not None and owner is not h:
                raise AssertionError(
                    f"prepare_for_forward_bucket: slot owner mismatch on "
                    f"{h.qualified_name} slot {h.slot_idx} (owner={owner})"
                )
            if h.role == MLP_GATE_UP_ROLE:
                needs_fill = owner is None or avail != required
            else:
                needs_fill = owner is None or avail < required
            if not needs_fill:
                continue
            if not issued:
                fork_event = torch.cuda.Event()
                torch.cuda.current_stream().record_event(fork_event)
                self.copy_stream.wait_event(fork_event)
                issued = True
            with torch.cuda.stream(self.copy_stream):
                if not self._dry_run:
                    if h.split_axis == INPUT_SPLIT_AXIS:
                        # Stage 7-C: read from the unified transposed
                        # `w_cpu` instead of the dropped duplicate.
                        assert h.w_cpu is not None
                        start = 0 if owner is None else avail
                        src = h.w_cpu[start:required, :]
                        h.w_prefetch_slots[h.slot_idx][start:required, :].copy_(
                            src, non_blocking=True
                        )
                    elif h.role == MLP_GATE_UP_ROLE:
                        assert h.w_cpu is not None
                        n_cpu_per_half_total = h.n_cpu // 2
                        slot = h.w_prefetch_slots[h.slot_idx]
                        slot[:required, :].copy_(
                            h.w_cpu[:required, :], non_blocking=True
                        )
                        slot[required : 2 * required, :].copy_(
                            h.w_cpu[
                                n_cpu_per_half_total : n_cpu_per_half_total + required,
                                :,
                            ],
                            non_blocking=True,
                        )
                    else:  # qkv/wo
                        assert h.w_cpu is not None
                        start = 0 if owner is None else avail
                        h.w_prefetch_slots[h.slot_idx][start:required, :].copy_(
                            h.w_cpu[start:required, :], non_blocking=True
                        )
                h.prefetch_owner_in_slot[h.slot_idx] = h
                h.prefetch_available_rows_in_slot[h.slot_idx] = required
        if issued:
            self._prefetch_in_capture[layer_idx] = in_capture
            self._copy_done_events[layer_idx].record(self.copy_stream)
            self._event_valid_for_eager[layer_idx] = not in_capture

    def wait(self, layer_idx: int) -> None:
        """Compute stream waits for this layer's prefetch. Branches on
        capture state — port of `prefetch.py:299-329`."""
        if torch.cuda.is_current_stream_capturing():
            if not self._prefetch_in_capture[layer_idx]:
                return
            torch.cuda.current_stream().wait_event(self._copy_done_events[layer_idx])
            self._prefetch_in_capture[layer_idx] = False
        else:
            if self._event_valid_for_eager[layer_idx]:
                torch.cuda.current_stream().wait_event(
                    self._copy_done_events[layer_idx]
                )
            else:
                torch.cuda.current_stream().wait_stream(self.copy_stream)

    def sync_prev_onload(self) -> None:
        """Drain copy_stream into the compute stream — port of
        `prefetch.py:331-338`."""
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    def join_after_forward(self) -> None:
        """Join any layers whose prefetch was started under capture but not
        yet waited — port of `prefetch.py:345-364`. Handles full and
        piecewise CUDA-graph modes."""
        for i, in_capture in enumerate(self._prefetch_in_capture):
            if in_capture:
                torch.cuda.current_stream().wait_event(self._copy_done_events[i])
                self._prefetch_in_capture[i] = False
