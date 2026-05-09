# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""COTS — Collaborative CPU-GPU offloading (thesis backend, Phase 1a).

Splits each WQKV / MLP1 / MLP2 weight along its tensor-parallel-native axis
so a fraction `f_cpu_store` of the bytes is pinned in CPU memory and is
GEMM'd on the CPU each forward pass, in parallel with the GPU's compute on
the GPU-resident slice. WO (`o_proj`) is NOT offloaded.

Phase 1a: no prefetch (`f_prefetch = 0`), strict sequential layer execution,
`enforce_eager=True` required. CPU GEMMs run on a single global
ThreadPoolExecutor(max_workers=1) — equivalent to KTransformers' single-thread
TaskQueue, swappable in Phase 1c for cudaLaunchHostFunc-based C++ binding
without touching the operator-level call sites.

Activation return from CPU uses an SM-issued Triton kernel that reads pinned
host memory via UVA and writes to a pre-allocated GPU buffer; bypasses CE0
(the H2D copy engine) per phase 0 §0.5.5.

Architecture (three layers):
  * **Storage** — `CotsLinearHandle`: per-Linear partition primitive (CPU/GPU
    weight slices, indices, weight_loader closure). No execution.
  * **Execution** — `CpuTaskRunner`: generic CPU work submitter. Phase 1c swap
    target (cudaLaunchHostFunc binding); operator code unchanged.
  * **Operators** — `CotsQKVOp` (per-Linear scatter for QKV), `CotsSwiGLUMLPOp`
    (block-level fused gate_up→SiLU→down). The MLP optimization is inherently
    block-level so we admit it explicitly: linears are storage partitions;
    QKV and MLP are execution operators.

See `David/Docs/implementation_roadmap.md §1a` and `David/Docs/phase0_findings.md`
for the design rationale and empirical numbers.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import prefetch_ops to register the prefetch custom ops at module load time.
# COTS reuses the same ops as `PrefetchOffloader`
# — they dispatch through `get_offloader()` and route to whichever backend
# is registered as the current offloader (`base.py`).
import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import CotsOffloadConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level state: shared single-worker executor (Phase 1c swap target).
# Operator instances reach the offloader's shared buffers via an explicit
# `offloader` reference passed at install time — no module-global lookup —
# so multiple offloader instances can coexist (e.g., generator + verifier
# engines in one process).
# ---------------------------------------------------------------------------
_GLOBAL_EXECUTOR: ThreadPoolExecutor | None = None


def _set_os_thread_name(name: str) -> None:
    """Set the Linux pthread name (visible in Nsight Systems / `top -H`)."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(15, name.encode("utf-8")[:15], 0, 0, 0)  # PR_SET_NAME=15
    except Exception:
        pass  # best-effort


def _get_executor() -> ThreadPoolExecutor:
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cots-cpu",
            initializer=lambda: _set_os_thread_name("cots-cpu"),
        )
    return _GLOBAL_EXECUTOR


# ---------------------------------------------------------------------------
# Triton UVA copy kernel — SM-issued read of pinned host memory + GPU write.
# Bypasses CE0 (the H2D copy engine), so prefetch DMA on CE0 doesn't queue
# behind activation returns (phase 0 §0.5.5: fg_s2c stays ~30 μs across all
# bg DMA chunk sizes).
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _uva_copy_kernel(
        src_ptr,
        dst_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, x, mask=mask)


def uva_copy_into_gpu(
    src_pinned: torch.Tensor,
    dst_gpu: torch.Tensor,
) -> None:
    """SM-issued copy from pinned-host (UVA) into a GPU buffer.

    Bypasses CE0; runs on the compute SMs, sharing PCIe link bandwidth with
    any concurrent CE0 DMA without queueing behind it.
    """
    assert src_pinned.is_pinned(), "src must be pinned host memory"
    assert dst_gpu.is_cuda, "dst must be on CUDA"
    assert src_pinned.shape == dst_gpu.shape, (
        f"shape mismatch: src={tuple(src_pinned.shape)}, dst={tuple(dst_gpu.shape)}"
    )
    assert src_pinned.dtype == dst_gpu.dtype, (
        f"dtype mismatch: src={src_pinned.dtype}, dst={dst_gpu.dtype}"
    )
    assert src_pinned.is_contiguous() and dst_gpu.is_contiguous(), (
        "uva_copy_into_gpu requires contiguous tensors"
    )
    if not HAS_TRITON:
        raise RuntimeError("cots requires Triton for the UVA activation-return kernel")
    n = src_pinned.numel()
    if n == 0:
        return
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _uva_copy_kernel[grid](src_pinned, dst_gpu, n_elements=n, BLOCK=BLOCK)


# ---------------------------------------------------------------------------
# K/V-biased column picker for WQKV. Single source of truth for per-shard
# CPU column counts. See `weight_offload_design.md §WQKV Column Choice`.
# ---------------------------------------------------------------------------
def _qkv_kv_biased_counts(
    q_size: int,
    kv_size: int,
    n_cpu_cols: int,
    *,
    head_dim: int,
    kv_biased: bool = True,
) -> tuple[int, int, int]:
    """Return (n_q_tail, n_k, n_v) — per-shard CPU column counts.

    For `kv_biased=True`, all three of n_q_tail / n_k / n_v are multiples of
    `head_dim`, n_k == n_v (paired KV head groups), and Q tail is whole heads
    (`weight_offload_design.md §201-205`,
    `phase0/bench_split_correctness.py:103,147`). The actual on-CPU count
    `n_q_tail + n_k + n_v` may differ from the requested `n_cpu_cols` due to
    head-boundary snapping.

    For `kv_biased=False`, TP-style proportional split (no head alignment) —
    used as ablation only.
    """
    total = q_size + 2 * kv_size
    if not (0 <= n_cpu_cols <= total):
        raise ValueError(f"n_cpu_cols={n_cpu_cols} out of range [0, {total}]")

    if not kv_biased:
        n_k = round(n_cpu_cols * kv_size / total)
        n_v = round(n_cpu_cols * kv_size / total)
        return (n_cpu_cols - n_k - n_v, n_k, n_v)

    kv_total = 2 * kv_size
    if n_cpu_cols <= kv_total:
        n_kv_heads = kv_size // head_dim
        n_pairs = min(round(n_cpu_cols / (2 * head_dim)), n_kv_heads)
        n_k = n_v = n_pairs * head_dim
        return (0, n_k, n_v)

    n_q_tail_raw = n_cpu_cols - kv_total
    n_q_heads = min(round(n_q_tail_raw / head_dim), q_size // head_dim)
    return (n_q_heads * head_dim, kv_size, kv_size)


def _qkv_kv_biased_indices(
    q_size: int,
    kv_size: int,
    n_cpu_cols: int,
    *,
    head_dim: int,
    kv_biased: bool = True,
) -> torch.Tensor:
    """CPU column indices in `[Q | K | V]` layout. Picks LAST cols of each
    shard (matches TP loader's narrow-on-rank-0-keeps-FIRST-cols).

    Returns indices in row order `[Q_tail (if any), K_cpu, V_cpu]` —
    matching the row layout `_w_cpu` uses.
    """
    n_q_tail, n_k, n_v = _qkv_kv_biased_counts(
        q_size, kv_size, n_cpu_cols, head_dim=head_dim, kv_biased=kv_biased
    )
    idx_q_tail = torch.arange(q_size - n_q_tail, q_size, dtype=torch.long, device="cpu")
    idx_k = torch.arange(
        q_size + kv_size - n_k,
        q_size + kv_size,
        dtype=torch.long,
        device="cpu",
    )
    idx_v = torch.arange(
        q_size + 2 * kv_size - n_v,
        q_size + 2 * kv_size,
        dtype=torch.long,
        device="cpu",
    )
    return torch.cat([idx_q_tail, idx_k, idx_v])


def _complement(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Return [0, n) \\ idx as a sorted LongTensor on CPU.

    Forced device='cpu' because vllm sets the default device to CUDA during
    model construction; we need the indices on CPU to feed load-time
    `index_select` calls on CPU tensors.
    """
    mask = torch.ones(n, dtype=torch.bool, device="cpu")
    mask[idx.cpu()] = False
    return torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.long)


# ---------------------------------------------------------------------------
# Storage layer: CotsLinearHandle
#
# One handle per offloaded Linear. Owns the GPU weight slice (replaces
# param.data), the CPU weight slice (`w_cpu`, pinned), the CUDA index
# tensors, and the wrapped weight_loader closure. No execution — operators
# read state from the handle and submit work via a `CpuTaskRunner`.
# ---------------------------------------------------------------------------
class CotsLinearHandle:
    """Per-Linear partition primitive: storage + load. No execution.

    Construction: use the kind-specific class methods (`for_qkv`, `for_col`,
    `for_row`) which compute snapped n_cpu and pick indices.
    """

    KINDS = ("qkv", "col", "row")

    def __init__(
        self,
        *,
        kind: str,
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
        kv_biased: bool = True,
        # Merged-col-only metadata:
        merged_partition_sizes: tuple[int, int] | None = None,
    ):
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind: {kind}")
        if cpu_indices.numel() != n_cpu:
            raise ValueError(
                f"cpu_indices.numel()={cpu_indices.numel()} != n_cpu={n_cpu}"
            )

        self.kind = kind
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
        if kind == "row":
            self.cpu_in_dim = n_cpu
            self.cpu_out_dim = out_dim
        else:  # col, qkv
            self.cpu_in_dim = in_dim
            self.cpu_out_dim = n_cpu

        # Per-shard split metadata for the loader closures.
        self.q_size = q_size
        self.kv_size = kv_size
        self.head_dim = head_dim
        self.kv_biased = kv_biased
        self.merged_partition_sizes = merged_partition_sizes
        self.n_q_tail = 0
        self.n_k = 0
        self.n_v = 0
        self.n_cpu_per_half = 0
        if kind == "qkv":
            assert q_size is not None and kv_size is not None
            assert head_dim is not None
            self.n_q_tail, self.n_k, self.n_v = _qkv_kv_biased_counts(
                q_size,
                kv_size,
                n_cpu,
                head_dim=head_dim,
                kv_biased=kv_biased,
            )
            assert self.n_q_tail + self.n_k + self.n_v == n_cpu, (
                f"QKV count mismatch at {qualified_name}: n_cpu={n_cpu} != "
                f"sum of (n_q_tail={self.n_q_tail}, n_k={self.n_k}, "
                f"n_v={self.n_v})."
            )
        elif kind == "col":
            assert merged_partition_sizes is not None
            assert merged_partition_sizes[0] == merged_partition_sizes[1], (
                f"MergedColumnParallelLinear's gate/up output partitions "
                f"must be equal-sized; got {merged_partition_sizes}"
            )
            assert n_cpu % 2 == 0, (
                f"col layer expects n_cpu divisible by 2; got n_cpu={n_cpu}"
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
        # shape `(max_n_prefetch, in_dim)` for col/qkv or
        # `(out_dim, max_n_prefetch)` for row, matching `w_cpu`'s layout.
        self.w_prefetch_slots: list[torch.Tensor] = []
        # Shape-group-shared per-slot state — buffer pool binds the SAME
        # list to every handle in a group so writes are visible across
        # all sharers of the physical slot.
        # `prefetch_owner_in_slot[k]`: handle that last filled slot k
        #   (None = empty). Operators assert owner is self before reading.
        # `prefetch_available_rows_in_slot[k]`: how many leading prefix
        #   rows of the slot are valid. Per-half row count for col
        #   (`gate[:a]` AND `up[:a]` valid → available_rows == a); total
        #   prefix rows for qkv/row. 0 = empty.
        self.prefetch_owner_in_slot: list[CotsLinearHandle | None] = []
        self.prefetch_available_rows_in_slot: list[int] = []
        # Row-handle only: pinned-CPU duplicate of the prefetched-cols
        # prefix in transposed layout (max_n_prefetch, out_dim). The
        # primary w_cpu is (out_dim, n_cpu); narrowing it on dim 1
        # yields a pitched (strided) H2D source that costs ~1.85x at
        # f_prefetch=0.15 (microprobe at down_proj shape). This buffer
        # makes the per-bucket H2D fully contiguous. Allocated by the
        # offloader once max_n_prefetch is known; populated by
        # _row_weight_loader. qkv/col prefetch via narrow(0, ...) is
        # already contiguous and needs no duplicate.
        self.w_row_prefetch_src_t: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Construction helpers — compute indices, snap n_cpu, build handle.
    # ------------------------------------------------------------------
    @classmethod
    def for_qkv(
        cls,
        linear: nn.Module,
        qualified_name: str,
        *,
        head_dim: int,
        kv_biased: bool,
        f_cpu_store: float,
    ) -> CotsLinearHandle | None:
        out_dim, in_dim = tuple(linear.weight.shape)
        parts = linear.output_partition_sizes
        assert len(parts) == 3, f"QKV expected 3 partitions, got {parts}"
        q_part, k_part, v_part = parts
        assert k_part == v_part, (
            f"QKV expected k_part == v_part, got k={k_part}, v={v_part}"
        )
        assert k_part % head_dim == 0, (
            f"QKV: kv_size={k_part} not a multiple of head_dim={head_dim}"
        )
        requested = int(round(f_cpu_store * out_dim))
        n_q_tail, n_k, n_v = _qkv_kv_biased_counts(
            q_part,
            k_part,
            requested,
            head_dim=head_dim,
            kv_biased=kv_biased,
        )
        n_cpu = n_q_tail + n_k + n_v
        if n_cpu == 0:
            return None
        cpu_indices = _qkv_kv_biased_indices(
            q_part,
            k_part,
            n_cpu,
            head_dim=head_dim,
            kv_biased=kv_biased,
        )
        return cls(
            kind="qkv",
            linear=linear,
            qualified_name=qualified_name,
            in_dim=in_dim,
            out_dim=out_dim,
            n_cpu=n_cpu,
            cpu_indices=cpu_indices,
            gpu_indices=_complement(cpu_indices, out_dim),
            dtype=linear.weight.dtype,
            q_size=q_part,
            kv_size=k_part,
            head_dim=head_dim,
            kv_biased=kv_biased,
        )

    @classmethod
    def for_col(
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
        n_cpu_per_half = int(round(f_cpu_store * half))
        n_cpu = 2 * n_cpu_per_half
        if n_cpu == 0:
            return None
        # LAST n_cpu_per_half rows of each half. Aligns with TP loader
        # convention (FIRST rows on rank 0 → exactly our GPU portion).
        base = torch.arange(half - n_cpu_per_half, half, dtype=torch.long, device="cpu")
        cpu_indices = torch.cat([base, base + half])
        return cls(
            kind="col",
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
    def for_row(
        cls,
        linear: nn.Module,
        qualified_name: str,
        *,
        f_cpu_store: float,
    ) -> CotsLinearHandle | None:
        out_dim, in_dim = tuple(linear.weight.shape)
        n_cpu = int(round(f_cpu_store * in_dim))
        if n_cpu == 0:
            return None
        # LAST n_cpu input cols. Preserves the MLP1↔MLP2 matched-index
        # invariant under uniform f_cpu_store.
        cpu_indices = torch.arange(
            in_dim - n_cpu, in_dim, dtype=torch.long, device="cpu"
        )
        return cls(
            kind="row",
            linear=linear,
            qualified_name=qualified_name,
            in_dim=in_dim,
            out_dim=out_dim,
            n_cpu=n_cpu,
            cpu_indices=cpu_indices,
            gpu_indices=_complement(cpu_indices, in_dim),
            dtype=linear.weight.dtype,
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
        if self.kind in ("col", "qkv"):
            gpu_slice_shape = (self.out_dim - self.n_cpu, self.in_dim)
            w_cpu_shape = (self.n_cpu, self.in_dim)
        else:  # row
            gpu_slice_shape = (self.out_dim, self.in_dim - self.n_cpu)
            w_cpu_shape = (self.out_dim, self.n_cpu)
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
        """Return the kind-specific weight_loader closure."""
        if self.kind == "row":
            return self._row_weight_loader
        if self.kind == "col":
            return self._merged_col_weight_loader
        return self._qkv_weight_loader

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

        for bucket, (_, f_prefetch) in dispatch_table.items():
            n_pref, pref_idx, cpu_idx = self._compute_bucket_split(f_prefetch)
            self.n_prefetch_by_bucket[bucket] = n_pref
            self.n_cpu_compute_by_bucket[bucket] = self.n_cpu - n_pref
            self.prefetch_indices_cuda_by_bucket[bucket] = pref_idx.to(device)
            self.cpu_compute_indices_cuda_by_bucket[bucket] = cpu_idx.to(device)

        self.max_n_prefetch = (
            max(self.n_prefetch_by_bucket.values()) if dispatch_table else 0
        )

    def _compute_bucket_split(
        self, f_prefetch: float
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Split `cpu_indices` into prefetched and CPU-computed subsets.

        Returns `(n_prefetch, prefetch_indices_cpu, cpu_compute_indices_cpu)`.
        Indices are still on CPU; the caller moves them to the device.

        Layout invariants:
          qkv: `cpu_indices` order is `[Q_tail | K_cpu | V_cpu]`. Prefetch
            takes the first `n_pref` indices.
          col: `cpu_indices` is `[gate_last_n_cpu_per_half | up_last_n_cpu_per_half]`.
            Prefetch takes the FIRST `n_pref_per_half` of each half — keeps
            the matched-index invariant with MLP2's input cols.
          row: `cpu_indices` is the LAST `n_cpu` input cols. Prefetch takes
            the first `n_pref` of those.

        For qkv, n_pref is snapped via the same picker as n_cpu so its
        snap grid matches: at f_prefetch == f_cpu_store the picker sees
        identical input → returns identical (n_q, n_k, n_v) → n_pref ==
        n_cpu by construction, no residual leak. col / row already share
        the same `round(f * dim)` rule for both n_cpu and n_pref so they
        align without extra work.
        """
        cap = self.n_cpu

        if self.kind == "qkv":
            assert self.q_size is not None and self.kv_size is not None
            assert self.head_dim is not None
            requested = int(round(f_prefetch * self.out_dim))
            n_q, n_k, n_v = _qkv_kv_biased_counts(
                self.q_size,
                self.kv_size,
                requested,
                head_dim=self.head_dim,
                kv_biased=self.kv_biased,
            )
            n_pref = min(n_q + n_k + n_v, cap)
        elif self.kind == "col":
            half = self.out_dim // 2
            n_pref_per_half = min(int(round(f_prefetch * half)), self.n_cpu_per_half)
            n_pref = 2 * n_pref_per_half
        else:  # row
            n_pref = min(int(round(f_prefetch * self.in_dim)), cap)

        # Index extraction is per-kind and depends on `cpu_indices`'s layout.
        if self.kind == "col":
            n_pref_per_half = n_pref // 2
            ncph = self.n_cpu_per_half
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
            return n_pref, pref_idx, cpu_idx

        # qkv / row: prefetch is a contiguous prefix of cpu_indices.
        return n_pref, self.cpu_indices[:n_pref], self.cpu_indices[n_pref:]

    # --- Loader closures (per-kind, accessing self by closure) ---

    def _row_weight_loader(self, param, loaded_weight):
        """RowParallelLinear (down_proj): single call, full
        (out_dim, in_dim) loaded_weight. GPU keeps FIRST keep_gpu input cols;
        CPU gets LAST n_cpu.
        """
        assert self.w_cpu is not None
        assert loaded_weight.shape == (self.out_dim, self.in_dim), (
            f"row loader at {self.qualified_name}: expected "
            f"({self.out_dim}, {self.in_dim}), got {tuple(loaded_weight.shape)}"
        )
        keep_gpu = self.in_dim - self.n_cpu
        param.data.copy_(loaded_weight[:, :keep_gpu], non_blocking=False)
        self.w_cpu.copy_(loaded_weight[:, keep_gpu:], non_blocking=False)
        # Phase 1b row-prefetch fix: also populate the transposed pinned
        # H2D source for the prefetched-cols prefix. The first
        # max_n_prefetch input cols of the CPU portion match the runtime
        # prefetch slice (`_compute_bucket_split` row branch), and per-
        # bucket `n_prefetch <= max_n_prefetch`. One-shot transpose is
        # paid here at load time so per-forward H2D is contiguous.
        if self.w_row_prefetch_src_t is not None:
            m = self.max_n_prefetch
            src_block = loaded_weight[:, keep_gpu : keep_gpu + m]  # (out_dim, m)
            self.w_row_prefetch_src_t.copy_(
                src_block.transpose(0, 1).contiguous(),  # (m, out_dim)
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
# Slot shape mirrors `w_cpu`'s layout per kind:
#   col / qkv : (max_n_prefetch, in_dim)   — prefetch dim 0
#   row       : (out_dim, max_n_prefetch)  — prefetch dim 1 (pitched H2D)
# Sized to `max_n_prefetch` (max across buckets); per-forward H2D narrows.
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

        # Group handles by (kind, slot_shape). Within a group, all handles
        # share K slots, rotated at runtime by `handle.slot_idx`.
        groups: dict[tuple, list[CotsLinearHandle]] = {}
        for h in handles:
            if h.max_n_prefetch == 0:
                h.w_prefetch_slots = []
                h.prefetch_owner_in_slot = []
                h.prefetch_available_rows_in_slot = []
                continue
            if h.kind == "row":
                # Phase 1b row-prefetch fix: transposed slot layout so
                # H2D narrow(0, ...) is contiguous (matches the new
                # w_row_prefetch_src_t source). Same numel as the prior
                # (out_dim, max_n_prefetch) shape.
                slot_shape = (h.max_n_prefetch, h.out_dim)
            else:  # col, qkv
                slot_shape = (h.max_n_prefetch, h.in_dim)
            groups.setdefault((h.kind, slot_shape), []).append(h)

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
# and the slot-rotation policy. Sibling of `CpuTaskRunner` in the execution
# layer. `CotsOffloader`'s four `BaseOffloader` lifecycle methods delegate
# to this class. No model knowledge — operates on opaque handles.
# Phase 1c does not touch this class; cudaLaunchHostFunc is `CpuTaskRunner`'s
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
        # Diagnostic: skip the actual H2D copy on the prefetch path while
        # keeping all bookkeeping (events, slot tracking, fork_event). Lets
        # `Bench 2` decompose collaborative-arm overhead into orchestration
        # vs PCIe transfer. Same diagnostic role as `prefetch_defer.py`.
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
                # fork_event ordering) but skip the actual H2D so the
                # measured cost is host orchestration only.
                if not self._dry_run:
                    if h.kind == "row":
                        # Phase 1b row-prefetch fix: source is the pinned
                        # transposed `w_row_prefetch_src_t` of shape
                        # (max_n_prefetch, out_dim); slot is also
                        # (max_n_prefetch, out_dim). Both narrow on dim 0
                        # → contiguous H2D, ~1.85x faster than the prior
                        # pitched `w_cpu.narrow(1, 0, n)`.
                        assert h.w_row_prefetch_src_t is not None, (
                            f"row handle {h.qualified_name} has prefetch "
                            f"requested but w_row_prefetch_src_t is None"
                        )
                        src = h.w_row_prefetch_src_t.narrow(0, 0, n)
                        dst = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n)
                        dst.copy_(src, non_blocking=True)
                    elif h.kind == "col":
                        # MergedCol w_cpu layout is [gate_full | up_full].
                        # Prefetch takes the first n_per_half rows of
                        # EACH half (matched-index invariant). Slot
                        # layout is FIXED-MAX `[gate_max | up_max]`
                        # (Phase 1b → Phase 1c refactor): gate region
                        # is `[0:max_half]`, up region is
                        # `[max_half:2*max_half]`, regardless of the
                        # active bucket. This makes max-fill at
                        # post_init slice-safe and lets the operator
                        # consume any active prefix `n_per_half` from
                        # both regions independently.
                        assert h.w_cpu is not None
                        n_per_half = n // 2
                        max_half = h.max_n_prefetch // 2
                        n_cpu_per_half_total = h.n_cpu // 2
                        slot = h.w_prefetch_slots[h.slot_idx]
                        slot[:n_per_half, :].copy_(
                            h.w_cpu[:n_per_half, :], non_blocking=True
                        )
                        slot[max_half : max_half + n_per_half, :].copy_(
                            h.w_cpu[
                                n_cpu_per_half_total : n_cpu_per_half_total
                                + n_per_half,
                                :,
                            ],
                            non_blocking=True,
                        )
                    else:  # qkv: w_cpu rows are [Q_tail | K | V]; prefetch
                        # takes a contiguous prefix.
                        assert h.w_cpu is not None
                        src = h.w_cpu.narrow(0, 0, n)
                        dst = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n)
                        dst.copy_(src, non_blocking=True)
                # Owner / available_rows on the shape-group-shared
                # metadata. Owner = this handle: lets the operator
                # assert it's reading its own weights, not another
                # layer's that overwrote the shared physical slot via
                # K=2 rotation. available_rows tracks how many leading
                # prefix rows are valid (per-half for col, total for
                # qkv/row).
                h.prefetch_owner_in_slot[h.slot_idx] = h
                if h.kind == "col":
                    h.prefetch_available_rows_in_slot[h.slot_idx] = n // 2
                else:
                    h.prefetch_available_rows_in_slot[h.slot_idx] = n

        self._copy_done_events[layer_idx].record(self.copy_stream)
        self._event_valid_for_eager[layer_idx] = not in_capture

    def prepare_for_forward_bucket(
        self, layer_idx: int, handles: list[CotsLinearHandle]
    ) -> None:
        """Idempotent boundary repair for `layer_idx` at `current_bucket`.
        Copies the missing suffix `[avail:required]` when the slot is
        underfilled relative to the active bucket; no-op when already
        sufficient. Owner mismatch is a hard error.

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
            required = (n_pref // 2) if h.kind == "col" else n_pref
            if required == 0:
                continue
            avail = h.prefetch_available_rows_in_slot[h.slot_idx]
            owner = h.prefetch_owner_in_slot[h.slot_idx]
            if owner is not h:
                raise AssertionError(
                    f"prepare_for_forward_bucket: slot owner mismatch on "
                    f"{h.qualified_name} slot {h.slot_idx} (owner={owner})"
                )
            if avail >= required:
                continue
            if not issued:
                fork_event = torch.cuda.Event()
                torch.cuda.current_stream().record_event(fork_event)
                self.copy_stream.wait_event(fork_event)
                issued = True
            with torch.cuda.stream(self.copy_stream):
                if not self._dry_run:
                    if h.kind == "row":
                        assert h.w_row_prefetch_src_t is not None
                        src = h.w_row_prefetch_src_t[avail:required, :]
                        h.w_prefetch_slots[h.slot_idx][avail:required, :].copy_(
                            src, non_blocking=True
                        )
                    elif h.kind == "col":
                        assert h.w_cpu is not None
                        n_cpu_per_half_total = h.n_cpu // 2
                        max_half = h.max_n_prefetch // 2
                        slot = h.w_prefetch_slots[h.slot_idx]
                        slot[avail:required, :].copy_(
                            h.w_cpu[avail:required, :], non_blocking=True
                        )
                        slot[max_half + avail : max_half + required, :].copy_(
                            h.w_cpu[
                                n_cpu_per_half_total + avail : n_cpu_per_half_total
                                + required,
                                :,
                            ],
                            non_blocking=True,
                        )
                    else:  # qkv
                        assert h.w_cpu is not None
                        h.w_prefetch_slots[h.slot_idx][avail:required, :].copy_(
                            h.w_cpu[avail:required, :], non_blocking=True
                        )
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


# ---------------------------------------------------------------------------
# Execution layer: PythonCotsRunner / NativeCotsRunner
#
# Phase 1a/1b shipped a single `CpuTaskRunner` (Python `ThreadPoolExecutor`
# + `future.result()`). Phase 1c splits this into two runners that share an
# operator-side facade (added incrementally — Stage 2 keeps the legacy
# `submit_with_d2h(fn, *args)` shape on PythonCotsRunner so operators don't
# break; Stage 3 flips operators to the uniform facade and wires
# NativeCotsRunner end-to-end).
#
#   * PythonCotsRunner — the Phase 1a/1b path, retained as a kill-switch
#     under `CotsOffloadConfig.cpu_runner = "python"`. Eager-only;
#     `enforce_eager=False` with python runner is rejected at post_init
#     (Stage 5).
#
#   * NativeCotsRunner — the Phase 1c production path under
#     `cpu_runner = "native"` (default). Wraps the C++ `CotsCpuInfer`
#     (vllm/_cots_C) and dispatches CPU work through cudaLaunchHostFunc
#     onto the current CUDA stream, removing the Python `executor.submit`
#     / `future.result` round-trip and enabling CUDA Graph capture.
#
# `CpuTaskRunner` is kept as a module-level alias of `PythonCotsRunner` for
# any external code that imported the old name.
#
# Strict sequential layer execution invariant means at most one task is
# in flight at any moment.
# ---------------------------------------------------------------------------
class PythonCotsRunner:
    """Phase 1a/1b-shaped CPU task runner — the kill-switch path under
    `CotsOffloadConfig.cpu_runner = "python"`. Body identical to the
    original `CpuTaskRunner`; only the class is renamed for the Phase 1c
    naming convention.

    Eager-only: `ThreadPoolExecutor.submit` is not graph-capturable, so
    selecting this runner with `enforce_eager=False` is a hard error
    (Stage 5 enforces; Stage 2 just plumbs the flag).
    """

    kind = "python"

    def __init__(self, dry_run: bool = False) -> None:
        self._future: Future | None = None
        self._dry_run = dry_run

    def submit_with_d2h(
        self,
        x_gpu: torch.Tensor,
        x_pinned_view: torch.Tensor,
        fn: Callable,
        *args,
    ) -> None:
        """Async D2H of `x_gpu` → `x_pinned_view`, record event, submit
        `fn(event, x_pinned_view, *args)` to the worker.

        The worker function MUST call `event.synchronize()` before reading
        `x_pinned_view` (otherwise the GEMM races the H2D copy).
        """
        x_pinned_view.copy_(x_gpu, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        if self._dry_run:
            fn = _cpu_dryrun_noop
        self._future = _get_executor().submit(fn, event, x_pinned_view, *args)

    def wait(self) -> None:
        """Block until the submitted task completes; re-raises worker errors."""
        assert self._future is not None, "submit_with_d2h() not called"
        self._future.result()
        self._future = None

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

    Stage 2 (this stage) ships the runner class definition and registry
    plumbing only; the legacy `submit_with_d2h(fn, *args)` API is NOT
    implemented because operators still call the old PythonCotsRunner
    shape until Stage 3 flips them to the uniform facade
    `submit_with_d2h(x, x_pinned, y_pinned, op_descriptor)` +
    `wait_and_uva(y_pinned, y_gpu, gpu_anchor_a, gpu_anchor_b)`. Until
    Stage 3 lands, `CotsOffloader.__init__` will reject
    `cpu_runner="native"` so users see a clean error rather than a
    confusing failure during the first forward.

    Multi-engine safety: each instance registers itself in the
    `cots_ops._COTS_RUNNERS` weak registry under a unique runner_id so
    two offloaders (FastTTS gen + ver) coexist with independent slab
    pools. `close()` explicitly drains the worker, then unregisters.
    """

    kind = "native"

    def __init__(self, dry_run: bool = False) -> None:
        # Lazy imports: cots_ops imports _cots_C which is built only on
        # CUDA; users on CPU-only / ROCm builds shouldn't hit ImportError
        # just by importing this module. The registry helpers live in
        # cots_ops alongside the custom-op registration.
        try:
            from vllm import _cots_C  # noqa: F401 — used via attr below
        except ImportError as e:
            raise RuntimeError(
                "NativeCotsRunner requires the `vllm._cots_C` extension, "
                "which builds only on CUDA targets. Either select "
                "`cpu_runner='python'` or rebuild vLLM with CUDA support."
            ) from e
        from vllm.model_executor.offloader import cots_ops

        self._infer = _cots_C.CotsCpuInfer()
        self._dry_run = dry_run
        # Strong reference to the cots_ops module so the registry's weak
        # entries don't get GC-collected while this runner is alive.
        self._cots_ops = cots_ops
        self._runner_id: int = cots_ops._register_runner(self)
        # Stage 3 will populate this from CotsOffloader._build_slab_table.
        # Format: {(layer_idx, bucket, op_kind): task_id}.
        self._task_id_for: dict[tuple[int, int, str], int] = {}
        self._installed: bool = False

    # Stage 2 placeholder methods that satisfy the operator-side typed
    # union `PythonCotsRunner | NativeCotsRunner` without falsely
    # claiming Stage 3 functionality. Operators currently call
    # `submit_with_d2h(fn, *args)` and `wait()` (Phase 1a/1b legacy
    # shape); routing them through NativeCotsRunner is a Stage 3 task
    # that flips the entire surface to the uniform
    # `submit_with_d2h(x, x_pin, y_pin, op_descriptor)` +
    # `wait_and_uva(...)` API. The Stage 2 native rejection in
    # CotsOffloader.__init__ ensures these methods are NEVER reached at
    # runtime. They exist only so mypy sees a consistent interface
    # across the runner union.
    def submit_with_d2h(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "NativeCotsRunner.submit_with_d2h is reserved for Phase 1c "
            "Stage 3, which flips operators to the uniform facade. "
            "Stage 2 rejects cpu_runner='native' at offloader "
            "construction so this should be unreachable. If you see "
            "this at runtime, the Stage 2 guard at "
            "CotsOffloader.__init__ has regressed."
        )

    def wait(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "NativeCotsRunner.wait is reserved for Phase 1c Stage 3."
        )

    def install(
        self,
        n_slabs: int,
        scratch_max_tokens: int,
        scratch_max_intermediate_per_half: int,
    ) -> None:
        """Allocate the C++ slab pool. Called once at offloader install
        time after the slab count + worst-case scratch sizes are known.
        Subsequent `populate_slab_*` calls fill in static fields per slab.
        Idempotent guard: re-install attempts raise.
        """
        if self._installed:
            raise RuntimeError(
                "NativeCotsRunner.install() called twice on the same instance"
            )
        self._infer.install(
            n_slabs=int(n_slabs),
            scratch_max_tokens=int(scratch_max_tokens),
            scratch_max_intermediate_per_half=int(scratch_max_intermediate_per_half),
        )
        self._installed = True

    def close(self) -> None:
        """Drain any in-flight worker task and drop the registry entry.
        Idempotent; safe to call from teardown."""
        try:
            if self._infer is not None:
                self._infer.sync_blocking()
        finally:
            self._cots_ops._unregister_runner(self._runner_id)

    def __del__(self) -> None:
        # Best-effort registry cleanup if the user forgot to call close().
        # Note: don't raise from __del__ — the GC log is unhelpful.
        #
        # FORWARD RISK (review finding #3, Stage 2 sign-off): this path
        # only unregisters; it does NOT drain the CUDA stream of any
        # in-flight `cudaLaunchHostFunc` callbacks scheduled via
        # `submit_on_stream` / `sync_on_stream`. Once Stage 3 wires
        # operators end-to-end, an offloader teardown mid-forward could
        # leave host callbacks pointing at a freed slab. Stage 3 must
        # either (a) add a BaseOffloader-level shutdown hook that drains
        # the compute stream and closes the runner before slabs are
        # freed, OR (b) make this `__del__` best-effort
        # `torch.cuda.current_stream().synchronize()` first (which is
        # also dangerous to do from a finalizer if CUDA is already
        # torn down). Tracked explicitly so it is not forgotten.
        with contextlib.suppress(Exception):
            self._cots_ops._unregister_runner(self._runner_id)


def _make_runner(config: CotsOffloadConfig) -> PythonCotsRunner | NativeCotsRunner:
    """Construct the offloader's single runner per `config.cpu_runner`.

    One runner per offloader (not per operator) — see plan §design-decision
    4 "One offloader-owned runner shared across all operators". This
    moves the Phase 1a/1b pattern of fresh `CpuTaskRunner()` per op
    (cots.py:1752 and :1807 in the legacy code) onto a single shared
    instance, which is the structural prerequisite for the native
    runner's per-offloader slab pool + runner_id.
    """
    # Default fallback "python" matches `CotsOffloadConfig.cpu_runner`
    # (vllm/config/offload.py) through Stage 4. Stage 5 will flip both
    # together once graph capture is verified end-to-end. Picking
    # "native" here would let an old config shim (no `cpu_runner`
    # field) silently route through the unwired native path during
    # Stage 2/3/4 and fail at the operator call site.
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


def _cpu_mlp_block_work(
    d2h_event: torch.cuda.Event,
    x_pinned: torch.Tensor,
    w_gate_cpu: torch.Tensor,
    w_up_cpu: torch.Tensor,
    w_mlp2_cpu: torch.Tensor,
    y2_pinned: torch.Tensor,
) -> None:
    """Fused MLP block: D2H wait → gate / up → SwiGLU → MLP2.

    Gate and up are passed separately so Phase 1b can hand the worker
    non-contiguous CPU-compute slices (each half's prefetched-row prefix
    is excluded). Phase 1a behavior at `f_prefetch=0`: the slices reduce
    to the full halves of `gu.w_cpu`. Matched-index invariant unchanged —
    `gate_out` and `up_out` are worker-local; only `y2_pinned` crosses
    to GPU via UVA.
    """
    d2h_event.synchronize()
    gate_out = F.linear(x_pinned, w_gate_cpu)
    up_out = F.linear(x_pinned, w_up_cpu)
    z = F.silu(gate_out) * up_out
    y2_pinned.copy_(F.linear(z, w_mlp2_cpu))


# ---------------------------------------------------------------------------
# Operator layer
#
# Operators encapsulate the forward semantics for each block of the model
# we offload. They consume `CotsLinearHandle`s for storage and a
# `CpuTaskRunner` for execution.
# ---------------------------------------------------------------------------
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
        assert offloader._x_pinned is not None
        assert offloader._y_pinned is not None
        assert offloader._y_gpu is not None

        h = self._handle
        assert h.w_cpu is not None
        num_tokens = x.shape[0]
        # Active-bucket dispatch: compute shape is decided by the bucket
        # of the CURRENT forward (capture-time constant under Phase 1c
        # graph capture), not by whatever was last filled into the slot.
        # Slot state (owner + available_rows) is asserted as a runtime
        # invariant, not used as a runtime lookup.
        streamer = offloader._streamer
        b = streamer.current_bucket if streamer is not None else None
        if b is None or h.max_n_prefetch == 0:
            n_pref = 0
            n_cpu = h.n_cpu
            cpu_idx = h.cpu_indices_cuda
            pref_idx = cpu_idx  # unused when n_pref == 0
        else:
            n_pref = h.n_prefetch_by_bucket[b]
            n_cpu = h.n_cpu_compute_by_bucket[b]
            pref_idx = h.prefetch_indices_cuda_by_bucket[b]
            cpu_idx = h.cpu_compute_indices_cuda_by_bucket[b]
            if n_pref > 0:
                assert h.prefetch_owner_in_slot[h.slot_idx] is h, (
                    f"slot owner mismatch on {h.qualified_name} slot {h.slot_idx}"
                )
                assert h.prefetch_available_rows_in_slot[h.slot_idx] >= n_pref, (
                    f"slot underfilled on {h.qualified_name}: have "
                    f"{h.prefetch_available_rows_in_slot[h.slot_idx]}, "
                    f"need {n_pref}"
                )

        # CPU compute path skipped when n_cpu_compute == 0 (pure-prefetch).
        y_dst: torch.Tensor | None = None
        if n_cpu > 0:
            x_in = offloader._x_pinned[: num_tokens * h.in_dim].view(
                num_tokens, h.in_dim
            )
            y_out = offloader._y_pinned[: num_tokens * n_cpu].view(num_tokens, n_cpu)
            y_dst = offloader._y_gpu[: num_tokens * n_cpu].view(num_tokens, n_cpu)
            w_cpu_compute = h.w_cpu.narrow(0, n_pref, n_cpu)
            self._runner.submit_with_d2h(
                x, x_in, _cpu_gemm_into_after_event, w_cpu_compute, y_out
            )

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
        if n_pref > 0 and h.w_prefetch_slots:
            slot_view = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n_pref)
            out_pref = F.linear(x, slot_view, None)

        if n_cpu > 0:
            self._runner.wait()
            assert y_dst is not None
            uva_copy_into_gpu(y_out, y_dst)

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
        assert offloader._x_pinned is not None
        assert offloader._y_pinned is not None
        assert offloader._y_gpu is not None

        gu_h = self._gate_up
        dn_h = self._down
        assert gu_h.w_cpu is not None
        assert dn_h.w_cpu is not None
        num_tokens = x.shape[0]
        # Active-bucket dispatch (see CotsQKVOp.apply for rationale).
        # Compute shape from the active bucket; slot state is asserted
        # as a runtime invariant. gu and dn share the active bucket by
        # construction, so no inter-handle bucket-equality check needed.
        streamer = offloader._streamer
        b = streamer.current_bucket if streamer is not None else None
        if b is None or gu_h.max_n_prefetch == 0:
            gu_n_pref = 0
            dn_n_pref = 0
            dn_n_cpu = dn_h.n_cpu
        else:
            gu_n_pref = gu_h.n_prefetch_by_bucket[b]
            dn_n_pref = dn_h.n_prefetch_by_bucket[b]
            dn_n_cpu = dn_h.n_cpu_compute_by_bucket[b]
            if gu_n_pref > 0:
                gu_n_per_half = gu_n_pref // 2
                assert gu_h.prefetch_owner_in_slot[gu_h.slot_idx] is gu_h, (
                    f"slot owner mismatch on {gu_h.qualified_name} slot {gu_h.slot_idx}"
                )
                assert (
                    gu_h.prefetch_available_rows_in_slot[gu_h.slot_idx] >= gu_n_per_half
                ), (
                    f"col slot underfilled on {gu_h.qualified_name}: have "
                    f"{gu_h.prefetch_available_rows_in_slot[gu_h.slot_idx]}, "
                    f"need {gu_n_per_half}"
                )
            if dn_n_pref > 0:
                assert dn_h.prefetch_owner_in_slot[dn_h.slot_idx] is dn_h, (
                    f"slot owner mismatch on {dn_h.qualified_name} slot {dn_h.slot_idx}"
                )
                assert (
                    dn_h.prefetch_available_rows_in_slot[dn_h.slot_idx] >= dn_n_pref
                ), (
                    f"row slot underfilled on {dn_h.qualified_name}: have "
                    f"{dn_h.prefetch_available_rows_in_slot[dn_h.slot_idx]}, "
                    f"need {dn_n_pref}"
                )

        n_pref_per_half = gu_n_pref // 2
        n_cpu_per_half_total = self._n_cpu_per_half  # original count per half

        # CPU compute path — skipped entirely when n_cpu_compute == 0
        # (pure-prefetch case). Without this fast-path the runner / D2H /
        # UVA overhead leaks into the prefetch-only regime.
        y2_pinned: torch.Tensor | None = None
        y2_gpu: torch.Tensor | None = None
        if dn_n_cpu > 0:
            x_pinned = offloader._x_pinned[: num_tokens * self._in_dim].view(
                num_tokens, self._in_dim
            )
            y2_pinned = offloader._y_pinned[: num_tokens * self._out_dim].view(
                num_tokens, self._out_dim
            )
            y2_gpu = offloader._y_gpu[: num_tokens * self._out_dim].view(
                num_tokens, self._out_dim
            )
            # CPU compute slices: gate / up exclude the first `n_pref_per_half`
            # rows of each half (those are prefetched). MLP2's input cols
            # exclude the first `dn_n_pref` cols.
            w_gate_compute = gu_h.w_cpu[n_pref_per_half:n_cpu_per_half_total, :]
            w_up_compute = gu_h.w_cpu[
                n_cpu_per_half_total + n_pref_per_half : 2 * n_cpu_per_half_total, :
            ]
            w_dn_compute = dn_h.w_cpu.narrow(1, dn_n_pref, dn_n_cpu)
            self._runner.submit_with_d2h(
                x,
                x_pinned,
                _cpu_mlp_block_work,
                w_gate_compute,
                w_up_compute,
                w_dn_compute,
                y2_pinned,
            )

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
        # Slot layout (Phase 1b → Phase 1c refactor): col slot is
        # FIXED-MAX `[gate_max | up_max]`. Gate region is `[0:max_half]`,
        # up region is `[max_half:2*max_half]`. The active bucket
        # consumes a per-half prefix `[:n_per_half]` from each region;
        # because gate and up are no longer adjacent in memory, MLP1
        # is two separate F.linear instead of one fused [gate|up] GEMM,
        # and silu*up is done explicitly (math-equivalent to
        # SiluAndMul.forward_native). MLP2/down slot is transposed
        # (Phase 1b row-prefetch fix): shape (dn_n_pref, out_dim).
        if gu_n_pref > 0 and gu_h.w_prefetch_slots and dn_h.w_prefetch_slots:
            n_per_half = gu_n_pref // 2
            max_half = gu_h.max_n_prefetch // 2
            gu_slot = gu_h.w_prefetch_slots[gu_h.slot_idx]
            dn_slot = dn_h.w_prefetch_slots[dn_h.slot_idx]
            gate_w = gu_slot[:n_per_half, :]
            up_w = gu_slot[max_half : max_half + n_per_half, :]
            pref_gate = F.linear(x, gate_w, None)
            pref_up = F.linear(x, up_w, None)
            pref_silu = F.silu(pref_gate) * pref_up
            pref_out = pref_silu.matmul(dn_slot[:dn_n_pref, :])
            out_gpu = pref_out if out_gpu is None else out_gpu.add_(pref_out)

        if dn_n_cpu > 0:
            self._runner.wait()
            assert y2_pinned is not None and y2_gpu is not None
            uva_copy_into_gpu(y2_pinned, y2_gpu)
            # When CPU is the sole contributor, clone — y2_gpu is a shared
            # activation buffer and would be clobbered by the next layer.
            out_gpu = y2_gpu.clone() if out_gpu is None else out_gpu.add_(y2_gpu)

        assert out_gpu is not None, "MLP block has no active path"
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
class CotsOffloader(BaseOffloader):
    """Collaborative CPU-GPU weight offloader (thesis Phase 1a).

    Three-pass per layer in `wrap_modules`:
      1. Build & install handles for each offloadable Linear.
      2. Install operator adapters: `CotsQKVOp` per QKV linear,
         `CotsSwiGLUMLPOp` per recognized MLP block (replaces parent.forward,
         installs `_RaiseOnDirectCall` guards on the MLP linears).
      3. Reject orphan col/row handles loudly (Phase 1a contract:
         MergedCol/Row offload requires an MLP block parent).
    """

    # `o_proj` intentionally absent — WO is GPU-resident in Phase 1/2 per
    # `weight_offload_design.md §WO Split Axis Decision`.
    _OFFLOAD_SUFFIXES = ("qkv_proj", "gate_up_proj", "down_proj")

    def __init__(
        self,
        config: CotsOffloadConfig,
        dispatch_table_factory: Callable[[list[int]], dict[int, tuple[float, float]]]
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
        if self.f_cpu_store > 0.0:
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
        self._capture_buckets: list[int] = []
        self._max_num_tokens: int = 0
        self._eager_fallback_entry: tuple[float, float] = (0.0, 0.0)

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
        self._runner: PythonCotsRunner | NativeCotsRunner | None = None
        if self.f_cpu_store > 0.0:
            # Stage 2 guard, BEFORE _make_runner so we don't construct
            # (and register, and spawn a C++ worker for) a NativeCotsRunner
            # only to throw it away. Constructing first would (a) pollute
            # the runner registry until GC, and (b) on non-CUDA builds,
            # raise an `_cots_C` import error that masks the intended
            # Stage-3 message. The native runner's operator-side wiring
            # lands in Stage 3 (uniform `submit_with_d2h(x, x_pin, y_pin,
            # op_descriptor)` + `wait_and_uva` + slab population); until
            # then, falling through to operator code that calls the
            # legacy `submit_with_d2h(fn, *args)` shape would just
            # AttributeError mid-forward. Fail loudly here instead.
            if getattr(config, "cpu_runner", "python") == "native":
                raise NotImplementedError(
                    "cots: cpu_runner='native' is reserved for Phase 1c "
                    "Stage 3+ where operators flip to the uniform facade. "
                    "Stage 2 has only stood up the substrate (cots_ops.py, "
                    "NativeCotsRunner class, runner registry, installer "
                    "refactor); operator call sites still expect the "
                    "PythonCotsRunner legacy shape. Set "
                    "cpu_runner='python' for now, OR wait for Stage 3 to "
                    "land before switching the default."
                )
            self._runner = _make_runner(config)

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

        # Allocate shared buffers AFTER all handles are registered (so we
        # know the worst-case shapes). All GPU allocations stay inside this
        # method (DeviceMemoryProfiler context).
        self._allocate_activation_buffers()

        # Phase 1b: build dispatch table, populate per-handle prefetch
        # geometry, and (if f_prefetch > 0) allocate the streamer + buffer
        # pool and install layer-level prefetch hooks. Phase 1a (f_prefetch=0)
        # leaves all of this no-op'd.
        self._build_dispatch_table()
        for h in self._handles:
            h.apply_prefetch_split_per_bucket(self._dispatch_table)
        # Install based on effective dispatch table, not config knob — a
        # Planner-emitted table can request prefetch even when config
        # f_prefetch == 0.
        if any(h.max_n_prefetch > 0 for h in self._handles):
            self._install_prefetch_machinery()

        logger.info_once(
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
            scope="local",
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
                    f"CotsOffloader (Phase 1a) only supports unquantized "
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
                    child, qualified_name, f_cpu_store=self.f_cpu_store
                )
            elif isinstance(child, RowParallelLinear):
                handle = CotsLinearHandle.for_row(
                    child, qualified_name, f_cpu_store=self.f_cpu_store
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
        """Phase 1a contract: TP=1 only. The loader closures assert full
        unsharded `loaded_weight` shapes (no per-rank narrow); native vLLM
        loaders narrow by TP rank before copying. Cleanly fail at wrap time
        rather than mismatch in a loader closure later.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if tp_size != 1:
            raise RuntimeError(
                f"CotsOffloader (Phase 1a) requires tensor_parallel_size=1; "
                f"got tp_size={tp_size}. Multi-rank TP is out of scope for "
                f"Phase 1a (loader closures assume full unsharded weights)."
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
        """Phase 1a contract: BF16-only (`offload.py` `cpu_dtype`). oneDNN
        BF16 is the only fast path on CPU per phase 0 §0.3.2 — torch.mm with
        FP16/FP32 falls back to scalar (100× slower).
        """
        if linear.weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"CotsOffloader (Phase 1a) requires bfloat16; "
                f"{qualified_name} has dtype={linear.weight.dtype}. "
                f"Launch with --dtype bfloat16."
            )

    @staticmethod
    def _check_no_orphan_col_row(handles: list[CotsLinearHandle]) -> None:
        """Phase 1a contract: every col/row handle must be in a fused block."""
        for h in handles:
            if h.kind in ("col", "row") and not h.in_block:
                raise RuntimeError(
                    f"cots Phase 1a: {h.qualified_name} is offloaded but not "
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

        # Row-handle only: allocate the transposed pinned-CPU prefetch
        # source. See CotsLinearHandle.w_row_prefetch_src_t for the
        # contiguous-vs-pitched-H2D rationale. Skipped when
        # max_n_prefetch == 0 so f_prefetch=0.0 is bit-exact to Phase 1a.
        for h in self._handles:
            if h.kind == "row" and h.max_n_prefetch > 0:
                h.w_row_prefetch_src_t = torch.empty(
                    (h.max_n_prefetch, h.out_dim),
                    dtype=h.dtype,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )

        n_layers = len(self._layer_modules)
        self._streamer = WeightPrefetchStreamer(n_layers=n_layers, dry_run=self.dry_run)
        self._streamer.buffer_pool = self._prefetch_buffer_pool

        for i, layer in enumerate(self._layer_modules):
            self._hook_layer_forward(i, layer)

        if self._layer_modules:
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

        Layer 0 is the only slot consumed before the current forward can
        issue a prefetch for it; all other layers are prefetched by their
        predecessor's pre-compute hook. The same repair is called at the
        Phase 1c FULL CUDA graph boundary.
        """
        del module
        anchor = args[0] if args else next(iter(kwargs.values()))
        self.prepare_before_forward(anchor.shape[0])

    def _bucket_for(self, num_tokens: int) -> int:
        """Bisect-up lookup on `_capture_buckets`. Returns the bucket key
        (matches `lookup_dispatch`'s rounding semantics, `planner_design.md
        §4.5`). Out-of-range returns the largest captured bucket."""
        from bisect import bisect_left

        i = bisect_left(self._capture_buckets, num_tokens)
        if i >= len(self._capture_buckets):
            return self._capture_buckets[-1]
        return self._capture_buckets[i]

    # --- BaseOffloader lifecycle delegation ---

    def prepare_before_forward(self, num_tokens: int) -> None:
        """Repair active-bucket state before a forward starts.

        This is deliberately limited to layer 0. Steady-state next-layer
        prefetches are emitted inside each layer wrapper so FULL CUDA graph
        capture records them as graph nodes rather than relying on replay-time
        Python state.
        """
        if self._streamer is None:
            return
        self._streamer.set_current_bucket(num_tokens, self._bucket_for)
        if self._layer_handles:
            self._streamer.prepare_for_forward_bucket(0, self._layer_handles[0])

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
        self._capture_buckets = sorted(set(capture_sizes))

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

    # --- post_init: bookkeeping only ---

    def post_init(self) -> None:
        """Verify enforce_eager and finalize bookkeeping. The dispatch table
        and per-bucket geometry are built in `wrap_modules` (Phase 1b — they
        must exist before the prefetch buffer pool is sized inside the
        DeviceMemoryProfiler context)."""
        if not self._handles:
            return
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if not vllm_config.model_config.enforce_eager:
            raise RuntimeError(
                "CotsOffloader requires enforce_eager=True. "
                "CUDA graph capture is deferred to Phase 1c (was Phase 4)."
            )

        self._eager_fallback_entry = self._dispatch_table[self._capture_buckets[-1]]

        # Post-init max-fill for layer 0. It is consumed before the current
        # forward can issue any pre-compute prefetch; every later layer is
        # prefetched by its predecessor's wrapper. Fill to max_n_prefetch
        # (per-half for col, total for qkv/row); active-bucket dispatch
        # consumes only the needed prefix.
        if self._streamer is not None and self._layer_handles:
            fork_event = torch.cuda.Event()
            torch.cuda.current_stream().record_event(fork_event)
            self._streamer.copy_stream.wait_event(fork_event)
            with torch.cuda.stream(self._streamer.copy_stream):
                for h in self._layer_handles[0]:
                    if h.max_n_prefetch == 0:
                        continue
                    if h.kind == "col":
                        assert h.w_cpu is not None
                        max_half = h.max_n_prefetch // 2
                        n_cpu_per_half_total = h.n_cpu // 2
                        slot = h.w_prefetch_slots[h.slot_idx]
                        slot[:max_half, :].copy_(
                            h.w_cpu[:max_half, :], non_blocking=True
                        )
                        slot[max_half : 2 * max_half, :].copy_(
                            h.w_cpu[
                                n_cpu_per_half_total : n_cpu_per_half_total + max_half,
                                :,
                            ],
                            non_blocking=True,
                        )
                        h.prefetch_available_rows_in_slot[h.slot_idx] = max_half
                    elif h.kind == "row":
                        assert h.w_row_prefetch_src_t is not None
                        m = h.max_n_prefetch
                        h.w_prefetch_slots[h.slot_idx][:m, :].copy_(
                            h.w_row_prefetch_src_t[:m, :],
                            non_blocking=True,
                        )
                        h.prefetch_available_rows_in_slot[h.slot_idx] = m
                    else:  # qkv
                        assert h.w_cpu is not None
                        m = h.max_n_prefetch
                        h.w_prefetch_slots[h.slot_idx][:m, :].copy_(
                            h.w_cpu[:m, :], non_blocking=True
                        )
                        h.prefetch_available_rows_in_slot[h.slot_idx] = m
                    h.prefetch_owner_in_slot[h.slot_idx] = h
            self._streamer.copy_stream.synchronize()

        total_offloaded = sum(
            h.w_cpu.numel() * h.w_cpu.element_size()
            for h in self._handles
            if h.w_cpu is not None
        )
        assert self._x_pinned is not None
        assert self._y_pinned is not None
        assert self._y_gpu is not None

        # Prefetch summary (zero / disabled when f_prefetch == 0).
        if self._prefetch_buffer_pool is not None:
            prefetch_gb = self._prefetch_buffer_pool.total_bytes / 1e9
            f_pref_per_bucket = sorted(
                {f_pref for _, f_pref in self._dispatch_table.values()}
            )
        else:
            prefetch_gb = 0.0
            f_pref_per_bucket = [0.0]

        logger.info(
            "[CotsOffloader] Initialized: %d offloaded linears, "
            "%d fused MLP blocks, "
            "GPU memory saved (weights): %.4f GB, "
            "shared activation buffers: %.4f GB pinned input + "
            "%.4f GB pinned output + %.4f GB GPU UVA-dest, "
            "prefetch buffer pool: %.4f GB (K=%d, f_prefetch values: %s), "
            "dispatch buckets: %s",
            len(self._handles),
            len(self._fused_ops),
            total_offloaded / 1e9,
            self._x_pinned.numel() * self._x_pinned.element_size() / 1e9,
            self._y_pinned.numel() * self._y_pinned.element_size() / 1e9,
            self._y_gpu.numel() * self._y_gpu.element_size() / 1e9,
            prefetch_gb,
            CotsPrefetchBufferPool.K,
            f_pref_per_bucket,
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
        logger.info(
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
        """
        from bisect import bisect_left

        i = bisect_left(self._capture_buckets, num_tokens)
        if i >= len(self._capture_buckets):
            return self._eager_fallback_entry
        return self._dispatch_table[self._capture_buckets[i]]
