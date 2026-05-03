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

from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Execution layer: CpuTaskRunner
#
# Generic CPU work submitter. Phase 1a: thin wrapper around the global
# ThreadPoolExecutor + a CUDA event handshake for D2H ordering.
# Phase 1c (was Phase 4): replace internals with cudaLaunchHostFunc binding
# to a C++ CPUInfer (kt-kernel/cpu_backend/cpuinfer.h:78-116). Operator code
# uses `submit_with_d2h` / `wait` and is unaffected by the swap.
#
# Strict sequential layer execution invariant means at most one task is
# in flight at any moment, so a single _future field per runner suffices.
# ---------------------------------------------------------------------------
class CpuTaskRunner:
    """Generic CPU-side task runner. Phase 1c swap target."""

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
    w_mlp1_cpu: torch.Tensor,
    w_mlp2_cpu: torch.Tensor,
    n_cpu_per_half: int,
    y2_pinned: torch.Tensor,
) -> None:
    """Fused MLP block: D2H wait → MLP1 → SwiGLU → MLP2.

    Matched-index invariant: CPU keeps `y1` and `z` locally — only `y2_pinned`
    crosses to GPU via UVA. `y1` / `z` are worker-local transients (PyTorch
    CPU caching allocator); not pinned and not pre-allocated. Phase 1c-
    compatible: only `x_pinned` / `y2_pinned` / weights need stable addresses.
    """
    d2h_event.synchronize()
    y1 = F.linear(x_pinned, w_mlp1_cpu)
    z = F.silu(y1[:, :n_cpu_per_half]) * y1[:, n_cpu_per_half:]
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
        runner: CpuTaskRunner,
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
        # Phase 1a invariant: dispatch lookup must yield f_prefetch=0.
        _, f_prefetch = offloader.lookup_dispatch(num_tokens)
        assert f_prefetch == 0.0, (
            f"Phase 1a: f_prefetch must be 0; got {f_prefetch} for {h.qualified_name}"
        )

        x_in = offloader._x_pinned[: num_tokens * h.in_dim].view(num_tokens, h.in_dim)
        y_out = offloader._y_pinned[: num_tokens * h.cpu_out_dim].view(
            num_tokens, h.cpu_out_dim
        )
        y_dst = offloader._y_gpu[: num_tokens * h.cpu_out_dim].view(
            num_tokens, h.cpu_out_dim
        )

        self._runner.submit_with_d2h(
            x, x_in, _cpu_gemm_into_after_event, h.w_cpu, y_out
        )
        out_gpu_slice = F.linear(x, layer.weight, None)
        self._runner.wait()
        uva_copy_into_gpu(y_out, y_dst)

        out = _scatter_col_outputs(out_gpu_slice, y_dst, h, num_tokens)
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
        runner: CpuTaskRunner,
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

        num_tokens = x.shape[0]
        x_pinned = offloader._x_pinned[: num_tokens * self._in_dim].view(
            num_tokens, self._in_dim
        )
        y2_pinned = offloader._y_pinned[: num_tokens * self._out_dim].view(
            num_tokens, self._out_dim
        )
        y2_gpu = offloader._y_gpu[: num_tokens * self._out_dim].view(
            num_tokens, self._out_dim
        )

        self._runner.submit_with_d2h(
            x,
            x_pinned,
            _cpu_mlp_block_work,
            self._gate_up.w_cpu,
            self._down.w_cpu,
            self._n_cpu_per_half,
            y2_pinned,
        )
        # GPU MLP block on its weight slices. F.linear bypasses the linears'
        # (raise-guarded) quant_method.apply.
        gpu_mlp1 = F.linear(x, self._gate_up_layer.weight, None)
        gpu_silu = self._act_fn(gpu_mlp1)
        out_gpu = F.linear(gpu_silu, self._down_layer.weight, None)

        self._runner.wait()
        uva_copy_into_gpu(y2_pinned, y2_gpu)
        out_gpu.add_(y2_gpu)
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


def _scatter_col_outputs(
    out_gpu_slice: torch.Tensor,
    out_cpu_on_gpu: torch.Tensor,
    handle: CotsLinearHandle,
    num_tokens: int,
) -> torch.Tensor:
    """Combine GPU and CPU column slices into the canonical layer output."""
    assert handle.gpu_indices_cuda is not None
    assert handle.cpu_indices_cuda is not None
    out = torch.empty(
        (num_tokens, handle.out_dim),
        dtype=out_gpu_slice.dtype,
        device=out_gpu_slice.device,
    )
    out.index_copy_(1, handle.gpu_indices_cuda, out_gpu_slice)
    out.index_copy_(1, handle.cpu_indices_cuda, out_cpu_on_gpu)
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

    def __init__(self, config: CotsOffloadConfig):
        self.config = config
        self.f_cpu_store = float(config.f_cpu_store)
        self.kv_biased = bool(config.kv_biased)
        self.dry_run = bool(config.dry_run)
        if not (0.0 <= self.f_cpu_store <= 1.0):
            raise ValueError(f"f_cpu_store must be in [0, 1], got {self.f_cpu_store}")
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

        # Shared activation I/O buffers — allocated at the end of
        # wrap_modules (so vLLM's DeviceMemoryProfiler sees them in
        # `model_memory_usage`). Flat 1D so per-forward views are always
        # contiguous regardless of which sub-module is active.
        self._x_pinned: torch.Tensor | None = None
        self._y_pinned: torch.Tensor | None = None
        self._y_gpu: torch.Tensor | None = None

        # Dispatch table populated in post_init.
        self._dispatch_table: dict[int, tuple[float, float]] = {}
        self._capture_buckets: list[int] = []
        self._max_num_tokens: int = 0
        self._eager_fallback_entry: tuple[float, float] = (0.0, 0.0)

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

        modules: list[nn.Module] = []
        for layer in modules_generator:
            modules.append(layer)
            if self.f_cpu_store == 0.0:
                continue
            layer_handles = self._build_handles(layer)
            self._install_qkv_ops(layer_handles)
            self._install_mlp_ops(layer, layer_handles)
            self._check_no_orphan_col_row(layer_handles)

        if self.f_cpu_store == 0.0:
            logger.info_once(
                "CotsOffloader: f_cpu_store=0, no offloading.", scope="local"
            )
            return modules

        # Allocate shared buffers AFTER all handles are registered (so we
        # know the worst-case shapes). All GPU allocations stay inside this
        # method (DeviceMemoryProfiler context).
        self._allocate_activation_buffers()
        logger.info_once(
            "CotsOffloader: wrapped %d linear modules and %d fused MLP blocks "
            "(f_cpu_store=%.4f, kv_biased=%s, cpu_num_threads=%d, dry_run=%s).",
            len(self._handles),
            len(self._fused_ops),
            self.f_cpu_store,
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
        for h in handles:
            if h.kind != "qkv":
                continue
            h.linear.quant_method = CotsQKVOp(
                handle=h,
                runner=CpuTaskRunner(dry_run=self.dry_run),
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

            mlp_op = CotsSwiGLUMLPOp(
                gate_up_layer=gu,
                down_layer=dp,
                gate_up_handle=gu_h,
                down_handle=dp_h,
                act_fn=af,
                runner=CpuTaskRunner(dry_run=self.dry_run),
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

    # --- Activation buffer allocation ---

    def _allocate_activation_buffers(self) -> None:
        """Allocate `_x_pinned`, `_y_pinned`, `_y_gpu` sized to the worst-case
        across registered handles. Flat 1D backing storage.
        """
        if not self._handles:
            return
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        device = torch.device("cuda")
        max_in_dim = max(h.cpu_in_dim for h in self._handles)
        max_cpu_out_dim = max(h.cpu_out_dim for h in self._handles)
        self._max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
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
        """Verify enforce_eager and build the per-bucket dispatch table.

        No GPU allocations or data movement — those happen in `wrap_modules`
        so vLLM's DeviceMemoryProfiler sees them in `model_memory_usage`.
        """
        if not self._handles:
            return
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if not vllm_config.model_config.enforce_eager:
            raise RuntimeError(
                "CotsOffloader (Phase 1a) requires enforce_eager=True. "
                "CUDA graph capture is deferred to Phase 1c (was Phase 4)."
            )

        capture_sizes = list(
            vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
        if not capture_sizes:
            capture_sizes = [self._max_num_tokens]
        self._capture_buckets = sorted(set(capture_sizes))
        self._dispatch_table = {
            b: (self.f_cpu_store, 0.0) for b in self._capture_buckets
        }
        self._eager_fallback_entry = (self.f_cpu_store, 0.0)

        for bucket, (_, f_prefetch) in self._dispatch_table.items():
            assert f_prefetch == 0.0, (
                f"Phase 1a: bucket {bucket} has non-zero f_prefetch "
                f"({f_prefetch}). Prefetch is Phase 1b."
            )

        total_offloaded = sum(
            h.w_cpu.numel() * h.w_cpu.element_size()
            for h in self._handles
            if h.w_cpu is not None
        )
        assert self._x_pinned is not None
        assert self._y_pinned is not None
        assert self._y_gpu is not None
        logger.info(
            "[CotsOffloader] Initialized: %d offloaded linears, "
            "%d fused MLP blocks, "
            "GPU memory saved (weights): %.4f GB, "
            "shared activation buffers: %.4f GB pinned input + "
            "%.4f GB pinned output + %.4f GB GPU UVA-dest, "
            "dispatch buckets: %s",
            len(self._handles),
            len(self._fused_ops),
            total_offloaded / 1e9,
            self._x_pinned.numel() * self._x_pinned.element_size() / 1e9,
            self._y_pinned.numel() * self._y_pinned.element_size() / 1e9,
            self._y_gpu.numel() * self._y_gpu.element_size() / 1e9,
            self._capture_buckets,
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
