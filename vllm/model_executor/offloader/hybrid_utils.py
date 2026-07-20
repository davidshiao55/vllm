# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the Hybrid offloader."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

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


def _has_pinned_host_storage(t: torch.Tensor) -> bool:
    """Storage-level page-locked check (§1c.20 safety belt).

    `Tensor.is_pinned()` reports the metadata bit on a tensor view,
    which can be lost across reinterpret/view operations even when
    the underlying storage IS page-locked. The Triton UVA kernel
    needs the storage to be page-locked, so we drop down to
    `untyped_storage().is_pinned()` if the metadata fast path says
    False. A pageable CPU tensor still returns False; non-CPU
    tensors are rejected immediately.

    NB: in the captured forward, Inductor's functionalization
    allocates a FRESH pageable CPU buffer rather than just dropping
    the metadata bit, so this helper alone does not unblock the
    captured path — that fix is the schema change (§1c.20: drop
    `y_pinned` from `hybrid_submit_gemm.mutates_args`, anchor
    `hybrid_sync_then_uva` on `x_gpu` instead). This helper is the
    safety belt for direct callers that still legitimately pass pinned
    tensors and views thereof.
    """
    if t.device.type != "cpu":
        return False
    if t.is_pinned():
        return True
    try:
        return bool(t.untyped_storage().is_pinned())
    except Exception:  # noqa: BLE001 — defensive: any failure → reject
        return False


def _uva_copy_trusted_host_into_gpu(
    src_pinned: torch.Tensor,
    dst_gpu: torch.Tensor,
) -> None:
    """§1c.20: same as `uva_copy_into_gpu` minus the page-locked
    storage assertion. Used ONLY by `hybrid_sync_then_uva`'s impl —
    the source tensor came from `HybridWeightTaskRunner::y_pinned_view`, which
    builds an `at::from_blob` view over a slab pointer that was
    populated at install time from a real `torch.empty(...,
    pin_memory=True)` allocation. The storage IS page-locked by
    construction; the Tensor metadata bit is unset because
    `at::from_blob` doesn't set it for foreign blobs.

    Inlined rather than calling `uva_copy_into_gpu(...)` so that the
    public helper's strict pinned check stays intact for direct tests and
    utilities, and so `is_pinned()`/storage-level checks aren't pointless
    work on the captured-graph hot path.
    """
    if not dst_gpu.is_cuda:
        raise RuntimeError("dst must be on CUDA")
    # §1c.35 commit-2: src is the slab's pinned view, sized to the
    # bucket capacity (via C++ y_pinned_view's clamp at
    # slab.bucket_capacity_tokens). dst is the operator's view
    # sized to int(y_gpu.shape[0]) which under Inductor
    # specialization may be larger (typically max). The Triton
    # kernel writes `src.numel()` elements into the prefix of dst;
    # tail rows of dst stay stale, harmless because downstream
    # uses only the first `live_count <= bucket` rows.
    if src_pinned.shape[1:] != dst_gpu.shape[1:]:
        raise RuntimeError(
            f"tail-dim mismatch: src={tuple(src_pinned.shape)}, "
            f"dst={tuple(dst_gpu.shape)}"
        )
    if src_pinned.numel() > dst_gpu.numel():
        raise RuntimeError(
            f"src.numel()={src_pinned.numel()} > dst.numel()={dst_gpu.numel()}"
        )
    if src_pinned.dtype != dst_gpu.dtype:
        raise RuntimeError(
            f"dtype mismatch: src={src_pinned.dtype}, dst={dst_gpu.dtype}"
        )
    if not (src_pinned.is_contiguous() and dst_gpu.is_contiguous()):
        raise RuntimeError(
            "_uva_copy_trusted_host_into_gpu requires contiguous tensors"
        )
    if not HAS_TRITON:
        raise RuntimeError(
            "hybrid requires Triton for the UVA activation-return kernel"
        )
    n = src_pinned.numel()
    if n == 0:
        return
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _uva_copy_kernel[grid](src_pinned, dst_gpu, n_elements=n, BLOCK=BLOCK)


def uva_copy_into_gpu(
    src_pinned: torch.Tensor,
    dst_gpu: torch.Tensor,
) -> None:
    """SM-issued copy from pinned-host (UVA) into a GPU buffer.

    Bypasses CE0; runs on the compute SMs, sharing PCIe link
    bandwidth with any concurrent CE0 DMA without queueing behind
    it. **Loss of CE0-bypass is unacceptable** (Phase 1b's measured
    1.85× PCIe BW recovery on row-prefetch depends on it), so the
    captured-forward path also routes through this helper via the
    `hybrid_sync_then_uva` custom op — but the §1c.20 schema fix
    ensures the input arrives as a real pinned-storage view, not an
    Inductor-cloned pageable buffer.
    """
    if not _has_pinned_host_storage(src_pinned):
        raise RuntimeError(
            "src must be pinned host memory (storage-level check; see "
            "phase1c_findings.md §1c.20). A pageable CPU tensor passed "
            "to the UVA kernel reads garbage from device-mapped host "
            "memory."
        )
    if not dst_gpu.is_cuda:
        raise RuntimeError("dst must be on CUDA")
    if src_pinned.shape != dst_gpu.shape:
        src_shape = tuple(src_pinned.shape)
        dst_shape = tuple(dst_gpu.shape)
        raise RuntimeError(f"shape mismatch: src={src_shape}, dst={dst_shape}")
    if src_pinned.dtype != dst_gpu.dtype:
        raise RuntimeError(
            f"dtype mismatch: src={src_pinned.dtype}, dst={dst_gpu.dtype}"
        )
    if not (src_pinned.is_contiguous() and dst_gpu.is_contiguous()):
        raise RuntimeError("uva_copy_into_gpu requires contiguous tensors")
    if not HAS_TRITON:
        raise RuntimeError(
            "hybrid requires Triton for the UVA activation-return kernel"
        )
    n = src_pinned.numel()
    if n == 0:
        return
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _uva_copy_kernel[grid](src_pinned, dst_gpu, n_elements=n, BLOCK=BLOCK)


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
# Storage layer: HybridLinearHandle
#
# One handle per offloaded Linear. Owns the GPU weight slice (replaces
# param.data), the CPU weight slice (`w_cpu`, pinned), the CUDA index
# tensors, and the wrapped weight_loader closure. No execution — operators
# read state from the handle and submit work via a Hybrid runner.
# ---------------------------------------------------------------------------
