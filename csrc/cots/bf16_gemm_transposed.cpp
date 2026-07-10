// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Stage 7 — Custom BF16 GEMM kernel for the row-major
// `(K, N)`-weight layout that oneDNN's bf16:bf16 AVX2 dispatch
// does not provide.
//
// `y[M, N] = x[M, K] @ w[K, N]`, all BF16, all row-major
// contiguous. Single translation unit, no JIT, no external
// dependency beyond ATen for the at::Tensor wrapper.
//
// What this kernel IS:
//   * **Inspired by** oneDNN's AVX2 `f32:bf16` upconversion/FMA
//     inner sequence
//     (`oneDNN/src/cpu/x64/brgemm/jit_brgemm_kernel.cpp:2880-2888`):
//       - Load 8 BF16 values, expand to FP32 via
//         `vpmovzxwd + vpslld 16`.
//       - FMA into the FP32 accumulator with `vfmadd231ps`.
//   * **Inspired by** oneDNN's `parallel(nthr, lambda)`
//     work-split pattern (`oneDNN/src/common/dnnl_thread.hpp:270`),
//     but realised via `at::parallel_for` so PyTorch's executor
//     (OMP or TBB) and nested-parallel guards apply.
//
// What this kernel is NOT:
//   * **Not a oneDNN BRGEMM port.** oneDNN's actual matmul kernel
//     is JIT-compiled assembly (Xbyak), with packing primitives,
//     post-ops (bias, scaling, eltwise), runtime ISA dispatch
//     (sse41/avx2/avx512/amx), AMX tile programming, blocked
//     `n16c` weight layouts, and a fully tuned register
//     allocator. We do none of that — this is ~500 LOC of plain
//     C++ + intrinsics, fixed-shape contiguous BF16 in/out, no
//     post-ops, no quantization scales, no AVX-512 / AMX paths,
//     no runtime ISA dispatch (the AVX2+FMA gate is a compile-
//     time `#if`).
//   * **Not a general-purpose GEMM.** It is specialized for COTS'
//     small-token BF16 down-projection slices.
//   * **Not a replacement for `at::linear`.** Only beats
//     `at::linear` on the specific `(K, N)` row-major-weight
//     layout because oneDNN has no bf16:bf16 AVX2 dispatch for
//     this layout (`brgemm_utils.cpp:150-156`).
//
// Why this exists (Stage 7 motivation):
// Phase 1b's row-prefetch fix kept a duplicate ~1 GiB pinned
// `w_row_prefetch_src_t` transposed buffer in the `(n_cpu_total,
// out_dim)` layout for contiguous H2D, while the CPU compute path
// kept a separate `(out_dim, n_cpu_total)` row-major copy purely
// because `at::linear`'s fast dispatch wants that layout. This
// kernel runs the CPU compute directly on the same transposed
// `(K, N)` buffer, unblocking Stage 7-C (drop the duplicate, save
// ~1 GiB pinned at `f_prefetch=0.30`).
//
// Limitations / scope:
//   * AVX2 + FMA only in the vLLM CMake target, which
//     force-compiles this TU with `-mavx2 -mfma`. The scalar branch
//     below is a source-local reference path for non-standard direct
//     compilation; it is not used by the production thesis extension.
//   * Output dtype is BF16. FP32 → BF16 conversion uses **RNE**
//     (round-to-nearest-even) matching the semantics of the
//     AVX-512 instruction `vcvtneps2bf16` — emulated in AVX2 via
//     `(bits + 0x7FFF + ((bits>>16) & 1)) >> 16`. See
//     `f32_to_bf16_rne` below.
//   * Intra-op parallelism via `at::parallel_for` on the outer
//     N-tile loop. Honors `at::set_num_threads(...)` which is
//     how the COTS worker is configured per bucket (§1c.04
//     bucket-aware policy).
//   * No bias, no post-ops, no quantization scales. The COTS
//     down-proj path is plain `y = x @ w` with the worker
//     copying into `y_pinned` after this kernel returns.
//
// Blocking strategy (oneDNN-inspired, but everything fits in
// AVX2 registers — no JIT, no FP32 accumulator buffer):
//   * `Nb = 16` (two AVX2 8-wide lanes per N-block).
//   * Register tile `M_TILE × N_INNER` chosen per M so the
//     `M_TILE × N_INNER × 2` accumulator footprint plus 2 ymm
//     for w-loads and 1 ymm for x-broadcast stays in the
//     16-register AVX2 budget:
//       - M=1 → N_INNER=4 (N_tile=64). Wins by sharing one
//         x-broadcast across 4 nb-tiles.
//       - M=2 → N_INNER=2 (N_tile=32). Hybrid.
//       - M=3 → N_INNER=2 (N_tile=32). Fused remainder path;
//         avoids reading the weight matrix three separate times.
//       - M=4 → N_INNER=1 (N_tile=Nb=16). Wins by **M-fusion**:
//         each w cache line is FMA'd against 4 m-rows in one
//         pass → 4× DRAM-traffic reduction vs a per-m outer
//         loop.
//       - Larger M → groups of four plus one fused 2/3-token
//         remainder. A one-token remainder keeps the conservative
//         N_INNER=1 path.
//   * `K` is the reduction dim, walked element-by-element with
//     **software prefetch 24 K-rows ahead** to overlap DRAM
//     latency with FMA compute. No K-blocking because the
//     accumulator fits entirely in registers for the M/N tile
//     sizes above.

#include <ATen/ATen.h>

#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define VLLM_COTS_HAS_AVX2_FMA 1
#else
  #define VLLM_COTS_HAS_AVX2_FMA 0
#endif

// Intra-op parallelism via ATen's executor — `at::parallel_for`
// (`ATen/Parallel-inl.h`). Inspired by oneDNN's `parallel(nthr,
// lambda)` pattern (`oneDNN/src/common/dnnl_thread.hpp:270`) but
// routed through PyTorch so:
//   * `at::set_num_threads(n)` is honored (this is how the COTS
//     worker is configured per bucket, §1c.04).
//   * Nested-parallel-region guards are handled by ATen via
//     `at::in_parallel_region()` (line 25 of `Parallel-inl.h`):
//     if a caller already holds a parallel region open, ATen
//     serializes our work instead of forking a second OMP team.
//   * No bare `#pragma omp` in the hot path — the threading
//     mechanism is selected at PyTorch build time.
//
// IMPORTANT (build dependency): `at::parallel_for` only takes the
// parallel path when our TU's `_OPENMP` macro is defined (this
// gates `INTRA_OP_PARALLEL` in `ATen/ParallelOpenMP.h`). Without
// `-fopenmp` on our compile, `parallel_for` silently falls
// through to a serial path — observed ~5× slowdown empirically.
// CMakeLists.txt adds `-fopenmp` to this source for that reason;
// the OMP runtime (libgomp) is already linked transitively via
// libtorch_cpu.
#include <ATen/Parallel.h>

namespace vllm {
namespace cots {

#if VLLM_COTS_HAS_AVX2_FMA

namespace {

// BF16 → FP32 for a single scalar. BF16 = upper 16 bits of FP32,
// so the conversion is a left shift by 16. Reinterpret cast keeps
// the bit pattern.
inline float bf16_to_f32_scalar(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// FP32 → BF16 for an 8-wide AVX2 FP32 vector with
// **round-to-nearest-even (RNE)**. Matches the semantics of the
// AVX-512 instruction `vcvtneps2bf16` (which is the canonical
// hardware BF16 down-conversion) using only AVX2 ops.
//
// Standard RNE bias formula for the FP32→BF16 truncate-with-round:
//   bf16_bits = (fp32_bits + 0x7FFF + ((fp32_bits >> 16) & 1)) >> 16
//
// Where:
//   * `fp32_bits >> 16` is the would-be BF16 bit pattern (the
//     truncate path). Its LSB is the "rounding bit" — whether the
//     final BF16 mantissa is odd or even.
//   * Adding `0x7FFF` rounds the discarded low-16-bit mantissa
//     half-up: any value > 0x8000 in the low 16 bits carries into
//     the BF16 mantissa.
//   * Adding `((fp32_bits >> 16) & 1)` implements the "ties to
//     even" rule: if the low 16 bits are exactly 0x8000 (a tie),
//     this nudges only the cases where the BF16 mantissa would
//     otherwise be odd, so the result mantissa is always even.
//
// NaN handling: the bias addition can flip an FP32 NaN into a
// signed-infinity-shaped pattern if the NaN's payload happens to
// be 0. oneDNN's `vcvtneps2bf16` emulation handles this by
// detecting NaNs and forcing a canonical quiet-NaN payload. Our
// COTS use case (BF16×BF16 reduction with finite inputs and FP32
// FMA accumulator) cannot produce NaN unless the inputs already
// contain NaN, in which case the production COTS path would have
// failed earlier. We document this and skip the NaN canonicalization
// to keep the inner loop compact; a future hardening pass could
// add `_mm256_cmp_ps(v, v, _CMP_UNORD_Q)` masking if needed.
inline __m128i f32_to_bf16_rne(__m256 v) {
  __m256i vi = _mm256_castps_si256(v);
  // Build the LSB of the would-be BF16 (i.e., (fp32 >> 16) & 1).
  // Used for the ties-to-even nudge.
  __m256i lsb =
      _mm256_and_si256(_mm256_srli_epi32(vi, 16), _mm256_set1_epi32(1));
  // RNE bias = 0x7FFF + LSB.
  __m256i bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x00007FFF));
  __m256i biased = _mm256_add_epi32(vi, bias);
  __m256i shifted = _mm256_srli_epi32(biased, 16);
  // Pack 8 × int32 → 8 × uint16 (BF16 bit pattern). vpackusdw
  // interleaves the two 128-bit halves, so we reorder back to
  // sequential lanes via the standard lo/hi extract + unpacklo64.
  __m256i packed = _mm256_packus_epi32(shifted, shifted);
  __m128i lo = _mm256_castsi256_si128(packed);
  __m128i hi = _mm256_extracti128_si256(packed, 1);
  return _mm_unpacklo_epi64(lo, hi);
}

// Load 8 BF16 values from `ptr` and expand to 8 FP32.
//
// Imitates the oneDNN sequence at
// src/cpu/x64/brgemm/jit_brgemm_kernel.cpp:2880-2883:
//     uni_vpmovzxwd(vmm_load, addr);
//     uni_vpslld(vmm_load, vmm_load, 16);
inline __m256 load8_bf16_as_f32(const uint16_t* ptr) {
  __m128i bf16x8 =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));  // 8×16-bit BF16
  __m256i ext = _mm256_cvtepu16_epi32(bf16x8);                 // vpmovzxwd
  __m256i shl = _mm256_slli_epi32(ext, 16);                    // vpslld 16
  return _mm256_castsi256_ps(shl);
}

// Register-tiled inner kernel. Mirrors oneDNN's M_blk × N_blk
// register tile (brgemm_matmul_utils.cpp:2108-2113 `M_chunk_elems`
// = `M_blk * M_chunk_size`): keeps `M_TILE × N_INNER × 2` AVX2
// accumulator registers live across the FULL K reduction so each
// w[k, nb..nb+15] cache line is read ONCE and FMA'd against all M
// rows — eliminates the M-fold redundant DRAM traffic the original
// per-(m, nb)-outer loop incurred.
//
// Register-budget analysis for AVX2 (16 ymm registers):
//   M_TILE × N_INNER × 2 accumulators  + 2 for w_lo/w_hi  + 1 for
//   x_bcast (broadcast per m, reused). The valid (M_TILE, N_INNER)
//   choices that keep accumulators ≤ 13 ymm:
//     (M_TILE=1, N_INNER=4) → 8 acc + 2 + 1 = 11 ymm
//     (M_TILE=1, N_INNER=6) → 12 + 2 + 1 = 15 ymm  (overflows; ok if
//                                                    compiler spills)
//     (M_TILE=4, N_INNER=2) → 16 acc + 2 + 1 → spill expected; we
//                              actually want 4+2+1=7 (1 nb tile),
//                              see below
//   The fallback case for other M reuses the original Nb=16 loop.
//
// K-tiling rationale: working set per (M_TILE × N_INNER) tile across
// the K loop is `K × M_TILE × 2` bytes (x) + `K × N_INNER × Nb × 2`
// bytes (w). At K=2842, N_INNER=4, that's 5.5 KB + 364 KB — the w
// part exceeds L1. Software prefetch (`_mm_prefetch`, MM_HINT_T0)
// of weights `PREFETCH_DIST` k-rows ahead overlaps DRAM latency
// with FMA compute. This is the same role oneDNN's `K_chunk_elems`
// + brgemm-internal prefetch plays — keep the L1 hot for the
// currently-accumulating K range.
constexpr int64_t Nb = 16;
constexpr int64_t PREFETCH_DIST = 24;  // k-rows ahead

template <int M_TILE, int N_INNER>
inline void gemm_tile_kernel(const uint16_t* x, const uint16_t* w, uint16_t* y,
                             int64_t M, int64_t K, int64_t N,
                             int64_t nt_start) {
  // Per-m accumulator pairs across the N_INNER nb sub-tiles in this
  // register tile. Initialized to zero.
  __m256 acc_lo[M_TILE][N_INNER];
  __m256 acc_hi[M_TILE][N_INNER];
  for (int m = 0; m < M_TILE; ++m) {
    for (int ni = 0; ni < N_INNER; ++ni) {
      acc_lo[m][ni] = _mm256_setzero_ps();
      acc_hi[m][ni] = _mm256_setzero_ps();
    }
  }

  for (int64_t k = 0; k < K; ++k) {
    // Software prefetch of w[k + PREFETCH_DIST, nt_start..] —
    // overlaps DRAM latency with this iteration's FMAs. Same idea
    // as oneDNN's L1-prefetch hints inside brgemm_kernel.cpp.
    if (k + PREFETCH_DIST < K) {
      const uint16_t* w_pf = w + (k + PREFETCH_DIST) * N + nt_start;
      // Two cache lines per N_INNER tile (32 bytes each = 1 line).
      for (int ni = 0; ni < N_INNER; ++ni) {
        _mm_prefetch(reinterpret_cast<const char*>(w_pf + ni * Nb),
                     _MM_HINT_T0);
      }
    }

    // Load N_INNER × Nb BF16 weights for row k. These cache lines
    // are then reused across all M_TILE FMAs — this is the whole
    // point of the M-fusion.
    __m256 w_lo[N_INNER];
    __m256 w_hi[N_INNER];
    for (int ni = 0; ni < N_INNER; ++ni) {
      const uint16_t* w_kn = w + k * N + nt_start + ni * Nb;
      w_lo[ni] = load8_bf16_as_f32(w_kn);
      w_hi[ni] = load8_bf16_as_f32(w_kn + 8);
    }

    // FMA: M_TILE × N_INNER tiles all accumulate against the same
    // w_lo/w_hi vector pair. M_TILE * N_INNER * 2 FMAs per k.
    for (int m = 0; m < M_TILE; ++m) {
      const __m256 x_bcast = _mm256_set1_ps(bf16_to_f32_scalar(x[m * K + k]));
      for (int ni = 0; ni < N_INNER; ++ni) {
        acc_lo[m][ni] = _mm256_fmadd_ps(x_bcast, w_lo[ni], acc_lo[m][ni]);
        acc_hi[m][ni] = _mm256_fmadd_ps(x_bcast, w_hi[ni], acc_hi[m][ni]);
      }
    }
  }

  // Truncate FP32 accumulators back to BF16 and store. The full
  // M_TILE × N_INNER tile is emitted at once.
  for (int m = 0; m < M_TILE; ++m) {
    for (int ni = 0; ni < N_INNER; ++ni) {
      const int64_t nb = nt_start + ni * Nb;
      __m128i out_lo = f32_to_bf16_rne(acc_lo[m][ni]);
      __m128i out_hi = f32_to_bf16_rne(acc_hi[m][ni]);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(y + m * N + nb), out_lo);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(y + m * N + nb + 8), out_hi);
    }
  }
}

// Run one M_TILE token group across the output dimension. The final partial
// output tile falls back to one 16-column block; production down-projection
// widths are divisible by every tile width used below, so that path is
// normally empty.
template <int M_TILE, int N_INNER>
void run_output_tiles(const uint16_t* x, const uint16_t* w, uint16_t* y,
                      int64_t K, int64_t N) {
  constexpr int64_t kTileWidth = N_INNER * Nb;
  const int64_t N_main = (N / Nb) * Nb;
  const int64_t n_tiles = N_main / kTileWidth;
  const int64_t covered = n_tiles * kTileWidth;
  at::parallel_for(0, n_tiles, 1, [=](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      gemm_tile_kernel<M_TILE, N_INNER>(x, w, y, M_TILE, K, N,
                                        tile * kTileWidth);
    }
  });
  for (int64_t n = covered; n < N_main; n += Nb) {
    gemm_tile_kernel<M_TILE, 1>(x, w, y, M_TILE, K, N, n);
  }
}

// Full four-token groups share each weight block across all four tokens.
// Parallelizing the joint (token-group, output-block) space gives enough work
// to every thread at large M without introducing nested parallel regions.
void run_m4_groups(const uint16_t* x, const uint16_t* w, uint16_t* y,
                   int64_t groups, int64_t K, int64_t N) {
  const int64_t n_blocks = N / Nb;
  at::parallel_for(0, groups * n_blocks, 1, [=](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t group = index / n_blocks;
      const int64_t block = index % n_blocks;
      gemm_tile_kernel<4, 1>(x + group * 4 * K, w, y + group * 4 * N, 4, K, N,
                             block * Nb);
    }
  });
}

}  // namespace

// Public entry. y[M, N] = x[M, K] @ w[K, N], all BF16 row-major.
//
// Requirements:
//   * x, w, y are bf16 pointers.
//   * Row-major contiguous (no strides argument; if the caller
//     has a strided weight, transpose to contiguous first).
//   * M >= 1, K >= 1, N must be a multiple of 8 for the main
//     kernel; remainder columns (N % 8) are handled by a tail
//     loop.
//   * No bias, no post-ops.
//
// Intra-op parallelism on the N-tile dimension via
// `at::parallel_for`. Each task writes a disjoint column range of
// y[M, N] so there is no cross-task contention and no reduction.
// `at::parallel_for` respects PyTorch's threading executor (OMP or
// TBB depending on build) and `at::set_num_threads(...)`, which is
// how the COTS worker is configured per bucket (§1c.04).
void bf16_gemm_transposed(const uint16_t* x, const uint16_t* w, uint16_t* y,
                          int64_t M, int64_t N, int64_t K) {
  const int64_t N_main = (N / Nb) * Nb;

  const int64_t m_groups = M / 4;
  if (M == 4) {
    // Preserve the dedicated output-only schedule: the joint group/output
    // index used for larger M would add an unnecessary integer divide here.
    run_output_tiles<4, 1>(x, w, y, K, N);
  } else if (m_groups > 0) {
    run_m4_groups(x, w, y, m_groups, K, N);
  }

  const int64_t m_tail_start = m_groups * 4;
  const uint16_t* tail_x = x + m_tail_start * K;
  uint16_t* tail_y = y + m_tail_start * N;
  switch (M - m_tail_start) {
    case 3:
      run_output_tiles<3, 2>(tail_x, w, tail_y, K, N);
      break;
    case 2:
      run_output_tiles<2, 2>(tail_x, w, tail_y, K, N);
      break;
    case 1:
      if (m_groups == 0) {
        // Preserve the dedicated M=1 tile: one activation broadcast feeds 64
        // output channels. A one-token remainder after full groups keeps the
        // conservative 16-column tile that was regression-free in production.
        run_output_tiles<1, 4>(tail_x, w, tail_y, K, N);
      } else {
        run_output_tiles<1, 1>(tail_x, w, tail_y, K, N);
      }
      break;
  }

  // Tail loop (N % Nb columns) — single-threaded, runs after the
  // parallel region above. Per-m, handles at most (Nb-1)=15 columns
  // per row. Production down-proj shapes are multiples of 16
  // (out_dim ∈ {3584, 4096, ...}), so this branch is almost
  // always a no-op; kept as a defensive path for non-aligned N.
  if (N_main < N) {
    for (int64_t m = 0; m < M; ++m) {
      const uint16_t* x_row = x + m * K;
      uint16_t* y_row = y + m * N;
      int64_t n = N_main;
      if (N - n >= 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int64_t k = 0; k < K; ++k) {
          const __m256 x_bcast = _mm256_set1_ps(bf16_to_f32_scalar(x_row[k]));
          const __m256 w_f32 = load8_bf16_as_f32(w + k * N + n);
          acc = _mm256_fmadd_ps(x_bcast, w_f32, acc);
        }
        __m128i out = f32_to_bf16_rne(acc);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(y_row + n), out);
        n += 8;
      }
      for (; n < N; ++n) {
        float acc_s = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          acc_s +=
              bf16_to_f32_scalar(x_row[k]) * bf16_to_f32_scalar(w[k * N + n]);
        }
        // Scalar RNE FP32 → BF16, same bias as f32_to_bf16_rne.
        uint32_t u;
        std::memcpy(&u, &acc_s, sizeof(u));
        const uint32_t lsb = (u >> 16) & 1u;
        u += 0x7FFFu + lsb;
        y_row[n] = static_cast<uint16_t>(u >> 16);
      }
    }
  }
}

#else  // !VLLM_COTS_HAS_AVX2_FMA

// Reference fallback for non-standard direct builds without AVX2/FMA.
// The vLLM CMake target force-compiles this TU with `-mavx2 -mfma`,
// so production COTS never takes this branch.
void bf16_gemm_transposed(const uint16_t* x, const uint16_t* w, uint16_t* y,
                          int64_t M, int64_t N, int64_t K) {
  auto bf16_to_f32 = [](uint16_t b) -> float {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };
  // Scalar RNE — matches f32_to_bf16_rne (the AVX2 intrinsic
  // version above). Bias by 0x7FFF + LSB-of-bf16-mantissa
  // implements round-to-nearest-with-ties-to-even.
  auto f32_to_bf16 = [](float f) -> uint16_t {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
  };
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += bf16_to_f32(x[m * K + k]) * bf16_to_f32(w[k * N + n]);
      }
      y[m * N + n] = f32_to_bf16(acc);
    }
  }
}

#endif

// at::Tensor entry. Validates dtype + contiguity then dispatches
// to the raw-pointer kernel. Designed to be called from
// CotsWeightTaskRunner's MLP-block worker for the transposed-storage path
// (Stage 7-C). Caller passes:
//   * x: (M, K) BF16, row-major contiguous.
//   * w: (K, N) BF16, row-major contiguous (this is the
//     transposed-storage layout — `w_cpu_t` from Stage 7's
//     storage unification proposal).
//   * y_out: (M, N) BF16, row-major contiguous, pre-allocated.
//
// Output is written in-place into y_out.
void bf16_gemm_transposed_at(const at::Tensor& x, const at::Tensor& w,
                             at::Tensor& y_out) {
  TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be bfloat16, got ",
              x.dtype());
  TORCH_CHECK(w.dtype() == at::kBFloat16, "w must be bfloat16, got ",
              w.dtype());
  TORCH_CHECK(y_out.dtype() == at::kBFloat16, "y_out must be bfloat16, got ",
              y_out.dtype());
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y_out.dim() == 2,
              "all tensors must be 2D");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y_out.is_contiguous(),
              "all tensors must be row-major contiguous (transposed-storage "
              "layout: w shape is (K, N), not (N, K))");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  TORCH_CHECK(w.size(0) == K, "w.size(0)=", w.size(0),
              " must equal x.size(1)=", K);
  const int64_t N = w.size(1);
  TORCH_CHECK(y_out.size(0) == M && y_out.size(1) == N,
              "y_out shape mismatch: expected (", M, ",", N, ") got (",
              y_out.size(0), ",", y_out.size(1), ")");

  bf16_gemm_transposed(reinterpret_cast<const uint16_t*>(x.data_ptr()),
                       reinterpret_cast<const uint16_t*>(w.data_ptr()),
                       reinterpret_cast<uint16_t*>(y_out.data_ptr()), M, N, K);
}

}  // namespace cots
}  // namespace vllm
