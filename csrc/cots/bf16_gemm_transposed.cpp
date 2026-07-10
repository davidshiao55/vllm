// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Transposed-storage BF16 GEMM for COTS down projections.
//
// x is (M, K), w is contiguous (K, N), and y is (M, N). The kernel
// vectorizes along N in 16-column blocks, accumulates in FP32, and emits BF16
// RNE. M_TILE and N_INNER keep the live accumulator set within AVX2's 16 YMM
// registers. No packing, bias, or post-ops are used.

#include "bf16_kernel_utils.h"
#include "bf16_kernels.h"

#include <ATen/Parallel.h>

#include <cstdint>

namespace vllm {
namespace cots {

namespace {

// Tile configurations: M=1 -> 1x64, M=2/3 -> Mx32, M=4 -> 4x16.
// Larger batches use four-token groups plus one fused remainder. Weight rows
// are prefetched 24 K-steps ahead.
constexpr int64_t kNBlock = 16;
constexpr int64_t kPrefetchDistance = 24;

template <int M_TILE, int N_INNER>
inline void gemm_tile_kernel(const uint16_t* x, const uint16_t* w, uint16_t* y,
                             int64_t K, int64_t N, int64_t nt_start) {
  __m256 acc_lo[M_TILE][N_INNER];
  __m256 acc_hi[M_TILE][N_INNER];
  for (int m = 0; m < M_TILE; ++m) {
    for (int ni = 0; ni < N_INNER; ++ni) {
      acc_lo[m][ni] = _mm256_setzero_ps();
      acc_hi[m][ni] = _mm256_setzero_ps();
    }
  }

  for (int64_t k = 0; k < K; ++k) {
    if (k + kPrefetchDistance < K) {
      const uint16_t* w_pf = w + (k + kPrefetchDistance) * N + nt_start;
      for (int ni = 0; ni < N_INNER; ++ni) {
        _mm_prefetch(reinterpret_cast<const char*>(w_pf + ni * kNBlock),
                     _MM_HINT_T0);
      }
    }

    __m256 w_lo[N_INNER];
    __m256 w_hi[N_INNER];
    for (int ni = 0; ni < N_INNER; ++ni) {
      const uint16_t* w_kn = w + k * N + nt_start + ni * kNBlock;
      w_lo[ni] = detail::load_bf16x8(w_kn);
      w_hi[ni] = detail::load_bf16x8(w_kn + 8);
    }

    for (int m = 0; m < M_TILE; ++m) {
      const __m256 x_bcast =
          _mm256_set1_ps(detail::bf16_to_float(x[m * K + k]));
      for (int ni = 0; ni < N_INNER; ++ni) {
        acc_lo[m][ni] = _mm256_fmadd_ps(x_bcast, w_lo[ni], acc_lo[m][ni]);
        acc_hi[m][ni] = _mm256_fmadd_ps(x_bcast, w_hi[ni], acc_hi[m][ni]);
      }
    }
  }

  for (int m = 0; m < M_TILE; ++m) {
    for (int ni = 0; ni < N_INNER; ++ni) {
      const int64_t nb = nt_start + ni * kNBlock;
      const __m128i out_lo = detail::float8_to_bf16_rne(acc_lo[m][ni]);
      const __m128i out_hi = detail::float8_to_bf16_rne(acc_hi[m][ni]);
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
  constexpr int64_t kTileWidth = N_INNER * kNBlock;
  const int64_t N_main = N / kNBlock * kNBlock;
  const int64_t n_tiles = N_main / kTileWidth;
  const int64_t covered = n_tiles * kTileWidth;
  at::parallel_for(0, n_tiles, 1, [=](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      gemm_tile_kernel<M_TILE, N_INNER>(x, w, y, K, N, tile * kTileWidth);
    }
  });
  for (int64_t n = covered; n < N_main; n += kNBlock) {
    gemm_tile_kernel<M_TILE, 1>(x, w, y, K, N, n);
  }
}

// Full four-token groups share each weight block across all four tokens.
// Parallelizing the joint (token-group, output-block) space gives enough work
// to every thread at large M without introducing nested parallel regions.
void run_m4_groups(const uint16_t* x, const uint16_t* w, uint16_t* y,
                   int64_t groups, int64_t K, int64_t N) {
  const int64_t n_blocks = N / kNBlock;
  at::parallel_for(0, groups * n_blocks, 1, [=](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t group = index / n_blocks;
      const int64_t block = index % n_blocks;
      gemm_tile_kernel<4, 1>(x + group * 4 * K, w, y + group * 4 * N, K, N,
                             block * kNBlock);
    }
  });
}

}  // namespace

void bf16_gemm_transposed(const uint16_t* x, const uint16_t* w, uint16_t* y,
                          int64_t M, int64_t N, int64_t K) {
  const int64_t N_main = N / kNBlock * kNBlock;

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

  // Production output widths are 16-aligned; retain a defensive N tail.
  if (N_main < N) {
    for (int64_t m = 0; m < M; ++m) {
      const uint16_t* x_row = x + m * K;
      uint16_t* y_row = y + m * N;
      int64_t n = N_main;
      if (N - n >= 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int64_t k = 0; k < K; ++k) {
          const __m256 x_bcast =
              _mm256_set1_ps(detail::bf16_to_float(x_row[k]));
          const __m256 w_f32 = detail::load_bf16x8(w + k * N + n);
          acc = _mm256_fmadd_ps(x_bcast, w_f32, acc);
        }
        const __m128i out = detail::float8_to_bf16_rne(acc);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(y_row + n), out);
        n += 8;
      }
      for (; n < N; ++n) {
        float acc_s = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          acc_s += detail::bf16_to_float(x_row[k]) *
                   detail::bf16_to_float(w[k * N + n]);
        }
        y_row[n] = detail::float_to_bf16_rne(acc_s);
      }
    }
  }
}

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
