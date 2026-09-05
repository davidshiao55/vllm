// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// AVX-512F/BW implementation of the complete Hybrid BF16 kernel table.
// BF16 values are expanded to FP32 in registers; AVX512_BF16 is intentionally
// not required. The selected natural and transposed microkernels use named
// accumulators to prevent hot-loop stack materialization by GCC.

#include "bf16_kernels_internal.h"
#include "bf16_transposed_blocking.h"

#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__FMA__)
  #error "Hybrid AVX-512 kernels require AVX512F, AVX512BW, and FMA"
#endif

#include <ATen/Parallel.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace vllm {
namespace hybrid {

namespace {

inline float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline uint16_t float_to_bf16_rne(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

inline __m512 load_bf16x16(const uint16_t* ptr) {
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  const __m512i extended = _mm512_cvtepu16_epi32(packed);
  return _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16));
}

inline __m256i float16_to_bf16_rne(__m512 value) {
  const __m512i bits = _mm512_castps_si512(value);
  const __m512i lsb =
      _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
  const __m512i bias = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x00007FFF));
  const __m512i shifted = _mm512_srli_epi32(_mm512_add_epi32(bits, bias), 16);
  return _mm512_cvtusepi32_epi16(shifted);
}

inline float horizontal_sum(__m512 value) {
  return _mm512_reduce_add_ps(value);
}

template <int M_TILE>
__attribute__((noinline)) void natural_tile5(const uint16_t* x,
                                             const uint16_t* w, uint16_t* y,
                                             int64_t K, int64_t N,
                                             int64_t m_start, int64_t n_start) {
  static_assert(M_TILE >= 1 && M_TILE <= 4);
  const __m512 zero = _mm512_setzero_ps();
  __m512 a00 = zero, a01 = zero, a02 = zero, a03 = zero, a04 = zero;
  __m512 a10 = zero, a11 = zero, a12 = zero, a13 = zero, a14 = zero;
  __m512 a20 = zero, a21 = zero, a22 = zero, a23 = zero, a24 = zero;
  __m512 a30 = zero, a31 = zero, a32 = zero, a33 = zero, a34 = zero;

  const int64_t K_main = K / 16 * 16;
  for (int64_t k = 0; k < K_main; k += 16) {
    const __m512 x0 = load_bf16x16(x + (m_start + 0) * K + k);
    const __m512 w0 = load_bf16x16(w + (n_start + 0) * K + k);
    const __m512 w1 = load_bf16x16(w + (n_start + 1) * K + k);
    const __m512 w2 = load_bf16x16(w + (n_start + 2) * K + k);
    const __m512 w3 = load_bf16x16(w + (n_start + 3) * K + k);
    const __m512 w4 = load_bf16x16(w + (n_start + 4) * K + k);
    a00 = _mm512_fmadd_ps(x0, w0, a00);
    a01 = _mm512_fmadd_ps(x0, w1, a01);
    a02 = _mm512_fmadd_ps(x0, w2, a02);
    a03 = _mm512_fmadd_ps(x0, w3, a03);
    a04 = _mm512_fmadd_ps(x0, w4, a04);
    if constexpr (M_TILE >= 2) {
      const __m512 x1 = load_bf16x16(x + (m_start + 1) * K + k);
      a10 = _mm512_fmadd_ps(x1, w0, a10);
      a11 = _mm512_fmadd_ps(x1, w1, a11);
      a12 = _mm512_fmadd_ps(x1, w2, a12);
      a13 = _mm512_fmadd_ps(x1, w3, a13);
      a14 = _mm512_fmadd_ps(x1, w4, a14);
    }
    if constexpr (M_TILE >= 3) {
      const __m512 x2 = load_bf16x16(x + (m_start + 2) * K + k);
      a20 = _mm512_fmadd_ps(x2, w0, a20);
      a21 = _mm512_fmadd_ps(x2, w1, a21);
      a22 = _mm512_fmadd_ps(x2, w2, a22);
      a23 = _mm512_fmadd_ps(x2, w3, a23);
      a24 = _mm512_fmadd_ps(x2, w4, a24);
    }
    if constexpr (M_TILE >= 4) {
      const __m512 x3 = load_bf16x16(x + (m_start + 3) * K + k);
      a30 = _mm512_fmadd_ps(x3, w0, a30);
      a31 = _mm512_fmadd_ps(x3, w1, a31);
      a32 = _mm512_fmadd_ps(x3, w2, a32);
      a33 = _mm512_fmadd_ps(x3, w3, a33);
      a34 = _mm512_fmadd_ps(x3, w4, a34);
    }
  }

  float s00 = horizontal_sum(a00), s01 = horizontal_sum(a01);
  float s02 = horizontal_sum(a02), s03 = horizontal_sum(a03);
  float s04 = horizontal_sum(a04);
  float s10 = 0, s11 = 0, s12 = 0, s13 = 0, s14 = 0;
  float s20 = 0, s21 = 0, s22 = 0, s23 = 0, s24 = 0;
  float s30 = 0, s31 = 0, s32 = 0, s33 = 0, s34 = 0;
  if constexpr (M_TILE >= 2) {
    s10 = horizontal_sum(a10);
    s11 = horizontal_sum(a11);
    s12 = horizontal_sum(a12);
    s13 = horizontal_sum(a13);
    s14 = horizontal_sum(a14);
  }
  if constexpr (M_TILE >= 3) {
    s20 = horizontal_sum(a20);
    s21 = horizontal_sum(a21);
    s22 = horizontal_sum(a22);
    s23 = horizontal_sum(a23);
    s24 = horizontal_sum(a24);
  }
  if constexpr (M_TILE >= 4) {
    s30 = horizontal_sum(a30);
    s31 = horizontal_sum(a31);
    s32 = horizontal_sum(a32);
    s33 = horizontal_sum(a33);
    s34 = horizontal_sum(a34);
  }

  for (int64_t k = K_main; k < K; ++k) {
    const float w0s = bf16_to_float(w[(n_start + 0) * K + k]);
    const float w1s = bf16_to_float(w[(n_start + 1) * K + k]);
    const float w2s = bf16_to_float(w[(n_start + 2) * K + k]);
    const float w3s = bf16_to_float(w[(n_start + 3) * K + k]);
    const float w4s = bf16_to_float(w[(n_start + 4) * K + k]);
    const float x0s = bf16_to_float(x[(m_start + 0) * K + k]);
    s00 += x0s * w0s;
    s01 += x0s * w1s;
    s02 += x0s * w2s;
    s03 += x0s * w3s;
    s04 += x0s * w4s;
    if constexpr (M_TILE >= 2) {
      const float x1s = bf16_to_float(x[(m_start + 1) * K + k]);
      s10 += x1s * w0s;
      s11 += x1s * w1s;
      s12 += x1s * w2s;
      s13 += x1s * w3s;
      s14 += x1s * w4s;
    }
    if constexpr (M_TILE >= 3) {
      const float x2s = bf16_to_float(x[(m_start + 2) * K + k]);
      s20 += x2s * w0s;
      s21 += x2s * w1s;
      s22 += x2s * w2s;
      s23 += x2s * w3s;
      s24 += x2s * w4s;
    }
    if constexpr (M_TILE >= 4) {
      const float x3s = bf16_to_float(x[(m_start + 3) * K + k]);
      s30 += x3s * w0s;
      s31 += x3s * w1s;
      s32 += x3s * w2s;
      s33 += x3s * w3s;
      s34 += x3s * w4s;
    }
  }

#define STORE_NATURAL_ROW(M)                                       \
  y[(m_start + M) * N + n_start + 0] = float_to_bf16_rne(s##M##0); \
  y[(m_start + M) * N + n_start + 1] = float_to_bf16_rne(s##M##1); \
  y[(m_start + M) * N + n_start + 2] = float_to_bf16_rne(s##M##2); \
  y[(m_start + M) * N + n_start + 3] = float_to_bf16_rne(s##M##3); \
  y[(m_start + M) * N + n_start + 4] = float_to_bf16_rne(s##M##4)
  STORE_NATURAL_ROW(0);
  if constexpr (M_TILE >= 2) {
    STORE_NATURAL_ROW(1);
  }
  if constexpr (M_TILE >= 3) {
    STORE_NATURAL_ROW(2);
  }
  if constexpr (M_TILE >= 4) {
    STORE_NATURAL_ROW(3);
  }
#undef STORE_NATURAL_ROW
}

template <int M_TILE>
inline void natural_single(const uint16_t* x, const uint16_t* w, uint16_t* y,
                           int64_t K, int64_t N, int64_t m_start, int64_t n) {
  __m512 acc[M_TILE];
  for (int m = 0; m < M_TILE; ++m) acc[m] = _mm512_setzero_ps();
  const uint16_t* w_row = w + n * K;
  const int64_t K_main = K / 16 * 16;
  for (int64_t k = 0; k < K_main; k += 16) {
    const __m512 wv = load_bf16x16(w_row + k);
    for (int m = 0; m < M_TILE; ++m) {
      const __m512 xv = load_bf16x16(x + (m_start + m) * K + k);
      acc[m] = _mm512_fmadd_ps(xv, wv, acc[m]);
    }
  }
  float sums[M_TILE];
  for (int m = 0; m < M_TILE; ++m) sums[m] = horizontal_sum(acc[m]);
  for (int64_t k = K_main; k < K; ++k) {
    const float ws = bf16_to_float(w_row[k]);
    for (int m = 0; m < M_TILE; ++m) {
      sums[m] += bf16_to_float(x[(m_start + m) * K + k]) * ws;
    }
  }
  for (int m = 0; m < M_TILE; ++m) {
    y[(m_start + m) * N + n] = float_to_bf16_rne(sums[m]);
  }
}

template <typename Tile4, typename Tile3, typename Tile2, typename Tile1>
inline void run_m_tiles(int64_t M, Tile4&& tile4, Tile3&& tile3, Tile2&& tile2,
                        Tile1&& tile1) {
  int64_t m = 0;
  for (; m + 4 <= M; m += 4) tile4(m);
  switch (M - m) {
    case 3:
      tile3(m);
      break;
    case 2:
      tile2(m);
      break;
    case 1:
      tile1(m);
      break;
  }
}

template <int M_TILE>
__attribute__((noinline)) void transposed_tile4(
    const uint16_t* x, const uint16_t* w, float* accum, int64_t K, int64_t N,
    int64_t m_start, int64_t n_start, int64_t k0, int64_t k_end) {
  static_assert(M_TILE >= 1 && M_TILE <= 4);
  const __m512 zero = _mm512_setzero_ps();
  __m512 a00 = zero, a01 = zero, a02 = zero, a03 = zero;
  __m512 a10 = zero, a11 = zero, a12 = zero, a13 = zero;
  __m512 a20 = zero, a21 = zero, a22 = zero, a23 = zero;
  __m512 a30 = zero, a31 = zero, a32 = zero, a33 = zero;

  if (k0 != 0) {
    a00 = _mm512_loadu_ps(accum + (m_start + 0) * N + n_start + 0);
    a01 = _mm512_loadu_ps(accum + (m_start + 0) * N + n_start + 16);
    a02 = _mm512_loadu_ps(accum + (m_start + 0) * N + n_start + 32);
    a03 = _mm512_loadu_ps(accum + (m_start + 0) * N + n_start + 48);
    if constexpr (M_TILE >= 2) {
      a10 = _mm512_loadu_ps(accum + (m_start + 1) * N + n_start + 0);
      a11 = _mm512_loadu_ps(accum + (m_start + 1) * N + n_start + 16);
      a12 = _mm512_loadu_ps(accum + (m_start + 1) * N + n_start + 32);
      a13 = _mm512_loadu_ps(accum + (m_start + 1) * N + n_start + 48);
    }
    if constexpr (M_TILE >= 3) {
      a20 = _mm512_loadu_ps(accum + (m_start + 2) * N + n_start + 0);
      a21 = _mm512_loadu_ps(accum + (m_start + 2) * N + n_start + 16);
      a22 = _mm512_loadu_ps(accum + (m_start + 2) * N + n_start + 32);
      a23 = _mm512_loadu_ps(accum + (m_start + 2) * N + n_start + 48);
    }
    if constexpr (M_TILE >= 4) {
      a30 = _mm512_loadu_ps(accum + (m_start + 3) * N + n_start + 0);
      a31 = _mm512_loadu_ps(accum + (m_start + 3) * N + n_start + 16);
      a32 = _mm512_loadu_ps(accum + (m_start + 3) * N + n_start + 32);
      a33 = _mm512_loadu_ps(accum + (m_start + 3) * N + n_start + 48);
    }
  }

  constexpr int64_t kPrefetchDistance = detail::kTransposedPrefetch;
  for (int64_t k = k0; k < k_end; ++k) {
    if (k + kPrefetchDistance < k_end) {
      const uint16_t* w_pf = w + (k + kPrefetchDistance) * N + n_start;
      _mm_prefetch(reinterpret_cast<const char*>(w_pf + 0), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(w_pf + 32), _MM_HINT_T0);
    }
    const __m512 w0 = load_bf16x16(w + k * N + n_start + 0);
    const __m512 w1 = load_bf16x16(w + k * N + n_start + 16);
    const __m512 w2 = load_bf16x16(w + k * N + n_start + 32);
    const __m512 w3 = load_bf16x16(w + k * N + n_start + 48);
    const __m512 x0 = _mm512_set1_ps(bf16_to_float(x[(m_start + 0) * K + k]));
    a00 = _mm512_fmadd_ps(x0, w0, a00);
    a01 = _mm512_fmadd_ps(x0, w1, a01);
    a02 = _mm512_fmadd_ps(x0, w2, a02);
    a03 = _mm512_fmadd_ps(x0, w3, a03);
    if constexpr (M_TILE >= 2) {
      const __m512 x1 = _mm512_set1_ps(bf16_to_float(x[(m_start + 1) * K + k]));
      a10 = _mm512_fmadd_ps(x1, w0, a10);
      a11 = _mm512_fmadd_ps(x1, w1, a11);
      a12 = _mm512_fmadd_ps(x1, w2, a12);
      a13 = _mm512_fmadd_ps(x1, w3, a13);
    }
    if constexpr (M_TILE >= 3) {
      const __m512 x2 = _mm512_set1_ps(bf16_to_float(x[(m_start + 2) * K + k]));
      a20 = _mm512_fmadd_ps(x2, w0, a20);
      a21 = _mm512_fmadd_ps(x2, w1, a21);
      a22 = _mm512_fmadd_ps(x2, w2, a22);
      a23 = _mm512_fmadd_ps(x2, w3, a23);
    }
    if constexpr (M_TILE >= 4) {
      const __m512 x3 = _mm512_set1_ps(bf16_to_float(x[(m_start + 3) * K + k]));
      a30 = _mm512_fmadd_ps(x3, w0, a30);
      a31 = _mm512_fmadd_ps(x3, w1, a31);
      a32 = _mm512_fmadd_ps(x3, w2, a32);
      a33 = _mm512_fmadd_ps(x3, w3, a33);
    }
  }

  _mm512_storeu_ps(accum + (m_start + 0) * N + n_start + 0, a00);
  _mm512_storeu_ps(accum + (m_start + 0) * N + n_start + 16, a01);
  _mm512_storeu_ps(accum + (m_start + 0) * N + n_start + 32, a02);
  _mm512_storeu_ps(accum + (m_start + 0) * N + n_start + 48, a03);
  if constexpr (M_TILE >= 2) {
    _mm512_storeu_ps(accum + (m_start + 1) * N + n_start + 0, a10);
    _mm512_storeu_ps(accum + (m_start + 1) * N + n_start + 16, a11);
    _mm512_storeu_ps(accum + (m_start + 1) * N + n_start + 32, a12);
    _mm512_storeu_ps(accum + (m_start + 1) * N + n_start + 48, a13);
  }
  if constexpr (M_TILE >= 3) {
    _mm512_storeu_ps(accum + (m_start + 2) * N + n_start + 0, a20);
    _mm512_storeu_ps(accum + (m_start + 2) * N + n_start + 16, a21);
    _mm512_storeu_ps(accum + (m_start + 2) * N + n_start + 32, a22);
    _mm512_storeu_ps(accum + (m_start + 2) * N + n_start + 48, a23);
  }
  if constexpr (M_TILE >= 4) {
    _mm512_storeu_ps(accum + (m_start + 3) * N + n_start + 0, a30);
    _mm512_storeu_ps(accum + (m_start + 3) * N + n_start + 16, a31);
    _mm512_storeu_ps(accum + (m_start + 3) * N + n_start + 32, a32);
    _mm512_storeu_ps(accum + (m_start + 3) * N + n_start + 48, a33);
  }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

template <int M_TILE, int I_TILE>
inline void gate_up_tile(const uint16_t* x, const uint16_t* w_gate,
                         const uint16_t* w_up, uint16_t* z, int64_t H,
                         int64_t I, int64_t m_start, int64_t i_start) {
  __m512 gate[I_TILE][M_TILE];
  __m512 up[I_TILE][M_TILE];
  for (int i = 0; i < I_TILE; ++i) {
    for (int m = 0; m < M_TILE; ++m) {
      gate[i][m] = _mm512_setzero_ps();
      up[i][m] = _mm512_setzero_ps();
    }
  }
  const int64_t H_main = H / 16 * 16;
  for (int64_t h = 0; h < H_main; h += 16) {
    __m512 xv[M_TILE];
    for (int m = 0; m < M_TILE; ++m) {
      xv[m] = load_bf16x16(x + (m_start + m) * H + h);
    }
    for (int i = 0; i < I_TILE; ++i) {
      const __m512 gv = load_bf16x16(w_gate + (i_start + i) * H + h);
      const __m512 uv = load_bf16x16(w_up + (i_start + i) * H + h);
      for (int m = 0; m < M_TILE; ++m) {
        gate[i][m] = _mm512_fmadd_ps(xv[m], gv, gate[i][m]);
        up[i][m] = _mm512_fmadd_ps(xv[m], uv, up[i][m]);
      }
    }
  }
  float gate_s[I_TILE][M_TILE];
  float up_s[I_TILE][M_TILE];
  for (int i = 0; i < I_TILE; ++i) {
    for (int m = 0; m < M_TILE; ++m) {
      gate_s[i][m] = horizontal_sum(gate[i][m]);
      up_s[i][m] = horizontal_sum(up[i][m]);
    }
  }
  for (int64_t h = H_main; h < H; ++h) {
    for (int m = 0; m < M_TILE; ++m) {
      const float xs = bf16_to_float(x[(m_start + m) * H + h]);
      for (int i = 0; i < I_TILE; ++i) {
        gate_s[i][m] += xs * bf16_to_float(w_gate[(i_start + i) * H + h]);
        up_s[i][m] += xs * bf16_to_float(w_up[(i_start + i) * H + h]);
      }
    }
  }
  for (int i = 0; i < I_TILE; ++i) {
    for (int m = 0; m < M_TILE; ++m) {
      const float gate_r = bf16_to_float(float_to_bf16_rne(gate_s[i][m]));
      const float up_r = bf16_to_float(float_to_bf16_rne(up_s[i][m]));
      z[(m_start + m) * I + i_start + i] =
          float_to_bf16_rne(silu(gate_r) * up_r);
    }
  }
}

template <int I_TILE>
void gate_up_impl(const uint16_t* x, const uint16_t* w_gate,
                  const uint16_t* w_up, uint16_t* z, int64_t M, int64_t H,
                  int64_t I) {
  const int64_t i_tiles = I / I_TILE;
  at::parallel_for(0, i_tiles, 1, [=](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      const int64_t i = tile * I_TILE;
      run_m_tiles(
          M,
          [&](int64_t m) {
            gate_up_tile<4, I_TILE>(x, w_gate, w_up, z, H, I, m, i);
          },
          [&](int64_t m) {
            gate_up_tile<3, I_TILE>(x, w_gate, w_up, z, H, I, m, i);
          },
          [&](int64_t m) {
            gate_up_tile<2, I_TILE>(x, w_gate, w_up, z, H, I, m, i);
          },
          [&](int64_t m) {
            gate_up_tile<1, I_TILE>(x, w_gate, w_up, z, H, I, m, i);
          });
    }
  });
  at::parallel_for(i_tiles * I_TILE, I, 1, [=](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      run_m_tiles(
          M,
          [&](int64_t m) {
            gate_up_tile<4, 1>(x, w_gate, w_up, z, H, I, m, i);
          },
          [&](int64_t m) {
            gate_up_tile<3, 1>(x, w_gate, w_up, z, H, I, m, i);
          },
          [&](int64_t m) {
            gate_up_tile<2, 1>(x, w_gate, w_up, z, H, I, m, i);
          },
          [&](int64_t m) {
            gate_up_tile<1, 1>(x, w_gate, w_up, z, H, I, m, i);
          });
    }
  });
}

}  // namespace

void bf16_gemm_natural_avx512(const uint16_t* x, const uint16_t* w, uint16_t* y,
                              int64_t M, int64_t N, int64_t K) {
  const int64_t n_tiles = N / 5;
  at::parallel_for(0, n_tiles, 1, [=](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      const int64_t n = tile * 5;
      run_m_tiles(
          M, [&](int64_t m) { natural_tile5<4>(x, w, y, K, N, m, n); },
          [&](int64_t m) { natural_tile5<3>(x, w, y, K, N, m, n); },
          [&](int64_t m) { natural_tile5<2>(x, w, y, K, N, m, n); },
          [&](int64_t m) { natural_tile5<1>(x, w, y, K, N, m, n); });
    }
  });
  at::parallel_for(n_tiles * 5, N, 1, [=](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      run_m_tiles(
          M, [&](int64_t m) { natural_single<4>(x, w, y, K, N, m, n); },
          [&](int64_t m) { natural_single<3>(x, w, y, K, N, m, n); },
          [&](int64_t m) { natural_single<2>(x, w, y, K, N, m, n); },
          [&](int64_t m) { natural_single<1>(x, w, y, K, N, m, n); });
    }
  });
}

void bf16_gemm_transposed_avx512(const uint16_t* x, const uint16_t* w,
                                 uint16_t* y, int64_t M, int64_t N, int64_t K) {
  detail::run_transposed_blocked(
      y, M, N, K,
      [=](float* accum, int64_t n0, int64_t n_end, int64_t k0, int64_t k_end) {
        int64_t n = n0;
        for (; n + 64 <= n_end; n += 64) {
          run_m_tiles(
              M,
              [&](int64_t m) {
                transposed_tile4<4>(x, w, accum, K, N, m, n, k0, k_end);
              },
              [&](int64_t m) {
                transposed_tile4<3>(x, w, accum, K, N, m, n, k0, k_end);
              },
              [&](int64_t m) {
                transposed_tile4<2>(x, w, accum, K, N, m, n, k0, k_end);
              },
              [&](int64_t m) {
                transposed_tile4<1>(x, w, accum, K, N, m, n, k0, k_end);
              });
        }
        // Defensive column tail, following exactly the same increasing-K FMA
        // order as vector tiles. Production down widths are 64-aligned.
        for (; n < n_end; ++n) {
          for (int64_t m = 0; m < M; ++m) {
            float acc = k0 == 0 ? 0.0f : accum[m * N + n];
            for (int64_t k = k0; k < k_end; ++k)
              acc = std::fma(bf16_to_float(x[m * K + k]),
                             bf16_to_float(w[k * N + n]), acc);
            accum[m * N + n] = acc;
          }
        }
      },
      [=](const float* accum, int64_t n0, int64_t n_end) {
        for (int64_t m = 0; m < M; ++m) {
          int64_t n = n0;
          for (; n + 16 <= n_end; n += 16)
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(y + m * N + n),
                float16_to_bf16_rne(_mm512_loadu_ps(accum + m * N + n)));
          for (; n < n_end; ++n)
            y[m * N + n] = float_to_bf16_rne(accum[m * N + n]);
        }
      });
}

void bf16_mlp_gate_up_silu_down_avx512(const uint16_t* x,
                                       const uint16_t* w_gate,
                                       const uint16_t* w_up,
                                       const uint16_t* w_down, uint16_t* y,
                                       uint16_t* z_scratch, int64_t M,
                                       int64_t H, int64_t I, int64_t O) {
  if (M <= 0 || H <= 0 || I <= 0 || O <= 0) return;
  gate_up_impl<3>(x, w_gate, w_up, z_scratch, M, H, I);
  bf16_gemm_transposed_avx512(z_scratch, w_down, y, M, O, I);
}

}  // namespace hybrid
}  // namespace vllm
