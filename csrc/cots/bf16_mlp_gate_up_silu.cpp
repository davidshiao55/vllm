// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// COTS CPU MLP block for decode/small-batch CPU compute.
//
// Computes:
//
//   z = silu(x @ w_gate^T) * (x @ w_up^T)
//   y = z @ w_down
//
// with BF16 x/weights/output, natural row-major gate/up weights `(I, H)`,
// transposed row-major down weights `(I, O)`, and worker-owned BF16 scratch for
// z. The gate/up/SwiGLU part is fused into one pass over x and the two input
// matrices; the down projection intentionally reuses the existing optimized
// BF16 transposed GEMM.

#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define VLLM_COTS_HAS_AVX2_FMA 1
#else
  #define VLLM_COTS_HAS_AVX2_FMA 0
#endif

namespace vllm {
namespace cots {

// Existing transposed BF16 down kernel from bf16_gemm_transposed.cpp.
void bf16_gemm_transposed(const uint16_t* x, const uint16_t* w, uint16_t* y,
                          int64_t M, int64_t N, int64_t K);

namespace {

inline float bf16_to_f32_scalar(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline uint16_t f32_to_bf16_rne_scalar(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

#if VLLM_COTS_HAS_AVX2_FMA

inline __m256 load8_bf16_as_f32(const uint16_t* ptr) {
  __m128i bf16x8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
  __m256i ext = _mm256_cvtepu16_epi32(bf16x8);
  __m256i shl = _mm256_slli_epi32(ext, 16);
  return _mm256_castsi256_ps(shl);
}

inline float hreduce_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(s);
  __m128 s2 = _mm_add_ps(s, sh);
  sh = _mm_movehl_ps(sh, s2);
  __m128 s3 = _mm_add_ss(s2, sh);
  return _mm_cvtss_f32(s3);
}

template <int M_TILE>
inline void gate_up_silu_tile(const uint16_t* x, const uint16_t* w_gate,
                              const uint16_t* w_up, uint16_t* z_scratch,
                              int64_t H, int64_t I, int64_t m_start,
                              int64_t i) {
  __m256 acc_gate[M_TILE];
  __m256 acc_up[M_TILE];
  for (int m = 0; m < M_TILE; ++m) {
    acc_gate[m] = _mm256_setzero_ps();
    acc_up[m] = _mm256_setzero_ps();
  }

  const uint16_t* wg = w_gate + i * H;
  const uint16_t* wu = w_up + i * H;
  const int64_t H_main = (H / 8) * 8;
  for (int64_t h = 0; h < H_main; h += 8) {
    const __m256 wg_vec = load8_bf16_as_f32(wg + h);
    const __m256 wu_vec = load8_bf16_as_f32(wu + h);
    for (int m = 0; m < M_TILE; ++m) {
      const __m256 x_vec = load8_bf16_as_f32(x + (m_start + m) * H + h);
      acc_gate[m] = _mm256_fmadd_ps(x_vec, wg_vec, acc_gate[m]);
      acc_up[m] = _mm256_fmadd_ps(x_vec, wu_vec, acc_up[m]);
    }
  }

  float gate_s[M_TILE];
  float up_s[M_TILE];
  for (int m = 0; m < M_TILE; ++m) {
    gate_s[m] = hreduce_ps(acc_gate[m]);
    up_s[m] = hreduce_ps(acc_up[m]);
  }
  for (int64_t h = H_main; h < H; ++h) {
    const float wg_s = bf16_to_f32_scalar(wg[h]);
    const float wu_s = bf16_to_f32_scalar(wu[h]);
    for (int m = 0; m < M_TILE; ++m) {
      const float x_s = bf16_to_f32_scalar(x[(m_start + m) * H + h]);
      gate_s[m] += x_s * wg_s;
      up_s[m] += x_s * wu_s;
    }
  }

  for (int m = 0; m < M_TILE; ++m) {
    const uint16_t gate_b = f32_to_bf16_rne_scalar(gate_s[m]);
    const uint16_t up_b = f32_to_bf16_rne_scalar(up_s[m]);
    const float gate_r = bf16_to_f32_scalar(gate_b);
    const float up_r = bf16_to_f32_scalar(up_b);
    z_scratch[(m_start + m) * I + i] =
        f32_to_bf16_rne_scalar(silu(gate_r) * up_r);
  }
}

void gate_up_silu_bf16_scratch(const uint16_t* x, const uint16_t* w_gate,
                               const uint16_t* w_up, uint16_t* z_scratch,
                               int64_t M, int64_t H, int64_t I) {
  at::parallel_for(
      0, I, /*grain=*/1,
      [x, w_gate, w_up, z_scratch, M, H, I](int64_t i_begin, int64_t i_end) {
        for (int64_t i = i_begin; i < i_end; ++i) {
          int64_t m = 0;
          for (; m + 4 <= M; m += 4) {
            gate_up_silu_tile<4>(x, w_gate, w_up, z_scratch, H, I, m, i);
          }
          for (; m < M; ++m) {
            gate_up_silu_tile<1>(x, w_gate, w_up, z_scratch, H, I, m, i);
          }
        }
      });
}

#else  // !VLLM_COTS_HAS_AVX2_FMA

void gate_up_silu_bf16_scratch(const uint16_t* x, const uint16_t* w_gate,
                               const uint16_t* w_up, uint16_t* z_scratch,
                               int64_t M, int64_t H, int64_t I) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t i = 0; i < I; ++i) {
      float gate = 0.0f;
      float up = 0.0f;
      for (int64_t h = 0; h < H; ++h) {
        const float xv = bf16_to_f32_scalar(x[m * H + h]);
        gate += xv * bf16_to_f32_scalar(w_gate[i * H + h]);
        up += xv * bf16_to_f32_scalar(w_up[i * H + h]);
      }
      const float gate_r = bf16_to_f32_scalar(f32_to_bf16_rne_scalar(gate));
      const float up_r = bf16_to_f32_scalar(f32_to_bf16_rne_scalar(up));
      z_scratch[m * I + i] = f32_to_bf16_rne_scalar(silu(gate_r) * up_r);
    }
  }
}

#endif

}  // namespace

void bf16_mlp_gate_up_silu_down(const uint16_t* x, const uint16_t* w_gate,
                                const uint16_t* w_up, const uint16_t* w_down,
                                uint16_t* y, uint16_t* z_scratch, int64_t M,
                                int64_t H, int64_t I, int64_t O) {
  if (M <= 0 || H <= 0 || I <= 0 || O <= 0) {
    return;
  }
  gate_up_silu_bf16_scratch(x, w_gate, w_up, z_scratch, M, H, I);
  bf16_gemm_transposed(z_scratch, w_down, y, M, O, I);
}

}  // namespace cots
}  // namespace vllm
