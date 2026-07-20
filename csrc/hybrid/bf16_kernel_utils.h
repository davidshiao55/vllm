// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#if !defined(__AVX2__) || !defined(__FMA__)
  #error "Hybrid BF16 kernels require AVX2 and FMA"
#endif

#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace vllm {
namespace hybrid {
namespace detail {

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

inline __m256 load_bf16x8(const uint16_t* ptr) {
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
  const __m256i extended = _mm256_cvtepu16_epi32(packed);
  return _mm256_castsi256_ps(_mm256_slli_epi32(extended, 16));
}

inline __m128i float8_to_bf16_rne(__m256 value) {
  const __m256i bits = _mm256_castps_si256(value);
  const __m256i lsb =
      _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
  const __m256i bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x00007FFF));
  const __m256i shifted = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
  const __m256i packed = _mm256_packus_epi32(shifted, shifted);
  const __m128i lo = _mm256_castsi256_si128(packed);
  const __m128i hi = _mm256_extracti128_si256(packed, 1);
  return _mm_unpacklo_epi64(lo, hi);
}

inline float horizontal_sum(__m256 value) {
  const __m128 lo = _mm256_castps256_ps128(value);
  const __m128 hi = _mm256_extractf128_ps(value, 1);
  const __m128 sum = _mm_add_ps(lo, hi);
  __m128 shuffled = _mm_movehdup_ps(sum);
  const __m128 pairs = _mm_add_ps(sum, shuffled);
  shuffled = _mm_movehl_ps(shuffled, pairs);
  return _mm_cvtss_f32(_mm_add_ss(pairs, shuffled));
}

// Accumulate two natural-layout weight rows for M_TILE tokens. Natural GEMM
// stores the two sums directly; fused gate/up applies its SwiGLU epilogue.
template <int M_TILE>
inline void dot_two_rows(const uint16_t* x, const uint16_t* w0,
                         const uint16_t* w1, int64_t K, int64_t m_start,
                         float (&sum0)[M_TILE], float (&sum1)[M_TILE]) {
  __m256 acc0[M_TILE];
  __m256 acc1[M_TILE];
  for (int m = 0; m < M_TILE; ++m) {
    acc0[m] = _mm256_setzero_ps();
    acc1[m] = _mm256_setzero_ps();
  }

  const int64_t K_main = K / 8 * 8;
  for (int64_t k = 0; k < K_main; k += 8) {
    const __m256 w0_vec = load_bf16x8(w0 + k);
    const __m256 w1_vec = load_bf16x8(w1 + k);
    for (int m = 0; m < M_TILE; ++m) {
      const __m256 x_vec = load_bf16x8(x + (m_start + m) * K + k);
      acc0[m] = _mm256_fmadd_ps(x_vec, w0_vec, acc0[m]);
      acc1[m] = _mm256_fmadd_ps(x_vec, w1_vec, acc1[m]);
    }
  }

  for (int m = 0; m < M_TILE; ++m) {
    sum0[m] = horizontal_sum(acc0[m]);
    sum1[m] = horizontal_sum(acc1[m]);
  }
  for (int64_t k = K_main; k < K; ++k) {
    const float w0_value = bf16_to_float(w0[k]);
    const float w1_value = bf16_to_float(w1[k]);
    for (int m = 0; m < M_TILE; ++m) {
      const float x_value = bf16_to_float(x[(m_start + m) * K + k]);
      sum0[m] += x_value * w0_value;
      sum1[m] += x_value * w1_value;
    }
  }
}

}  // namespace detail
}  // namespace hybrid
}  // namespace vllm
