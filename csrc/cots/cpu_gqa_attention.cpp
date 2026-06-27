// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// COTS CPU GQA attention kernels.
//
// This is intentionally thesis-owned and narrow: BF16 paged attention for GQA
// models with head_dim=128 and at most 8 query heads per KV head. It contains
// the Phase 2 suffix-attention entry points plus standalone decode/prefill CPU
// attention kernels used by the measurement harnesses.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA 1
#else
  #define VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA 0
#endif

namespace vllm {
namespace cots {
namespace {

constexpr int64_t kMaxHeadsPerKV = 8;
constexpr int64_t kSupportedHeadDim = 128;
constexpr int64_t kVecWidth = 8;
constexpr int64_t kSuffixTwoPassMaxSeqLen = 128;

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

#if VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA

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

inline void store8_f32_as_bf16(__m256 v, uint16_t* ptr) {
  const __m256i x = _mm256_castps_si256(v);
  const __m256i lsb =
      _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(1));
  const __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
  const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(x, bias), 16);
  const __m128i lo = _mm256_castsi256_si128(rounded);
  const __m128i hi = _mm256_extracti128_si256(rounded, 1);
  const __m128i packed = _mm_packus_epi32(lo, hi);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), packed);
}

struct CpuGqaAttentionStrides {
  int64_t q_b;
  int64_t q_h;
  int64_t q_d;
  int64_t kv_block;
  int64_t kv_head;
  int64_t kv_token;
  int64_t kv_d;
  int64_t out_b;
  int64_t out_h;
  int64_t out_d;
  int64_t bt_b;
  int64_t bt_block;
  int64_t lse_h;
  int64_t lse_b;
};

struct SuffixScatterParams {
  const uint16_t* key;
  const uint16_t* value;
  const int64_t* block_ids;
  const int64_t* block_offsets;
  int64_t count;
  int64_t key_b;
  int64_t key_h;
  int64_t key_d;
  int64_t value_b;
  int64_t value_h;
  int64_t value_d;
};

inline int32_t block_table_value(const int32_t* block_table,
                                 const CpuGqaAttentionStrides& strides,
                                 int64_t seq_idx, int64_t block_col) {
  return block_table[seq_idx * strides.bt_b + block_col * strides.bt_block];
}

template <typename scalar_t>
inline void zero_one_head(scalar_t* output,
                          const CpuGqaAttentionStrides& strides,
                          int64_t seq_idx, int64_t q_head) {
  scalar_t* out = output + seq_idx * strides.out_b + q_head * strides.out_h;
  for (int64_t d = 0; d < kSupportedHeadDim; ++d) {
    out[d * strides.out_d] = 0;
  }
}

inline const uint16_t* kv_ptr(const uint16_t* cache,
                              const CpuGqaAttentionStrides& strides,
                              int64_t block_id, int64_t kv_head,
                              int64_t token_in_block) {
  return cache + block_id * strides.kv_block + kv_head * strides.kv_head +
         token_in_block * strides.kv_token;
}

inline uint16_t* kv_ptr_mut(uint16_t* cache,
                            const CpuGqaAttentionStrides& strides,
                            int64_t block_id, int64_t kv_head,
                            int64_t token_in_block) {
  return cache + block_id * strides.kv_block + kv_head * strides.kv_head +
         token_in_block * strides.kv_token;
}

inline const uint16_t* scatter_key_head_ptr(const SuffixScatterParams& scatter,
                                            int64_t seq_idx, int64_t kv_head) {
  return scatter.key + seq_idx * scatter.key_b + kv_head * scatter.key_h;
}

inline const uint16_t* scatter_value_head_ptr(
    const SuffixScatterParams& scatter, int64_t seq_idx, int64_t kv_head) {
  return scatter.value + seq_idx * scatter.value_b + kv_head * scatter.value_h;
}

inline void prefetch_bf16_head(const uint16_t* ptr) {
  _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(ptr + 32), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(ptr + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(ptr + 96), _MM_HINT_T0);
}

template <int64_t HeadsPerKV>
void gqa_suffix_attention_one_group(
    const uint16_t* query, uint16_t* key_cache, uint16_t* value_cache,
    const int32_t* block_table, const int32_t* seq_lens, uint16_t* output,
    float* output_lse, const CpuGqaAttentionStrides& strides,
    int64_t block_size, int64_t num_blocks, int64_t seq_idx, int64_t kv_head,
    float scale, const SuffixScatterParams* scatter) {
  const int64_t seq_len = seq_lens[seq_idx];
  constexpr int64_t heads_per_kv = HeadsPerKV;
  const int64_t q_head_base = kv_head * heads_per_kv;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();

  if (scatter != nullptr && seq_idx < scatter->count) {
    const int64_t scatter_block_id = scatter->block_ids[seq_idx];
    const int64_t scatter_block_offset = scatter->block_offsets[seq_idx];
    const uint16_t* scatter_k_head =
        scatter_key_head_ptr(*scatter, seq_idx, kv_head);
    const uint16_t* scatter_v_head =
        scatter_value_head_ptr(*scatter, seq_idx, kv_head);
    uint16_t* dst_key = kv_ptr_mut(key_cache, strides, scatter_block_id,
                                   kv_head, scatter_block_offset);
    uint16_t* dst_value = kv_ptr_mut(value_cache, strides, scatter_block_id,
                                     kv_head, scatter_block_offset);
    for (int64_t d = 0; d < kSupportedHeadDim; ++d) {
      dst_key[d * strides.kv_d] = scatter_k_head[d * scatter->key_d];
      dst_value[d * strides.kv_d] = scatter_v_head[d * scatter->value_d];
    }
  }

  if (seq_len <= 0) {
    for (int64_t q = 0; q < heads_per_kv; ++q) {
      const int64_t q_head = q_head_base + q;
      output_lse[q_head * strides.lse_h + seq_idx * strides.lse_b] = neg_inf;
      zero_one_head(output, strides, seq_idx, q_head);
    }
    return;
  }

  const uint16_t* q_ptrs[HeadsPerKV];
  for (int64_t q = 0; q < heads_per_kv; ++q) {
    const int64_t q_head = q_head_base + q;
    q_ptrs[q] = query + seq_idx * strides.q_b + q_head * strides.q_h;
  }

  alignas(32) std::array<float, HeadsPerKV * kSupportedHeadDim> q_f32;
  for (int64_t q = 0; q < heads_per_kv; ++q) {
    float* q_dst = q_f32.data() + q * kSupportedHeadDim;
    for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
      const __m256 q_vec = load8_bf16_as_f32(q_ptrs[q] + d * strides.q_d);
      _mm256_store_ps(q_dst + d, q_vec);
    }
  }

  if (seq_len <= kSuffixTwoPassMaxSeqLen) {
    alignas(32) thread_local std::array<float, kMaxHeadsPerKV *
                                                   kSuffixTwoPassMaxSeqLen>
        logits;
    float two_pass_max_logits[HeadsPerKV];
    std::fill(two_pass_max_logits, two_pass_max_logits + heads_per_kv, neg_inf);

    for (int64_t block_col = 0, t = 0; t < seq_len; ++block_col) {
      const int32_t block_id =
          block_table_value(block_table, strides, seq_idx, block_col);
  #ifndef NDEBUG
      TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                  "block_table contains an out-of-range block id: ", block_id,
                  " valid range is [0, ", num_blocks, ")");
  #endif
      const int64_t block_token_begin = block_col * block_size;
      const int64_t block_token_end =
          std::min(seq_len, block_token_begin + block_size);

      for (; t < block_token_end; ++t) {
        const int64_t token_in_block = t - block_token_begin;
        const uint16_t* k_ptr =
            kv_ptr(key_cache, strides, block_id, kv_head, token_in_block);
        if (t + 1 < block_token_end) {
          const int64_t next_token_in_block = token_in_block + 1;
          prefetch_bf16_head(kv_ptr(key_cache, strides, block_id, kv_head,
                                    next_token_in_block));
        }

        __m256 acc[HeadsPerKV];
        for (int64_t q = 0; q < heads_per_kv; ++q) {
          acc[q] = _mm256_setzero_ps();
        }

        for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
          const __m256 k_vec = load8_bf16_as_f32(k_ptr + d * strides.kv_d);
          for (int64_t q = 0; q < heads_per_kv; ++q) {
            const __m256 q_vec =
                _mm256_load_ps(q_f32.data() + q * kSupportedHeadDim + d);
            acc[q] = _mm256_fmadd_ps(q_vec, k_vec, acc[q]);
          }
        }

        for (int64_t q = 0; q < heads_per_kv; ++q) {
          const float logit = hreduce_ps(acc[q]) * scale;
          logits[q * kSuffixTwoPassMaxSeqLen + t] = logit;
          two_pass_max_logits[q] = std::max(two_pass_max_logits[q], logit);
        }
      }
    }

    alignas(32) std::array<float, HeadsPerKV * kSupportedHeadDim> out_f32;
    std::fill(out_f32.begin(), out_f32.end(), 0.0f);
    float two_pass_sums[HeadsPerKV];
    std::fill(two_pass_sums, two_pass_sums + heads_per_kv, 0.0f);

    for (int64_t block_col = 0, t = 0; t < seq_len; ++block_col) {
      const int32_t block_id =
          block_table_value(block_table, strides, seq_idx, block_col);
      const int64_t block_token_begin = block_col * block_size;
      const int64_t block_token_end =
          std::min(seq_len, block_token_begin + block_size);

      for (; t < block_token_end; ++t) {
        const int64_t token_in_block = t - block_token_begin;
        const uint16_t* v_ptr =
            kv_ptr(value_cache, strides, block_id, kv_head, token_in_block);
        if (t + 1 < block_token_end) {
          const int64_t next_token_in_block = token_in_block + 1;
          prefetch_bf16_head(kv_ptr(value_cache, strides, block_id, kv_head,
                                    next_token_in_block));
        }

        __m256 beta_vec[HeadsPerKV];
        for (int64_t q = 0; q < heads_per_kv; ++q) {
          const float beta = std::exp(logits[q * kSuffixTwoPassMaxSeqLen + t] -
                                      two_pass_max_logits[q]);
          two_pass_sums[q] += beta;
          beta_vec[q] = _mm256_set1_ps(beta);
        }

        for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
          const __m256 v_vec = load8_bf16_as_f32(v_ptr + d * strides.kv_d);
          for (int64_t q = 0; q < heads_per_kv; ++q) {
            float* out_dst = out_f32.data() + q * kSupportedHeadDim + d;
            const __m256 prev = _mm256_load_ps(out_dst);
            const __m256 next = _mm256_fmadd_ps(beta_vec[q], v_vec, prev);
            _mm256_store_ps(out_dst, next);
          }
        }
      }
    }

    for (int64_t q = 0; q < heads_per_kv; ++q) {
      const float inv_sum = 1.0f / two_pass_sums[q];
      const float lse = std::log(two_pass_sums[q]) + two_pass_max_logits[q];
      const int64_t q_head = q_head_base + q;
      output_lse[q_head * strides.lse_h + seq_idx * strides.lse_b] = lse;
      const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
      for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
        const __m256 out_vec = _mm256_mul_ps(
            _mm256_load_ps(out_f32.data() + q * kSupportedHeadDim + d),
            inv_sum_vec);
        uint16_t* out_ptr = output + seq_idx * strides.out_b +
                            q_head * strides.out_h + d * strides.out_d;
        store8_f32_as_bf16(out_vec, out_ptr);
      }
    }
    return;
  }

  alignas(32) std::array<float, HeadsPerKV * kSupportedHeadDim> out_f32;
  std::fill(out_f32.begin(), out_f32.end(), 0.0f);

  float max_logits[HeadsPerKV];
  std::fill(max_logits, max_logits + heads_per_kv, neg_inf);
  float sums[HeadsPerKV];
  std::fill(sums, sums + heads_per_kv, 0.0f);

  for (int64_t block_col = 0, t = 0; t < seq_len; ++block_col) {
    const int32_t block_id =
        block_table_value(block_table, strides, seq_idx, block_col);
  #ifndef NDEBUG
    TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                "block_table contains an out-of-range block id: ", block_id,
                " valid range is [0, ", num_blocks, ")");
  #endif
    const int64_t block_token_begin = block_col * block_size;
    const int64_t block_token_end =
        std::min(seq_len, block_token_begin + block_size);

    for (; t < block_token_end; ++t) {
      const int64_t token_in_block = t - block_token_begin;
      const uint16_t* k_ptr =
          kv_ptr(key_cache, strides, block_id, kv_head, token_in_block);
      const uint16_t* v_ptr =
          kv_ptr(value_cache, strides, block_id, kv_head, token_in_block);

      if (t + 1 < block_token_end) {
        const int64_t next_token_in_block = token_in_block + 1;
        prefetch_bf16_head(
            kv_ptr(key_cache, strides, block_id, kv_head, next_token_in_block));
        prefetch_bf16_head(kv_ptr(value_cache, strides, block_id, kv_head,
                                  next_token_in_block));
      }

      __m256 acc[HeadsPerKV];
      for (int64_t q = 0; q < heads_per_kv; ++q) {
        acc[q] = _mm256_setzero_ps();
      }

      for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
        const __m256 k_vec = load8_bf16_as_f32(k_ptr + d * strides.kv_d);
        for (int64_t q = 0; q < heads_per_kv; ++q) {
          const __m256 q_vec =
              _mm256_load_ps(q_f32.data() + q * kSupportedHeadDim + d);
          acc[q] = _mm256_fmadd_ps(q_vec, k_vec, acc[q]);
        }
      }

      float alpha[HeadsPerKV];
      float beta[HeadsPerKV];
      for (int64_t q = 0; q < heads_per_kv; ++q) {
        const float logit = hreduce_ps(acc[q]) * scale;
        const float old_max = max_logits[q];
        const float new_max = std::max(old_max, logit);
        alpha[q] = std::exp(old_max - new_max);
        beta[q] = std::exp(logit - new_max);
        sums[q] = sums[q] * alpha[q] + beta[q];
        max_logits[q] = new_max;
      }

      __m256 alpha_vec[HeadsPerKV];
      __m256 beta_vec[HeadsPerKV];
      for (int64_t q = 0; q < heads_per_kv; ++q) {
        alpha_vec[q] = _mm256_set1_ps(alpha[q]);
        beta_vec[q] = _mm256_set1_ps(beta[q]);
      }

      for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
        const __m256 v_vec = load8_bf16_as_f32(v_ptr + d * strides.kv_d);
        for (int64_t q = 0; q < heads_per_kv; ++q) {
          float* out_dst = out_f32.data() + q * kSupportedHeadDim + d;
          const __m256 prev = _mm256_load_ps(out_dst);
          const __m256 scaled = _mm256_mul_ps(prev, alpha_vec[q]);
          const __m256 next = _mm256_fmadd_ps(beta_vec[q], v_vec, scaled);
          _mm256_store_ps(out_dst, next);
        }
      }
    }
  }

  for (int64_t q = 0; q < heads_per_kv; ++q) {
    const float inv_sum = 1.0f / sums[q];
    const float lse = std::log(sums[q]) + max_logits[q];
    const int64_t q_head = q_head_base + q;
    output_lse[q_head * strides.lse_h + seq_idx * strides.lse_b] = lse;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
      const __m256 out_vec = _mm256_mul_ps(
          _mm256_load_ps(out_f32.data() + q * kSupportedHeadDim + d),
          inv_sum_vec);
      uint16_t* out_ptr = output + seq_idx * strides.out_b +
                          q_head * strides.out_h + d * strides.out_d;
      store8_f32_as_bf16(out_vec, out_ptr);
    }
  }
}

// Existing hybrid-KV suffix path. Keep this algorithm stable while alternate
// decode/prefill probes are evaluated.
template <int64_t HeadsPerKV>
void run_gqa_suffix_attention_groups(
    const uint16_t* q_ptr, uint16_t* k_ptr, uint16_t* v_ptr,
    const int32_t* bt_ptr, const int32_t* len_ptr, uint16_t* out_ptr,
    float* lse_ptr, const CpuGqaAttentionStrides& strides, int64_t batch,
    int64_t block_size, int64_t num_blocks, int64_t num_kv_heads, float scale_f,
    const SuffixScatterParams* scatter = nullptr) {
  at::parallel_for(0, batch * num_kv_heads, /*grain_size=*/8,
                   [&](int64_t begin, int64_t end) {
                     for (int64_t task = begin; task < end; ++task) {
                       const int64_t seq_idx = task / num_kv_heads;
                       const int64_t kv_head = task - seq_idx * num_kv_heads;
                       gqa_suffix_attention_one_group<HeadsPerKV>(
                           q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr,
                           lse_ptr, strides, block_size, num_blocks, seq_idx,
                           kv_head, scale_f, scatter);
                     }
                   });
}

// Dynamic two-pass paged attention used by benchmark-only decode/prefill entry
// points.
template <int64_t HeadsPerKV>
void gqa_paged_attention_twopass_dynamic_group(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_table, uint16_t* output,
    float* output_lse, const CpuGqaAttentionStrides& strides,
    int64_t block_size, int64_t num_blocks, int64_t query_row,
    int64_t block_table_row, int64_t output_row, int64_t seq_len,
    int64_t kv_head, float scale) {
  constexpr int64_t heads_per_kv = HeadsPerKV;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  const int64_t q_head_base = kv_head * heads_per_kv;

  if (seq_len <= 0) {
    for (int64_t q = 0; q < heads_per_kv; ++q) {
      const int64_t q_head = q_head_base + q;
      output_lse[q_head * strides.lse_h + output_row * strides.lse_b] = neg_inf;
      zero_one_head(output, strides, output_row, q_head);
    }
    return;
  }

  const uint16_t* q_ptrs[HeadsPerKV];
  for (int64_t q = 0; q < heads_per_kv; ++q) {
    const int64_t q_head = q_head_base + q;
    q_ptrs[q] = query + query_row * strides.q_b + q_head * strides.q_h;
  }

  alignas(32) std::array<float, HeadsPerKV * kSupportedHeadDim> q_f32;
  for (int64_t q = 0; q < heads_per_kv; ++q) {
    float* q_dst = q_f32.data() + q * kSupportedHeadDim;
    for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
      const __m256 q_vec = load8_bf16_as_f32(q_ptrs[q] + d * strides.q_d);
      _mm256_store_ps(q_dst + d, q_vec);
    }
  }

  thread_local std::vector<float> logits_storage;
  const int64_t logits_count = heads_per_kv * seq_len;
  if (static_cast<int64_t>(logits_storage.size()) < logits_count) {
    logits_storage.resize(logits_count);
  }
  float* logits = logits_storage.data();

  float max_logits[HeadsPerKV];
  std::fill(max_logits, max_logits + heads_per_kv, neg_inf);

  for (int64_t block_col = 0, t = 0; t < seq_len; ++block_col) {
    const int32_t block_id =
        block_table_value(block_table, strides, block_table_row, block_col);
  #ifndef NDEBUG
    TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                "block_table contains an out-of-range block id: ", block_id,
                " valid range is [0, ", num_blocks, ")");
  #endif
    const int64_t block_token_begin = block_col * block_size;
    const int64_t block_token_end =
        std::min(seq_len, block_token_begin + block_size);

    for (; t < block_token_end; ++t) {
      const int64_t token_in_block = t - block_token_begin;
      const uint16_t* k_ptr =
          kv_ptr(key_cache, strides, block_id, kv_head, token_in_block);
      if (t + 1 < block_token_end) {
        const int64_t next_token_in_block = token_in_block + 1;
        prefetch_bf16_head(
            kv_ptr(key_cache, strides, block_id, kv_head, next_token_in_block));
      }

      __m256 acc[HeadsPerKV];
      for (int64_t q = 0; q < heads_per_kv; ++q) {
        acc[q] = _mm256_setzero_ps();
      }

      for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
        const __m256 k_vec = load8_bf16_as_f32(k_ptr + d * strides.kv_d);
        for (int64_t q = 0; q < heads_per_kv; ++q) {
          const __m256 q_vec =
              _mm256_load_ps(q_f32.data() + q * kSupportedHeadDim + d);
          acc[q] = _mm256_fmadd_ps(q_vec, k_vec, acc[q]);
        }
      }

      for (int64_t q = 0; q < heads_per_kv; ++q) {
        const float logit = hreduce_ps(acc[q]) * scale;
        logits[q * seq_len + t] = logit;
        max_logits[q] = std::max(max_logits[q], logit);
      }
    }
  }

  alignas(32) std::array<float, HeadsPerKV * kSupportedHeadDim> out_f32;
  std::fill(out_f32.begin(), out_f32.end(), 0.0f);
  float sums[HeadsPerKV];
  std::fill(sums, sums + heads_per_kv, 0.0f);

  for (int64_t block_col = 0, t = 0; t < seq_len; ++block_col) {
    const int32_t block_id =
        block_table_value(block_table, strides, block_table_row, block_col);
    const int64_t block_token_begin = block_col * block_size;
    const int64_t block_token_end =
        std::min(seq_len, block_token_begin + block_size);

    for (; t < block_token_end; ++t) {
      const int64_t token_in_block = t - block_token_begin;
      const uint16_t* v_ptr =
          kv_ptr(value_cache, strides, block_id, kv_head, token_in_block);
      if (t + 1 < block_token_end) {
        const int64_t next_token_in_block = token_in_block + 1;
        prefetch_bf16_head(kv_ptr(value_cache, strides, block_id, kv_head,
                                  next_token_in_block));
      }

      __m256 beta_vec[HeadsPerKV];
      for (int64_t q = 0; q < heads_per_kv; ++q) {
        const float beta = std::exp(logits[q * seq_len + t] - max_logits[q]);
        sums[q] += beta;
        beta_vec[q] = _mm256_set1_ps(beta);
      }

      for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
        const __m256 v_vec = load8_bf16_as_f32(v_ptr + d * strides.kv_d);
        for (int64_t q = 0; q < heads_per_kv; ++q) {
          float* out_dst = out_f32.data() + q * kSupportedHeadDim + d;
          const __m256 prev = _mm256_load_ps(out_dst);
          const __m256 next = _mm256_fmadd_ps(beta_vec[q], v_vec, prev);
          _mm256_store_ps(out_dst, next);
        }
      }
    }
  }

  for (int64_t q = 0; q < heads_per_kv; ++q) {
    const float inv_sum = 1.0f / sums[q];
    const float lse = std::log(sums[q]) + max_logits[q];
    const int64_t q_head = q_head_base + q;
    output_lse[q_head * strides.lse_h + output_row * strides.lse_b] = lse;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    for (int64_t d = 0; d < kSupportedHeadDim; d += kVecWidth) {
      const __m256 out_vec = _mm256_mul_ps(
          _mm256_load_ps(out_f32.data() + q * kSupportedHeadDim + d),
          inv_sum_vec);
      uint16_t* out_ptr = output + output_row * strides.out_b +
                          q_head * strides.out_h + d * strides.out_d;
      store8_f32_as_bf16(out_vec, out_ptr);
    }
  }
}

template <int64_t HeadsPerKV>
void run_gqa_decode_attention_groups(
    const uint16_t* q_ptr, const uint16_t* k_ptr, const uint16_t* v_ptr,
    const int32_t* bt_ptr, const int32_t* len_ptr, uint16_t* out_ptr,
    float* lse_ptr, const CpuGqaAttentionStrides& strides, int64_t batch,
    int64_t block_size, int64_t num_blocks, int64_t num_kv_heads,
    float scale_f) {
  at::parallel_for(0, batch * num_kv_heads, /*grain_size=*/8,
                   [&](int64_t begin, int64_t end) {
                     for (int64_t task = begin; task < end; ++task) {
                       const int64_t seq_idx = task / num_kv_heads;
                       const int64_t kv_head = task - seq_idx * num_kv_heads;
                       gqa_paged_attention_twopass_dynamic_group<HeadsPerKV>(
                           q_ptr, k_ptr, v_ptr, bt_ptr, out_ptr, lse_ptr,
                           strides, block_size, num_blocks, seq_idx, seq_idx,
                           seq_idx, len_ptr[seq_idx], kv_head, scale_f);
                     }
                   });
}

template <int64_t HeadsPerKV>
void run_gqa_prefill_attention_groups(
    const uint16_t* q_ptr, const uint16_t* k_ptr, const uint16_t* v_ptr,
    const int32_t* bt_ptr, const int64_t* query_to_seq_ptr,
    const int32_t* len_ptr, uint16_t* out_ptr, float* lse_ptr,
    const CpuGqaAttentionStrides& strides, int64_t num_tokens,
    int64_t block_size, int64_t num_blocks, int64_t num_kv_heads,
    float scale_f) {
  at::parallel_for(0, num_tokens * num_kv_heads, /*grain_size=*/8,
                   [&](int64_t begin, int64_t end) {
                     for (int64_t task = begin; task < end; ++task) {
                       const int64_t token_idx = task / num_kv_heads;
                       const int64_t kv_head = task - token_idx * num_kv_heads;
                       const int64_t seq_idx = query_to_seq_ptr[token_idx];
                       gqa_paged_attention_twopass_dynamic_group<HeadsPerKV>(
                           q_ptr, k_ptr, v_ptr, bt_ptr, out_ptr, lse_ptr,
                           strides, block_size, num_blocks, token_idx, seq_idx,
                           token_idx, len_ptr[token_idx], kv_head, scale_f);
                     }
                   });
}

#endif  // VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA

void validate_gqa_suffix_attention_tensors(const at::Tensor& query,
                                           const at::Tensor& key_cache,
                                           const at::Tensor& value_cache,
                                           const at::Tensor& block_table,
                                           const at::Tensor& seq_lens,
                                           const at::Tensor& output,
                                           const at::Tensor& output_lse) {
  TORCH_CHECK(query.device().is_cpu(), "query must be a CPU tensor");
  TORCH_CHECK(key_cache.device().is_cpu(), "key_cache must be a CPU tensor");
  TORCH_CHECK(value_cache.device().is_cpu(),
              "value_cache must be a CPU tensor");
  TORCH_CHECK(block_table.device().is_cpu(),
              "block_table must be a CPU tensor");
  TORCH_CHECK(seq_lens.device().is_cpu(), "seq_lens must be a CPU tensor");
  TORCH_CHECK(output.device().is_cpu(), "output must be a CPU tensor");
  TORCH_CHECK(output_lse.device().is_cpu(), "output_lse must be a CPU tensor");

  TORCH_CHECK(query.dtype() == at::kBFloat16, "query must be bfloat16");
  TORCH_CHECK(key_cache.dtype() == at::kBFloat16, "key_cache must be bfloat16");
  TORCH_CHECK(value_cache.dtype() == at::kBFloat16,
              "value_cache must be bfloat16");
  TORCH_CHECK(output.dtype() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(output_lse.dtype() == at::kFloat, "output_lse must be float32");
  TORCH_CHECK(block_table.dtype() == at::kInt,
              "block_table must be int32 for the COTS suffix kernel");
  TORCH_CHECK(seq_lens.dtype() == at::kInt,
              "seq_lens must be int32 for the COTS suffix kernel");

  TORCH_CHECK(query.dim() == 3, "query must have shape [B, num_q_heads, 128]");
  TORCH_CHECK(output.dim() == 3,
              "output must have shape [B, num_q_heads, 128]");
  TORCH_CHECK(
      key_cache.dim() == 4,
      "key_cache must have shape [blocks, num_kv_heads, block_size, 128]");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must have shape [blocks, num_kv_heads, block_size, 128]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(output_lse.dim() == 2,
              "output_lse must have shape [num_q_heads, B]");

  const int64_t batch = query.size(0);
  const int64_t num_q_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t block_size = key_cache.size(2);
  TORCH_CHECK(batch >= 0, "query batch must be non-negative");
  TORCH_CHECK(num_q_heads > 0, "query must have at least one head");
  TORCH_CHECK(num_kv_heads > 0, "key_cache must have at least one KV head");
  TORCH_CHECK(block_size > 0, "key_cache block_size must be positive");
  TORCH_CHECK(head_dim == kSupportedHeadDim,
              "COTS suffix attention currently supports head_dim=128, got ",
              head_dim);
  TORCH_CHECK(key_cache.size(3) == head_dim,
              "key_cache head_dim must match query head_dim");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads, got ",
              num_q_heads, " and ", num_kv_heads);
  const int64_t heads_per_kv = num_q_heads / num_kv_heads;
  TORCH_CHECK(heads_per_kv <= kMaxHeadsPerKV,
              "COTS suffix attention supports at most ", kMaxHeadsPerKV,
              " query heads per KV head, got ", heads_per_kv);

  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    const int64_t seq_len = len_ptr[seq_idx];
    TORCH_CHECK(seq_len >= 0, "seq_lens contains a negative sequence length");
    const int64_t needed_blocks = (seq_len + block_size - 1) / block_size;
    TORCH_CHECK(needed_blocks <= block_table.size(1),
                "block_table has too few columns for seq_lens: needed ",
                needed_blocks, ", got ", block_table.size(1));
  }
  TORCH_CHECK(output.sizes() == query.sizes(), "output shape must match query");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "value_cache shape must match key_cache");
  TORCH_CHECK(block_table.size(0) == batch,
              "block_table batch dimension must match query");
  TORCH_CHECK(seq_lens.size(0) == batch,
              "seq_lens size must match query batch");
  TORCH_CHECK(output_lse.size(0) == num_q_heads && output_lse.size(1) == batch,
              "output_lse must have shape [num_q_heads, B]");

  TORCH_CHECK(query.stride(2) == 1, "query head_dim must be contiguous");
  TORCH_CHECK(key_cache.stride(3) == 1,
              "key_cache head_dim must be contiguous");
  TORCH_CHECK(value_cache.stride(3) == 1,
              "value_cache head_dim must be contiguous");
  TORCH_CHECK(output.stride(2) == 1, "output head_dim must be contiguous");
  TORCH_CHECK(block_table.is_contiguous(),
              "block_table must be contiguous in the COTS suffix kernel");
  TORCH_CHECK(seq_lens.is_contiguous(),
              "seq_lens must be contiguous in the COTS suffix kernel");
  TORCH_CHECK(output_lse.stride(1) == 1,
              "output_lse sequence dimension must be contiguous");

  const auto* bt_ptr = block_table.data_ptr<int32_t>();
  const int64_t bt_b = block_table.stride(0);
  const int64_t bt_block = block_table.stride(1);
  const int64_t num_blocks = key_cache.size(0);
  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    const int64_t seq_len = len_ptr[seq_idx];
    const int64_t needed_blocks = (seq_len + block_size - 1) / block_size;
    for (int64_t block_col = 0; block_col < needed_blocks; ++block_col) {
      const int32_t block_id = bt_ptr[seq_idx * bt_b + block_col * bt_block];
      TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                  "block_table contains an out-of-range block id at row ",
                  seq_idx, ", col ", block_col, ": ", block_id,
                  " valid range is [0, ", num_blocks, ")");
    }
  }
}

void validate_gqa_prefill_attention_tensors(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_table,
    const at::Tensor& query_to_seq, const at::Tensor& seq_lens,
    const at::Tensor& output, const at::Tensor& output_lse) {
  TORCH_CHECK(query.device().is_cpu(), "query must be a CPU tensor");
  TORCH_CHECK(key_cache.device().is_cpu(), "key_cache must be a CPU tensor");
  TORCH_CHECK(value_cache.device().is_cpu(),
              "value_cache must be a CPU tensor");
  TORCH_CHECK(block_table.device().is_cpu(),
              "block_table must be a CPU tensor");
  TORCH_CHECK(query_to_seq.device().is_cpu(),
              "query_to_seq must be a CPU tensor");
  TORCH_CHECK(seq_lens.device().is_cpu(), "seq_lens must be a CPU tensor");
  TORCH_CHECK(output.device().is_cpu(), "output must be a CPU tensor");
  TORCH_CHECK(output_lse.device().is_cpu(), "output_lse must be a CPU tensor");

  TORCH_CHECK(query.dtype() == at::kBFloat16, "query must be bfloat16");
  TORCH_CHECK(key_cache.dtype() == at::kBFloat16, "key_cache must be bfloat16");
  TORCH_CHECK(value_cache.dtype() == at::kBFloat16,
              "value_cache must be bfloat16");
  TORCH_CHECK(output.dtype() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(output_lse.dtype() == at::kFloat, "output_lse must be float32");
  TORCH_CHECK(block_table.dtype() == at::kInt,
              "block_table must be int32 for the COTS prefill kernel");
  TORCH_CHECK(query_to_seq.dtype() == at::kLong,
              "query_to_seq must be int64 for the COTS prefill kernel");
  TORCH_CHECK(seq_lens.dtype() == at::kInt,
              "seq_lens must be int32 for the COTS prefill kernel");

  TORCH_CHECK(query.dim() == 3, "query must have shape [T, num_q_heads, 128]");
  TORCH_CHECK(output.dim() == 3,
              "output must have shape [T, num_q_heads, 128]");
  TORCH_CHECK(
      key_cache.dim() == 4,
      "key_cache must have shape [blocks, num_kv_heads, block_size, 128]");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must have shape [blocks, num_kv_heads, block_size, 128]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(query_to_seq.dim() == 1, "query_to_seq must be 1D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(output_lse.dim() == 2,
              "output_lse must have shape [num_q_heads, T]");

  const int64_t num_tokens = query.size(0);
  const int64_t num_q_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t block_size = key_cache.size(2);
  TORCH_CHECK(num_tokens >= 0, "query token count must be non-negative");
  TORCH_CHECK(num_q_heads > 0, "query must have at least one head");
  TORCH_CHECK(num_kv_heads > 0, "key_cache must have at least one KV head");
  TORCH_CHECK(block_size > 0, "key_cache block_size must be positive");
  TORCH_CHECK(head_dim == kSupportedHeadDim,
              "COTS prefill attention currently supports head_dim=128, got ",
              head_dim);
  TORCH_CHECK(key_cache.size(3) == head_dim,
              "key_cache head_dim must match query head_dim");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads, got ",
              num_q_heads, " and ", num_kv_heads);
  const int64_t heads_per_kv = num_q_heads / num_kv_heads;
  TORCH_CHECK(heads_per_kv <= kMaxHeadsPerKV,
              "COTS prefill attention supports at most ", kMaxHeadsPerKV,
              " query heads per KV head, got ", heads_per_kv);

  TORCH_CHECK(output.sizes() == query.sizes(), "output shape must match query");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "value_cache shape must match key_cache");
  TORCH_CHECK(query_to_seq.size(0) == num_tokens,
              "query_to_seq size must match query token count");
  TORCH_CHECK(seq_lens.size(0) == num_tokens,
              "seq_lens size must match query token count");
  TORCH_CHECK(
      output_lse.size(0) == num_q_heads && output_lse.size(1) == num_tokens,
      "output_lse must have shape [num_q_heads, T]");

  TORCH_CHECK(query.stride(2) == 1, "query head_dim must be contiguous");
  TORCH_CHECK(key_cache.stride(3) == 1,
              "key_cache head_dim must be contiguous");
  TORCH_CHECK(value_cache.stride(3) == 1,
              "value_cache head_dim must be contiguous");
  TORCH_CHECK(output.stride(2) == 1, "output head_dim must be contiguous");
  TORCH_CHECK(block_table.is_contiguous(),
              "block_table must be contiguous in the COTS prefill kernel");
  TORCH_CHECK(query_to_seq.is_contiguous(),
              "query_to_seq must be contiguous in the COTS prefill kernel");
  TORCH_CHECK(seq_lens.is_contiguous(),
              "seq_lens must be contiguous in the COTS prefill kernel");
  TORCH_CHECK(output_lse.stride(1) == 1,
              "output_lse sequence dimension must be contiguous");

  const auto* bt_ptr = block_table.data_ptr<int32_t>();
  const auto* req_ptr = query_to_seq.data_ptr<int64_t>();
  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  const int64_t bt_b = block_table.stride(0);
  const int64_t bt_block = block_table.stride(1);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_reqs = block_table.size(0);
  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t req_idx = req_ptr[token_idx];
    TORCH_CHECK(req_idx >= 0 && req_idx < num_reqs,
                "query_to_seq contains an out-of-range request id");
    const int64_t seq_len = len_ptr[token_idx];
    TORCH_CHECK(seq_len >= 0, "seq_lens contains a negative sequence length");
    const int64_t needed_blocks = (seq_len + block_size - 1) / block_size;
    TORCH_CHECK(needed_blocks <= block_table.size(1),
                "block_table has too few columns for seq_lens: needed ",
                needed_blocks, ", got ", block_table.size(1));
    for (int64_t block_col = 0; block_col < needed_blocks; ++block_col) {
      const int32_t block_id = bt_ptr[req_idx * bt_b + block_col * bt_block];
      TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                  "block_table contains an out-of-range block id at row ",
                  req_idx, ", col ", block_col, ": ", block_id,
                  " valid range is [0, ", num_blocks, ")");
    }
  }
}

}  // namespace

void gqa_bf16_suffix_attention_unchecked_at(const at::Tensor& query,
                                            const at::Tensor& key_cache,
                                            const at::Tensor& value_cache,
                                            const at::Tensor& block_table,
                                            const at::Tensor& seq_lens,
                                            double scale, at::Tensor& output,
                                            at::Tensor& output_lse);
void gqa_bf16_suffix_attention_scatter_unchecked_at(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_table,
    const at::Tensor& seq_lens, double scale, at::Tensor& output,
    at::Tensor& output_lse, const at::Tensor& scatter_key,
    const at::Tensor& scatter_value, const at::Tensor& scatter_block_ids,
    const at::Tensor& scatter_block_offsets);

void gqa_bf16_decode_attention_at(const at::Tensor& query,
                                  const at::Tensor& key_cache,
                                  const at::Tensor& value_cache,
                                  const at::Tensor& block_table,
                                  const at::Tensor& seq_lens, double scale,
                                  at::Tensor& output, at::Tensor& output_lse) {
  validate_gqa_suffix_attention_tensors(
      query, key_cache, value_cache, block_table, seq_lens, output, output_lse);
#if VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA
  const int64_t batch = query.size(0);
  const int64_t block_size = key_cache.size(2);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t heads_per_kv = query.size(1) / num_kv_heads;
  const auto strides = CpuGqaAttentionStrides{
      query.stride(0),      query.stride(1),       query.stride(2),
      key_cache.stride(0),  key_cache.stride(1),   key_cache.stride(2),
      key_cache.stride(3),  output.stride(0),      output.stride(1),
      output.stride(2),     block_table.stride(0), block_table.stride(1),
      output_lse.stride(0), output_lse.stride(1),
  };

  const auto* q_ptr = reinterpret_cast<const uint16_t*>(query.data_ptr());
  const auto* k_ptr = reinterpret_cast<const uint16_t*>(key_cache.data_ptr());
  const auto* v_ptr = reinterpret_cast<const uint16_t*>(value_cache.data_ptr());
  const auto* bt_ptr = block_table.data_ptr<int32_t>();
  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  auto* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr());
  auto* lse_ptr = output_lse.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);

  switch (heads_per_kv) {
    case 1:
      run_gqa_decode_attention_groups<1>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 2:
      run_gqa_decode_attention_groups<2>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 3:
      run_gqa_decode_attention_groups<3>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 4:
      run_gqa_decode_attention_groups<4>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 5:
      run_gqa_decode_attention_groups<5>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 6:
      run_gqa_decode_attention_groups<6>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 7:
      run_gqa_decode_attention_groups<7>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 8:
      run_gqa_decode_attention_groups<8>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    default:
      TORCH_CHECK(false, "unsupported COTS GQA heads_per_kv=", heads_per_kv);
  }
#else
  TORCH_CHECK(false, "COTS decode attention requires AVX2+FMA");
#endif
}

void gqa_bf16_prefill_attention_at(const at::Tensor& query,
                                   const at::Tensor& key_cache,
                                   const at::Tensor& value_cache,
                                   const at::Tensor& block_table,
                                   const at::Tensor& query_to_seq,
                                   const at::Tensor& seq_lens, double scale,
                                   at::Tensor& output, at::Tensor& output_lse) {
  validate_gqa_prefill_attention_tensors(query, key_cache, value_cache,
                                         block_table, query_to_seq, seq_lens,
                                         output, output_lse);
#if VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA
  const int64_t num_tokens = query.size(0);
  const int64_t block_size = key_cache.size(2);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t heads_per_kv = query.size(1) / num_kv_heads;
  const auto strides = CpuGqaAttentionStrides{
      query.stride(0),      query.stride(1),       query.stride(2),
      key_cache.stride(0),  key_cache.stride(1),   key_cache.stride(2),
      key_cache.stride(3),  output.stride(0),      output.stride(1),
      output.stride(2),     block_table.stride(0), block_table.stride(1),
      output_lse.stride(0), output_lse.stride(1),
  };

  const auto* q_ptr = reinterpret_cast<const uint16_t*>(query.data_ptr());
  const auto* k_ptr = reinterpret_cast<const uint16_t*>(key_cache.data_ptr());
  const auto* v_ptr = reinterpret_cast<const uint16_t*>(value_cache.data_ptr());
  const auto* bt_ptr = block_table.data_ptr<int32_t>();
  const auto* req_ptr = query_to_seq.data_ptr<int64_t>();
  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  auto* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr());
  auto* lse_ptr = output_lse.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);

  switch (heads_per_kv) {
    case 1:
      run_gqa_prefill_attention_groups<1>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 2:
      run_gqa_prefill_attention_groups<2>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 3:
      run_gqa_prefill_attention_groups<3>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 4:
      run_gqa_prefill_attention_groups<4>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 5:
      run_gqa_prefill_attention_groups<5>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 6:
      run_gqa_prefill_attention_groups<6>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 7:
      run_gqa_prefill_attention_groups<7>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 8:
      run_gqa_prefill_attention_groups<8>(
          q_ptr, k_ptr, v_ptr, bt_ptr, req_ptr, len_ptr, out_ptr, lse_ptr,
          strides, num_tokens, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    default:
      TORCH_CHECK(false, "unsupported COTS GQA heads_per_kv=", heads_per_kv);
  }
#else
  TORCH_CHECK(false, "COTS prefill attention requires AVX2+FMA");
#endif
}

void gqa_bf16_suffix_attention_at(const at::Tensor& query,
                                  const at::Tensor& key_cache,
                                  const at::Tensor& value_cache,
                                  const at::Tensor& block_table,
                                  const at::Tensor& seq_lens, double scale,
                                  at::Tensor& output, at::Tensor& output_lse) {
  validate_gqa_suffix_attention_tensors(
      query, key_cache, value_cache, block_table, seq_lens, output, output_lse);
  gqa_bf16_suffix_attention_unchecked_at(query, key_cache, value_cache,
                                         block_table, seq_lens, scale, output,
                                         output_lse);
}

void gqa_bf16_suffix_attention_unchecked_at(const at::Tensor& query,
                                            const at::Tensor& key_cache,
                                            const at::Tensor& value_cache,
                                            const at::Tensor& block_table,
                                            const at::Tensor& seq_lens,
                                            double scale, at::Tensor& output,
                                            at::Tensor& output_lse) {
#if VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA
  const int64_t batch = query.size(0);
  const int64_t block_size = key_cache.size(2);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t heads_per_kv = query.size(1) / num_kv_heads;
  const auto strides = CpuGqaAttentionStrides{
      query.stride(0),      query.stride(1),       query.stride(2),
      key_cache.stride(0),  key_cache.stride(1),   key_cache.stride(2),
      key_cache.stride(3),  output.stride(0),      output.stride(1),
      output.stride(2),     block_table.stride(0), block_table.stride(1),
      output_lse.stride(0), output_lse.stride(1),
  };

  const auto* q_ptr = reinterpret_cast<const uint16_t*>(query.data_ptr());
  auto* k_ptr = reinterpret_cast<uint16_t*>(key_cache.data_ptr());
  auto* v_ptr = reinterpret_cast<uint16_t*>(value_cache.data_ptr());
  const auto* bt_ptr = block_table.data_ptr<int32_t>();
  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  auto* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr());
  auto* lse_ptr = output_lse.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);

  switch (heads_per_kv) {
    case 1:
      run_gqa_suffix_attention_groups<1>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 2:
      run_gqa_suffix_attention_groups<2>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 3:
      run_gqa_suffix_attention_groups<3>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 4:
      run_gqa_suffix_attention_groups<4>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 5:
      run_gqa_suffix_attention_groups<5>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 6:
      run_gqa_suffix_attention_groups<6>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 7:
      run_gqa_suffix_attention_groups<7>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    case 8:
      run_gqa_suffix_attention_groups<8>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f);
      break;
    default:
      TORCH_CHECK(false, "unsupported COTS GQA heads_per_kv=", heads_per_kv);
  }
#else
  TORCH_CHECK(false, "COTS suffix attention requires AVX2+FMA");
#endif
}

void gqa_bf16_suffix_attention_scatter_unchecked_at(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_table,
    const at::Tensor& seq_lens, double scale, at::Tensor& output,
    at::Tensor& output_lse, const at::Tensor& scatter_key,
    const at::Tensor& scatter_value, const at::Tensor& scatter_block_ids,
    const at::Tensor& scatter_block_offsets) {
#if VLLM_COTS_CPU_GQA_ATTN_HAS_AVX2_FMA
  const int64_t batch = query.size(0);
  const int64_t block_size = key_cache.size(2);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t heads_per_kv = query.size(1) / num_kv_heads;
  const int64_t scatter_count = scatter_block_ids.numel();
  TORCH_CHECK(scatter_count >= 0 && scatter_count <= batch,
              "COTS suffix fused scatter count must be in [0, batch], got ",
              scatter_count, " for batch ", batch);
  TORCH_CHECK(scatter_block_offsets.numel() == scatter_count,
              "COTS suffix fused scatter ids/offsets length mismatch");
  TORCH_CHECK(scatter_key.dim() == 3 && scatter_value.dim() == 3,
              "COTS suffix fused scatter K/V must be [N, num_kv_heads, 128]");
  TORCH_CHECK(scatter_key.size(0) >= scatter_count &&
                  scatter_value.size(0) >= scatter_count,
              "COTS suffix fused scatter K/V row count is smaller than "
              "scatter_count");
  TORCH_CHECK(scatter_key.size(1) == num_kv_heads &&
                  scatter_value.size(1) == num_kv_heads &&
                  scatter_key.size(2) == kSupportedHeadDim &&
                  scatter_value.size(2) == kSupportedHeadDim,
              "COTS suffix fused scatter K/V shape must match KV heads and "
              "head_dim=128");
  TORCH_CHECK(scatter_key.dtype() == at::kBFloat16 &&
                  scatter_value.dtype() == at::kBFloat16,
              "COTS suffix fused scatter K/V must be BF16");
  TORCH_CHECK(scatter_block_ids.dtype() == at::kLong &&
                  scatter_block_offsets.dtype() == at::kLong,
              "COTS suffix fused scatter block metadata must be int64");
  TORCH_CHECK(scatter_block_ids.is_contiguous() &&
                  scatter_block_offsets.is_contiguous(),
              "COTS suffix fused scatter block metadata must be contiguous");

  const auto* scatter_block_ids_ptr = scatter_block_ids.data_ptr<int64_t>();
  const auto* scatter_block_offsets_ptr =
      scatter_block_offsets.data_ptr<int64_t>();
  for (int64_t i = 0; i < scatter_count; ++i) {
    const int64_t block_id = scatter_block_ids_ptr[i];
    const int64_t block_offset = scatter_block_offsets_ptr[i];
    TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                "COTS suffix fused scatter block id out of range");
    TORCH_CHECK(block_offset >= 0 && block_offset < block_size,
                "COTS suffix fused scatter block offset out of range");
  }

  const auto strides = CpuGqaAttentionStrides{
      query.stride(0),      query.stride(1),       query.stride(2),
      key_cache.stride(0),  key_cache.stride(1),   key_cache.stride(2),
      key_cache.stride(3),  output.stride(0),      output.stride(1),
      output.stride(2),     block_table.stride(0), block_table.stride(1),
      output_lse.stride(0), output_lse.stride(1),
  };
  const SuffixScatterParams scatter{
      reinterpret_cast<const uint16_t*>(scatter_key.data_ptr()),
      reinterpret_cast<const uint16_t*>(scatter_value.data_ptr()),
      scatter_block_ids_ptr,
      scatter_block_offsets_ptr,
      scatter_count,
      scatter_key.stride(0),
      scatter_key.stride(1),
      scatter_key.stride(2),
      scatter_value.stride(0),
      scatter_value.stride(1),
      scatter_value.stride(2),
  };

  const auto* q_ptr = reinterpret_cast<const uint16_t*>(query.data_ptr());
  auto* k_ptr = reinterpret_cast<uint16_t*>(key_cache.data_ptr());
  auto* v_ptr = reinterpret_cast<uint16_t*>(value_cache.data_ptr());
  const auto* bt_ptr = block_table.data_ptr<int32_t>();
  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  auto* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr());
  auto* lse_ptr = output_lse.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);

  switch (heads_per_kv) {
    case 1:
      run_gqa_suffix_attention_groups<1>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 2:
      run_gqa_suffix_attention_groups<2>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 3:
      run_gqa_suffix_attention_groups<3>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 4:
      run_gqa_suffix_attention_groups<4>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 5:
      run_gqa_suffix_attention_groups<5>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 6:
      run_gqa_suffix_attention_groups<6>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 7:
      run_gqa_suffix_attention_groups<7>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    case 8:
      run_gqa_suffix_attention_groups<8>(
          q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr, lse_ptr, strides,
          batch, block_size, num_blocks, num_kv_heads, scale_f, &scatter);
      break;
    default:
      TORCH_CHECK(false, "unsupported COTS GQA heads_per_kv=", heads_per_kv);
  }
#else
  TORCH_CHECK(false, "COTS suffix attention requires AVX2+FMA");
#endif
}

void gqa_bf16_scatter_suffix_kv_unchecked_at(const at::Tensor& key,
                                             const at::Tensor& value,
                                             const at::Tensor& block_ids,
                                             const at::Tensor& block_offsets,
                                             at::Tensor& key_cache,
                                             at::Tensor& value_cache) {
  const int64_t n = block_ids.numel();
  const int64_t num_kv_heads = key.size(1);
  const int64_t head_dim = key.size(2);
  const auto* key_ptr = reinterpret_cast<const uint16_t*>(key.data_ptr());
  const auto* value_ptr = reinterpret_cast<const uint16_t*>(value.data_ptr());
  auto* key_cache_ptr = reinterpret_cast<uint16_t*>(key_cache.data_ptr());
  auto* value_cache_ptr = reinterpret_cast<uint16_t*>(value_cache.data_ptr());
  const auto* block_ids_ptr = block_ids.data_ptr<int64_t>();
  const auto* block_offsets_ptr = block_offsets.data_ptr<int64_t>();

  const int64_t key_b = key.stride(0);
  const int64_t key_h = key.stride(1);
  const int64_t value_b = value.stride(0);
  const int64_t value_h = value.stride(1);
  const int64_t cache_block = key_cache.stride(0);
  const int64_t cache_head = key_cache.stride(1);
  const int64_t cache_token = key_cache.stride(2);
  const int64_t value_cache_block = value_cache.stride(0);
  const int64_t value_cache_head = value_cache.stride(1);
  const int64_t value_cache_token = value_cache.stride(2);

  for (int64_t i = 0; i < n; ++i) {
    const int64_t block_id = block_ids_ptr[i];
    const int64_t block_offset = block_offsets_ptr[i];
    for (int64_t h = 0; h < num_kv_heads; ++h) {
      const uint16_t* src_key = key_ptr + i * key_b + h * key_h;
      const uint16_t* src_value = value_ptr + i * value_b + h * value_h;
      uint16_t* dst_key = key_cache_ptr + block_id * cache_block +
                          h * cache_head + block_offset * cache_token;
      uint16_t* dst_value = value_cache_ptr + block_id * value_cache_block +
                            h * value_cache_head +
                            block_offset * value_cache_token;
      std::memcpy(dst_key, src_key, head_dim * sizeof(uint16_t));
      std::memcpy(dst_value, src_value, head_dim * sizeof(uint16_t));
    }
  }
}

void gqa_bf16_scatter_suffix_kv_at(const at::Tensor& key,
                                   const at::Tensor& value,
                                   const at::Tensor& block_ids,
                                   const at::Tensor& block_offsets,
                                   at::Tensor& key_cache,
                                   at::Tensor& value_cache) {
  TORCH_CHECK(key.device().is_cpu(), "key must be a CPU tensor");
  TORCH_CHECK(value.device().is_cpu(), "value must be a CPU tensor");
  TORCH_CHECK(block_ids.device().is_cpu(), "block_ids must be a CPU tensor");
  TORCH_CHECK(block_offsets.device().is_cpu(),
              "block_offsets must be a CPU tensor");
  TORCH_CHECK(key_cache.device().is_cpu(), "key_cache must be a CPU tensor");
  TORCH_CHECK(value_cache.device().is_cpu(),
              "value_cache must be a CPU tensor");

  TORCH_CHECK(key.dtype() == at::kBFloat16, "key must be bfloat16");
  TORCH_CHECK(value.dtype() == at::kBFloat16, "value must be bfloat16");
  TORCH_CHECK(key_cache.dtype() == at::kBFloat16, "key_cache must be bfloat16");
  TORCH_CHECK(value_cache.dtype() == at::kBFloat16,
              "value_cache must be bfloat16");
  TORCH_CHECK(block_ids.dtype() == at::kLong,
              "block_ids must be int64 for the COTS scatter fast path");
  TORCH_CHECK(block_offsets.dtype() == at::kLong,
              "block_offsets must be int64 for the COTS scatter fast path");

  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key/value must have shape [N, num_kv_heads, 128]");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");
  const int64_t num_kv_heads = key.size(1);
  const int64_t head_dim = key.size(2);
  TORCH_CHECK(num_kv_heads > 0, "key/value must have at least one KV head");
  TORCH_CHECK(head_dim == kSupportedHeadDim,
              "COTS suffix KV scatter currently supports head_dim=128, got ",
              head_dim);
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
              "key/value head_dim must be contiguous");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "key/value caches must have shape [blocks, num_kv_heads, "
              "block_size, 128]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "key/value cache shapes must match");
  TORCH_CHECK(
      key_cache.size(1) == num_kv_heads && key_cache.size(3) == head_dim,
      "key/value cache shape must match key/value KV heads and head_dim");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "key/value cache head_dim must be contiguous");
  TORCH_CHECK(block_ids.dim() == 1 && block_offsets.dim() == 1,
              "block_ids/block_offsets must be 1D tensors");
  TORCH_CHECK(block_ids.numel() == block_offsets.numel(),
              "block_ids/block_offsets lengths must match");
  TORCH_CHECK(key.size(0) >= block_ids.numel(),
              "key/value must contain one row for each block id");

  const int64_t n = block_ids.numel();
  const int64_t block_size = key_cache.size(2);
  const auto* block_ids_ptr = block_ids.data_ptr<int64_t>();
  const auto* block_offsets_ptr = block_offsets.data_ptr<int64_t>();

  for (int64_t i = 0; i < n; ++i) {
    const int64_t block_id = block_ids_ptr[i];
    const int64_t block_offset = block_offsets_ptr[i];
    TORCH_CHECK(block_id >= 0 && block_id < key_cache.size(0),
                "block_ids contains an out-of-range block id");
    TORCH_CHECK(block_offset >= 0 && block_offset < block_size,
                "block_offsets contains an out-of-range block offset");
  }
  gqa_bf16_scatter_suffix_kv_unchecked_at(key, value, block_ids, block_offsets,
                                          key_cache, value_cache);
}

}  // namespace cots
}  // namespace vllm
