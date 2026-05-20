// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// COTS Phase 2 CPU suffix attention prototype.
//
// This is intentionally thesis-owned and narrow: decode-only BF16 suffix
// attention for Qwen2.5-7B's GQA shape (28 query heads, 4 KV heads,
// head_dim=128). vLLM's generic CPU attention is kept as a reference path;
// this kernel exists so the COTS path can specialize the q_per_kv=7 case
// instead of falling back to MHA-like work.

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
  #define VLLM_COTS_SUFFIX_ATTN_HAS_AVX2_FMA 1
#else
  #define VLLM_COTS_SUFFIX_ATTN_HAS_AVX2_FMA 0
#endif

namespace vllm {
namespace cots {
namespace {

constexpr int64_t kQwenNumQHeads = 28;
constexpr int64_t kQwenNumKVHeads = 4;
constexpr int64_t kQwenHeadsPerKV = 7;
constexpr int64_t kQwenHeadDim = 128;
constexpr int64_t kVecWidth = 8;
constexpr int64_t kStackProbSeqLen = 128;

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

#if VLLM_COTS_SUFFIX_ATTN_HAS_AVX2_FMA

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

struct SuffixAttentionStrides {
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

inline int32_t block_table_value(const int32_t* block_table,
                                 const SuffixAttentionStrides& strides,
                                 int64_t seq_idx, int64_t block_col) {
  return block_table[seq_idx * strides.bt_b + block_col * strides.bt_block];
}

template <typename scalar_t>
inline void zero_one_head(scalar_t* output,
                          const SuffixAttentionStrides& strides,
                          int64_t seq_idx, int64_t q_head) {
  scalar_t* out = output + seq_idx * strides.out_b + q_head * strides.out_h;
  for (int64_t d = 0; d < kQwenHeadDim; ++d) {
    out[d * strides.out_d] = 0;
  }
}

inline const uint16_t* kv_ptr(const uint16_t* cache,
                              const SuffixAttentionStrides& strides,
                              int64_t block_id, int64_t kv_head,
                              int64_t token_in_block) {
  return cache + block_id * strides.kv_block + kv_head * strides.kv_head +
         token_in_block * strides.kv_token;
}

void qwen_suffix_attention_one_group(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_table,
    const int32_t* seq_lens, uint16_t* output, float* output_lse,
    const SuffixAttentionStrides& strides, int64_t block_size,
    int64_t num_blocks, int64_t seq_idx, int64_t kv_head, float scale) {
  const int64_t seq_len = seq_lens[seq_idx];
  const int64_t q_head_base = kv_head * kQwenHeadsPerKV;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();

  if (seq_len <= 0) {
    for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
      const int64_t q_head = q_head_base + q;
      output_lse[q_head * strides.lse_h + seq_idx * strides.lse_b] = neg_inf;
      zero_one_head(output, strides, seq_idx, q_head);
    }
    return;
  }

  float max_logits[kQwenHeadsPerKV];
  std::fill(std::begin(max_logits), std::end(max_logits), neg_inf);
  float sums[kQwenHeadsPerKV];
  std::fill(std::begin(sums), std::end(sums), 0.0f);

  std::array<float, kQwenHeadsPerKV * kStackProbSeqLen> stack_probs;
  std::vector<float> heap_probs;
  float* probs = nullptr;
  if (seq_len <= kStackProbSeqLen) {
    probs = stack_probs.data();
  } else {
    heap_probs.resize(kQwenHeadsPerKV * seq_len);
    probs = heap_probs.data();
  }

  std::array<const uint16_t*, kStackProbSeqLen> stack_v_ptrs;
  std::vector<const uint16_t*> heap_v_ptrs;
  const uint16_t** v_ptrs = nullptr;
  if (seq_len <= kStackProbSeqLen) {
    v_ptrs = stack_v_ptrs.data();
  } else {
    heap_v_ptrs.resize(seq_len);
    v_ptrs = heap_v_ptrs.data();
  }

  const uint16_t* q_ptrs[kQwenHeadsPerKV];
  for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
    const int64_t q_head = q_head_base + q;
    q_ptrs[q] = query + seq_idx * strides.q_b + q_head * strides.q_h;
  }

  alignas(32) std::array<float, kQwenHeadsPerKV * kQwenHeadDim> q_f32;
  for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
    float* q_dst = q_f32.data() + q * kQwenHeadDim;
    for (int64_t d = 0; d < kQwenHeadDim; d += kVecWidth) {
      const __m256 q_vec = load8_bf16_as_f32(q_ptrs[q] + d * strides.q_d);
      _mm256_store_ps(q_dst + d, q_vec);
    }
  }

  for (int64_t block_col = 0, t = 0; t < seq_len; ++block_col) {
    const int32_t block_id =
        block_table_value(block_table, strides, seq_idx, block_col);
    TORCH_CHECK(block_id >= 0 && block_id < num_blocks,
                "block_table contains an out-of-range block id: ", block_id,
                " valid range is [0, ", num_blocks, ")");
    const int64_t block_token_begin = block_col * block_size;
    const int64_t block_token_end =
        std::min(seq_len, block_token_begin + block_size);

    for (; t + 1 < block_token_end; t += 2) {
      const int64_t token_in_block = t - block_token_begin;
      const uint16_t* k0_ptr =
          kv_ptr(key_cache, strides, block_id, kv_head, token_in_block);
      const uint16_t* k1_ptr =
          kv_ptr(key_cache, strides, block_id, kv_head, token_in_block + 1);
      v_ptrs[t] =
          kv_ptr(value_cache, strides, block_id, kv_head, token_in_block);
      v_ptrs[t + 1] =
          kv_ptr(value_cache, strides, block_id, kv_head, token_in_block + 1);

      __m256 acc0[kQwenHeadsPerKV];
      __m256 acc1[kQwenHeadsPerKV];
      for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
        acc0[q] = _mm256_setzero_ps();
        acc1[q] = _mm256_setzero_ps();
      }

      for (int64_t d = 0; d < kQwenHeadDim; d += kVecWidth) {
        const __m256 k0_vec = load8_bf16_as_f32(k0_ptr + d * strides.kv_d);
        const __m256 k1_vec = load8_bf16_as_f32(k1_ptr + d * strides.kv_d);
        for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
          const __m256 q_vec =
              _mm256_load_ps(q_f32.data() + q * kQwenHeadDim + d);
          acc0[q] = _mm256_fmadd_ps(q_vec, k0_vec, acc0[q]);
          acc1[q] = _mm256_fmadd_ps(q_vec, k1_vec, acc1[q]);
        }
      }

      for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
        const float logit0 = hreduce_ps(acc0[q]) * scale;
        const float logit1 = hreduce_ps(acc1[q]) * scale;
        probs[q * seq_len + t] = logit0;
        probs[q * seq_len + t + 1] = logit1;
        max_logits[q] = std::max(max_logits[q], std::max(logit0, logit1));
      }
    }

    if (t < block_token_end) {
      const int64_t token_in_block = t - block_token_begin;
      const uint16_t* k_ptr =
          kv_ptr(key_cache, strides, block_id, kv_head, token_in_block);
      v_ptrs[t] =
          kv_ptr(value_cache, strides, block_id, kv_head, token_in_block);

      __m256 acc[kQwenHeadsPerKV];
      for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
        acc[q] = _mm256_setzero_ps();
      }

      for (int64_t d = 0; d < kQwenHeadDim; d += kVecWidth) {
        const __m256 k_vec = load8_bf16_as_f32(k_ptr + d * strides.kv_d);
        for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
          const __m256 q_vec =
              _mm256_load_ps(q_f32.data() + q * kQwenHeadDim + d);
          acc[q] = _mm256_fmadd_ps(q_vec, k_vec, acc[q]);
        }
      }

      for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
        const float logit = hreduce_ps(acc[q]) * scale;
        probs[q * seq_len + t] = logit;
        max_logits[q] = std::max(max_logits[q], logit);
      }
      ++t;
    }
  }

  float inv_sums[kQwenHeadsPerKV];
  for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
    for (int64_t t = 0; t < seq_len; ++t) {
      const float prob = std::exp(probs[q * seq_len + t] - max_logits[q]);
      probs[q * seq_len + t] = prob;
      sums[q] += prob;
    }
    inv_sums[q] = 1.0f / sums[q];
    for (int64_t t = 0; t < seq_len; ++t) {
      probs[q * seq_len + t] *= inv_sums[q];
    }
    const float lse = std::log(sums[q]) + max_logits[q];
    const int64_t q_head = q_head_base + q;
    output_lse[q_head * strides.lse_h + seq_idx * strides.lse_b] = lse;
  }

  for (int64_t d = 0; d < kQwenHeadDim; d += kVecWidth) {
    __m256 out_acc[kQwenHeadsPerKV];
    for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
      out_acc[q] = _mm256_setzero_ps();
    }
    for (int64_t t = 0; t < seq_len; ++t) {
      const __m256 v_vec = load8_bf16_as_f32(v_ptrs[t] + d * strides.kv_d);
      for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
        const __m256 prob_vec = _mm256_set1_ps(probs[q * seq_len + t]);
        out_acc[q] = _mm256_fmadd_ps(prob_vec, v_vec, out_acc[q]);
      }
    }
    for (int64_t q = 0; q < kQwenHeadsPerKV; ++q) {
      const int64_t q_head = q_head_base + q;
      uint16_t* out_ptr = output + seq_idx * strides.out_b +
                          q_head * strides.out_h + d * strides.out_d;
      store8_f32_as_bf16(out_acc[q], out_ptr);
    }
  }
}

#endif  // VLLM_COTS_SUFFIX_ATTN_HAS_AVX2_FMA

void validate_qwen_suffix_attention_tensors(const at::Tensor& query,
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
              "block_table must be int32 for the prototype");
  TORCH_CHECK(seq_lens.dtype() == at::kInt,
              "seq_lens must be int32 for the prototype");

  TORCH_CHECK(query.dim() == 3, "query must have shape [B, 28, 128]");
  TORCH_CHECK(output.dim() == 3, "output must have shape [B, 28, 128]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must have shape [blocks, 4, block_size, 128]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must have shape [blocks, 4, block_size, 128]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(output_lse.dim() == 2, "output_lse must have shape [28, B]");

  const int64_t batch = query.size(0);
  const int64_t block_size = key_cache.size(2);
  TORCH_CHECK(block_size > 0, "key_cache block_size must be positive");
  const auto* len_ptr = seq_lens.data_ptr<int32_t>();
  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    const int64_t seq_len = len_ptr[seq_idx];
    TORCH_CHECK(seq_len >= 0, "seq_lens contains a negative sequence length");
    const int64_t needed_blocks = (seq_len + block_size - 1) / block_size;
    TORCH_CHECK(needed_blocks <= block_table.size(1),
                "block_table has too few columns for seq_lens: needed ",
                needed_blocks, ", got ", block_table.size(1));
  }
  TORCH_CHECK(query.size(1) == kQwenNumQHeads && query.size(2) == kQwenHeadDim,
              "query must have shape [B, 28, 128]");
  TORCH_CHECK(output.size(0) == batch && output.size(1) == kQwenNumQHeads &&
                  output.size(2) == kQwenHeadDim,
              "output shape must match query");
  TORCH_CHECK(
      key_cache.size(1) == kQwenNumKVHeads && key_cache.size(3) == kQwenHeadDim,
      "key_cache must have shape [blocks, 4, block_size, 128]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "value_cache shape must match key_cache");
  TORCH_CHECK(block_table.size(0) == batch,
              "block_table batch dimension must match query");
  TORCH_CHECK(seq_lens.size(0) == batch,
              "seq_lens size must match query batch");
  TORCH_CHECK(
      output_lse.size(0) == kQwenNumQHeads && output_lse.size(1) == batch,
      "output_lse must have shape [28, B]");

  TORCH_CHECK(query.stride(2) == 1, "query head_dim must be contiguous");
  TORCH_CHECK(key_cache.stride(3) == 1,
              "key_cache head_dim must be contiguous");
  TORCH_CHECK(value_cache.stride(3) == 1,
              "value_cache head_dim must be contiguous");
  TORCH_CHECK(output.stride(2) == 1, "output head_dim must be contiguous");
  TORCH_CHECK(block_table.is_contiguous(),
              "block_table must be contiguous in the prototype");
  TORCH_CHECK(seq_lens.is_contiguous(),
              "seq_lens must be contiguous in the prototype");
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

}  // namespace

void qwen_bf16_suffix_attention_at(const at::Tensor& query,
                                   const at::Tensor& key_cache,
                                   const at::Tensor& value_cache,
                                   const at::Tensor& block_table,
                                   const at::Tensor& seq_lens, double scale,
                                   at::Tensor& output, at::Tensor& output_lse) {
  validate_qwen_suffix_attention_tensors(
      query, key_cache, value_cache, block_table, seq_lens, output, output_lse);
#if VLLM_COTS_SUFFIX_ATTN_HAS_AVX2_FMA
  const int64_t batch = query.size(0);
  const int64_t block_size = key_cache.size(2);
  const int64_t num_blocks = key_cache.size(0);
  const auto strides = SuffixAttentionStrides{
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

  at::parallel_for(0, batch * kQwenNumKVHeads, /*grain_size=*/1,
                   [&](int64_t begin, int64_t end) {
                     for (int64_t task = begin; task < end; ++task) {
                       const int64_t seq_idx = task / kQwenNumKVHeads;
                       const int64_t kv_head = task - seq_idx * kQwenNumKVHeads;
                       qwen_suffix_attention_one_group(
                           q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, out_ptr,
                           lse_ptr, strides, block_size, num_blocks, seq_idx,
                           kv_head, scale_f);
                     }
                   });
#else
  TORCH_CHECK(false, "COTS Qwen suffix attention requires AVX2+FMA");
#endif
}

void qwen_bf16_scatter_suffix_kv_at(const at::Tensor& key,
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
              "key/value must have shape [N, 4, 128]");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");
  TORCH_CHECK(key.size(1) == kQwenNumKVHeads && key.size(2) == kQwenHeadDim,
              "key/value must have shape [N, 4, 128]");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
              "key/value head_dim must be contiguous");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "key/value caches must have shape [blocks, 4, block_size, 128]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "key/value cache shapes must match");
  TORCH_CHECK(
      key_cache.size(1) == kQwenNumKVHeads && key_cache.size(3) == kQwenHeadDim,
      "key/value caches must have shape [blocks, 4, block_size, 128]");
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
    TORCH_CHECK(block_id >= 0 && block_id < key_cache.size(0),
                "block_ids contains an out-of-range block id");
    TORCH_CHECK(block_offset >= 0 && block_offset < block_size,
                "block_offsets contains an out-of-range block offset");
    for (int64_t h = 0; h < kQwenNumKVHeads; ++h) {
      const uint16_t* src_key = key_ptr + i * key_b + h * key_h;
      const uint16_t* src_value = value_ptr + i * value_b + h * value_h;
      uint16_t* dst_key = key_cache_ptr + block_id * cache_block +
                          h * cache_head + block_offset * cache_token;
      uint16_t* dst_value = value_cache_ptr + block_id * value_cache_block +
                            h * value_cache_head +
                            block_offset * value_cache_token;
      std::memcpy(dst_key, src_key, kQwenHeadDim * sizeof(uint16_t));
      std::memcpy(dst_value, src_value, kQwenHeadDim * sizeof(uint16_t));
    }
  }
}

}  // namespace cots
}  // namespace vllm
