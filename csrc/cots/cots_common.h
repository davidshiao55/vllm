// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared host-side helpers for COTS native task runners.

#ifndef VLLM_COTS_COMMON_H_
#define VLLM_COTS_COMMON_H_

#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace vllm {
namespace cots {

inline bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

inline bool cots_nvtx_enabled() {
  static const bool enabled = []() { return env_flag("VLLM_COTS_NVTX"); }();
  return enabled;
}

struct NvtxScope {
  explicit NvtxScope(const char* name) : active_(cots_nvtx_enabled()) {
    if (active_) nvtxRangePushA(name);
  }
  ~NvtxScope() {
    if (active_) nvtxRangePop();
  }
  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;

 private:
  bool active_;
};

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace cots
}  // namespace vllm

#endif  // VLLM_COTS_COMMON_H_
