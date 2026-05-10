// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// §1c.29 M3 — captured wait kernel that replaces the
// `cudaLaunchHostFunc(sync_cb)` node in the COTS captured graph
// when `cots_m3_wait_kernel=True`. Compiled as CUDA so nvcc
// handles `__global__` and PTX inline asm; rest of `_cots_C`
// stays as plain C++ host code under the same target language.
//
// Two kernels — production and diag — to keep the production
// hot path branch-free. The diag kernel is launched only when
// VLLM_COTS_DIAG=1 (decision made on the host side, captured
// into the graph at trace time).

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm {
namespace cots {

namespace {

// Production wait kernel. Reads the captured-stream's expected
// seq from req_slot, then spins on done_slot until the worker
// publishes >= expected. Uses PTX `nanosleep` (sm_70+) to avoid
// burning SM cycles on the spin loop.
//
// The volatile reads are critical for host-mapped pinned
// visibility — the worker writes done_slot from the CPU side and
// the GPU read must not be cached.
__global__ void m3_wait_kernel(volatile uint32_t* req_slot,
                               volatile uint32_t* done_slot) {
  uint32_t expected = *req_slot;
  uint32_t done = *done_slot;
  while (done < expected) {
    asm volatile("nanosleep.u32 100;" ::: "memory");
    done = *done_slot;
  }
}

// Diag wait kernel. Identical control flow plus three counter
// increments. `immediate_resume_count` increments when the GPU
// window covered the CPU work (done >= req on first read);
// `lagging_wait_count` increments when the kernel had to spin
// at all; `spin_iters_total` accumulates the spin-loop iteration
// count for the lagging waits. Together they tell us how often
// the wait actually serializes (the §1c.29 design's canary).
//
// Atomics used to be safe under any future multi-block / multi-
// stream variant. With the current single-thread-block launch,
// regular adds would suffice — keeping atomic to make the
// counters correct under any later expansion.
__global__ void m3_wait_kernel_diag(volatile uint32_t* req_slot,
                                    volatile uint32_t* done_slot,
                                    int64_t* spin_iters_total,
                                    int64_t* immediate_count,
                                    int64_t* lagging_count) {
  uint32_t expected = *req_slot;
  uint32_t done = *done_slot;
  if (done >= expected) {
    atomicAdd(reinterpret_cast<unsigned long long*>(immediate_count), 1ull);
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(lagging_count), 1ull);
  int64_t iters = 0;
  do {
    asm volatile("nanosleep.u32 100;" ::: "memory");
    done = *done_slot;
    ++iters;
  } while (done < expected);
  atomicAdd(reinterpret_cast<unsigned long long*>(spin_iters_total),
            static_cast<unsigned long long>(iters));
}

}  // namespace

// C-linkage launcher entry points — called from cots_cpu_infer.cpp
// (compiled as C++ host code under the same target). Hide the
// `<<<>>>` syntax + kernel symbols behind plain functions so the
// .cpp file doesn't need to be CUDA.

extern "C" void launch_m3_wait_kernel_production(uint32_t* dev_req_slot,
                                                 uint32_t* dev_done_slot,
                                                 cudaStream_t stream) {
  m3_wait_kernel<<<1, 1, 0, stream>>>(dev_req_slot, dev_done_slot);
}

extern "C" void launch_m3_wait_kernel_diag(
    uint32_t* dev_req_slot, uint32_t* dev_done_slot, int64_t* spin_iters_total,
    int64_t* immediate_count, int64_t* lagging_count, cudaStream_t stream) {
  m3_wait_kernel_diag<<<1, 1, 0, stream>>>(dev_req_slot, dev_done_slot,
                                           spin_iters_total, immediate_count,
                                           lagging_count);
}

}  // namespace cots
}  // namespace vllm
