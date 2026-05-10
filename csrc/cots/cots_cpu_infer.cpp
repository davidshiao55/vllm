// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 1c — native CPU runner implementation.

#include "cots_cpu_infer.h"

#include <ATen/ops/linear.h>
#include <ATen/ops/silu.h>
#include <c10/core/InferenceMode.h>
#include <nvtx3/nvToolsExt.h>
#include <pthread.h>
#include <sched.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vllm {
namespace cots {

namespace {

constexpr int kMaxCpus = 64;

// §1c.24 instrumentation. NVTX ranges around the hot paths so nsys
// timeline can attribute time to the specific COTS phase. Gated by
// `VLLM_COTS_DIAG=1` (read once at first call) — when no profiler is
// attached `nvtxRangePush*` is a near-no-op but it still costs ~10ns
// in dispatch, which adds up across 7k+ hot-path calls per generate.
// Diagnostic-only; the production hot path stays clean.
namespace nvtx_internal {
inline bool diag_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("VLLM_COTS_DIAG");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return enabled;
}
}  // namespace nvtx_internal

struct NvtxScope {
  explicit NvtxScope(const char* name) {
    if (nvtx_internal::diag_enabled()) nvtxRangePushA(name);
  }
  ~NvtxScope() {
    if (nvtx_internal::diag_enabled()) nvtxRangePop();
  }
  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;
};

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Phase 1c is bfloat16-locked (see header comment + cpu_dtype literal in
// `vllm/config/offload.py`). Centralizing the dtype here keeps the slab
// struct lean and avoids any int-enum dance over pybind.
constexpr auto kCpuDtype = at::kBFloat16;

// Build an ATen tensor view over a pinned-CPU pointer with a contiguous
// row-major layout. Shape is {rows, cols}.
at::Tensor ContigCpuViewFromBlob(void* ptr, int64_t rows, int64_t cols) {
  auto opts = at::TensorOptions().dtype(kCpuDtype).device(at::kCPU);
  return at::from_blob(ptr, {rows, cols}, opts);
}

// Build an ATen tensor view over a column-narrowed pinned-CPU pointer.
// `ptr` is the post-narrow data pointer (offset already baked in).
// strides {row_stride, col_stride} reflect the source tensor.
at::Tensor StridedCpuViewFromBlob(void* ptr, int64_t rows, int64_t cols,
                                  int64_t row_stride, int64_t col_stride) {
  auto opts = at::TensorOptions().dtype(kCpuDtype).device(at::kCPU);
  return at::from_blob(ptr, {rows, cols}, {row_stride, col_stride}, opts);
}

}  // namespace

CotsCpuInfer::CotsCpuInfer() : task_queue_(std::make_unique<TaskQueue>()) {
  sync_args_.infer = static_cast<void*>(this);
  sync_args_.allow_n_pending = 0;
}

CotsCpuInfer::~CotsCpuInfer() {
  // Drain before destructing the queue so any in-flight task completes.
  if (task_queue_) {
    task_queue_->sync(0);
  }
  // §1c.29 M3: free per-slab host-mapped pinned slots. These are
  // allocated lazily via cudaHostAlloc(cudaHostAllocMapped) by
  // install_m3_for_task; teardown must release them or the host
  // pinned region leaks. Walk the slabs_ array (sized at install
  // time) and free any with m3_installed=true.
  if (slabs_) {
    for (int64_t i = 0; i < slab_count_; ++i) {
      TaskSlab& s = slabs_[i];
      if (s.m3_installed.load(std::memory_order_acquire)) {
        if (s.host_req_slot != nullptr) {
          cudaFreeHost(s.host_req_slot);
          s.host_req_slot = nullptr;
        }
        if (s.host_done_slot != nullptr) {
          cudaFreeHost(s.host_done_slot);
          s.host_done_slot = nullptr;
        }
        s.m3_installed.store(false, std::memory_order_release);
      }
    }
  }
  // Free M3 diag counter cells if allocated.
  if (m3_immediate_resume_host_ != nullptr) {
    cudaFreeHost(m3_immediate_resume_host_);
    m3_immediate_resume_host_ = nullptr;
  }
  if (m3_lagging_wait_host_ != nullptr) {
    cudaFreeHost(m3_lagging_wait_host_);
    m3_lagging_wait_host_ = nullptr;
  }
  if (m3_spin_iters_host_ != nullptr) {
    cudaFreeHost(m3_spin_iters_host_);
    m3_spin_iters_host_ = nullptr;
  }
}

void CotsCpuInfer::install(int64_t n_slabs, int64_t scratch_max_tokens,
                           int64_t scratch_max_intermediate_per_half) {
  TORCH_CHECK(n_slabs >= 0, "install: n_slabs must be >= 0, got ", n_slabs);
  TORCH_CHECK(!slabs_,
              "install: CotsCpuInfer already installed; call once per "
              "CotsCpuInfer instance");

  // Sized-once heap array. Address-stable for the lifetime of *this so
  // captured CUDA graphs can record &slabs_[id] as host-callback userData.
  if (n_slabs > 0) {
    slabs_ = std::unique_ptr<TaskSlab[]>(new TaskSlab[n_slabs]);
    for (int64_t i = 0; i < n_slabs; ++i) {
      slabs_[i].self = static_cast<void*>(this);
      slabs_[i].op_kind = TaskSlab::kDryrunNoop;
    }
  }
  slab_count_ = n_slabs;

  scratch_max_tokens_ = scratch_max_tokens;
  scratch_max_intermediate_per_half_ = scratch_max_intermediate_per_half;
  if (scratch_max_tokens > 0 && scratch_max_intermediate_per_half > 0) {
    auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
    // Worker-local scratch for silu(gate)*up. One max-sized tensor.
    scratch_silu_up_ = at::empty(
        {scratch_max_tokens, scratch_max_intermediate_per_half}, opts);
  }
}

void CotsCpuInfer::check_error() {
  if (!has_error_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  // Throwing std::runtime_error: pybind11 maps it to Python RuntimeError
  // automatically. Mirrors the Python runner's `future.result()` re-raise.
  throw std::runtime_error(msg);
}

int32_t CotsCpuInfer::slab_bucket_capacity_tokens(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "slab_bucket_capacity_tokens: task_id ", task_id,
              " out of range");
  return slabs_[task_id].bucket_capacity_tokens;
}

int32_t CotsCpuInfer::slab_num_tokens(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "slab_num_tokens: task_id ", task_id, " out of range");
  return slabs_[task_id].num_tokens.load(std::memory_order_acquire);
}

void CotsCpuInfer::populate_slab_qkv(int64_t task_id, int32_t n_threads,
                                     int32_t bucket_capacity_tokens,
                                     uintptr_t x_pinned_ptr, int32_t in_dim,
                                     uintptr_t y_pinned_ptr,
                                     int32_t cpu_out_dim, uintptr_t w_cpu_ptr,
                                     int32_t w_cpu_rows) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_qkv: task_id ", task_id, " out of range");
  TORCH_CHECK(bucket_capacity_tokens >= 0,
              "populate_slab_qkv: bucket_capacity_tokens=",
              bucket_capacity_tokens, " < 0");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kQkv;
  s.n_threads = n_threads;
  s.bucket_capacity_tokens = bucket_capacity_tokens;
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
  s.w_cpu_ptr = reinterpret_cast<void*>(w_cpu_ptr);
  s.w_cpu_rows = w_cpu_rows;
}

void CotsCpuInfer::populate_slab_mlp(
    int64_t task_id, int32_t n_threads, int32_t bucket_capacity_tokens,
    uintptr_t x_pinned_ptr, int32_t in_dim, uintptr_t y_pinned_ptr,
    int32_t cpu_out_dim, uintptr_t w_gate_ptr, int32_t w_gate_rows,
    uintptr_t w_up_ptr, int32_t w_up_rows, uintptr_t w_down_ptr,
    int32_t w_down_rows, int32_t w_down_cols, int64_t w_down_stride_row,
    int64_t w_down_stride_col, int32_t intermediate_per_half) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_mlp: task_id ", task_id, " out of range");
  TORCH_CHECK(bucket_capacity_tokens >= 0,
              "populate_slab_mlp: bucket_capacity_tokens=",
              bucket_capacity_tokens, " < 0");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kMlpBlock;
  s.n_threads = n_threads;
  s.bucket_capacity_tokens = bucket_capacity_tokens;
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
  s.w_gate_ptr = reinterpret_cast<void*>(w_gate_ptr);
  s.w_gate_rows = w_gate_rows;
  s.w_up_ptr = reinterpret_cast<void*>(w_up_ptr);
  s.w_up_rows = w_up_rows;
  s.w_down_ptr = reinterpret_cast<void*>(w_down_ptr);
  s.w_down_rows = w_down_rows;
  s.w_down_cols = w_down_cols;
  s.w_down_stride_row = w_down_stride_row;
  s.w_down_stride_col = w_down_stride_col;
  s.intermediate_per_half = intermediate_per_half;
}

void CotsCpuInfer::populate_slab_dryrun(int64_t task_id,
                                        int32_t bucket_capacity_tokens,
                                        uintptr_t x_pinned_ptr, int32_t in_dim,
                                        uintptr_t y_pinned_ptr,
                                        int32_t cpu_out_dim) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_dryrun: task_id ", task_id, " out of range");
  TORCH_CHECK(bucket_capacity_tokens >= 0,
              "populate_slab_dryrun: bucket_capacity_tokens=",
              bucket_capacity_tokens, " < 0");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kDryrunNoop;
  s.bucket_capacity_tokens = bucket_capacity_tokens;
  // §1c.20: dryrun must carry the x_pinned + y_pinned pointers so
  // submit_on_stream's D2H and sync's y_pinned_view both resolve.
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
}

void CotsCpuInfer::submit_on_stream(int64_t task_id, int32_t num_tokens,
                                    uintptr_t x_gpu_ptr, int64_t x_cols,
                                    int64_t x_stride0, int64_t x_stride1,
                                    uintptr_t cuda_stream) {
  NvtxScope nvtx_scope("cots:submit_on_stream");
  // Surface any prior worker error BEFORE queueing more work.
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "submit_on_stream: task_id ", task_id, " out of range");
  TaskSlab* slab = &slabs_[task_id];
  TORCH_CHECK(num_tokens >= 0, "submit_on_stream: num_tokens=", num_tokens,
              " < 0");
  // §1c.20: bound check against scratch_max_tokens_ catches Planner
  // mistakes where an oversized batch reaches a slab. Skipped when
  // scratch_max_tokens_==0 (test fixtures that don't allocate scratch
  // / don't run real tokens through the slab — diagnostic-only path).
  TORCH_CHECK(scratch_max_tokens_ == 0 || num_tokens <= scratch_max_tokens_,
              "submit_on_stream: num_tokens=", num_tokens,
              " exceeds scratch_max_tokens=", scratch_max_tokens_,
              " (would write past the pinned buffer's tail)");
  slab->num_tokens.store(num_tokens, std::memory_order_release);
  // §1c.21 counters: bump submit_count + num_tokens histogram for
  // this op kind. Bin index = position of msb of num_tokens, clamped
  // to [0, 7]. nt<=1 → 0, nt<=2 → 1, ..., nt<=64 → 6, nt>64 → 7.
  {
    int hist_bin;
    if (num_tokens <= 1)
      hist_bin = 0;
    else if (num_tokens <= 2)
      hist_bin = 1;
    else if (num_tokens <= 4)
      hist_bin = 2;
    else if (num_tokens <= 8)
      hist_bin = 3;
    else if (num_tokens <= 16)
      hist_bin = 4;
    else if (num_tokens <= 32)
      hist_bin = 5;
    else if (num_tokens <= 64)
      hist_bin = 6;
    else
      hist_bin = 7;
    switch (slab->op_kind) {
      case TaskSlab::kQkv:
        submit_count_qkv_.fetch_add(1, std::memory_order_relaxed);
        nt_hist_qkv_[hist_bin].fetch_add(1, std::memory_order_relaxed);
        break;
      case TaskSlab::kMlpBlock:
        submit_count_mlp_.fetch_add(1, std::memory_order_relaxed);
        nt_hist_mlp_[hist_bin].fetch_add(1, std::memory_order_relaxed);
        break;
      case TaskSlab::kDryrunNoop:
      default:
        submit_count_dryrun_.fetch_add(1, std::memory_order_relaxed);
        nt_hist_dryrun_[hist_bin].fetch_add(1, std::memory_order_relaxed);
        break;
    }
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  // §1c.26 ablation: skip the captured cudaMemcpyAsync entirely when
  // ablate_d2h_ is set. Probe-only — gated upstream to dryrun + DIAG;
  // worker reads stale x_pinned, output is garbage, but in dryrun
  // worker doesn't compute so this is safe for the diagnostic.
  const bool skip_d2h = ablate_d2h_.load(std::memory_order_relaxed);
  // §1c.20: D2H from x_gpu to slab's pinned input buffer.
  // Stride-aware: contiguous → 1D cudaMemcpyAsync, row-strided →
  // 2D cudaMemcpy2DAsync. Both graph-capturable. Skipped when
  // `x_gpu_ptr == 0` (test fixtures that exercise dispatch only).
  if (x_gpu_ptr != 0 && !skip_d2h) {
    TORCH_CHECK(slab->x_pinned_ptr != nullptr,
                "submit_on_stream: slab.x_pinned_ptr is null at task_id=",
                task_id, " (slab not populated?)");
    TORCH_CHECK(slab->in_dim > 0, "submit_on_stream: slab.in_dim is ",
                slab->in_dim, " at task_id=", task_id);
    TORCH_CHECK(
        x_cols == slab->in_dim, "submit_on_stream: x_gpu.shape[1]=", x_cols,
        " disagrees with slab.in_dim=", slab->in_dim, " at task_id=", task_id);
    TORCH_CHECK(x_stride1 == 1, "submit_on_stream: x_gpu.stride(1)=", x_stride1,
                " (must be 1 — no transposed-stride layouts; only "
                "row-strided contiguous-along-feature inputs are "
                "supported)");
    TORCH_CHECK(x_stride0 >= x_cols,
                "submit_on_stream: x_gpu.stride(0)=", x_stride0,
                " < x_cols=", x_cols, " (rows would overlap; reject)");
    const size_t width_bytes =
        static_cast<size_t>(slab->in_dim) * sizeof(at::BFloat16);
    cudaError_t copy_err;
    NvtxScope d2h_scope("cots:d2h_record");
    if (x_stride0 == x_cols) {
      // Contiguous row layout — single 1D copy is fastest.
      const size_t bytes = static_cast<size_t>(num_tokens) * width_bytes;
      d2h_1d_count_.fetch_add(1, std::memory_order_relaxed);
      d2h_record_bytes_1d_.fetch_add(static_cast<int64_t>(bytes),
                                     std::memory_order_relaxed);
      copy_err = cudaMemcpyAsync(slab->x_pinned_ptr,
                                 reinterpret_cast<void*>(x_gpu_ptr), bytes,
                                 cudaMemcpyDeviceToHost, stream);
    } else {
      // Row-strided (e.g., a `[:, :hidden_dim]` slice over a wider
      // base, or a hidden-state view that the model produced with
      // padding). 2D copy walks rows at the source stride and writes
      // tightly-packed rows at the destination.
      const size_t src_pitch =
          static_cast<size_t>(x_stride0) * sizeof(at::BFloat16);
      const size_t bytes_2d = width_bytes * static_cast<size_t>(num_tokens);
      d2h_2d_count_.fetch_add(1, std::memory_order_relaxed);
      d2h_record_bytes_2d_.fetch_add(static_cast<int64_t>(bytes_2d),
                                     std::memory_order_relaxed);
      copy_err = cudaMemcpy2DAsync(
          slab->x_pinned_ptr, /*dpitch=*/width_bytes,
          reinterpret_cast<void*>(x_gpu_ptr), /*spitch=*/src_pitch,
          /*width=*/width_bytes, /*height=*/static_cast<size_t>(num_tokens),
          cudaMemcpyDeviceToHost, stream);
    }
    TORCH_CHECK(copy_err == cudaSuccess, "submit_on_stream: D2H copy failed (",
                (x_stride0 == x_cols ? "1D" : "2D"),
                "): ", cudaGetErrorString(copy_err));
  }
  // §1c.26/§1c.27 ablation: skip the captured submit/dispatch
  // cudaLaunchHostFunc when either the broad `ablate_hostfn_` or
  // the narrow `ablate_submit_hostfn_` flag is set. Worker is
  // never enqueued; in dryrun there's nothing to enqueue anyway.
  const bool skip_submit_hostfn =
      ablate_hostfn_.load(std::memory_order_relaxed) ||
      ablate_submit_hostfn_.load(std::memory_order_relaxed);
  if (!skip_submit_hostfn) {
    NvtxScope launch_scope("cots:launch_dispatch_cb");
    cudaError_t err = cudaLaunchHostFunc(
        stream, &CotsCpuInfer::DispatchCallback, static_cast<void*>(slab));
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc(DispatchCallback) failed: ",
                cudaGetErrorString(err));
  }
}

void CotsCpuInfer::sync_on_stream(uintptr_t cuda_stream) {
  NvtxScope nvtx_scope("cots:sync_on_stream");
  check_error();
  // §1c.26/§1c.27 ablation: skip captured sync host_fn when either
  // the broad `ablate_hostfn_` or the narrow `ablate_sync_hostfn_`
  // flag is set. In dryrun there is nothing to drain.
  const bool skip_sync_hostfn =
      ablate_hostfn_.load(std::memory_order_relaxed) ||
      ablate_sync_hostfn_.load(std::memory_order_relaxed);
  if (skip_sync_hostfn) {
    return;
  }
  // sync_args_ is a stable member of *this; safe to take its address as
  // userData for cudaLaunchHostFunc, including across CUDA graph replays.
  cudaError_t err = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(cuda_stream), &CotsCpuInfer::SyncCallback,
      static_cast<void*>(&sync_args_));
  TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc(SyncCallback) failed: ",
              cudaGetErrorString(err));
}

void CotsCpuInfer::sync_or_wait_on_stream(int64_t task_id,
                                          uintptr_t cuda_stream) {
  // §1c.29 commit 2 — unified entry. Per-slab branch: if M3 is
  // installed for this task, the captured node is the wait kernel
  // (reads the worker-published done_slot=seq). Otherwise the
  // captured node stays the legacy SyncCallback host_fn that
  // blocks the driver thread on TaskQueue::sync(0). Both
  // mechanisms can coexist within the same offloader / same
  // CotsCpuInfer instance — the branch is per-slab, not per-
  // runner — but in practice the offloader sets the flag for ALL
  // slabs at install time (cots_m3_wait_kernel=True is binary at
  // the runner level) so the branch is uniform across a single
  // offloader's slabs.
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "sync_or_wait_on_stream: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  if (s.m3_installed.load(std::memory_order_acquire)) {
    m3_wait_on_stream(task_id, cuda_stream);
  } else {
    sync_on_stream(cuda_stream);
  }
}

bool CotsCpuInfer::m3_installed_for_task(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "m3_installed_for_task: task_id ", task_id, " out of range");
  return slabs_[task_id].m3_installed.load(std::memory_order_acquire);
}

void CotsCpuInfer::submit_dryrun_burst(int64_t n) {
  check_error();
  for (int64_t i = 0; i < n; ++i) {
    task_queue_->enqueue([] {
      // Pure no-op: no slab read, no scratch use. Used by
      // test_taskqueue_stress to validate FIFO + drain semantics.
    });
  }
}

void CotsCpuInfer::sync_blocking() {
  task_queue_->sync(0);
  // Surface any worker error that fired while we were waiting.
  check_error();
}

void CotsCpuInfer::set_ablations(bool ablate_d2h, bool ablate_hostfn,
                                 bool ablate_submit_hostfn,
                                 bool ablate_sync_hostfn) {
  ablate_d2h_.store(ablate_d2h, std::memory_order_relaxed);
  ablate_hostfn_.store(ablate_hostfn, std::memory_order_relaxed);
  ablate_submit_hostfn_.store(ablate_submit_hostfn, std::memory_order_relaxed);
  ablate_sync_hostfn_.store(ablate_sync_hostfn, std::memory_order_relaxed);
}

void CotsCpuInfer::set_worker_affinity(uint64_t cpu_set) {
  if (cpu_set == 0) {
    return;  // No-op; leave kernel default. (Stage 4 fills this out.)
  }
  // Intersect with the process's existing affinity to avoid EINVAL under
  // a restrictive cgroup mask. (Phase 1c plan §risk-3 / Stage 4.)
  cpu_set_t requested;
  CPU_ZERO(&requested);
  // Iterate up to min(kMaxCpus, 64) — uint64 mask has 64 bits, no more.
  // uint64_t shift up to bit 63 is well-defined (unlike signed int64).
  for (int i = 0; i < kMaxCpus && i < 64; ++i) {
    if (cpu_set & (uint64_t{1} << i)) CPU_SET(i, &requested);
  }
  cpu_set_t process_mask;
  CPU_ZERO(&process_mask);
  if (sched_getaffinity(0, sizeof(process_mask), &process_mask) != 0) {
    return;
  }
  cpu_set_t effective;
  CPU_ZERO(&effective);
  for (int i = 0; i < kMaxCpus; ++i) {
    if (CPU_ISSET(i, &requested) && CPU_ISSET(i, &process_mask)) {
      CPU_SET(i, &effective);
    }
  }
  if (CPU_COUNT(&effective) == 0) {
    return;  // empty intersection; warn-and-skip per plan §Stage 4.
  }
  // Run the actual pthread_setaffinity_np on the worker thread itself by
  // submitting a tiny task. Stage 4 expands this; Stage 1 stores a pending
  // mask only via this side channel.
  task_queue_->enqueue([effective] {
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &effective);
  });
}

std::string CotsCpuInfer::take_error() {
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  return msg;
}

void CotsCpuInfer::run_at_linear_inline(at::Tensor x, at::Tensor w,
                                        at::Tensor y_out) {
  // Stage 1 microbench helper: directly drive `at::linear` from C++.
  // No TaskQueue, no host callback. Lets the test compare wall-clock
  // against Python `F.linear` on bit-identical tensors and catch the
  // catastrophic-scalar-fallback path documented in
  // `David/Docs/phase0_findings.md §0.3.2`.
  c10::InferenceMode g;
  // c10::AutoDispatchBelowAutograd is implied by InferenceMode.
  y_out.copy_(at::linear(x, w));
}

at::Tensor CotsCpuInfer::y_pinned_view(int64_t task_id,
                                       int32_t num_tokens) const {
  // §1c.20: build an `at::from_blob` CPU tensor view over the slab's
  // pinned output buffer. Used by `cots_sync_then_uva`'s impl on the
  // captured-graph hot path so the CPU output tensor is NEVER a
  // custom-op argument visible to Inductor.
  if (task_id < 0 || task_id >= slab_count_) {
    throw std::out_of_range(
        "CotsCpuInfer::y_pinned_view: task_id=" + std::to_string(task_id) +
        " out of range [0, " + std::to_string(slab_count_) + ")");
  }
  const TaskSlab& slab = slabs_[task_id];
  if (slab.y_pinned_ptr == nullptr) {
    throw std::runtime_error(
        "CotsCpuInfer::y_pinned_view: slab.y_pinned_ptr is null at task_id=" +
        std::to_string(task_id) + " (slab not populated?)");
  }
  if (num_tokens < 0) {
    throw std::invalid_argument("CotsCpuInfer::y_pinned_view: num_tokens=" +
                                std::to_string(num_tokens) + " < 0");
  }
  if (scratch_max_tokens_ != 0 && num_tokens > scratch_max_tokens_) {
    throw std::invalid_argument(
        "CotsCpuInfer::y_pinned_view: num_tokens=" +
        std::to_string(num_tokens) +
        " exceeds scratch_max_tokens=" + std::to_string(scratch_max_tokens_) +
        " (would read past the pinned buffer's tail)");
  }
  // dtype is hard-coded bfloat16 (matches populate_slab_*'s contract).
  // The pointer's underlying allocation came from `_y_pinned` (a
  // `torch.empty(..., pin_memory=True)`) and is page-locked; we trust
  // the install-time invariant rather than re-validating here.
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
  auto sizes = std::array<int64_t, 2>{static_cast<int64_t>(num_tokens),
                                      static_cast<int64_t>(slab.cpu_out_dim)};
  return at::from_blob(slab.y_pinned_ptr, sizes, options);
}

// --- static cudaLaunchHostFunc callbacks -----------------------------------

// Submit-side host callback. Runs on the CUDA driver thread; must NOT
// block (CUDA stream is paused while we run). Just enqueues to the
// TaskQueue worker.
void CotsCpuInfer::DispatchCallback(void* user_data) {
  NvtxScope nvtx_scope("cots:dispatch_cb");
  TaskSlab* slab = static_cast<TaskSlab*>(user_data);
  CotsCpuInfer* self = static_cast<CotsCpuInfer*>(slab->self);
  // §1c.24 attribution: stamp enqueue time so the worker can later
  // compute queue_wait = worker_start - enqueue_time. Gated together
  // with the NVTX scopes by VLLM_COTS_DIAG=1; in production-default
  // mode neither now_ns() nor the atomic write fires. Worker reads
  // enqueue_time_ns conditionally on the same flag, so a
  // diag-disabled run leaves it at its initial value (0).
  if (nvtx_internal::diag_enabled()) {
    slab->enqueue_time_ns.store(now_ns(), std::memory_order_release);
    self->dispatch_cb_count_.fetch_add(1, std::memory_order_relaxed);
  }
  // §1c.29 commit 2 — M3 sequence publish. When M3 is installed
  // for this slab, increment the slab-local seq, write it into
  // host_req_slot (with a release fence so the value precedes the
  // worker enqueue), and capture it into the worker lambda. The
  // worker publishes `done_slot = seq` after finishing (or on
  // exception), and the captured `m3_wait_kernel` on the GPU side
  // spins until it sees `done >= req`. seq=0 in the lambda
  // signals "no M3 publish needed" — RunSlabOnWorker skips the
  // done_slot store, preserving the legacy non-M3 path bit-for-bit.
  uint32_t seq = 0;
  if (slab->m3_installed.load(std::memory_order_acquire)) {
    // Wrap behavior: uint32_t monotonically increases. At ~1k ops/
    // generate this overflows after 2^32 ≈ 4.3e6 generates, far
    // beyond any practical run. Documented inline rather than
    // reset-on-wrap to keep the hot path branch-free.
    seq = slab->next_seq.fetch_add(1, std::memory_order_relaxed) + 1u;
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile uint32_t*>(slab->host_req_slot) = seq;
  }
  // Copy the pointer + seq into the lambda capture; this is safe because
  // `slab` is a member of `self->slabs_` which is reserve-once.
  self->task_queue_->enqueue(
      [self, slab, seq] { self->RunSlabOnWorker(slab, seq); });
}

// Sync-side host callback. Blocks the CUDA driver thread until the
// TaskQueue drains. The user_data is `&self->sync_args_` (stable member).
void CotsCpuInfer::SyncCallback(void* user_data) {
  NvtxScope nvtx_scope("cots:sync_cb_wait");
  SyncArgs* args = static_cast<SyncArgs*>(user_data);
  CotsCpuInfer* self = static_cast<CotsCpuInfer*>(args->infer);
  // §1c.24 attribution: time the sync wait — distinguishes "driver
  // blocked waiting for the worker" from "driver doing other work
  // then unblocking immediately". Same VLLM_COTS_DIAG gate as the
  // dispatch counter; in production-default mode the timestamps
  // and atomic adds are skipped.
  if (nvtx_internal::diag_enabled()) {
    const int64_t t0 = now_ns();
    self->task_queue_->sync(args->allow_n_pending);
    const int64_t t1 = now_ns();
    self->sync_cb_wait_total_ns_.fetch_add(t1 - t0, std::memory_order_relaxed);
    self->sync_cb_count_.fetch_add(1, std::memory_order_relaxed);
  } else {
    self->task_queue_->sync(args->allow_n_pending);
  }
}

// --- worker-thread task body ----------------------------------------------

void CotsCpuInfer::RunSlabOnWorker(TaskSlab* slab, uint32_t seq) {
  // §1c.24 attribution: stamp worker start + queue wait, gated by
  // VLLM_COTS_DIAG. Production-default leaves worker_t0 at 0 (the
  // worker_busy_total_ns add at the end is also gated). NVTX scope
  // is independently gated inside NvtxScope's ctor.
  const bool diag = nvtx_internal::diag_enabled();
  const int64_t worker_t0 = diag ? now_ns() : 0;
  if (diag) {
    const int64_t enq = slab->enqueue_time_ns.load(std::memory_order_acquire);
    if (enq > 0) {
      worker_queue_wait_total_ns_.fetch_add(worker_t0 - enq,
                                            std::memory_order_relaxed);
    }
  }
  const char* nvtx_name = "cots:worker";
  switch (slab->op_kind) {
    case TaskSlab::kQkv:
      nvtx_name = "cots:worker_qkv";
      break;
    case TaskSlab::kMlpBlock:
      nvtx_name = "cots:worker_mlp";
      break;
    case TaskSlab::kDryrunNoop:
      nvtx_name = "cots:worker_dryrun";
      break;
  }
  NvtxScope nvtx_scope(nvtx_name);
  // Worker exception policy: every body wrapped in try/catch so that
  // pending-decrement / cv_ notify in TaskQueue::Worker still happens
  // (we return normally from this function), and the next Python-side
  // submit/sync call surfaces the error as a Python RuntimeError.
  try {
    c10::InferenceMode g;

    // Bucket-aware thread policy: only call set_num_threads when it would
    // change. Stage 4 populates slab->n_threads from the per-bucket map;
    // Stage 1 leaves n_threads = 1, which means we call set_num_threads(1)
    // on the very first task and then stay at 1. (No regression vs the
    // Phase 1a Python runner default, which used scalar `cpu_num_threads`
    // via `torch.set_num_threads` once at offloader init.)
    if (slab->n_threads > 0 && slab->n_threads != worker_current_n_threads_) {
      at::set_num_threads(slab->n_threads);
      worker_current_n_threads_ = slab->n_threads;
    }
    last_observed_num_threads_.store(at::get_num_threads(),
                                     std::memory_order_release);

    // §1c.21: prefer runtime_num_tokens_ (live unpadded count, set
    // OUT OF GRAPH by `set_runtime_num_tokens` before each captured
    // replay) over `slab->num_tokens` (captured bucket capacity).
    // Sentinel 0 → fall back to slab capacity; bounded by
    // effective_n <= slab->num_tokens so the worker never reads past
    // the slab's pinned buffer.
    const int32_t slab_cap = slab->num_tokens.load(std::memory_order_acquire);
    const int32_t override_n =
        runtime_num_tokens_.load(std::memory_order_acquire);
    const int32_t n = (override_n > 0) ? override_n : slab_cap;
    TORCH_CHECK(n <= slab_cap, "RunSlabOnWorker: runtime_num_tokens=", n,
                " exceeds slab capacity (slab.num_tokens=", slab_cap,
                ") at task_id=", static_cast<int64_t>(slab - slabs_.get()),
                " — would read past the pinned buffer's tail");

    // §1c.21 fix-validation: bin the effective_n the worker
    // actually used. If submit-time num_tokens histogram is mostly
    // `nt_gt_64` but this is mostly `nt_le_1`, the override is
    // working as intended.
    {
      int hist_bin;
      if (n <= 1)
        hist_bin = 0;
      else if (n <= 2)
        hist_bin = 1;
      else if (n <= 4)
        hist_bin = 2;
      else if (n <= 8)
        hist_bin = 3;
      else if (n <= 16)
        hist_bin = 4;
      else if (n <= 32)
        hist_bin = 5;
      else if (n <= 64)
        hist_bin = 6;
      else
        hist_bin = 7;
      worker_effective_n_hist_[hist_bin].fetch_add(1,
                                                   std::memory_order_relaxed);
    }
    // §1c.22 byte accounting (replay-time). RunSlabOnWorker fires
    // PER REPLAY because the captured host callback re-executes
    // each time the graph replays. Two paired counters:
    //
    // - worker_*_live_bytes: bytes the worker actually reads/writes
    //   for the live-token override (n).
    // - *_replay_bucket_bytes: bytes attributable to the captured
    //   cudaMemcpyAsync (input D2H) and Triton UVA kernel (output
    //   H2D) — sized by the descriptor bucket capacity.
    //
    // §1c.22 review-fix: read the IMMUTABLE
    // `bucket_capacity_tokens` populated at install time, NOT the
    // mutable `slab->num_tokens` (which can be overwritten by later
    // submit_on_stream calls during graph capture / PIECEWISE Python
    // re-execution and is thus unreliable as a stable bucket
    // estimate). bucket_capacity_tokens is still an ESTIMATE of the
    // captured graph node's actual byte param — only authoritative
    // value would come from inspecting the cuGraphNode at capture
    // time, which is out of scope. Stable and descriptor-attributed
    // is the best we can do without graph introspection.
    const int64_t bucket_n = static_cast<int64_t>(slab->bucket_capacity_tokens);
    if (slab->in_dim > 0) {
      const int64_t row_bytes_in = static_cast<int64_t>(slab->in_dim) *
                                   static_cast<int64_t>(sizeof(at::BFloat16));
      d2h_replay_bucket_bytes_.fetch_add(bucket_n * row_bytes_in,
                                         std::memory_order_relaxed);
      if (n > 0) {
        worker_input_live_bytes_.fetch_add(
            static_cast<int64_t>(n) * row_bytes_in, std::memory_order_relaxed);
      }
    }
    if (slab->cpu_out_dim > 0) {
      const int64_t row_bytes_out = static_cast<int64_t>(slab->cpu_out_dim) *
                                    static_cast<int64_t>(sizeof(at::BFloat16));
      uva_replay_bucket_bytes_.fetch_add(bucket_n * row_bytes_out,
                                         std::memory_order_relaxed);
      if (n > 0) {
        worker_output_live_bytes_.fetch_add(
            static_cast<int64_t>(n) * row_bytes_out, std::memory_order_relaxed);
      }
    }

    switch (slab->op_kind) {
      case TaskSlab::kDryrunNoop: {
        // Stage 2 substrate gate: install all wrappers but skip real
        // CPU work. Mirrors `_cpu_dryrun_noop` (cots.py:1161).
        break;
      }
      case TaskSlab::kQkv: {
        // y_view <- at::linear(x_view, w_view).
        auto x_view =
            ContigCpuViewFromBlob(slab->x_pinned_ptr, n, slab->in_dim);
        auto w_view = ContigCpuViewFromBlob(slab->w_cpu_ptr, slab->w_cpu_rows,
                                            slab->in_dim);
        auto y_view =
            ContigCpuViewFromBlob(slab->y_pinned_ptr, n, slab->cpu_out_dim);
        y_view.copy_(at::linear(x_view, w_view));
        break;
      }
      case TaskSlab::kMlpBlock: {
        // gate / up are contiguous prefix views of their respective halves
        // of the gate_up CPU buffer (Phase 1b populates them this way).
        auto x_view =
            ContigCpuViewFromBlob(slab->x_pinned_ptr, n, slab->in_dim);
        auto w_gate = ContigCpuViewFromBlob(slab->w_gate_ptr, slab->w_gate_rows,
                                            slab->in_dim);
        auto w_up = ContigCpuViewFromBlob(slab->w_up_ptr, slab->w_up_rows,
                                          slab->in_dim);
        auto gate_out =
            at::linear(x_view, w_gate);  // (n, intermediate_per_half)
        auto up_out = at::linear(x_view, w_up);

        // Worker-local scratch — single max-sized tensor.
        TORCH_CHECK(scratch_silu_up_.defined(),
                    "MLP slab dispatched but scratch_silu_up_ not "
                    "allocated; install() must be called with non-zero "
                    "scratch sizes.");
        TORCH_CHECK(n <= scratch_max_tokens_, "MLP slab num_tokens (", n,
                    ") exceeds scratch_max_tokens_ (", scratch_max_tokens_,
                    ")");
        TORCH_CHECK(
            slab->intermediate_per_half <= scratch_max_intermediate_per_half_,
            "MLP slab intermediate_per_half (", slab->intermediate_per_half,
            ") exceeds scratch_max_intermediate_per_half_ (",
            scratch_max_intermediate_per_half_, ")");
        auto z = scratch_silu_up_.narrow(0, 0, n).narrow(
            1, 0, slab->intermediate_per_half);
        z.copy_(at::silu(gate_out) * up_out);

        // Down-proj: column-strided view.
        auto w_down = StridedCpuViewFromBlob(
            slab->w_down_ptr, slab->w_down_rows, slab->w_down_cols,
            slab->w_down_stride_row, slab->w_down_stride_col);
        auto y_view =
            ContigCpuViewFromBlob(slab->y_pinned_ptr, n, slab->cpu_out_dim);
        y_view.copy_(at::linear(z, w_down));
        break;
      }
    }
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = std::string("[cots worker] ") + e.what();
    has_error_.store(true, std::memory_order_release);
  } catch (...) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = "[cots worker] unknown exception";
    has_error_.store(true, std::memory_order_release);
  }
  // §1c.29 commit 2 — finally-publish for M3. Even on exception we
  // MUST write done_slot=seq so the captured m3_wait_kernel can
  // exit its spin loop. Without this publish a worker throw would
  // wedge the GPU stream in an infinite spin and the next Python-
  // side check would never see the error (because the next
  // submit/sync is itself behind the wedged stream). On the
  // success path, has_error_ stays false and the consumer reads
  // y_pinned normally; on the failure path, has_error_ + msg
  // surface at the next Python-side submit/sync as a
  // RuntimeError, but the GPU stream is unblocked first so that
  // check actually runs. seq=0 means "no M3 installed for this
  // slab" — skip the publish to keep the legacy path bit-for-bit
  // identical.
  if (seq != 0 && slab->m3_installed.load(std::memory_order_acquire)) {
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile uint32_t*>(slab->host_done_slot) = seq;
  }
  // §1c.24: worker compute duration (NVTX scope ends when the
  // function returns; this counter sums it for the bench summary).
  // Gated together with the start-of-function timestamp.
  if (diag) {
    worker_busy_total_ns_.fetch_add(now_ns() - worker_t0,
                                    std::memory_order_relaxed);
    worker_run_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

// --- §1c.21 live-token override ---------------------------------------

void CotsCpuInfer::note_uva_request(int32_t num_tokens, int32_t cpu_out_dim) {
  // §1c.22 bookkeeping. No bound checks beyond non-negativity —
  // this is a measurement-only path; the actual UVA kernel does
  // its own validation.
  if (num_tokens <= 0 || cpu_out_dim <= 0) return;
  const int64_t bytes = static_cast<int64_t>(num_tokens) *
                        static_cast<int64_t>(cpu_out_dim) *
                        static_cast<int64_t>(sizeof(at::BFloat16));
  uva_record_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  uva_record_count_.fetch_add(1, std::memory_order_relaxed);
}

void CotsCpuInfer::set_runtime_num_tokens(int32_t n) {
  TORCH_CHECK(n >= 0, "set_runtime_num_tokens: n=", n,
              " < 0; pass 0 to clear the override.");
  // Release store: the worker's acquire load in RunSlabOnWorker pairs
  // with this. The caller's responsibility is to set this BEFORE the
  // captured graph replay begins (i.e., from
  // CotsOffloader.prepare_before_forward, called by
  // cudagraph_utils.py:267 outside the captured region).
  runtime_num_tokens_.store(n, std::memory_order_release);
  runtime_set_calls_.fetch_add(1, std::memory_order_relaxed);
  runtime_last_value_.store(n, std::memory_order_relaxed);
}

// --- §1c.21 perf-investigation counters --------------------------------

std::vector<std::pair<std::string, int64_t>> CotsCpuInfer::get_counters()
    const {
  std::vector<std::pair<std::string, int64_t>> out;
  out.reserve(40);
  auto load = [](const std::atomic<int64_t>& a) {
    return a.load(std::memory_order_relaxed);
  };
  out.emplace_back("submit_count_qkv", load(submit_count_qkv_));
  out.emplace_back("submit_count_mlp", load(submit_count_mlp_));
  out.emplace_back("submit_count_dryrun", load(submit_count_dryrun_));
  static const char* kBinNames[8] = {"nt_le_1",  "nt_le_2",  "nt_le_4",
                                     "nt_le_8",  "nt_le_16", "nt_le_32",
                                     "nt_le_64", "nt_gt_64"};
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("nt_qkv_") + kBinNames[i],
                     load(nt_hist_qkv_[i]));
  }
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("nt_mlp_") + kBinNames[i],
                     load(nt_hist_mlp_[i]));
  }
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("nt_dryrun_") + kBinNames[i],
                     load(nt_hist_dryrun_[i]));
  }
  // §1c.22 — record-time counters (capture/warmup only).
  out.emplace_back("d2h_1d_count", load(d2h_1d_count_));
  out.emplace_back("d2h_2d_count", load(d2h_2d_count_));
  out.emplace_back("d2h_record_bytes_1d", load(d2h_record_bytes_1d_));
  out.emplace_back("d2h_record_bytes_2d", load(d2h_record_bytes_2d_));
  // §1c.22 — replay-time counters (incremented in RunSlabOnWorker
  // which fires per replay because the captured host callback
  // re-executes). Apples-to-apples with worker_*_live_bytes.
  out.emplace_back("d2h_replay_bucket_bytes", load(d2h_replay_bucket_bytes_));
  out.emplace_back("uva_replay_bucket_bytes", load(uva_replay_bucket_bytes_));
  out.emplace_back("runtime_set_calls", load(runtime_set_calls_));
  out.emplace_back("runtime_last_value", load(runtime_last_value_));
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("worker_eff_n_") + kBinNames[i],
                     load(worker_effective_n_hist_[i]));
  }
  out.emplace_back("worker_input_live_bytes", load(worker_input_live_bytes_));
  out.emplace_back("worker_output_live_bytes", load(worker_output_live_bytes_));
  // §1c.22 — record-time UVA accounting (captured Triton grid
  // size, set once during graph capture by Python-side
  // note_uva_request).
  out.emplace_back("uva_record_bytes", load(uva_record_bytes_));
  out.emplace_back("uva_record_count", load(uva_record_count_));
  // §1c.24 attribution counters. ns totals + invocation counts.
  out.emplace_back("dispatch_cb_count", load(dispatch_cb_count_));
  out.emplace_back("sync_cb_count", load(sync_cb_count_));
  out.emplace_back("sync_cb_wait_total_ns", load(sync_cb_wait_total_ns_));
  out.emplace_back("worker_run_count", load(worker_run_count_));
  out.emplace_back("worker_busy_total_ns", load(worker_busy_total_ns_));
  out.emplace_back("worker_queue_wait_total_ns",
                   load(worker_queue_wait_total_ns_));
  // §1c.29 M3 diag counters. Populated only when
  // `VLLM_COTS_DIAG=1` AND a captured graph that fires
  // `m3_wait_kernel_diag` runs (production-default path skips
  // these). Stored as host-mapped pinned int64_t cells so the
  // GPU can atomicAdd; the host pointer is read here.
  out.emplace_back("m3_immediate_resume_count",
                   m3_immediate_resume_host_ ? *m3_immediate_resume_host_ : 0);
  out.emplace_back("m3_lagging_wait_count",
                   m3_lagging_wait_host_ ? *m3_lagging_wait_host_ : 0);
  out.emplace_back("m3_wait_spin_iters_total",
                   m3_spin_iters_host_ ? *m3_spin_iters_host_ : 0);
  return out;
}

void CotsCpuInfer::reset_counters() {
  submit_count_qkv_.store(0, std::memory_order_relaxed);
  submit_count_mlp_.store(0, std::memory_order_relaxed);
  submit_count_dryrun_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < 8; ++i) {
    nt_hist_qkv_[i].store(0, std::memory_order_relaxed);
    nt_hist_mlp_[i].store(0, std::memory_order_relaxed);
    nt_hist_dryrun_[i].store(0, std::memory_order_relaxed);
  }
  d2h_1d_count_.store(0, std::memory_order_relaxed);
  d2h_2d_count_.store(0, std::memory_order_relaxed);
  d2h_record_bytes_1d_.store(0, std::memory_order_relaxed);
  d2h_record_bytes_2d_.store(0, std::memory_order_relaxed);
  runtime_set_calls_.store(0, std::memory_order_relaxed);
  runtime_last_value_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < 8; ++i) {
    worker_effective_n_hist_[i].store(0, std::memory_order_relaxed);
  }
  worker_input_live_bytes_.store(0, std::memory_order_relaxed);
  worker_output_live_bytes_.store(0, std::memory_order_relaxed);
  uva_record_bytes_.store(0, std::memory_order_relaxed);
  uva_record_count_.store(0, std::memory_order_relaxed);
  d2h_replay_bucket_bytes_.store(0, std::memory_order_relaxed);
  uva_replay_bucket_bytes_.store(0, std::memory_order_relaxed);
  // §1c.24 attribution counters.
  dispatch_cb_count_.store(0, std::memory_order_relaxed);
  sync_cb_count_.store(0, std::memory_order_relaxed);
  sync_cb_wait_total_ns_.store(0, std::memory_order_relaxed);
  worker_run_count_.store(0, std::memory_order_relaxed);
  worker_busy_total_ns_.store(0, std::memory_order_relaxed);
  worker_queue_wait_total_ns_.store(0, std::memory_order_relaxed);
  // §1c.29 M3 diag counters. Lazy-allocated; only zero them
  // if they exist (i.e., M3 was installed at least once).
  if (m3_immediate_resume_host_) *m3_immediate_resume_host_ = 0;
  if (m3_lagging_wait_host_) *m3_lagging_wait_host_ = 0;
  if (m3_spin_iters_host_) *m3_spin_iters_host_ = 0;
}

// §1c.29 M3 — install per-slab host-mapped pinned slots.
// Forward declarations of the launchers in cots_m3_wait_kernel.cu
// so we can call them from this C++ TU without a header pulling
// in CUDA-specific symbols.
extern "C" void launch_m3_wait_kernel_production(uint32_t*, uint32_t*,
                                                 cudaStream_t);
extern "C" void launch_m3_wait_kernel_diag(uint32_t*, uint32_t*, int64_t*,
                                           int64_t*, int64_t*, cudaStream_t);

// §1c.29 helper: lazily allocate the diag counter cells the
// first time install_m3_for_task is called. Host-mapped pinned
// int64_t each so the GPU kernel can atomicAdd directly. Reads
// in get_counters/reset_counters use the host pointer.
static void ensure_m3_diag_cell(int64_t** host_ptr, int64_t** dev_ptr,
                                const char* name) {
  if (*host_ptr != nullptr) return;
  void* hp = nullptr;
  cudaError_t e = cudaHostAlloc(&hp, sizeof(int64_t), cudaHostAllocMapped);
  TORCH_CHECK(e == cudaSuccess, "install_m3_for_task: cudaHostAlloc(", name,
              ") failed: ", cudaGetErrorString(e));
  *static_cast<int64_t*>(hp) = 0;
  void* dp = nullptr;
  cudaError_t e2 = cudaHostGetDevicePointer(&dp, hp, 0);
  if (e2 != cudaSuccess) {
    cudaFreeHost(hp);
    TORCH_CHECK(false, "install_m3_for_task: cudaHostGetDevicePointer(", name,
                ") failed: ", cudaGetErrorString(e2));
  }
  *host_ptr = static_cast<int64_t*>(hp);
  *dev_ptr = static_cast<int64_t*>(dp);
}

void CotsCpuInfer::install_m3_for_task(int64_t task_id) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "install_m3_for_task: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(!s.m3_installed.load(std::memory_order_acquire),
              "install_m3_for_task: M3 already installed for task_id=", task_id,
              " (idempotency violation)");
  // Lazy-alloc the per-runner diag counter cells, but ONLY when
  // VLLM_COTS_DIAG=1 — production M3 should not pay the pinned-
  // allocation surface for cells the diag kernel will never read.
  // Per reviewer (§1c.29 commit 1 fix): diag-only allocation
  // surface keeps the production failure space minimal.
  // m3_wait_on_stream re-checks diag_enabled() at each launch and
  // selects the diag kernel only if both the env is set AND the
  // cells are allocated; in production these cells stay nullptr
  // and the production launcher (which doesn't take counter ptrs)
  // is used instead.
  if (nvtx_internal::diag_enabled()) {
    ensure_m3_diag_cell(&m3_immediate_resume_host_, &m3_immediate_resume_dev_,
                        "m3_immediate_resume");
    ensure_m3_diag_cell(&m3_lagging_wait_host_, &m3_lagging_wait_dev_,
                        "m3_lagging_wait");
    ensure_m3_diag_cell(&m3_spin_iters_host_, &m3_spin_iters_dev_,
                        "m3_spin_iters");
  }
  // Allocate one uint32_t per slot, host-mapped pinned. We keep
  // host_*_ptr (CPU-visible) and dev_*_ptr (GPU-visible — same
  // memory, different virtual address) on the slab. Hard-fails
  // on any allocation/mapping error per §1c.29 safety gate
  // (c)/(d) — silent fallback under graph capture would put
  // different slabs on different mechanisms.
  void* host_req = nullptr;
  void* host_done = nullptr;
  cudaError_t e1 =
      cudaHostAlloc(&host_req, sizeof(uint32_t), cudaHostAllocMapped);
  TORCH_CHECK(e1 == cudaSuccess,
              "install_m3_for_task: cudaHostAlloc(req_slot) failed: ",
              cudaGetErrorString(e1));
  cudaError_t e2 =
      cudaHostAlloc(&host_done, sizeof(uint32_t), cudaHostAllocMapped);
  if (e2 != cudaSuccess) {
    cudaFreeHost(host_req);  // partial-failure cleanup
    TORCH_CHECK(false, "install_m3_for_task: cudaHostAlloc(done_slot) failed: ",
                cudaGetErrorString(e2));
  }
  *static_cast<uint32_t*>(host_req) = 0;
  *static_cast<uint32_t*>(host_done) = 0;
  void* dev_req = nullptr;
  void* dev_done = nullptr;
  cudaError_t e3 = cudaHostGetDevicePointer(&dev_req, host_req, 0);
  cudaError_t e4 = cudaHostGetDevicePointer(&dev_done, host_done, 0);
  if (e3 != cudaSuccess || e4 != cudaSuccess) {
    cudaFreeHost(host_req);
    cudaFreeHost(host_done);
    TORCH_CHECK(false, "install_m3_for_task: cudaHostGetDevicePointer failed: ",
                cudaGetErrorString(e3 != cudaSuccess ? e3 : e4));
  }
  s.host_req_slot = host_req;
  s.dev_req_slot = dev_req;
  s.host_done_slot = host_done;
  s.dev_done_slot = dev_done;
  s.next_seq.store(0, std::memory_order_relaxed);
  s.m3_installed.store(true, std::memory_order_release);
}

void CotsCpuInfer::m3_wait_on_stream(int64_t task_id, uintptr_t cuda_stream) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "m3_wait_on_stream: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(s.m3_installed.load(std::memory_order_acquire),
              "m3_wait_on_stream: M3 not installed for task_id=", task_id,
              "; call install_m3_for_task first");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (nvtx_internal::diag_enabled()) {
    // Diag counter cells are lazy-allocated by install_m3_for_task,
    // so they must exist by now (we already passed the m3_installed
    // gate above).
    TORCH_CHECK(m3_immediate_resume_dev_ != nullptr &&
                    m3_lagging_wait_dev_ != nullptr &&
                    m3_spin_iters_dev_ != nullptr,
                "m3_wait_on_stream: diag mode active but diag counter "
                "cells not allocated (logic bug — install_m3_for_task "
                "should have allocated them)");
    launch_m3_wait_kernel_diag(static_cast<uint32_t*>(s.dev_req_slot),
                               static_cast<uint32_t*>(s.dev_done_slot),
                               m3_spin_iters_dev_, m3_immediate_resume_dev_,
                               m3_lagging_wait_dev_, stream);
  } else {
    launch_m3_wait_kernel_production(static_cast<uint32_t*>(s.dev_req_slot),
                                     static_cast<uint32_t*>(s.dev_done_slot),
                                     stream);
  }
  // §1c.29 commit 1 review-fix-2: surface launch errors here, not
  // at the next stream sync. cudaGetLastError catches sync errors
  // (invalid kernel config, bad stream, etc.); async kernel-runtime
  // errors will still surface at the next sync, but this gives a
  // tight blame-line for any launch-config bug in commit 2.
  cudaError_t le = cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "m3_wait_on_stream: kernel launch failed: ",
              cudaGetErrorString(le));
}

uint32_t CotsCpuInfer::m3_get_req_slot(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "m3_get_req_slot: task_id out of range");
  const TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(s.m3_installed.load(std::memory_order_acquire),
              "m3_get_req_slot: M3 not installed for task_id=", task_id);
  return *static_cast<volatile uint32_t*>(s.host_req_slot);
}

uint32_t CotsCpuInfer::m3_get_done_slot(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "m3_get_done_slot: task_id out of range");
  const TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(s.m3_installed.load(std::memory_order_acquire),
              "m3_get_done_slot: M3 not installed for task_id=", task_id);
  return *static_cast<volatile uint32_t*>(s.host_done_slot);
}

void CotsCpuInfer::m3_set_req_slot(int64_t task_id, uint32_t value) {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "m3_set_req_slot: task_id out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(s.m3_installed.load(std::memory_order_acquire),
              "m3_set_req_slot: M3 not installed for task_id=", task_id);
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint32_t*>(s.host_req_slot) = value;
  std::atomic_thread_fence(std::memory_order_release);
}

void CotsCpuInfer::m3_set_done_slot(int64_t task_id, uint32_t value) {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "m3_set_done_slot: task_id out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(s.m3_installed.load(std::memory_order_acquire),
              "m3_set_done_slot: M3 not installed for task_id=", task_id);
  // Worker-side publish ordering (§1c.29 reminder): in production
  // the worker writes y_pinned, releases via fence, THEN writes
  // done_slot=seq. The test helper omits y_pinned (no real CPU
  // GEMM in the smoke), but keeps the release fence for
  // consistency.
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint32_t*>(s.host_done_slot) = value;
  std::atomic_thread_fence(std::memory_order_release);
}

}  // namespace cots
}  // namespace vllm
