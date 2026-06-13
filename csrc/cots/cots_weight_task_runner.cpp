// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 1c — native weight task runner implementation.

#include "cots_weight_task_runner.h"

#include "cots_common.h"

#include <ATen/ops/linear.h>
#include <c10/core/InferenceMode.h>
#include <pthread.h>
#include <sched.h>
#include <torch/torch.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vllm {
namespace cots {

// Forward declaration — defined in bf16_gemm_transposed.cpp.
// Used by run_bf16_gemm_transposed_inline below to call into
// the Stage 7 custom AVX2 BF16 GEMM kernel without a header.
void bf16_gemm_transposed_at(const at::Tensor& x, const at::Tensor& w,
                             at::Tensor& y_out);

// Forward declaration — defined in
// bf16_gemm_natural.cpp. Stage 7-D probe entry
// for the natural (N, K) row-major BF16 GEMM kernel.
void bf16_gemm_natural_at(const at::Tensor& x, const at::Tensor& w,
                          at::Tensor& y_out);

// Forward declaration — defined in bf16_mlp_gate_up_silu.cpp. Production MLP
// CPU path: gate/up/SwiGLU into BF16 scratch, then transposed down GEMM.
void bf16_mlp_gate_up_silu_down(const uint16_t* x, const uint16_t* w_gate,
                                const uint16_t* w_up, const uint16_t* w_down,
                                uint16_t* y, uint16_t* z_scratch, int64_t M,
                                int64_t H, int64_t I, int64_t O);

namespace {

constexpr int kMaxCpus = 64;

// Instrumentation gates. `VLLM_COTS_DIAG=1` is the umbrella shortcut; split
// flags let benchmark runs enable counters or the diagnostic wait kernel
// independently. NVTX is shared across COTS runners via NvtxScope in
// cots_common.h.
namespace cots_diag {
inline bool legacy_enabled() {
  static const bool enabled = []() { return env_flag("VLLM_COTS_DIAG"); }();
  return enabled;
}

inline bool counters_enabled() {
  static const bool enabled = []() {
    return legacy_enabled() || env_flag("VLLM_COTS_COUNTERS");
  }();
  return enabled;
}

inline bool wait_kernel_diag_enabled() {
  static const bool enabled = []() {
    return legacy_enabled() || env_flag("VLLM_COTS_WAIT_KERNEL_DIAG");
  }();
  return enabled;
}
}  // namespace cots_diag

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

}  // namespace

CotsWeightTaskRunner::CotsWeightTaskRunner()
    : task_queue_(std::make_unique<TaskQueue>()) {
  sync_args_.runner = static_cast<void*>(this);
  sync_args_.allow_n_pending = 0;
}

CotsWeightTaskRunner::~CotsWeightTaskRunner() {
  // Drain before destructing the queue so any in-flight task completes.
  if (task_queue_) {
    task_queue_->sync(0);
  }
  // §1c.29 wait-kernel sync: free per-slab host-mapped pinned slots. These are
  // allocated lazily via cudaHostAlloc(cudaHostAllocMapped) by
  // install_wait_kernel_sync_for_task; teardown must release them or the host
  // pinned region leaks. Walk the slabs_ array (sized at install
  // time) and free any with wait_kernel_sync_installed=true.
  if (slabs_) {
    for (int64_t i = 0; i < slab_count_; ++i) {
      TaskSlab& s = slabs_[i];
      if (s.wait_kernel_sync_installed.load(std::memory_order_acquire)) {
        if (s.host_req_slot != nullptr) {
          cudaFreeHost(s.host_req_slot);
          s.host_req_slot = nullptr;
        }
        if (s.host_done_slot != nullptr) {
          cudaFreeHost(s.host_done_slot);
          s.host_done_slot = nullptr;
        }
        s.wait_kernel_sync_installed.store(false, std::memory_order_release);
      }
    }
  }
  // Free wait-kernel sync diag counter cells if allocated.
  if (wait_kernel_immediate_resume_host_ != nullptr) {
    cudaFreeHost(wait_kernel_immediate_resume_host_);
    wait_kernel_immediate_resume_host_ = nullptr;
  }
  if (wait_kernel_lagging_wait_host_ != nullptr) {
    cudaFreeHost(wait_kernel_lagging_wait_host_);
    wait_kernel_lagging_wait_host_ = nullptr;
  }
  if (wait_kernel_spin_iters_host_ != nullptr) {
    cudaFreeHost(wait_kernel_spin_iters_host_);
    wait_kernel_spin_iters_host_ = nullptr;
  }
}

void CotsWeightTaskRunner::install(int64_t n_slabs, int64_t max_num_tokens) {
  TORCH_CHECK(n_slabs >= 0, "install: n_slabs must be >= 0, got ", n_slabs);
  TORCH_CHECK(!slabs_,
              "install: CotsWeightTaskRunner already installed; call once per "
              "CotsWeightTaskRunner instance");

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
  max_num_tokens_ = max_num_tokens;
}

void CotsWeightTaskRunner::check_error() {
  if (!has_error_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  // Throwing std::runtime_error: pybind11 maps it to Python RuntimeError
  // automatically. Mirrors the Python runner's `future.result()` re-raise.
  throw std::runtime_error(msg);
}

int32_t CotsWeightTaskRunner::slab_bucket_capacity_tokens(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "slab_bucket_capacity_tokens: task_id ", task_id,
              " out of range");
  return slabs_[task_id].bucket_capacity_tokens;
}

int32_t CotsWeightTaskRunner::slab_num_tokens(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "slab_num_tokens: task_id ", task_id, " out of range");
  return slabs_[task_id].num_tokens.load(std::memory_order_acquire);
}

void CotsWeightTaskRunner::populate_slab_qkv(
    int64_t task_id, int32_t n_threads, int32_t bucket_capacity_tokens,
    uintptr_t x_pinned_ptr, int32_t in_dim, uintptr_t y_pinned_ptr,
    int32_t cpu_out_dim, uintptr_t w_cpu_ptr, int32_t w_cpu_rows) {
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

void CotsWeightTaskRunner::populate_slab_mlp(
    int64_t task_id, int32_t n_threads, int32_t bucket_capacity_tokens,
    uintptr_t x_pinned_ptr, int32_t in_dim, uintptr_t y_pinned_ptr,
    int32_t cpu_out_dim, uintptr_t w_gate_ptr, int32_t w_gate_rows,
    uintptr_t w_up_ptr, int32_t w_up_rows, uintptr_t w_down_ptr,
    int32_t w_down_rows, int32_t w_down_cols) {
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
}

void CotsWeightTaskRunner::populate_slab_dryrun(
    int64_t task_id, int32_t bucket_capacity_tokens, uintptr_t x_pinned_ptr,
    int32_t in_dim, uintptr_t y_pinned_ptr, int32_t cpu_out_dim) {
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

void CotsWeightTaskRunner::submit_on_stream(int64_t task_id, int32_t num_tokens,
                                            uintptr_t x_gpu_ptr, int64_t x_cols,
                                            int64_t x_stride0,
                                            int64_t x_stride1,
                                            uintptr_t cuda_stream) {
  NvtxScope nvtx_scope("cots:submit_on_stream");
  // Surface any prior worker error BEFORE queueing more work.
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "submit_on_stream: task_id ", task_id, " out of range");
  TaskSlab* slab = &slabs_[task_id];
  TORCH_CHECK(num_tokens >= 0, "submit_on_stream: num_tokens=", num_tokens,
              " < 0");
  // §1c.20: bound check against max_num_tokens_ catches Planner
  // mistakes where an oversized batch reaches a slab. Skipped when
  // max_num_tokens_==0 (test fixtures that don't run real tokens
  // through the slab — diagnostic-only path).
  TORCH_CHECK(max_num_tokens_ == 0 || num_tokens <= max_num_tokens_,
              "submit_on_stream: num_tokens=", num_tokens,
              " exceeds max_num_tokens=", max_num_tokens_,
              " (would write past the pinned buffer's tail)");
  // §1c.35 commit-2: clamp the Python-passed num_tokens at the
  // slab's bucket capacity. The passed value comes from
  // `int(x_gpu.shape[0])` which under vLLM's PIECEWISE compile +
  // BACKED dynamic shapes resolves to the SymInt hint
  // (typically max_num_batched_tokens or another large constant),
  // baked into the captured custom-op argument and shared across
  // all per-bucket FULL captures. `slab->bucket_capacity_tokens`
  // is IMMUTABLE (set at install time per (layer, bucket,
  // op_kind) from `op_descriptor[1]`) and IS the dispatched
  // bucket. Clamping here makes the captured D2H byte count and
  // `slab.num_tokens` tight per-bucket — matching what
  // §1c.21's runtime override clamps the worker GEMM to (live).
  // For B=1 decode at bucket=1: effective_n=1 even without the
  // override.
  num_tokens = std::min(num_tokens, slab->bucket_capacity_tokens);
  slab->num_tokens.store(num_tokens, std::memory_order_release);
  // §1c.21 counters: bump submit_count + num_tokens histogram for
  // this op kind. Diag-gated (§1c.34 cleanup C) — production-default
  // path skips the atomic adds entirely.
  if (cots_diag::counters_enabled()) {
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
  // §1c.20: D2H from x_gpu to slab's pinned input buffer.
  // Stride-aware: contiguous → 1D cudaMemcpyAsync, row-strided →
  // 2D cudaMemcpy2DAsync. Both graph-capturable. Skipped when
  // `x_gpu_ptr == 0` (test fixtures that exercise dispatch only).
  if (x_gpu_ptr != 0) {
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
    // §1c.34 cleanup C: D2H byte/count counters are diagnostic only;
    // diag-gate them so the hot path skips the atomic adds.
    const bool d2h_diag = cots_diag::counters_enabled();
    if (x_stride0 == x_cols) {
      // Contiguous row layout — single 1D copy is fastest.
      const size_t bytes = static_cast<size_t>(num_tokens) * width_bytes;
      if (d2h_diag) {
        d2h_1d_count_.fetch_add(1, std::memory_order_relaxed);
        d2h_record_bytes_1d_.fetch_add(static_cast<int64_t>(bytes),
                                       std::memory_order_relaxed);
      }
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
      if (d2h_diag) {
        d2h_2d_count_.fetch_add(1, std::memory_order_relaxed);
        d2h_record_bytes_2d_.fetch_add(static_cast<int64_t>(bytes_2d),
                                       std::memory_order_relaxed);
      }
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
  NvtxScope launch_scope("cots:launch_dispatch_cb");
  cudaError_t err =
      cudaLaunchHostFunc(stream, &CotsWeightTaskRunner::DispatchCallback,
                         static_cast<void*>(slab));
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaLaunchHostFunc(DispatchCallback) failed: ", cudaGetErrorString(err));
}

void CotsWeightTaskRunner::sync_on_stream(uintptr_t cuda_stream) {
  NvtxScope nvtx_scope("cots:sync_on_stream");
  check_error();
  // sync_args_ is a stable member of *this; safe to take its address as
  // userData for cudaLaunchHostFunc, including across CUDA graph replays.
  cudaError_t err = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(cuda_stream),
      &CotsWeightTaskRunner::SyncCallback, static_cast<void*>(&sync_args_));
  TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc(SyncCallback) failed: ",
              cudaGetErrorString(err));
}

void CotsWeightTaskRunner::sync_or_wait_on_stream(int64_t task_id,
                                                  uintptr_t cuda_stream) {
  // §1c.29 commit 2 — unified entry. Per-slab branch: if wait-kernel sync is
  // installed for this task, the captured node is the wait kernel
  // (reads the worker-published done_slot=seq). Otherwise the
  // captured node stays the host-callback SyncCallback node that
  // blocks the driver thread on TaskQueue::sync(0). Both
  // mechanisms can coexist within the same offloader / same
  // CotsWeightTaskRunner instance — the branch is per-slab, not per-
  // runner — but in practice the offloader sets the flag for ALL
  // slabs at install time (weight_capture_sync_mode="wait_kernel" is binary at
  // the runner level) so the branch is uniform across a single
  // offloader's slabs.
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "sync_or_wait_on_stream: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  if (s.wait_kernel_sync_installed.load(std::memory_order_acquire)) {
    // §1c.29 commit 2 review-fix: route through the no-check
    // launcher so a prior worker error does not block the wait
    // kernel from being recorded/launched. Without this, a worker
    // throw would set has_error_, then the next captured
    // sync_or_wait_on_stream would raise from check_error before
    // the wait kernel was launched, wedging the stream with no
    // done_slot consumer. Errors are still surfaced by
    // check_error() at the next safe Python entry point (the next
    // submit_on_stream, sync_blocking, etc.).
    wait_kernel_sync_on_stream_no_check(task_id, cuda_stream);
  } else {
    // Legacy path: keep check_error() in sync_on_stream. The
    // captured SyncCallback host_fn blocks the driver thread on
    // TaskQueue::sync(0); if the host_fn never gets recorded
    // (because check_error raised), the stream isn't really
    // "wedged" — it just hasn't been told to drain anything. The
    // worker still completes; the next call surfaces the error.
    sync_on_stream(cuda_stream);
  }
}

bool CotsWeightTaskRunner::wait_kernel_sync_installed_for_task(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "wait_kernel_sync_installed_for_task: task_id ", task_id,
              " out of range");
  return slabs_[task_id].wait_kernel_sync_installed.load(
      std::memory_order_acquire);
}

void CotsWeightTaskRunner::sync_blocking() {
  task_queue_->sync(0);
  // Surface any worker error that fired while we were waiting.
  check_error();
}

void CotsWeightTaskRunner::set_worker_affinity(uint64_t cpu_set) {
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

std::string CotsWeightTaskRunner::take_error() {
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  return msg;
}

void CotsWeightTaskRunner::run_at_linear_inline(at::Tensor x, at::Tensor w,
                                                at::Tensor y_out) {
  // Stage 1 microbench helper: directly drive `at::linear` from C++.
  // No TaskQueue, no host callback. Lets the test compare wall-clock
  // against Python `F.linear` on bit-identical tensors and catch the
  // catastrophic-scalar-fallback path documented in
  // `docs/phase0_findings.md §0.3.2`.
  c10::InferenceMode g;
  // c10::AutoDispatchBelowAutograd is implied by InferenceMode.
  y_out.copy_(at::linear(x, w));
}

void CotsWeightTaskRunner::run_bf16_gemm_transposed_inline(at::Tensor x,
                                                           at::Tensor w,
                                                           at::Tensor y_out) {
  // Correctness/diagnostic helper: drive the custom BF16
  // row-major-weight GEMM kernel (csrc/cots/bf16_gemm_transposed.cpp)
  // inline. No TaskQueue, no host callback. The wrapped function does
  // its own dtype/contiguity validation and TORCH_CHECK calls on shape;
  // we just hold InferenceMode so the dispatcher does not record
  // autograd metadata.
  c10::InferenceMode g;
  bf16_gemm_transposed_at(x, w, y_out);
}

void CotsWeightTaskRunner::run_bf16_gemm_natural_inline(at::Tensor x,
                                                        at::Tensor w,
                                                        at::Tensor y_out) {
  // Stage 7-D probe — sibling kernel for the natural (N, K) row-major
  // BF16 GEMM layout. Same harness pattern as the Path H helper above.
  c10::InferenceMode g;
  bf16_gemm_natural_at(x, w, y_out);
}

void CotsWeightTaskRunner::run_bf16_mlp_inline(at::Tensor x, at::Tensor w_gate,
                                               at::Tensor w_up,
                                               at::Tensor w_down,
                                               at::Tensor y_out) {
  c10::InferenceMode g;
  TORCH_CHECK(x.device().is_cpu() && w_gate.device().is_cpu() &&
                  w_up.device().is_cpu() && w_down.device().is_cpu() &&
                  y_out.device().is_cpu(),
              "run_bf16_mlp_inline: all tensors must be CPU tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  w_gate.scalar_type() == at::kBFloat16 &&
                  w_up.scalar_type() == at::kBFloat16 &&
                  w_down.scalar_type() == at::kBFloat16 &&
                  y_out.scalar_type() == at::kBFloat16,
              "run_bf16_mlp_inline: all tensors must be bfloat16");
  TORCH_CHECK(x.is_contiguous() && w_gate.is_contiguous() &&
                  w_up.is_contiguous() && w_down.is_contiguous() &&
                  y_out.is_contiguous(),
              "run_bf16_mlp_inline: all tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w_gate.dim() == 2 && w_up.dim() == 2 &&
                  w_down.dim() == 2 && y_out.dim() == 2,
              "run_bf16_mlp_inline: all tensors must be rank-2");

  const int64_t M = x.size(0);
  const int64_t H = x.size(1);
  const int64_t I = w_gate.size(0);
  const int64_t O = w_down.size(1);
  TORCH_CHECK(w_gate.size(1) == H,
              "run_bf16_mlp_inline: w_gate shape must be (I, H)");
  TORCH_CHECK(w_up.size(0) == I && w_up.size(1) == H,
              "run_bf16_mlp_inline: w_up shape must match w_gate");
  TORCH_CHECK(w_down.size(0) == I,
              "run_bf16_mlp_inline: w_down shape must be (I, O)");
  TORCH_CHECK(y_out.size(0) == M && y_out.size(1) == O,
              "run_bf16_mlp_inline: y_out shape must be (M, O)");

  const int64_t z_elems = M * I;
  mlp_scratch_bf16_.resize(static_cast<size_t>(std::max<int64_t>(z_elems, 0)));
  bf16_mlp_gate_up_silu_down(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      reinterpret_cast<const uint16_t*>(w_gate.data_ptr()),
      reinterpret_cast<const uint16_t*>(w_up.data_ptr()),
      reinterpret_cast<const uint16_t*>(w_down.data_ptr()),
      reinterpret_cast<uint16_t*>(y_out.data_ptr()), mlp_scratch_bf16_.data(),
      M, H, I, O);
}

at::Tensor CotsWeightTaskRunner::y_pinned_view(int64_t task_id,
                                               int32_t num_tokens) const {
  // §1c.20: build an `at::from_blob` CPU tensor view over the slab's
  // pinned output buffer. Used by `cots_sync_then_uva`'s impl on the
  // captured-graph hot path so the CPU output tensor is NEVER a
  // custom-op argument visible to Inductor.
  if (task_id < 0 || task_id >= slab_count_) {
    throw std::out_of_range("CotsWeightTaskRunner::y_pinned_view: task_id=" +
                            std::to_string(task_id) + " out of range [0, " +
                            std::to_string(slab_count_) + ")");
  }
  const TaskSlab& slab = slabs_[task_id];
  if (slab.y_pinned_ptr == nullptr) {
    throw std::runtime_error(
        "CotsWeightTaskRunner::y_pinned_view: slab.y_pinned_ptr is null at "
        "task_id=" +
        std::to_string(task_id) + " (slab not populated?)");
  }
  if (num_tokens < 0) {
    throw std::invalid_argument(
        "CotsWeightTaskRunner::y_pinned_view: num_tokens=" +
        std::to_string(num_tokens) + " < 0");
  }
  if (max_num_tokens_ != 0 && num_tokens > max_num_tokens_) {
    throw std::invalid_argument(
        "CotsWeightTaskRunner::y_pinned_view: num_tokens=" +
        std::to_string(num_tokens) +
        " exceeds max_num_tokens=" + std::to_string(max_num_tokens_) +
        " (would read past the pinned buffer's tail)");
  }
  // §1c.35 commit-2: clamp at the slab's bucket capacity (same
  // rationale as submit_on_stream above). Makes the returned view
  // bucket-sized — so the captured UVA Triton kernel's grid is
  // sized to bucket * cpu_out_dim, not max * cpu_out_dim. The
  // operator's destination view (sized to the
  // Inductor-baked int(y_gpu.shape[0])) may be larger; the
  // Triton kernel writes src.numel() elements into the prefix
  // of dst, leaving the tail rows untouched. Downstream consumes
  // only the first `live_count <= bucket` rows (see also
  // _uva_copy_trusted_host_into_gpu's relaxed shape assertion).
  num_tokens = std::min(num_tokens, slab.bucket_capacity_tokens);
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
void CotsWeightTaskRunner::DispatchCallback(void* user_data) {
  NvtxScope nvtx_scope("cots:dispatch_cb");
  TaskSlab* slab = static_cast<TaskSlab*>(user_data);
  CotsWeightTaskRunner* self = static_cast<CotsWeightTaskRunner*>(slab->self);
  // §1c.24 attribution: stamp enqueue time so the worker can later
  // compute queue_wait = worker_start - enqueue_time. Gated together
  // with the NVTX scopes by VLLM_COTS_DIAG=1; in production-default
  // mode neither now_ns() nor the atomic write fires. Worker reads
  // enqueue_time_ns conditionally on the same flag, so a
  // diag-disabled run leaves it at its initial value (0).
  if (cots_diag::counters_enabled()) {
    slab->enqueue_time_ns.store(now_ns(), std::memory_order_release);
    self->dispatch_cb_count_.fetch_add(1, std::memory_order_relaxed);
  }
  // §1c.29 commit 2 — wait-kernel sync sequence publish. When wait-kernel sync
  // is installed for this slab, increment the slab-local seq, capture it into
  // the worker lambda, ENQUEUE the lambda FIRST, THEN publish
  // host_req_slot=seq. Per §1c.29 commit 2 review-fix this is the
  // strictly stronger order: the GPU wait kernel cannot observe
  // req=seq before the worker for that seq is queued. The two
  // CPU operations execute back-to-back and the wait kernel only
  // fires later as a captured stream node; in practice the GPU
  // never observes the in-between state, but the cleaner order
  // matches the standalone smoke and costs nothing.
  //
  // Wrap behavior: uint32_t monotonically increases. At ~1k ops/
  // generate this overflows after 2^32 ≈ 4.3e6 generates, far
  // beyond any practical run. Documented inline rather than
  // reset-on-wrap to keep the hot path branch-free.
  //
  // seq=0 in the lambda signals "no wait-kernel sync publish needed" —
  // RunSlabOnWorker skips the done_slot store, preserving the
  // legacy non-wait-kernel-sync path bit-for-bit.
  uint32_t seq = 0;
  const bool m3 =
      slab->wait_kernel_sync_installed.load(std::memory_order_acquire);
  if (m3) {
    seq = slab->next_seq.fetch_add(1, std::memory_order_relaxed) + 1u;
  }
  // Enqueue worker BEFORE publishing req_slot.
  self->task_queue_->enqueue(
      [self, slab, seq] { self->RunSlabOnWorker(slab, seq); });
  if (m3) {
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile uint32_t*>(slab->host_req_slot) = seq;
  }
}

// Sync-side host callback. Blocks the CUDA driver thread until the
// TaskQueue drains. The user_data is `&self->sync_args_` (stable member).
void CotsWeightTaskRunner::SyncCallback(void* user_data) {
  NvtxScope nvtx_scope("cots:sync_cb_wait");
  SyncArgs* args = static_cast<SyncArgs*>(user_data);
  CotsWeightTaskRunner* self = static_cast<CotsWeightTaskRunner*>(args->runner);
  // §1c.24 attribution: time the sync wait — distinguishes "driver
  // blocked waiting for the worker" from "driver doing other work
  // then unblocking immediately". Same VLLM_COTS_DIAG gate as the
  // dispatch counter; in production-default mode the timestamps
  // and atomic adds are skipped.
  if (cots_diag::counters_enabled()) {
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

void CotsWeightTaskRunner::RunSlabOnWorker(TaskSlab* slab, uint32_t seq) {
  // §1c.24 attribution: stamp worker start + queue wait, gated by
  // VLLM_COTS_DIAG. Production-default leaves worker_t0 at 0 (the
  // worker_busy_total_ns add at the end is also gated). NVTX scope
  // is independently gated inside NvtxScope's ctor.
  const bool diag = cots_diag::counters_enabled();
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

    // §1c.21: prefer live_num_tokens override (stored in runtime_num_tokens_,
    // set OUT OF GRAPH by `set_live_num_tokens` before each captured replay)
    // over `slab->num_tokens` (captured bucket capacity). Sentinel 0 → fall
    // back to slab capacity.
    //
    // §1c.31 (commit-3-real fix): the override is a CAP, not a
    // required row count. If it exceeds slab capacity, clamp to
    // slab_cap and bump worker_clamp_override_count_ for
    // observability. Pinned-buffer reads past slab_cap rows is UB,
    // so the clamp is the safe behavior. This commonly happens in
    // eager mode where set_live_num_tokens() applies globally to
    // whatever slab fires next, regardless of which bucket sized
    // that slab (e.g., B=4 prefill at input_len=8 → 32 tokens, but
    // an MLP slab keyed by the smallest bucket has capacity 8).
    const int32_t slab_cap = slab->num_tokens.load(std::memory_order_acquire);
    const int32_t override_n =
        runtime_num_tokens_.load(std::memory_order_acquire);
    int32_t n;
    if (override_n > 0) {
      if (override_n > slab_cap) {
        n = slab_cap;
        // §1c.34 cleanup C: clamp counter is diag-gated. Production
        // doesn't need observability of clamp events — the clamp
        // itself is a safety behavior (correct vs reading past the
        // pinned buffer); only the COUNT is observational.
        if (diag) {
          worker_clamp_override_count_.fetch_add(1, std::memory_order_relaxed);
        }
      } else {
        n = override_n;
      }
    } else {
      n = slab_cap;
    }

    // §1c.21 fix-validation + §1c.22 byte accounting: bin the
    // effective_n the worker actually used, plus per-replay bucket
    // and live byte counters. All diag-gated (§1c.34 cleanup C);
    // production-default skips the atomic adds entirely.
    if (diag) {
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
      // §1c.22 byte accounting. Read the IMMUTABLE
      // `bucket_capacity_tokens` populated at install time (NOT
      // mutable `slab->num_tokens` which can be overwritten by
      // later submit_on_stream calls during graph capture /
      // PIECEWISE Python re-execution).
      const int64_t bucket_n =
          static_cast<int64_t>(slab->bucket_capacity_tokens);
      if (slab->in_dim > 0) {
        const int64_t row_bytes_in = static_cast<int64_t>(slab->in_dim) *
                                     static_cast<int64_t>(sizeof(at::BFloat16));
        d2h_replay_bucket_bytes_.fetch_add(bucket_n * row_bytes_in,
                                           std::memory_order_relaxed);
        if (n > 0) {
          worker_input_live_bytes_.fetch_add(
              static_cast<int64_t>(n) * row_bytes_in,
              std::memory_order_relaxed);
        }
      }
      if (slab->cpu_out_dim > 0) {
        const int64_t row_bytes_out =
            static_cast<int64_t>(slab->cpu_out_dim) *
            static_cast<int64_t>(sizeof(at::BFloat16));
        uva_replay_bucket_bytes_.fetch_add(bucket_n * row_bytes_out,
                                           std::memory_order_relaxed);
        if (n > 0) {
          worker_output_live_bytes_.fetch_add(
              static_cast<int64_t>(n) * row_bytes_out,
              std::memory_order_relaxed);
        }
      }
    }

    switch (slab->op_kind) {
      case TaskSlab::kDryrunNoop: {
        // Stage 2 substrate gate: install all wrappers but skip real
        // CPU work. Mirrors `_cpu_dryrun_noop` (cots.py:1161).
        break;
      }
      case TaskSlab::kQkv: {
        // Stage 7-C: replace at::linear (oneDNN bf16:bf16 emulation
        // path, ~30 µs dispatch + scratch reorder per call) with our
        // custom natural-layout BF16 GEMM. Same (N, K) row-major
        // weight layout the slab already carries; output is written
        // in-place into y_view's pinned-memory backing.
        auto x_view =
            ContigCpuViewFromBlob(slab->x_pinned_ptr, n, slab->in_dim);
        auto w_view = ContigCpuViewFromBlob(slab->w_cpu_ptr, slab->w_cpu_rows,
                                            slab->in_dim);
        auto y_view =
            ContigCpuViewFromBlob(slab->y_pinned_ptr, n, slab->cpu_out_dim);
        bf16_gemm_natural_at(x_view, w_view, y_view);
        break;
      }
      case TaskSlab::kMlpBlock: {
        TORCH_CHECK(slab->w_gate_rows == slab->w_up_rows,
                    "MLP CPU path requires gate/up row counts to match, got ",
                    slab->w_gate_rows, " and ", slab->w_up_rows);
        TORCH_CHECK(slab->w_down_rows == slab->w_gate_rows,
                    "MLP CPU path requires down K to match intermediate rows, "
                    "got down_rows=",
                    slab->w_down_rows, " gate_rows=", slab->w_gate_rows);
        TORCH_CHECK(slab->w_down_cols == slab->cpu_out_dim,
                    "MLP CPU path output dim mismatch: down_cols=",
                    slab->w_down_cols, " cpu_out_dim=", slab->cpu_out_dim);
        const int64_t z_elems =
            static_cast<int64_t>(n) * static_cast<int64_t>(slab->w_down_rows);
        mlp_scratch_bf16_.resize(static_cast<size_t>(
            std::max<int64_t>(z_elems, static_cast<int64_t>(0))));
        bf16_mlp_gate_up_silu_down(
            reinterpret_cast<const uint16_t*>(slab->x_pinned_ptr),
            reinterpret_cast<const uint16_t*>(slab->w_gate_ptr),
            reinterpret_cast<const uint16_t*>(slab->w_up_ptr),
            reinterpret_cast<const uint16_t*>(slab->w_down_ptr),
            reinterpret_cast<uint16_t*>(slab->y_pinned_ptr),
            mlp_scratch_bf16_.data(), n, slab->in_dim, slab->w_down_rows,
            slab->w_down_cols);
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
  // §1c.29 commit 2 — finally-publish for wait-kernel sync. Even on exception
  // we MUST write done_slot=seq so the captured cots_wait_done_kernel can exit
  // its spin loop. Without this publish a worker throw would wedge the GPU
  // stream in an infinite spin and the next Python- side check would never see
  // the error (because the next submit/sync is itself behind the wedged
  // stream). On the success path, has_error_ stays false and the consumer reads
  // y_pinned normally; on the failure path, has_error_ + msg
  // surface at the next Python-side submit/sync as a
  // RuntimeError, but the GPU stream is unblocked first so that
  // check actually runs. seq=0 means "no wait-kernel sync installed for this
  // slab" — skip the publish to keep the legacy path bit-for-bit
  // identical.
  if (seq != 0 &&
      slab->wait_kernel_sync_installed.load(std::memory_order_acquire)) {
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

void CotsWeightTaskRunner::note_uva_request(int32_t num_tokens,
                                            int32_t cpu_out_dim) {
  // §1c.22 bookkeeping. Diag-gated (§1c.34 cleanup C) —
  // measurement-only path; no functional dependency.
  if (num_tokens <= 0 || cpu_out_dim <= 0) return;
  if (!cots_diag::counters_enabled()) return;
  const int64_t bytes = static_cast<int64_t>(num_tokens) *
                        static_cast<int64_t>(cpu_out_dim) *
                        static_cast<int64_t>(sizeof(at::BFloat16));
  uva_record_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  uva_record_count_.fetch_add(1, std::memory_order_relaxed);
}

void CotsWeightTaskRunner::set_live_num_tokens(int32_t n) {
  TORCH_CHECK(n >= 0, "set_live_num_tokens: n=", n,
              " < 0; pass 0 to clear the override.");
  // Release store: the worker's acquire load in RunSlabOnWorker pairs
  // with this. The caller's responsibility is to set this BEFORE the
  // captured graph replay begins (i.e., from CotsOffloader.on_dispatch
  // outside the captured region). This store
  // is FUNCTIONAL (drives the worker's effective_n) and must stay
  // always-on; the live_set_calls / live_last_value counters
  // alongside are diagnostic only and diag-gated (§1c.34 cleanup C).
  runtime_num_tokens_.store(n, std::memory_order_release);
  if (cots_diag::counters_enabled()) {
    live_set_calls_.fetch_add(1, std::memory_order_relaxed);
    live_last_value_.store(n, std::memory_order_relaxed);
  }
}

// --- §1c.21 perf-investigation counters --------------------------------

std::vector<std::pair<std::string, int64_t>>
CotsWeightTaskRunner::get_counters() const {
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
  out.emplace_back("live_set_calls", load(live_set_calls_));
  out.emplace_back("live_last_value", load(live_last_value_));
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
  out.emplace_back("worker_clamp_override_count",
                   load(worker_clamp_override_count_));
  // §1c.29 wait-kernel sync diag counters. Populated only when
  // `VLLM_COTS_DIAG=1` AND a captured graph that fires
  // `cots_wait_done_kernel_diag` runs (production-default path skips
  // these). Stored as host-mapped pinned int64_t cells so the
  // GPU can atomicAdd; the host pointer is read here.
  out.emplace_back("wait_kernel_immediate_resume_count",
                   wait_kernel_immediate_resume_host_
                       ? *wait_kernel_immediate_resume_host_
                       : 0);
  out.emplace_back(
      "wait_kernel_lagging_wait_count",
      wait_kernel_lagging_wait_host_ ? *wait_kernel_lagging_wait_host_ : 0);
  out.emplace_back(
      "wait_kernel_spin_iters_total",
      wait_kernel_spin_iters_host_ ? *wait_kernel_spin_iters_host_ : 0);
  return out;
}

void CotsWeightTaskRunner::reset_counters() {
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
  live_set_calls_.store(0, std::memory_order_relaxed);
  live_last_value_.store(0, std::memory_order_relaxed);
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
  worker_clamp_override_count_.store(0, std::memory_order_relaxed);
  // §1c.29 wait-kernel sync diag counters. Lazy-allocated; only zero them
  // if they exist (i.e., wait-kernel sync was installed at least once).
  if (wait_kernel_immediate_resume_host_)
    *wait_kernel_immediate_resume_host_ = 0;
  if (wait_kernel_lagging_wait_host_) *wait_kernel_lagging_wait_host_ = 0;
  if (wait_kernel_spin_iters_host_) *wait_kernel_spin_iters_host_ = 0;
}

// §1c.29 wait-kernel sync — install per-slab host-mapped pinned slots.
// Shared wait-kernel diagnostics and launch checks live in cots_common.h.
void CotsWeightTaskRunner::install_wait_kernel_sync_for_task(int64_t task_id) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "install_wait_kernel_sync_for_task: task_id ", task_id,
              " out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(!s.wait_kernel_sync_installed.load(std::memory_order_acquire),
              "install_wait_kernel_sync_for_task: wait-kernel sync already "
              "installed for task_id=",
              task_id, " (idempotency violation)");
  // Lazy-alloc the per-runner diag counter cells, but ONLY when
  // VLLM_COTS_DIAG=1 — production wait-kernel sync should not pay the pinned-
  // allocation surface for cells the diag kernel will never read.
  // Per reviewer (§1c.29 commit 1 fix): diag-only allocation
  // surface keeps the production failure space minimal.
  // wait_kernel_sync_on_stream re-checks diag_enabled() at each launch and
  // selects the diag kernel only if both the env is set AND the
  // cells are allocated; in production these cells stay nullptr
  // and the production launcher (which doesn't take counter ptrs)
  // is used instead.
  if (cots_diag::wait_kernel_diag_enabled()) {
    ensure_mapped_i64_cell(
        &wait_kernel_immediate_resume_host_, &wait_kernel_immediate_resume_dev_,
        "install_wait_kernel_sync_for_task", "wait_kernel_immediate_resume");
    ensure_mapped_i64_cell(
        &wait_kernel_lagging_wait_host_, &wait_kernel_lagging_wait_dev_,
        "install_wait_kernel_sync_for_task", "wait_kernel_lagging_wait");
    ensure_mapped_i64_cell(
        &wait_kernel_spin_iters_host_, &wait_kernel_spin_iters_dev_,
        "install_wait_kernel_sync_for_task", "wait_kernel_spin_iters");
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
  TORCH_CHECK(
      e1 == cudaSuccess,
      "install_wait_kernel_sync_for_task: cudaHostAlloc(req_slot) failed: ",
      cudaGetErrorString(e1));
  cudaError_t e2 =
      cudaHostAlloc(&host_done, sizeof(uint32_t), cudaHostAllocMapped);
  if (e2 != cudaSuccess) {
    cudaFreeHost(host_req);  // partial-failure cleanup
    TORCH_CHECK(
        false,
        "install_wait_kernel_sync_for_task: cudaHostAlloc(done_slot) failed: ",
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
    TORCH_CHECK(
        false,
        "install_wait_kernel_sync_for_task: cudaHostGetDevicePointer failed: ",
        cudaGetErrorString(e3 != cudaSuccess ? e3 : e4));
  }
  s.host_req_slot = host_req;
  s.dev_req_slot = dev_req;
  s.host_done_slot = host_done;
  s.dev_done_slot = dev_done;
  s.next_seq.store(0, std::memory_order_relaxed);
  s.wait_kernel_sync_installed.store(true, std::memory_order_release);
}

void CotsWeightTaskRunner::wait_kernel_sync_on_stream(int64_t task_id,
                                                      uintptr_t cuda_stream) {
  check_error();
  wait_kernel_sync_on_stream_no_check(task_id, cuda_stream);
}

void CotsWeightTaskRunner::wait_kernel_sync_on_stream_no_check(
    int64_t task_id, uintptr_t cuda_stream) {
  // §1c.29 commit 2 review-fix — DO NOT call check_error() here.
  // This launcher is used by `sync_or_wait_on_stream` on the
  // captured-graph hot path; if a prior worker task set
  // has_error_, raising here would prevent the wait kernel from
  // being recorded/launched and the stream would be wedged with
  // no done_slot consumer. The error is surfaced at the next
  // Python-side entry point that is safe to short-circuit
  // (submit_on_stream's check_error, sync_blocking's, etc.).
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "wait_kernel_sync_on_stream: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(
      s.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_sync_on_stream: wait-kernel sync not installed for task_id=",
      task_id, "; call install_wait_kernel_sync_for_task first");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  launch_wait_done_kernel(
      static_cast<uint32_t*>(s.dev_req_slot),
      static_cast<uint32_t*>(s.dev_done_slot),
      cots_diag::wait_kernel_diag_enabled(), wait_kernel_spin_iters_dev_,
      wait_kernel_immediate_resume_dev_, wait_kernel_lagging_wait_dev_, stream,
      "wait_kernel_sync_on_stream");
}

uint32_t CotsWeightTaskRunner::wait_kernel_get_req_slot(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "wait_kernel_get_req_slot: task_id out of range");
  const TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(
      s.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_get_req_slot: wait-kernel sync not installed for task_id=",
      task_id);
  return *static_cast<volatile uint32_t*>(s.host_req_slot);
}

uint32_t CotsWeightTaskRunner::wait_kernel_get_done_slot(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "wait_kernel_get_done_slot: task_id out of range");
  const TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(
      s.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_get_done_slot: wait-kernel sync not installed for task_id=",
      task_id);
  return *static_cast<volatile uint32_t*>(s.host_done_slot);
}

void CotsWeightTaskRunner::wait_kernel_set_req_slot(int64_t task_id,
                                                    uint32_t value) {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "wait_kernel_set_req_slot: task_id out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(
      s.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_set_req_slot: wait-kernel sync not installed for task_id=",
      task_id);
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint32_t*>(s.host_req_slot) = value;
  std::atomic_thread_fence(std::memory_order_release);
}

void CotsWeightTaskRunner::wait_kernel_set_done_slot(int64_t task_id,
                                                     uint32_t value) {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "wait_kernel_set_done_slot: task_id out of range");
  TaskSlab& s = slabs_[task_id];
  TORCH_CHECK(
      s.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_set_done_slot: wait-kernel sync not installed for task_id=",
      task_id);
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
