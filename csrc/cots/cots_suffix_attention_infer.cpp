// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 2 native runner for prepared CPU suffix attention tasks.

#include "cots_suffix_attention_infer.h"

#include <c10/core/InferenceMode.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>

namespace vllm {
namespace cots {

extern "C" void launch_cots_wait_done_kernel_production(uint32_t*, uint32_t*,
                                                        cudaStream_t);
extern "C" void launch_cots_wait_done_kernel_diag(uint32_t*, uint32_t*,
                                                  int64_t*, int64_t*, int64_t*,
                                                  cudaStream_t);

void qwen_bf16_suffix_attention_at(const at::Tensor& query,
                                   const at::Tensor& key_cache,
                                   const at::Tensor& value_cache,
                                   const at::Tensor& block_table,
                                   const at::Tensor& seq_lens, double scale,
                                   at::Tensor& output, at::Tensor& output_lse);
void qwen_bf16_scatter_suffix_kv_at(const at::Tensor& key,
                                    const at::Tensor& value,
                                    const at::Tensor& block_ids,
                                    const at::Tensor& block_offsets,
                                    at::Tensor& key_cache,
                                    at::Tensor& value_cache);

namespace {

constexpr int64_t kQwenNumQHeads = 28;
constexpr int64_t kQwenNumKVHeads = 4;
constexpr int64_t kQwenHeadDim = 128;

namespace cots_suffix_diag {
inline bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

inline bool nvtx_enabled() {
  static const bool enabled = []() {
    return env_flag("VLLM_COTS_DIAG") || env_flag("VLLM_COTS_NVTX");
  }();
  return enabled;
}

inline bool counters_enabled() {
  static const bool enabled = []() {
    return env_flag("VLLM_COTS_DIAG") || env_flag("VLLM_COTS_SUFFIX_COUNTERS");
  }();
  return enabled;
}

inline bool wait_kernel_diag_enabled() {
  static const bool enabled = []() {
    return env_flag("VLLM_COTS_DIAG") ||
           env_flag("VLLM_COTS_WAIT_KERNEL_DIAG") ||
           env_flag("VLLM_COTS_SUFFIX_WAIT_KERNEL_DIAG");
  }();
  return enabled;
}
}  // namespace cots_suffix_diag

struct NvtxScope {
  explicit NvtxScope(const char* name) {
    if (cots_suffix_diag::nvtx_enabled()) nvtxRangePushA(name);
  }
  ~NvtxScope() {
    if (cots_suffix_diag::nvtx_enabled()) nvtxRangePop();
  }
  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;
};

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void ensure_wait_kernel_diag_cell(int64_t** host_ptr, int64_t** dev_ptr,
                                  const char* name) {
  if (*host_ptr != nullptr) return;
  void* hp = nullptr;
  cudaError_t e = cudaHostAlloc(&hp, sizeof(int64_t), cudaHostAllocMapped);
  TORCH_CHECK(e == cudaSuccess,
              "install_wait_kernel_sync_for_task: cudaHostAlloc(", name,
              ") failed: ", cudaGetErrorString(e));
  *static_cast<int64_t*>(hp) = 0;
  void* dp = nullptr;
  cudaError_t e2 = cudaHostGetDevicePointer(&dp, hp, 0);
  if (e2 != cudaSuccess) {
    cudaFreeHost(hp);
    TORCH_CHECK(false,
                "install_wait_kernel_sync_for_task: cudaHostGetDevicePointer(",
                name, ") failed: ", cudaGetErrorString(e2));
  }
  *host_ptr = static_cast<int64_t*>(hp);
  *dev_ptr = static_cast<int64_t*>(dp);
}

at::Tensor Bf16View(void* ptr, at::IntArrayRef sizes) {
  auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
  return at::from_blob(ptr, sizes, opts);
}

at::Tensor Bf16StridedView(void* ptr, at::IntArrayRef sizes,
                           at::IntArrayRef strides) {
  auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
  return at::from_blob(ptr, sizes, strides, opts);
}

at::Tensor IntView(void* ptr, at::IntArrayRef sizes) {
  auto opts = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
  return at::from_blob(ptr, sizes, opts);
}

at::Tensor LongView(void* ptr, at::IntArrayRef sizes) {
  auto opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  return at::from_blob(ptr, sizes, opts);
}

at::Tensor FloatView(void* ptr, at::IntArrayRef sizes) {
  auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  return at::from_blob(ptr, sizes, opts);
}

}  // namespace

struct CotsSuffixAttentionInfer::SubmittedTask {
  CotsSuffixAttentionInfer* self = nullptr;
  SuffixAttentionTask* sync_task = nullptr;
  bool owned_by_callback = true;
  int64_t task_id = -1;
  uint32_t seq = 0;
  int64_t enqueue_time_ns = 0;

  void* query_ptr = nullptr;
  void* key_cache_ptr = nullptr;
  void* value_cache_ptr = nullptr;
  void* block_table_ptr = nullptr;
  void* seq_lens_ptr = nullptr;
  void* output_ptr = nullptr;
  void* output_lse_ptr = nullptr;
  void* scatter_block_ids_ptr = nullptr;
  void* scatter_block_offsets_ptr = nullptr;
  void* scatter_key_ptr = nullptr;
  void* scatter_value_ptr = nullptr;

  int32_t task_capacity = 0;
  int32_t query_capacity = 0;
  int64_t query_stride0 = 0;
  int64_t query_stride1 = 0;
  int64_t query_stride2 = 0;
  int32_t num_cpu_blocks = 0;
  int32_t block_size = 0;
  int32_t max_suffix_blocks = 0;
  int32_t scatter_count = 0;
  std::vector<uint16_t> qkv_snapshot;
  std::vector<uint16_t> query_snapshot;
  std::vector<uint16_t> scatter_key_snapshot;
  std::vector<uint16_t> scatter_value_snapshot;
  int64_t query_snapshot_stride0 = 0;
  int64_t query_snapshot_stride1 = 0;
  int64_t query_snapshot_stride2 = 1;
  std::vector<int32_t> block_table_snapshot;
  std::vector<int32_t> seq_lens_snapshot;
  std::vector<int64_t> scatter_block_ids_snapshot;
  std::vector<int64_t> scatter_block_offsets_snapshot;
  int32_t runtime_num_tokens = -1;
  int32_t runtime_scatter_count = -1;
  bool use_runtime_count_snapshot = false;
  bool scatter_from_qkv = false;
  bool scatter_from_separate_kv = false;
  bool snapshot_inputs = true;
  double scale = 0.0;
};

CotsSuffixAttentionInfer::CotsSuffixAttentionInfer()
    : task_queue_(std::make_unique<TaskQueue>()) {
  sync_args_.infer = static_cast<void*>(this);
  sync_args_.allow_n_pending = 0;
}

CotsSuffixAttentionInfer::~CotsSuffixAttentionInfer() {
  if (task_queue_) {
    task_queue_->sync(0);
  }
  if (tasks_) {
    for (int64_t i = 0; i < task_count_; ++i) {
      SuffixAttentionTask& t = tasks_[i];
      if (t.wait_kernel_sync_installed.load(std::memory_order_acquire)) {
        if (t.host_req_slot != nullptr) {
          cudaFreeHost(t.host_req_slot);
          t.host_req_slot = nullptr;
        }
        if (t.host_done_slot != nullptr) {
          cudaFreeHost(t.host_done_slot);
          t.host_done_slot = nullptr;
        }
        t.wait_kernel_sync_installed.store(false, std::memory_order_release);
      }
    }
  }
  if (wait_kernel_spin_iters_host_ != nullptr) {
    cudaFreeHost(wait_kernel_spin_iters_host_);
    wait_kernel_spin_iters_host_ = nullptr;
  }
  if (wait_kernel_immediate_resume_host_ != nullptr) {
    cudaFreeHost(wait_kernel_immediate_resume_host_);
    wait_kernel_immediate_resume_host_ = nullptr;
  }
  if (wait_kernel_lagging_wait_host_ != nullptr) {
    cudaFreeHost(wait_kernel_lagging_wait_host_);
    wait_kernel_lagging_wait_host_ = nullptr;
  }
}

void CotsSuffixAttentionInfer::install(int64_t n_tasks) {
  TORCH_CHECK(n_tasks >= 0, "install: n_tasks must be >= 0, got ", n_tasks);
  TORCH_CHECK(!tasks_, "install: CotsSuffixAttentionInfer already installed");
  if (n_tasks > 0) {
    tasks_ = std::unique_ptr<SuffixAttentionTask[]>(
        new SuffixAttentionTask[n_tasks]);
    for (int64_t i = 0; i < n_tasks; ++i) {
      tasks_[i].self = static_cast<void*>(this);
    }
  }
  task_count_ = n_tasks;
}

void CotsSuffixAttentionInfer::populate_task(
    int64_t task_id, uintptr_t query_ptr, int32_t query_capacity,
    int64_t query_stride0, int64_t query_stride1, int64_t query_stride2,
    uintptr_t key_cache_ptr, int32_t num_cpu_blocks, int32_t block_size,
    uintptr_t value_cache_ptr, uintptr_t block_table_ptr,
    int32_t max_suffix_blocks, uintptr_t seq_lens_ptr, uintptr_t output_ptr,
    uintptr_t output_lse_ptr, uintptr_t scatter_block_ids_ptr,
    uintptr_t scatter_block_offsets_ptr, uintptr_t scatter_key_ptr,
    uintptr_t scatter_value_ptr, int32_t scatter_count, bool scatter_from_qkv,
    bool scatter_from_separate_kv, bool snapshot_inputs, double scale) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < task_count_, "populate_task: task_id ",
              task_id, " out of range");
  TORCH_CHECK(query_capacity >= 0,
              "populate_task: query_capacity must be >= 0");
  TORCH_CHECK(
      query_stride0 > 0 && query_stride1 > 0 && query_stride2 == 1,
      "populate_task: query strides must be positive with stride2=1, got ",
      query_stride0, ", ", query_stride1, ", ", query_stride2);
  TORCH_CHECK(num_cpu_blocks > 0, "populate_task: num_cpu_blocks must be > 0");
  TORCH_CHECK(block_size > 0, "populate_task: block_size must be > 0");
  TORCH_CHECK(max_suffix_blocks > 0,
              "populate_task: max_suffix_blocks must be > 0");
  TORCH_CHECK(scatter_count >= 0, "populate_task: scatter_count must be >= 0");
  TORCH_CHECK(
      !(scatter_from_qkv && scatter_from_separate_kv),
      "populate_task: scatter source must be qkv or separate kv, not both");
  if (scatter_count > 0) {
    TORCH_CHECK(scatter_block_ids_ptr != 0,
                "populate_task: scatter_block_ids_ptr is null");
    TORCH_CHECK(scatter_block_offsets_ptr != 0,
                "populate_task: scatter_block_offsets_ptr is null");
    TORCH_CHECK(
        scatter_from_qkv || scatter_from_separate_kv,
        "populate_task: non-zero scatter_count requires a scatter source");
    if (scatter_from_separate_kv) {
      TORCH_CHECK(scatter_key_ptr != 0,
                  "populate_task: scatter_key_ptr is null");
      TORCH_CHECK(scatter_value_ptr != 0,
                  "populate_task: scatter_value_ptr is null");
    }
  }
  if (cots_suffix_diag::counters_enabled()) {
    populate_count_.fetch_add(1, std::memory_order_relaxed);
  }
  SuffixAttentionTask& t = tasks_[task_id];
  t.query_ptr = reinterpret_cast<void*>(query_ptr);
  t.query_capacity = query_capacity;
  t.query_stride0 = query_stride0;
  t.query_stride1 = query_stride1;
  t.query_stride2 = query_stride2;
  t.key_cache_ptr = reinterpret_cast<void*>(key_cache_ptr);
  t.num_cpu_blocks = num_cpu_blocks;
  t.block_size = block_size;
  t.value_cache_ptr = reinterpret_cast<void*>(value_cache_ptr);
  t.block_table_ptr = reinterpret_cast<void*>(block_table_ptr);
  t.max_suffix_blocks = max_suffix_blocks;
  t.seq_lens_ptr = reinterpret_cast<void*>(seq_lens_ptr);
  t.output_ptr = reinterpret_cast<void*>(output_ptr);
  t.output_lse_ptr = reinterpret_cast<void*>(output_lse_ptr);
  t.scatter_block_ids_ptr = reinterpret_cast<void*>(scatter_block_ids_ptr);
  t.scatter_block_offsets_ptr =
      reinterpret_cast<void*>(scatter_block_offsets_ptr);
  t.scatter_key_ptr = reinterpret_cast<void*>(scatter_key_ptr);
  t.scatter_value_ptr = reinterpret_cast<void*>(scatter_value_ptr);
  t.scatter_count = scatter_count;
  t.scatter_from_qkv = scatter_from_qkv;
  t.scatter_from_separate_kv = scatter_from_separate_kv;
  t.snapshot_inputs = snapshot_inputs;
  t.scale = scale;
}

void CotsSuffixAttentionInfer::submit_prepared_on_stream(
    int64_t task_id, uintptr_t cuda_stream) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "submit_prepared_on_stream: task_id ", task_id, " out of range");
  SuffixAttentionTask* task = &tasks_[task_id];
  TORCH_CHECK(task->query_ptr != nullptr,
              "submit_prepared_on_stream: query_ptr is null");
  TORCH_CHECK(task->key_cache_ptr != nullptr,
              "submit_prepared_on_stream: key_cache_ptr is null");
  TORCH_CHECK(task->value_cache_ptr != nullptr,
              "submit_prepared_on_stream: value_cache_ptr is null");
  TORCH_CHECK(task->block_table_ptr != nullptr,
              "submit_prepared_on_stream: block_table_ptr is null");
  TORCH_CHECK(task->seq_lens_ptr != nullptr,
              "submit_prepared_on_stream: seq_lens_ptr is null");
  TORCH_CHECK(task->output_ptr != nullptr,
              "submit_prepared_on_stream: output_ptr is null");
  TORCH_CHECK(task->output_lse_ptr != nullptr,
              "submit_prepared_on_stream: output_lse_ptr is null");
  const int32_t num_tokens = task->query_capacity;
  TORCH_CHECK(num_tokens > 0,
              "submit_prepared_on_stream: num_tokens must be > 0");
  task->num_tokens.store(num_tokens, std::memory_order_release);

  auto submitted = std::make_unique<SubmittedTask>();
  submitted->self = this;
  submitted->sync_task = task;
  submitted->task_id = task_id;
  submitted->task_capacity = num_tokens;
  submitted->query_capacity = task->query_capacity;
  submitted->query_ptr = task->query_ptr;
  submitted->query_stride0 = task->query_stride0;
  submitted->query_stride1 = task->query_stride1;
  submitted->query_stride2 = task->query_stride2;
  submitted->key_cache_ptr = task->key_cache_ptr;
  submitted->num_cpu_blocks = task->num_cpu_blocks;
  submitted->block_size = task->block_size;
  submitted->value_cache_ptr = task->value_cache_ptr;
  submitted->block_table_ptr = task->block_table_ptr;
  submitted->max_suffix_blocks = task->max_suffix_blocks;
  submitted->seq_lens_ptr = task->seq_lens_ptr;
  submitted->output_ptr = task->output_ptr;
  submitted->output_lse_ptr = task->output_lse_ptr;
  submitted->scatter_block_ids_ptr = task->scatter_block_ids_ptr;
  submitted->scatter_block_offsets_ptr = task->scatter_block_offsets_ptr;
  submitted->scatter_key_ptr = task->scatter_key_ptr;
  submitted->scatter_value_ptr = task->scatter_value_ptr;
  submitted->scatter_count = task->scatter_count;
  submitted->scatter_from_qkv = task->scatter_from_qkv;
  submitted->scatter_from_separate_kv = task->scatter_from_separate_kv;
  submitted->snapshot_inputs = task->snapshot_inputs;
  submitted->scale = task->scale;

  auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaError_t capture_err = cudaStreamIsCapturing(stream, &capture_status);
  TORCH_CHECK(capture_err == cudaSuccess,
              "cudaStreamIsCapturing(SuffixDispatchCallback) failed: ",
              cudaGetErrorString(capture_err));
  if (capture_status == cudaStreamCaptureStatusNone) {
    // Eager submissions have exact per-task capacities. Do not let an older
    // queued CPU task observe live-count overrides from a later scheduler wave.
    submitted->use_runtime_count_snapshot = true;
    submitted->runtime_num_tokens = -1;
    submitted->runtime_scatter_count = -1;

    const int64_t block_table_elems =
        static_cast<int64_t>(submitted->query_capacity) *
        submitted->max_suffix_blocks;
    const auto* block_table_src =
        static_cast<const int32_t*>(submitted->block_table_ptr);
    submitted->block_table_snapshot.assign(block_table_src,
                                           block_table_src + block_table_elems);
    const auto* seq_lens_src =
        static_cast<const int32_t*>(submitted->seq_lens_ptr);
    submitted->seq_lens_snapshot.assign(
        seq_lens_src, seq_lens_src + submitted->query_capacity);
    if (submitted->scatter_count > 0) {
      const auto* scatter_ids_src =
          static_cast<const int64_t*>(submitted->scatter_block_ids_ptr);
      const auto* scatter_offsets_src =
          static_cast<const int64_t*>(submitted->scatter_block_offsets_ptr);
      submitted->scatter_block_ids_snapshot.assign(
          scatter_ids_src, scatter_ids_src + submitted->scatter_count);
      submitted->scatter_block_offsets_snapshot.assign(
          scatter_offsets_src, scatter_offsets_src + submitted->scatter_count);
    }
  }

  if (cots_suffix_diag::counters_enabled()) {
    submit_count_.fetch_add(1, std::memory_order_relaxed);
  }

  void* callback_data = nullptr;
  const bool capture_submission = capture_status != cudaStreamCaptureStatusNone;
  if (capture_submission) {
    submitted->owned_by_callback = false;
    callback_data = static_cast<void*>(submitted.get());
  } else {
    callback_data = static_cast<void*>(submitted.release());
  }

  cudaError_t err =
      cudaLaunchHostFunc(stream, &DispatchCallback, callback_data);
  if (err != cudaSuccess && !capture_submission) {
    delete static_cast<SubmittedTask*>(callback_data);
  }
  TORCH_CHECK(err == cudaSuccess,
              "cudaLaunchHostFunc(SuffixDispatchCallback) failed: ",
              cudaGetErrorString(err));
  if (capture_submission) {
    graph_submitted_tasks_.push_back(std::move(submitted));
  }
}

void CotsSuffixAttentionInfer::set_runtime_counts(int32_t num_tokens,
                                                  int32_t scatter_count) {
  TORCH_CHECK(num_tokens >= -1, "set_runtime_counts: num_tokens=", num_tokens,
              " < -1; pass -1 to clear the row override.");
  TORCH_CHECK(scatter_count >= -1,
              "set_runtime_counts: scatter_count=", scatter_count,
              " < -1; pass -1 to clear the scatter override.");
  runtime_num_tokens_.store(num_tokens, std::memory_order_release);
  runtime_scatter_count_.store(scatter_count, std::memory_order_release);
}

void CotsSuffixAttentionInfer::sync_on_stream(uintptr_t cuda_stream) {
  check_error();
  auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t err = cudaLaunchHostFunc(stream, &SyncCallback,
                                       static_cast<void*>(&sync_args_));
  TORCH_CHECK(err == cudaSuccess,
              "cudaLaunchHostFunc(SuffixSyncCallback) failed: ",
              cudaGetErrorString(err));
}

void CotsSuffixAttentionInfer::sync_or_wait_on_stream(int64_t task_id,
                                                      uintptr_t cuda_stream) {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "sync_or_wait_on_stream: task_id ", task_id, " out of range");
  SuffixAttentionTask& task = tasks_[task_id];
  if (task.wait_kernel_sync_installed.load(std::memory_order_acquire)) {
    wait_kernel_sync_on_stream_no_check(task_id, cuda_stream);
  } else {
    sync_on_stream(cuda_stream);
  }
}

void CotsSuffixAttentionInfer::install_wait_kernel_sync_for_task(
    int64_t task_id) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "install_wait_kernel_sync_for_task: task_id ", task_id,
              " out of range");
  SuffixAttentionTask& task = tasks_[task_id];
  TORCH_CHECK(!task.wait_kernel_sync_installed.load(std::memory_order_acquire),
              "install_wait_kernel_sync_for_task: wait-kernel sync already "
              "installed for task_id=",
              task_id);

  if (cots_suffix_diag::wait_kernel_diag_enabled()) {
    ensure_wait_kernel_diag_cell(&wait_kernel_spin_iters_host_,
                                 &wait_kernel_spin_iters_dev_,
                                 "suffix_spin_iters");
    ensure_wait_kernel_diag_cell(&wait_kernel_immediate_resume_host_,
                                 &wait_kernel_immediate_resume_dev_,
                                 "suffix_immediate_resume");
    ensure_wait_kernel_diag_cell(&wait_kernel_lagging_wait_host_,
                                 &wait_kernel_lagging_wait_dev_,
                                 "suffix_lagging_wait");
  }

  void* host_req = nullptr;
  void* host_done = nullptr;
  cudaError_t e1 =
      cudaHostAlloc(&host_req, sizeof(uint32_t), cudaHostAllocMapped);
  TORCH_CHECK(e1 == cudaSuccess,
              "install_wait_kernel_sync_for_task: cudaHostAlloc(req_slot) "
              "failed: ",
              cudaGetErrorString(e1));
  cudaError_t e2 =
      cudaHostAlloc(&host_done, sizeof(uint32_t), cudaHostAllocMapped);
  if (e2 != cudaSuccess) {
    cudaFreeHost(host_req);
    TORCH_CHECK(false,
                "install_wait_kernel_sync_for_task: "
                "cudaHostAlloc(done_slot) failed: ",
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
    TORCH_CHECK(false,
                "install_wait_kernel_sync_for_task: "
                "cudaHostGetDevicePointer failed: ",
                cudaGetErrorString(e3 != cudaSuccess ? e3 : e4));
  }

  task.host_req_slot = host_req;
  task.dev_req_slot = dev_req;
  task.host_done_slot = host_done;
  task.dev_done_slot = dev_done;
  task.next_seq.store(0, std::memory_order_relaxed);
  task.wait_kernel_sync_installed.store(true, std::memory_order_release);
}

bool CotsSuffixAttentionInfer::wait_kernel_sync_installed_for_task(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "wait_kernel_sync_installed_for_task: task_id ", task_id,
              " out of range");
  return tasks_[task_id].wait_kernel_sync_installed.load(
      std::memory_order_acquire);
}

void CotsSuffixAttentionInfer::wait_kernel_sync_on_stream(
    int64_t task_id, uintptr_t cuda_stream) {
  check_error();
  wait_kernel_sync_on_stream_no_check(task_id, cuda_stream);
}

void CotsSuffixAttentionInfer::wait_kernel_sync_on_stream_no_check(
    int64_t task_id, uintptr_t cuda_stream) {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "wait_kernel_sync_on_stream: task_id ", task_id, " out of range");
  SuffixAttentionTask& task = tasks_[task_id];
  TORCH_CHECK(task.wait_kernel_sync_installed.load(std::memory_order_acquire),
              "wait_kernel_sync_on_stream: wait-kernel sync not installed for "
              "task_id=",
              task_id, "; call install_wait_kernel_sync_for_task first");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (cots_suffix_diag::counters_enabled()) {
    wait_kernel_launch_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (cots_suffix_diag::wait_kernel_diag_enabled()) {
    TORCH_CHECK(
        wait_kernel_spin_iters_dev_ != nullptr &&
            wait_kernel_immediate_resume_dev_ != nullptr &&
            wait_kernel_lagging_wait_dev_ != nullptr,
        "wait_kernel_sync_on_stream: suffix diag mode active but counter "
        "cells are not allocated");
    launch_cots_wait_done_kernel_diag(
        static_cast<uint32_t*>(task.dev_req_slot),
        static_cast<uint32_t*>(task.dev_done_slot), wait_kernel_spin_iters_dev_,
        wait_kernel_immediate_resume_dev_, wait_kernel_lagging_wait_dev_,
        stream);
  } else {
    launch_cots_wait_done_kernel_production(
        static_cast<uint32_t*>(task.dev_req_slot),
        static_cast<uint32_t*>(task.dev_done_slot), stream);
  }
  cudaError_t le = cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess,
              "wait_kernel_sync_on_stream: kernel launch failed: ",
              cudaGetErrorString(le));
}

uint32_t CotsSuffixAttentionInfer::wait_kernel_get_req_slot(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "wait_kernel_get_req_slot: task_id out of range");
  const SuffixAttentionTask& task = tasks_[task_id];
  TORCH_CHECK(
      task.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_get_req_slot: wait-kernel sync not installed for task_id=",
      task_id);
  return *static_cast<volatile uint32_t*>(task.host_req_slot);
}

uint32_t CotsSuffixAttentionInfer::wait_kernel_get_done_slot(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "wait_kernel_get_done_slot: task_id out of range");
  const SuffixAttentionTask& task = tasks_[task_id];
  TORCH_CHECK(
      task.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_get_done_slot: wait-kernel sync not installed for task_id=",
      task_id);
  return *static_cast<volatile uint32_t*>(task.host_done_slot);
}

void CotsSuffixAttentionInfer::wait_kernel_set_req_slot(int64_t task_id,
                                                        uint32_t value) {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "wait_kernel_set_req_slot: task_id out of range");
  SuffixAttentionTask& task = tasks_[task_id];
  TORCH_CHECK(
      task.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_set_req_slot: wait-kernel sync not installed for task_id=",
      task_id);
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint32_t*>(task.host_req_slot) = value;
}

void CotsSuffixAttentionInfer::wait_kernel_set_done_slot(int64_t task_id,
                                                         uint32_t value) {
  TORCH_CHECK(task_id >= 0 && task_id < task_count_,
              "wait_kernel_set_done_slot: task_id out of range");
  SuffixAttentionTask& task = tasks_[task_id];
  TORCH_CHECK(
      task.wait_kernel_sync_installed.load(std::memory_order_acquire),
      "wait_kernel_set_done_slot: wait-kernel sync not installed for task_id=",
      task_id);
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint32_t*>(task.host_done_slot) = value;
}

void CotsSuffixAttentionInfer::sync_blocking() {
  task_queue_->sync(0);
  check_error();
}

std::string CotsSuffixAttentionInfer::take_error() {
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  return msg;
}

void CotsSuffixAttentionInfer::check_error() {
  if (!has_error_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  throw std::runtime_error(msg);
}

void CotsSuffixAttentionInfer::DispatchCallback(void* user_data) {
  NvtxScope nvtx_scope("cots:suffix_dispatch_cb");
  SubmittedTask* raw_submitted = static_cast<SubmittedTask*>(user_data);
  std::shared_ptr<SubmittedTask> submitted =
      raw_submitted->owned_by_callback
          ? std::shared_ptr<SubmittedTask>(raw_submitted)
          : std::shared_ptr<SubmittedTask>(raw_submitted,
                                           [](SubmittedTask*) {});
  CotsSuffixAttentionInfer* self = submitted->self;
  SuffixAttentionTask* sync_task = submitted->sync_task;
  if (cots_suffix_diag::counters_enabled()) {
    submitted->enqueue_time_ns = now_ns();
    self->dispatch_cb_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (submitted->use_runtime_count_snapshot && submitted->snapshot_inputs) {
    // Eager submissions reach this callback only after the stream-ordered D2H
    // staging copies are complete. Snapshot Q/QKV here, not at submit time,
    // so the CPU worker owns stable inputs without racing incomplete copies or
    // later staging-buffer reuse.
    const auto* query_src = static_cast<const uint16_t*>(submitted->query_ptr);
    const int64_t snapshot_heads = submitted->scatter_from_qkv
                                       ? kQwenNumQHeads + 2 * kQwenNumKVHeads
                                       : kQwenNumQHeads;
    std::vector<uint16_t>& query_dst = submitted->scatter_from_qkv
                                           ? submitted->qkv_snapshot
                                           : submitted->query_snapshot;
    query_dst.resize(static_cast<int64_t>(submitted->query_capacity) *
                     snapshot_heads * kQwenHeadDim);
    for (int64_t b = 0; b < submitted->query_capacity; ++b) {
      for (int64_t h = 0; h < snapshot_heads; ++h) {
        const uint16_t* src = query_src + b * submitted->query_stride0 +
                              h * submitted->query_stride1;
        uint16_t* dst =
            query_dst.data() + (b * snapshot_heads + h) * kQwenHeadDim;
        for (int64_t d = 0; d < kQwenHeadDim; ++d) {
          dst[d] = src[d * submitted->query_stride2];
        }
      }
    }
    submitted->query_snapshot_stride0 = snapshot_heads * kQwenHeadDim;
    submitted->query_snapshot_stride1 = kQwenHeadDim;
    submitted->query_snapshot_stride2 = 1;

    if (submitted->scatter_from_separate_kv && submitted->scatter_count > 0) {
      const auto* key_src =
          static_cast<const uint16_t*>(submitted->scatter_key_ptr);
      const auto* value_src =
          static_cast<const uint16_t*>(submitted->scatter_value_ptr);
      const int64_t kv_elems = static_cast<int64_t>(submitted->scatter_count) *
                               kQwenNumKVHeads * kQwenHeadDim;
      submitted->scatter_key_snapshot.assign(key_src, key_src + kv_elems);
      submitted->scatter_value_snapshot.assign(value_src, value_src + kv_elems);
    }
  }
  const bool wait_kernel =
      sync_task->wait_kernel_sync_installed.load(std::memory_order_acquire);
  if (wait_kernel) {
    submitted->seq =
        sync_task->next_seq.fetch_add(1, std::memory_order_relaxed) + 1u;
  }
  self->task_queue_->enqueue(
      [self, submitted] { self->RunTaskOnWorker(std::move(submitted)); });
  if (wait_kernel) {
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile uint32_t*>(sync_task->host_req_slot) = submitted->seq;
  }
}

void CotsSuffixAttentionInfer::SyncCallback(void* user_data) {
  NvtxScope nvtx_scope("cots:suffix_sync_cb_wait");
  SyncArgs* args = static_cast<SyncArgs*>(user_data);
  CotsSuffixAttentionInfer* self =
      static_cast<CotsSuffixAttentionInfer*>(args->infer);
  if (cots_suffix_diag::counters_enabled()) {
    const int64_t t0 = now_ns();
    self->task_queue_->sync(args->allow_n_pending);
    const int64_t t1 = now_ns();
    self->legacy_sync_cb_count_.fetch_add(1, std::memory_order_relaxed);
    self->legacy_sync_cb_wait_total_ns_.fetch_add(t1 - t0,
                                                  std::memory_order_relaxed);
  } else {
    self->task_queue_->sync(args->allow_n_pending);
  }
}

void CotsSuffixAttentionInfer::RunTaskOnWorker(
    std::shared_ptr<SubmittedTask> task) {
  NvtxScope nvtx_scope("cots:suffix_worker");
  struct DonePublisher {
    std::shared_ptr<SubmittedTask> task;
    ~DonePublisher() {
      SuffixAttentionTask* sync_task = task->sync_task;
      if (task->seq != 0 && sync_task->wait_kernel_sync_installed.load(
                                std::memory_order_acquire)) {
        std::atomic_thread_fence(std::memory_order_release);
        *static_cast<volatile uint32_t*>(sync_task->host_done_slot) = task->seq;
      }
    }
  } done{task};

  const bool diag = cots_suffix_diag::counters_enabled();
  const int64_t worker_t0 = diag ? now_ns() : 0;
  if (diag) {
    const int64_t enq = task->enqueue_time_ns;
    if (enq > 0) {
      worker_queue_wait_total_ns_.fetch_add(worker_t0 - enq,
                                            std::memory_order_relaxed);
    }
  }
  struct WorkerMetricsPublisher {
    CotsSuffixAttentionInfer* self;
    bool diag;
    int64_t worker_t0;
    ~WorkerMetricsPublisher() {
      if (!diag) return;
      self->worker_busy_total_ns_.fetch_add(now_ns() - worker_t0,
                                            std::memory_order_relaxed);
      self->worker_run_count_.fetch_add(1, std::memory_order_relaxed);
    }
  } metrics_done{this, diag, worker_t0};

  try {
    c10::InferenceMode g;
    const int32_t task_capacity = task->task_capacity;
    TORCH_CHECK(task_capacity > 0,
                "suffix attention task capacity must be > 0");
    TORCH_CHECK(task_capacity <= task->query_capacity,
                "suffix attention task capacity exceeds query capacity");

    const int32_t override_n =
        task->use_runtime_count_snapshot
            ? task->runtime_num_tokens
            : runtime_num_tokens_.load(std::memory_order_acquire);
    const int32_t batch =
        override_n >= 0 ? std::min(override_n, task_capacity) : task_capacity;
    TORCH_CHECK(batch >= 0, "suffix attention task batch must be >= 0");

    const int32_t override_scatter =
        task->use_runtime_count_snapshot
            ? task->runtime_scatter_count
            : runtime_scatter_count_.load(std::memory_order_acquire);
    int32_t scatter_count = task->scatter_count;
    if (override_scatter >= 0) {
      TORCH_CHECK(override_scatter <= task->scatter_count,
                  "runtime suffix scatter count exceeds captured capacity: ",
                  override_scatter, " > ", task->scatter_count);
      scatter_count = override_scatter;
    }
    if (diag) {
      worker_capacity_rows_.fetch_add(task_capacity, std::memory_order_relaxed);
      worker_live_rows_.fetch_add(batch, std::memory_order_relaxed);
      worker_padded_rows_.fetch_add(task_capacity - batch,
                                    std::memory_order_relaxed);
      worker_scatter_rows_.fetch_add(scatter_count, std::memory_order_relaxed);
      if (batch == 0) {
        worker_zero_live_count_.fetch_add(1, std::memory_order_relaxed);
      }
    }

    auto output_all = Bf16View(task->output_ptr,
                               {task_capacity, kQwenNumQHeads, kQwenHeadDim});
    auto output_lse_all =
        FloatView(task->output_lse_ptr, {kQwenNumQHeads, task_capacity});
    if (batch == 0) {
      NvtxScope zero_live_scope("cots:suffix_zero_live");
      output_all.zero_();
      output_lse_all.fill_(-std::numeric_limits<float>::infinity());
      return;
    }

    void* query_ptr =
        task->qkv_snapshot.empty()
            ? (task->query_snapshot.empty()
                   ? task->query_ptr
                   : static_cast<void*>(task->query_snapshot.data()))
            : static_cast<void*>(task->qkv_snapshot.data());
    const int64_t query_stride0 =
        (task->qkv_snapshot.empty() && task->query_snapshot.empty())
            ? task->query_stride0
            : task->query_snapshot_stride0;
    const int64_t query_stride1 =
        (task->qkv_snapshot.empty() && task->query_snapshot.empty())
            ? task->query_stride1
            : task->query_snapshot_stride1;
    const int64_t query_stride2 =
        (task->qkv_snapshot.empty() && task->query_snapshot.empty())
            ? task->query_stride2
            : task->query_snapshot_stride2;
    auto query =
        Bf16StridedView(query_ptr, {batch, kQwenNumQHeads, kQwenHeadDim},
                        {query_stride0, query_stride1, query_stride2});
    auto key_cache =
        Bf16View(task->key_cache_ptr, {task->num_cpu_blocks, kQwenNumKVHeads,
                                       task->block_size, kQwenHeadDim});
    auto value_cache =
        Bf16View(task->value_cache_ptr, {task->num_cpu_blocks, kQwenNumKVHeads,
                                         task->block_size, kQwenHeadDim});
    void* block_table_ptr =
        task->block_table_snapshot.empty()
            ? task->block_table_ptr
            : static_cast<void*>(task->block_table_snapshot.data());
    void* seq_lens_ptr =
        task->seq_lens_snapshot.empty()
            ? task->seq_lens_ptr
            : static_cast<void*>(task->seq_lens_snapshot.data());
    auto block_table =
        IntView(block_table_ptr, {batch, task->max_suffix_blocks});
    auto seq_lens = IntView(seq_lens_ptr, {batch});
    auto output = output_all.narrow(0, 0, batch);
    auto output_lse = output_lse_all.narrow(1, 0, batch);

    if ((task->scatter_from_qkv || task->scatter_from_separate_kv) &&
        scatter_count > 0) {
      TORCH_CHECK(scatter_count <= batch,
                  "suffix scatter_count exceeds task batch");
      void* scatter_block_ids_ptr =
          task->scatter_block_ids_snapshot.empty()
              ? task->scatter_block_ids_ptr
              : static_cast<void*>(task->scatter_block_ids_snapshot.data());
      void* scatter_block_offsets_ptr =
          task->scatter_block_offsets_snapshot.empty()
              ? task->scatter_block_offsets_ptr
              : static_cast<void*>(task->scatter_block_offsets_snapshot.data());
      auto scatter_block_ids = LongView(scatter_block_ids_ptr, {scatter_count});
      auto scatter_block_offsets =
          LongView(scatter_block_offsets_ptr, {scatter_count});
      NvtxScope scatter_scope("cots:suffix_scatter");
      if (task->scatter_from_qkv) {
        auto* qkv_ptr = static_cast<uint16_t*>(query_ptr);
        auto key =
            Bf16StridedView(qkv_ptr + kQwenNumQHeads * query_stride1,
                            {scatter_count, kQwenNumKVHeads, kQwenHeadDim},
                            {query_stride0, query_stride1, query_stride2});
        auto value = Bf16StridedView(
            qkv_ptr + (kQwenNumQHeads + kQwenNumKVHeads) * query_stride1,
            {scatter_count, kQwenNumKVHeads, kQwenHeadDim},
            {query_stride0, query_stride1, query_stride2});
        qwen_bf16_scatter_suffix_kv_at(key, value, scatter_block_ids,
                                       scatter_block_offsets, key_cache,
                                       value_cache);
      } else {
        TORCH_CHECK(task->scatter_key_ptr != nullptr,
                    "suffix separate scatter key ptr is null");
        TORCH_CHECK(task->scatter_value_ptr != nullptr,
                    "suffix separate scatter value ptr is null");
        void* scatter_key_ptr =
            task->scatter_key_snapshot.empty()
                ? task->scatter_key_ptr
                : static_cast<void*>(task->scatter_key_snapshot.data());
        void* scatter_value_ptr =
            task->scatter_value_snapshot.empty()
                ? task->scatter_value_ptr
                : static_cast<void*>(task->scatter_value_snapshot.data());
        auto key = Bf16View(scatter_key_ptr,
                            {scatter_count, kQwenNumKVHeads, kQwenHeadDim});
        auto value = Bf16View(scatter_value_ptr,
                              {scatter_count, kQwenNumKVHeads, kQwenHeadDim});
        qwen_bf16_scatter_suffix_kv_at(key, value, scatter_block_ids,
                                       scatter_block_offsets, key_cache,
                                       value_cache);
      }
    }

    {
      NvtxScope attention_scope("cots:suffix_attention");
      qwen_bf16_suffix_attention_at(query, key_cache, value_cache, block_table,
                                    seq_lens, task->scale, output, output_lse);
    }

    if (batch < task_capacity) {
      output_all.narrow(0, batch, task_capacity - batch).zero_();
      output_lse_all.narrow(1, batch, task_capacity - batch)
          .fill_(-std::numeric_limits<float>::infinity());
    }
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = std::string("[cots suffix worker] ") + e.what();
    has_error_.store(true, std::memory_order_release);
  } catch (...) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = "[cots suffix worker] unknown exception";
    has_error_.store(true, std::memory_order_release);
  }
}

std::vector<std::pair<std::string, int64_t>>
CotsSuffixAttentionInfer::get_counters() const {
  auto load = [](const std::atomic<int64_t>& a) {
    return a.load(std::memory_order_relaxed);
  };
  std::vector<std::pair<std::string, int64_t>> out;
  out.reserve(24);
  out.emplace_back("suffix_populate_count", load(populate_count_));
  out.emplace_back("suffix_submit_count", load(submit_count_));
  out.emplace_back("suffix_dispatch_cb_count", load(dispatch_cb_count_));
  out.emplace_back("suffix_legacy_sync_cb_count", load(legacy_sync_cb_count_));
  out.emplace_back("suffix_legacy_sync_cb_wait_total_ns",
                   load(legacy_sync_cb_wait_total_ns_));
  out.emplace_back("suffix_wait_kernel_launch_count",
                   load(wait_kernel_launch_count_));
  out.emplace_back("suffix_worker_run_count", load(worker_run_count_));
  out.emplace_back("suffix_worker_busy_total_ns", load(worker_busy_total_ns_));
  out.emplace_back("suffix_worker_queue_wait_total_ns",
                   load(worker_queue_wait_total_ns_));
  out.emplace_back("suffix_worker_capacity_rows", load(worker_capacity_rows_));
  out.emplace_back("suffix_worker_live_rows", load(worker_live_rows_));
  out.emplace_back("suffix_worker_padded_rows", load(worker_padded_rows_));
  out.emplace_back("suffix_worker_scatter_rows", load(worker_scatter_rows_));
  out.emplace_back("suffix_worker_zero_live_count",
                   load(worker_zero_live_count_));
  out.emplace_back("suffix_wait_kernel_immediate_resume_count",
                   wait_kernel_immediate_resume_host_
                       ? *wait_kernel_immediate_resume_host_
                       : 0);
  out.emplace_back(
      "suffix_wait_kernel_lagging_wait_count",
      wait_kernel_lagging_wait_host_ ? *wait_kernel_lagging_wait_host_ : 0);
  out.emplace_back(
      "suffix_wait_kernel_spin_iters_total",
      wait_kernel_spin_iters_host_ ? *wait_kernel_spin_iters_host_ : 0);
  return out;
}

void CotsSuffixAttentionInfer::reset_counters() {
  populate_count_.store(0, std::memory_order_relaxed);
  submit_count_.store(0, std::memory_order_relaxed);
  dispatch_cb_count_.store(0, std::memory_order_relaxed);
  legacy_sync_cb_count_.store(0, std::memory_order_relaxed);
  legacy_sync_cb_wait_total_ns_.store(0, std::memory_order_relaxed);
  wait_kernel_launch_count_.store(0, std::memory_order_relaxed);
  worker_run_count_.store(0, std::memory_order_relaxed);
  worker_busy_total_ns_.store(0, std::memory_order_relaxed);
  worker_queue_wait_total_ns_.store(0, std::memory_order_relaxed);
  worker_capacity_rows_.store(0, std::memory_order_relaxed);
  worker_live_rows_.store(0, std::memory_order_relaxed);
  worker_padded_rows_.store(0, std::memory_order_relaxed);
  worker_scatter_rows_.store(0, std::memory_order_relaxed);
  worker_zero_live_count_.store(0, std::memory_order_relaxed);
  if (wait_kernel_immediate_resume_host_)
    *wait_kernel_immediate_resume_host_ = 0;
  if (wait_kernel_lagging_wait_host_) *wait_kernel_lagging_wait_host_ = 0;
  if (wait_kernel_spin_iters_host_) *wait_kernel_spin_iters_host_ = 0;
}

}  // namespace cots
}  // namespace vllm
