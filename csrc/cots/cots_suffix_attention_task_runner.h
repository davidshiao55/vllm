// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 2 native runner for prepared CPU suffix attention tasks.
//
// This is the first graph-compatible substrate slice for hybrid KV. The
// current Python path still prepares the CPU query/KV metadata, but the suffix
// attention work itself is launched through cudaLaunchHostFunc so a later graph
// path can replay CPU suffix work instead of running it only at capture time.

#ifndef VLLM_COTS_SUFFIX_ATTENTION_TASK_RUNNER_H_
#define VLLM_COTS_SUFFIX_ATTENTION_TASK_RUNNER_H_

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "task_queue.h"

namespace vllm {
namespace cots {

struct alignas(64) SuffixAttentionTask {
  void* self = nullptr;

  // Phase 2 graph sync: same host-mapped req/done slot pattern as the
  // Phase 1 native runner. Dispatch publishes req=seq after queueing the CPU
  // task; the worker publishes done=seq after writing CPU outputs. A captured
  // CUDA wait kernel spins on done >= req instead of recording a sync
  // cudaLaunchHostFunc.
  void* host_req_slot{nullptr};
  void* dev_req_slot{nullptr};
  void* host_done_slot{nullptr};
  void* dev_done_slot{nullptr};
  std::atomic<uint32_t> next_seq{0};
  std::atomic<bool> wait_kernel_sync_installed{false};

  // Diagnostic-only timestamp written by DispatchCallback when
  // VLLM_COTS_SUFFIX_COUNTERS=1 or VLLM_COTS_DIAG=1.
  std::atomic<int64_t> enqueue_time_ns{0};

  std::atomic<int32_t> num_tokens{0};
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

  int32_t query_capacity = 0;
  int32_t num_q_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int64_t query_stride0 = 0;
  int64_t query_stride1 = 0;
  int64_t query_stride2 = 0;
  int32_t num_cpu_blocks = 0;
  int32_t block_size = 0;
  int32_t max_suffix_blocks = 0;
  int32_t scatter_count = 0;
  bool scatter_from_qkv = false;
  bool scatter_from_separate_kv = false;
  bool snapshot_inputs = true;
  double scale = 0.0;
};

class CotsSuffixAttentionTaskRunner {
 public:
  CotsSuffixAttentionTaskRunner();
  ~CotsSuffixAttentionTaskRunner();

  CotsSuffixAttentionTaskRunner(const CotsSuffixAttentionTaskRunner&) = delete;
  CotsSuffixAttentionTaskRunner& operator=(
      const CotsSuffixAttentionTaskRunner&) = delete;

  void install(int64_t n_tasks);

  void populate_task(
      int64_t task_id, uintptr_t query_ptr, int32_t query_capacity,
      int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
      int64_t query_stride0, int64_t query_stride1, int64_t query_stride2,
      uintptr_t key_cache_ptr, int32_t num_cpu_blocks, int32_t block_size,
      uintptr_t value_cache_ptr, uintptr_t block_table_ptr,
      int32_t max_suffix_blocks, uintptr_t seq_lens_ptr, uintptr_t output_ptr,
      uintptr_t output_lse_ptr, uintptr_t scatter_block_ids_ptr,
      uintptr_t scatter_block_offsets_ptr, uintptr_t scatter_key_ptr,
      uintptr_t scatter_value_ptr, int32_t scatter_count, bool scatter_from_qkv,
      bool scatter_from_separate_kv, bool snapshot_inputs, double scale);

  void submit_prepared_on_stream(int64_t task_id, uintptr_t cuda_stream);
  void set_runtime_counts(int32_t num_tokens, int32_t scatter_count);
  void sync_on_stream(uintptr_t cuda_stream);
  void sync_or_wait_on_stream(int64_t task_id, uintptr_t cuda_stream);
  void sync_blocking();

  void install_wait_kernel_sync_for_task(int64_t task_id);
  bool wait_kernel_sync_installed_for_task(int64_t task_id) const;
  void wait_kernel_sync_on_stream(int64_t task_id, uintptr_t cuda_stream);
  uint32_t wait_kernel_get_req_slot(int64_t task_id) const;
  uint32_t wait_kernel_get_done_slot(int64_t task_id) const;
  void wait_kernel_set_req_slot(int64_t task_id, uint32_t value);
  void wait_kernel_set_done_slot(int64_t task_id, uint32_t value);

  std::vector<std::pair<std::string, int64_t>> get_counters() const;
  void reset_counters();

  bool has_error() const { return has_error_.load(std::memory_order_acquire); }
  std::string take_error();
  void check_error();

 private:
  struct SyncArgs {
    void* runner = nullptr;
    size_t allow_n_pending = 0;
  };

  struct SubmittedTask;

  static void DispatchCallback(void* user_data);
  static void SyncCallback(void* user_data);

  void RunTaskOnWorker(std::shared_ptr<SubmittedTask> task);
  void wait_kernel_sync_on_stream_no_check(int64_t task_id,
                                           uintptr_t cuda_stream);

  std::unique_ptr<TaskQueue> task_queue_;
  std::unique_ptr<SuffixAttentionTask[]> tasks_;
  std::vector<std::unique_ptr<SubmittedTask>> graph_submitted_tasks_;
  int64_t task_count_ = 0;
  SyncArgs sync_args_{};

  // Live-row override for CUDA graph replay. Prepared tasks capture
  // bucket-sized CPU pointers and capacities; replay publishes the live decode
  // rows and live scatter rows out-of-graph so the worker skips padded
  // attention/scatter work. Sentinel num_tokens=-1 falls back to the captured
  // task capacity. A num_tokens value of 0 is a real graph replay state: skip
  // CPU suffix attention and write neutral output/LSE for the captured bucket.
  // Sentinel scatter_count=-1 falls back to the captured task scatter count.
  std::atomic<int32_t> runtime_num_tokens_{-1};
  std::atomic<int32_t> runtime_scatter_count_{-1};

  std::atomic<int64_t> populate_count_{0};
  std::atomic<int64_t> submit_count_{0};
  std::atomic<int64_t> submit_prepare_total_ns_{0};
  std::atomic<int64_t> submit_metadata_snapshot_total_ns_{0};
  std::atomic<int64_t> submit_launch_hostfunc_total_ns_{0};
  std::atomic<int64_t> dispatch_cb_count_{0};
  std::atomic<int64_t> dispatch_cb_total_ns_{0};
  std::atomic<int64_t> dispatch_cb_snapshot_total_ns_{0};
  std::atomic<int64_t> dispatch_cb_enqueue_total_ns_{0};
  std::atomic<int64_t> host_callback_sync_count_{0};
  std::atomic<int64_t> host_callback_sync_wait_total_ns_{0};
  std::atomic<int64_t> wait_kernel_launch_count_{0};
  std::atomic<int64_t> worker_run_count_{0};
  std::atomic<int64_t> worker_requested_num_threads_{0};
  std::atomic<int64_t> worker_observed_num_threads_{0};
  std::atomic<int64_t> worker_busy_total_ns_{0};
  std::atomic<int64_t> worker_queue_wait_total_ns_{0};
  std::atomic<int64_t> worker_scatter_total_ns_{0};
  std::atomic<int64_t> worker_attention_total_ns_{0};
  std::atomic<int64_t> worker_capacity_rows_{0};
  std::atomic<int64_t> worker_live_rows_{0};
  std::atomic<int64_t> worker_padded_rows_{0};
  std::atomic<int64_t> worker_scatter_rows_{0};
  std::atomic<int64_t> worker_zero_live_count_{0};

  uint32_t* wait_kernel_slots_host_ = nullptr;
  uint32_t* wait_kernel_slots_dev_ = nullptr;
  int64_t wait_kernel_slots_capacity_ = 0;

  int64_t* wait_kernel_spin_iters_host_ = nullptr;
  int64_t* wait_kernel_spin_iters_dev_ = nullptr;
  int64_t* wait_kernel_immediate_resume_host_ = nullptr;
  int64_t* wait_kernel_immediate_resume_dev_ = nullptr;
  int64_t* wait_kernel_lagging_wait_host_ = nullptr;
  int64_t* wait_kernel_lagging_wait_dev_ = nullptr;

  std::atomic<bool> has_error_{false};
  std::mutex error_mtx_;
  std::string last_error_msg_;

  int32_t worker_current_num_threads_ = 0;
};

}  // namespace cots
}  // namespace vllm

#endif  // VLLM_COTS_SUFFIX_ATTENTION_TASK_RUNNER_H_
