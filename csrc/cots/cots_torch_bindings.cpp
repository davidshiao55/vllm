// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// pybind11 bindings for the COTS native task runners. Exposes the
// weight and suffix-attention task-runner classes; no torch.ops.* registration
// in C++ — the torch.ops.vllm.cots_* ops are registered Python-side in
// vllm/model_executor/offloader/cots_ops.py via direct_register_custom_op.

#include <torch/extension.h>

#include "cots_weight_task_runner.h"
#include "cots_suffix_attention_task_runner.h"

namespace py = pybind11;
using vllm::cots::CotsSuffixAttentionTaskRunner;
using vllm::cots::CotsWeightTaskRunner;

namespace vllm::cots {
void gqa_bf16_suffix_attention_at(const at::Tensor& query,
                                  const at::Tensor& key_cache,
                                  const at::Tensor& value_cache,
                                  const at::Tensor& block_table,
                                  const at::Tensor& seq_lens, double scale,
                                  at::Tensor& output, at::Tensor& output_lse);
void gqa_bf16_scatter_suffix_kv_at(const at::Tensor& key,
                                   const at::Tensor& value,
                                   const at::Tensor& block_ids,
                                   const at::Tensor& block_offsets,
                                   at::Tensor& key_cache,
                                   at::Tensor& value_cache);
}  // namespace vllm::cots

PYBIND11_MODULE(_cots_C, m) {
  m.doc() = "COTS native weight/suffix task runners (vllm/csrc/cots/).";

  m.def("gqa_bf16_suffix_attention", &vllm::cots::gqa_bf16_suffix_attention_at,
        py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
        py::arg("block_table"), py::arg("seq_lens"), py::arg("scale"),
        py::arg("output"), py::arg("output_lse"));
  m.def("gqa_bf16_scatter_suffix_kv",
        &vllm::cots::gqa_bf16_scatter_suffix_kv_at, py::arg("key"),
        py::arg("value"), py::arg("block_ids"), py::arg("block_offsets"),
        py::arg("key_cache"), py::arg("value_cache"));
  py::class_<CotsSuffixAttentionTaskRunner>(m, "CotsSuffixAttentionTaskRunner")
      .def(py::init<>())
      .def("install", &CotsSuffixAttentionTaskRunner::install,
           py::arg("n_tasks"))
      .def("populate_task", &CotsSuffixAttentionTaskRunner::populate_task,
           py::arg("task_id"), py::arg("query_ptr"), py::arg("query_capacity"),
           py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
           py::arg("query_stride0"), py::arg("query_stride1"),
           py::arg("query_stride2"), py::arg("key_cache_ptr"),
           py::arg("num_cpu_blocks"), py::arg("block_size"),
           py::arg("value_cache_ptr"), py::arg("block_table_ptr"),
           py::arg("max_suffix_blocks"), py::arg("seq_lens_ptr"),
           py::arg("output_ptr"), py::arg("output_lse_ptr"),
           py::arg("scatter_block_ids_ptr"),
           py::arg("scatter_block_offsets_ptr"), py::arg("scatter_key_ptr"),
           py::arg("scatter_value_ptr"), py::arg("scatter_count"),
           py::arg("scatter_from_qkv"), py::arg("scatter_from_separate_kv"),
           py::arg("snapshot_inputs"), py::arg("scale"))
      .def("submit_prepared_on_stream",
           &CotsSuffixAttentionTaskRunner::submit_prepared_on_stream,
           py::arg("task_id"), py::arg("cuda_stream"))
      .def("set_runtime_counts",
           &CotsSuffixAttentionTaskRunner::set_runtime_counts,
           py::arg("num_tokens"), py::arg("scatter_count"))
      .def("sync_on_stream", &CotsSuffixAttentionTaskRunner::sync_on_stream,
           py::arg("cuda_stream"))
      .def("sync_or_wait_on_stream",
           &CotsSuffixAttentionTaskRunner::sync_or_wait_on_stream,
           py::arg("task_id"), py::arg("cuda_stream"))
      .def("install_wait_kernel_sync_for_task",
           &CotsSuffixAttentionTaskRunner::install_wait_kernel_sync_for_task,
           py::arg("task_id"))
      .def("wait_kernel_sync_installed_for_task",
           &CotsSuffixAttentionTaskRunner::wait_kernel_sync_installed_for_task,
           py::arg("task_id"))
      .def("wait_kernel_sync_on_stream",
           &CotsSuffixAttentionTaskRunner::wait_kernel_sync_on_stream,
           py::arg("task_id"), py::arg("cuda_stream"))
      .def("wait_kernel_get_req_slot",
           &CotsSuffixAttentionTaskRunner::wait_kernel_get_req_slot,
           py::arg("task_id"))
      .def("wait_kernel_get_done_slot",
           &CotsSuffixAttentionTaskRunner::wait_kernel_get_done_slot,
           py::arg("task_id"))
      .def("wait_kernel_set_req_slot",
           &CotsSuffixAttentionTaskRunner::wait_kernel_set_req_slot,
           py::arg("task_id"), py::arg("value"))
      .def("wait_kernel_set_done_slot",
           &CotsSuffixAttentionTaskRunner::wait_kernel_set_done_slot,
           py::arg("task_id"), py::arg("value"))
      .def("sync_blocking", &CotsSuffixAttentionTaskRunner::sync_blocking)
      .def("get_counters",
           [](const CotsSuffixAttentionTaskRunner& self) {
             py::dict out;
             for (auto& [name, value] : self.get_counters()) {
               out[py::str(name)] = value;
             }
             return out;
           })
      .def("reset_counters", &CotsSuffixAttentionTaskRunner::reset_counters)
      .def("has_error", &CotsSuffixAttentionTaskRunner::has_error)
      .def("take_error", &CotsSuffixAttentionTaskRunner::take_error)
      .def("check_error", &CotsSuffixAttentionTaskRunner::check_error);

  py::class_<CotsWeightTaskRunner>(m, "CotsWeightTaskRunner")
      .def(py::init<>())
      .def("install", &CotsWeightTaskRunner::install, py::arg("n_slabs"),
           py::arg("max_num_tokens"))
      .def("populate_slab_qkv", &CotsWeightTaskRunner::populate_slab_qkv,
           py::arg("task_id"), py::arg("n_threads"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"),
           py::arg("w_cpu_ptr"), py::arg("w_cpu_rows"))
      .def("populate_slab_mlp", &CotsWeightTaskRunner::populate_slab_mlp,
           py::arg("task_id"), py::arg("n_threads"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"),
           py::arg("w_gate_ptr"), py::arg("w_gate_rows"), py::arg("w_up_ptr"),
           py::arg("w_up_rows"), py::arg("w_down_ptr"), py::arg("w_down_rows"),
           py::arg("w_down_cols"))
      .def("populate_slab_dryrun", &CotsWeightTaskRunner::populate_slab_dryrun,
           py::arg("task_id"), py::arg("bucket_capacity_tokens"),
           py::arg("x_pinned_ptr"), py::arg("in_dim"), py::arg("y_pinned_ptr"),
           py::arg("cpu_out_dim"))
      .def("slab_bucket_capacity_tokens",
           &CotsWeightTaskRunner::slab_bucket_capacity_tokens,
           py::arg("task_id"))
      .def("slab_num_tokens", &CotsWeightTaskRunner::slab_num_tokens,
           py::arg("task_id"))
      .def("submit_on_stream", &CotsWeightTaskRunner::submit_on_stream,
           py::arg("task_id"), py::arg("num_tokens"), py::arg("x_gpu_ptr"),
           py::arg("x_cols"), py::arg("x_stride0"), py::arg("x_stride1"),
           py::arg("cuda_stream"))
      .def("sync_on_stream", &CotsWeightTaskRunner::sync_on_stream,
           py::arg("cuda_stream"))
      // §1c.29 commit 2 — unified entry. Always called by
      // `cots_sync_then_uva`'s impl; per-slab branch into
      // sync_on_stream or wait_kernel_sync_on_stream lives in C++ so the
      // Python side does not need to know which mechanism each
      // task uses.
      .def("sync_or_wait_on_stream",
           &CotsWeightTaskRunner::sync_or_wait_on_stream, py::arg("task_id"),
           py::arg("cuda_stream"))
      .def("wait_kernel_sync_installed_for_task",
           &CotsWeightTaskRunner::wait_kernel_sync_installed_for_task,
           py::arg("task_id"))
      .def("sync_blocking", &CotsWeightTaskRunner::sync_blocking)
      .def("set_worker_affinity", &CotsWeightTaskRunner::set_worker_affinity,
           py::arg("cpu_set"))
      // §1c.29 wait-kernel sync — install + wait launcher + test helpers.
      .def("install_wait_kernel_sync_for_task",
           &CotsWeightTaskRunner::install_wait_kernel_sync_for_task,
           py::arg("task_id"))
      .def("wait_kernel_sync_on_stream",
           &CotsWeightTaskRunner::wait_kernel_sync_on_stream,
           py::arg("task_id"), py::arg("cuda_stream"))
      .def("wait_kernel_get_req_slot",
           &CotsWeightTaskRunner::wait_kernel_get_req_slot, py::arg("task_id"))
      .def("wait_kernel_get_done_slot",
           &CotsWeightTaskRunner::wait_kernel_get_done_slot, py::arg("task_id"))
      .def("wait_kernel_set_req_slot",
           &CotsWeightTaskRunner::wait_kernel_set_req_slot, py::arg("task_id"),
           py::arg("value"))
      .def("wait_kernel_set_done_slot",
           &CotsWeightTaskRunner::wait_kernel_set_done_slot, py::arg("task_id"),
           py::arg("value"))
      .def("last_observed_num_threads",
           &CotsWeightTaskRunner::last_observed_num_threads)
      .def("has_error", &CotsWeightTaskRunner::has_error)
      .def("take_error", &CotsWeightTaskRunner::take_error)
      .def("check_error", &CotsWeightTaskRunner::check_error)
      .def("run_at_linear_inline", &CotsWeightTaskRunner::run_at_linear_inline,
           py::arg("x"), py::arg("w"), py::arg("y_out"))
      // Custom AVX2 BF16 GEMM kernel inline. Same signature shape as
      // run_at_linear_inline; w must be (K, N) row-major BF16 (the
      // transposed-storage layout that oneDNN doesn't fast-path on
      // AVX2).
      .def("run_bf16_gemm_transposed_inline",
           &CotsWeightTaskRunner::run_bf16_gemm_transposed_inline, py::arg("x"),
           py::arg("w"), py::arg("y_out"))
      // Natural (N, K) BF16 GEMM kernel.
      .def("run_bf16_gemm_natural_inline",
           &CotsWeightTaskRunner::run_bf16_gemm_natural_inline, py::arg("x"),
           py::arg("w"), py::arg("y_out"))
      .def("y_pinned_view", &CotsWeightTaskRunner::y_pinned_view,
           py::arg("task_id"), py::arg("num_tokens"))
      .def("set_live_num_tokens", &CotsWeightTaskRunner::set_live_num_tokens,
           py::arg("n"))
      .def("note_uva_request", &CotsWeightTaskRunner::note_uva_request,
           py::arg("num_tokens"), py::arg("cpu_out_dim"))
      .def("get_counters",
           [](const CotsWeightTaskRunner& self) {
             // §1c.21: return as a Python dict for ergonomic
             // dump-and-print at the bench harness level.
             py::dict out;
             for (auto& [name, value] : self.get_counters()) {
               out[py::str(name)] = value;
             }
             return out;
           })
      .def("reset_counters", &CotsWeightTaskRunner::reset_counters);
}
