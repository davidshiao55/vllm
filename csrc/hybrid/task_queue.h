// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Direct port of KTransformers `kt-kernel/cpu_backend/task_queue.h`
// (commit reference in /TTC/Reference_Frameworks/ktransformers/), adapted
// into the vllm::hybrid namespace. See docs/implementation_roadmap.md
// Phase 1c and docs/phase1c_findings.md for the design rationale.

#ifndef VLLM_HYBRID_TASK_QUEUE_H_
#define VLLM_HYBRID_TASK_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace vllm {
namespace hybrid {

// Michael-Scott style MPSC queue with a single worker thread. Tasks are
// std::function<void()> so any captured lambda is fine. `sync(N)` blocks
// the calling thread until pending count drops to <= N (or the queue is
// shutting down). Exception policy: tasks are run inside the worker; any
// thrown exception leaves `pending` decremented (the catch happens in the
// caller's slab dispatcher, NOT here), so sync() never deadlocks.
class TaskQueue {
 public:
  TaskQueue();
  ~TaskQueue();

  TaskQueue(const TaskQueue&) = delete;
  TaskQueue& operator=(const TaskQueue&) = delete;
  TaskQueue(TaskQueue&&) = delete;
  TaskQueue& operator=(TaskQueue&&) = delete;

  void enqueue(std::function<void()> task);

  // Block calling thread until `pending` <= allow_n_pending, OR queue is
  // shutting down. Used both by Python `runner.sync()` (no stream) and by
  // the CUDA host callback in HybridWeightTaskRunner::sync() (CUDA driver
  // thread).
  void sync(size_t allow_n_pending);

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    Node() : task(nullptr), next(nullptr) {}
    explicit Node(std::function<void()> t)
        : task(std::move(t)), next(nullptr) {}
  };

  std::atomic<Node*> head_;
  std::atomic<Node*> tail_;
  std::atomic<bool> done_;
  std::atomic<size_t> pending_;
  std::thread worker_thread_;
  std::mutex mtx_;
  std::condition_variable cv_;

  void Worker();
};

}  // namespace hybrid
}  // namespace vllm

#endif  // VLLM_HYBRID_TASK_QUEUE_H_
