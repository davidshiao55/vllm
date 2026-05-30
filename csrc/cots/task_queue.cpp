// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Direct port of KTransformers `kt-kernel/cpu_backend/task_queue.cpp`,
// adapted into vllm::cots and renamed to plan-aligned identifiers.
// One worker thread, MPSC linked list with head/tail atomics + a guard
// mutex/condvar for sleep/wake. Phase 1c host-callback design relies on
// (a) `enqueue` being lock-free fast path on the producer side and
// (b) `sync(allow_n_pending)` blocking the CUDA driver thread on cv_
// rather than busy-waiting.

#include "task_queue.h"

#include <pthread.h>

#include <utility>

namespace vllm {
namespace cots {

TaskQueue::TaskQueue() : done_(false), pending_(0) {
  Node* dummy = new Node();
  head_.store(dummy, std::memory_order_relaxed);
  tail_.store(dummy, std::memory_order_relaxed);
  worker_thread_ = std::thread(&TaskQueue::Worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    done_.store(true, std::memory_order_release);
  }
  cv_.notify_all();
  if (worker_thread_.joinable()) worker_thread_.join();

  Node* node = head_.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending_.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(std::move(task));
  Node* prev = tail_.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  // Take the mutex briefly to avoid the lost-wake-up race against worker
  // sleeping on `cv_.wait`. (Same pattern as kt-kernel.)
  {
    std::lock_guard<std::mutex> lock(mtx_);
  }
  cv_.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [&] {
    return pending_.load(std::memory_order_acquire) <= allow_n_pending ||
           done_.load(std::memory_order_acquire);
  });
}

void TaskQueue::Worker() {
  // Visible in `top -H`, `htop`, and Nsight Systems traces.
  pthread_setname_np(pthread_self(), "cots-cpu-wkr");

  Node* curr = head_.load(std::memory_order_relaxed);
  while (!done_.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      if (next->task) {
        // Task body is responsible for its own try/catch; if it throws,
        // we still need to decrement pending and notify cv_ so that
        // sync() doesn't deadlock. The slab dispatcher in
        // cots_weight_task_runner.cpp wraps every task in try/catch and stores
        // the error into CotsWeightTaskRunner state instead of letting it
        // propagate here.
        next->task();
      }
      delete curr;
      curr = next;
      head_.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx_);
        pending_.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv_.notify_all();
    } else {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr ||
               done_.load(std::memory_order_acquire);
      });
    }
  }
}

}  // namespace cots
}  // namespace vllm
