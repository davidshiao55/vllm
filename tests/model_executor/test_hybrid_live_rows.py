# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

hybrid_c = pytest.importorskip("vllm._hybrid_C")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Hybrid live-row tests require CUDA"
)


class QkvFixture:
    def __init__(self, bucket: int = 64, in_dim: int = 32, out_dim: int = 17):
        self.bucket = bucket
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.runner = hybrid_c.HybridWeightTaskRunner()
        self.x_pinned = torch.empty(
            bucket, in_dim, dtype=torch.bfloat16, pin_memory=True
        )
        self.y_pinned = torch.empty(
            bucket, out_dim, dtype=torch.bfloat16, pin_memory=True
        )
        self.weight = (torch.randn(out_dim, in_dim, dtype=torch.float32) * 0.125).to(
            torch.bfloat16
        )
        self.runner.install(n_slabs=1, max_num_tokens=bucket)
        self.runner.populate_slab_qkv(
            task_id=0,
            n_threads=1,
            bucket_capacity_tokens=bucket,
            x_pinned_ptr=self.x_pinned.data_ptr(),
            in_dim=in_dim,
            y_pinned_ptr=self.y_pinned.data_ptr(),
            cpu_out_dim=out_dim,
            w_cpu_ptr=self.weight.data_ptr(),
            w_cpu_rows=out_dim,
        )

    def submit(self, x_gpu: torch.Tensor, transfer_rows: int) -> None:
        stream = torch.cuda.current_stream()
        self.runner.submit_on_stream(
            task_id=0,
            num_tokens=transfer_rows,
            x_gpu_ptr=x_gpu.data_ptr(),
            x_cols=x_gpu.shape[1],
            x_stride0=x_gpu.stride(0),
            x_stride1=x_gpu.stride(1),
            cuda_stream=stream.cuda_stream,
        )

    def publish(self, live_rows: int) -> None:
        stream = torch.cuda.current_stream()
        self.runner.publish_live_num_tokens_on_stream(live_rows, stream.cuda_stream)

    def sync_on_stream(self) -> None:
        stream = torch.cuda.current_stream()
        self.runner.sync_on_stream(stream.cuda_stream)

    def close(self) -> None:
        torch.accelerator.synchronize()
        self.runner.sync_blocking()


@pytest.fixture
def qkv_fixture():
    torch.manual_seed(20260817)
    fixture = QkvFixture()
    try:
        yield fixture
    finally:
        fixture.close()


@pytest.mark.parametrize("earlier_rows,later_rows", [(64, 8), (48, 8)])
def test_later_publication_cannot_shrink_queued_task(
    qkv_fixture: QkvFixture, earlier_rows: int, later_rows: int
) -> None:
    """Reproduce the prefill->decode ordering without a host-side fence.

    The later publication is intentionally issued before the CUDA stream has
    reached the earlier dispatch callback. A host-global live count makes the
    earlier task compute only ``later_rows``; stream ordering plus a task-owned
    snapshot must preserve all ``earlier_rows``.
    """
    x = (torch.randn(qkv_fixture.bucket, qkv_fixture.in_dim) * 0.125).to(
        device="cuda", dtype=torch.bfloat16
    )
    sentinel = -123.0
    qkv_fixture.y_pinned.fill_(sentinel)

    qkv_fixture.publish(earlier_rows)
    qkv_fixture.submit(x, transfer_rows=earlier_rows)
    qkv_fixture.sync_on_stream()
    # No synchronize here: the host is allowed to publish the next forward
    # while the previous CUDA callback / CPU task is still pending.
    qkv_fixture.publish(later_rows)
    torch.accelerator.synchronize()
    qkv_fixture.runner.sync_blocking()

    expected = (x[:earlier_rows].cpu().float() @ qkv_fixture.weight.float().t()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        qkv_fixture.y_pinned[:earlier_rows], expected, rtol=0.04, atol=0.04
    )
    if earlier_rows < qkv_fixture.bucket:
        assert torch.all(
            qkv_fixture.y_pinned[earlier_rows:]
            == torch.tensor(sentinel, dtype=torch.bfloat16)
        )


def test_cuda_graph_replay_uses_stream_ordered_live_rows(
    qkv_fixture: QkvFixture,
) -> None:
    """Live rows remain dynamic when native submission is graph-captured."""
    x = (torch.randn(qkv_fixture.bucket, qkv_fixture.in_dim) * 0.125).to(
        device="cuda", dtype=torch.bfloat16
    )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qkv_fixture.submit(x, transfer_rows=qkv_fixture.bucket)
        qkv_fixture.sync_on_stream()

    sentinel = -321.0
    qkv_fixture.y_pinned.fill_(sentinel)
    live_rows = 48
    qkv_fixture.publish(live_rows)
    graph.replay()
    # Queue the next forward's smaller publication immediately. It must remain
    # behind this replay's captured dispatch callback.
    qkv_fixture.publish(8)
    torch.accelerator.synchronize()
    qkv_fixture.runner.sync_blocking()

    expected = (x[:live_rows].cpu().float() @ qkv_fixture.weight.float().t()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        qkv_fixture.y_pinned[:live_rows], expected, rtol=0.04, atol=0.04
    )
    assert torch.all(
        qkv_fixture.y_pinned[live_rows:] == torch.tensor(sentinel, dtype=torch.bfloat16)
    )
