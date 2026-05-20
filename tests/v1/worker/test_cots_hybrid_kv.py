# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

_cots_C = pytest.importorskip("vllm._cots_C")
if not hasattr(_cots_C, "CotsSuffixAttentionInfer"):
    pytest.skip(
        "CotsSuffixAttentionInfer is not built; rebuild vLLM",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.cots_hybrid_attention import (  # noqa: E402
    cots_hybrid_kv_cache_update,
)
from vllm.v1.worker.cots_hybrid_kv import CotsHybridKVStore  # noqa: E402


def _make_store(num_layers: int = 2, max_num_reqs: int = 4) -> CotsHybridKVStore:
    return CotsHybridKVStore(
        layer_names=[f"layer.{i}.attn" for i in range(num_layers)],
        split_blocks=2,
        cpu_pool_bytes=16 * 1024,
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        max_num_reqs=max_num_reqs,
        max_model_len=32,
        pin_memory=False,
    )


def _cpu_block_ids(*groups):
    return [None if group is None else (list(group),) for group in groups]


def test_cots_hybrid_store_builds_suffix_local_metadata() -> None:
    store = _make_store()
    seq_lens = torch.tensor([8, 9, 15], dtype=torch.int32)
    prompt_lens = torch.tensor([4, 4, 4], dtype=torch.int32)
    is_prefilling = torch.tensor([False, False, False])
    cpu_block_ids = _cpu_block_ids([], [5], [6, 7])

    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a", "b", "c"],
        seq_lens_cpu=seq_lens,
        prompt_lens_cpu=prompt_lens,
        is_prefilling_cpu=is_prefilling,
        max_query_len=1,
        num_actual_tokens=3,
        cpu_block_ids_by_req=cpu_block_ids,
    )

    assert metadata is not None
    assert metadata.split_blocks == 2
    assert metadata.cpu_seq_lens.tolist() == [0, 1, 7]
    assert metadata.cpu_block_table.shape[0] == 3
    assert metadata.cpu_block_table[:, :2].shape == (3, 2)
    assert metadata.cpu_block_table[:, :2].tolist() == [[0, 0], [5, 0], [6, 7]]

    # Building metadata for another layer reuses the scheduler-provided CPU
    # suffix block IDs while swapping to that layer's CPU cache tensors.
    metadata_l1 = store.build_decode_metadata(
        layer_name="layer.1.attn",
        req_ids=["a", "b", "c"],
        seq_lens_cpu=seq_lens,
        prompt_lens_cpu=prompt_lens,
        is_prefilling_cpu=is_prefilling,
        max_query_len=1,
        num_actual_tokens=3,
        cpu_block_ids_by_req=cpu_block_ids,
    )

    assert metadata_l1 is not None
    assert metadata_l1.cpu_block_table[:, :2].tolist() == [[0, 0], [5, 0], [6, 7]]
    assert metadata_l1.cpu_key_cache.data_ptr() != metadata.cpu_key_cache.data_ptr()


def test_cots_hybrid_store_stays_on_gpu_before_split_or_prefill() -> None:
    store = _make_store()
    prompt_lens = torch.tensor([4, 4], dtype=torch.int32)

    assert (
        store.build_decode_metadata(
            layer_name="layer.0.attn",
            req_ids=["a", "b"],
            seq_lens_cpu=torch.tensor([7, 8], dtype=torch.int32),
            prompt_lens_cpu=prompt_lens,
            is_prefilling_cpu=torch.tensor([False, False]),
            max_query_len=1,
            num_actual_tokens=2,
        )
        is None
    )

    assert (
        store.build_decode_metadata(
            layer_name="layer.0.attn",
            req_ids=["a", "b"],
            seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
            prompt_lens_cpu=prompt_lens,
            is_prefilling_cpu=torch.tensor([False, True]),
            max_query_len=1,
            num_actual_tokens=2,
        )
        is None
    )


def test_cots_hybrid_store_gpu_tail_before_planner_split() -> None:
    store = _make_store(num_layers=1)
    prompt_lens = torch.tensor([6, 6], dtype=torch.int32)
    is_prefilling = torch.tensor([False, False])

    # Prompt finished before the planner split. Generated tail tokens at
    # positions 6 and 7 are still GPU-owned and must not need CPU suffix KV.
    for position in (6, 7):
        metadata = store.build_decode_metadata(
            layer_name="layer.0.attn",
            req_ids=["a", "b"],
            seq_lens_cpu=torch.tensor([position + 1, position + 1], dtype=torch.int32),
            prompt_lens_cpu=prompt_lens,
            is_prefilling_cpu=is_prefilling,
            max_query_len=1,
            num_actual_tokens=2,
            req_indices_cpu=[0, 1],
            positions_cpu=[position, position],
        )

        assert metadata is None

    # The first token at the split starts CPU suffix-local position 0.
    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a", "b"],
        seq_lens_cpu=torch.tensor([9, 9], dtype=torch.int32),
        prompt_lens_cpu=prompt_lens,
        is_prefilling_cpu=is_prefilling,
        max_query_len=1,
        num_actual_tokens=2,
        cpu_block_ids_by_req=_cpu_block_ids([1], [2]),
        req_indices_cpu=[0, 1],
        positions_cpu=[8, 8],
    )

    assert metadata is not None
    assert metadata.cpu_seq_lens.tolist() == [1, 1]
    assert metadata.cpu_block_table[:, :1].tolist() == [[1], [2]]
    assert metadata.scatter_block_offsets is not None
    assert metadata.scatter_block_offsets.tolist() == [0, 0]


def test_cots_hybrid_store_supports_cpu_suffix_prefill_rows() -> None:
    store = _make_store(num_layers=1)

    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["prefill"],
        seq_lens_cpu=torch.tensor([12], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([12], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([True]),
        max_query_len=4,
        num_actual_tokens=4,
        cpu_block_ids_by_req=_cpu_block_ids([5]),
        req_indices_cpu=[0, 0, 0, 0],
        positions_cpu=[8, 9, 10, 11],
    )

    assert metadata is not None
    assert metadata.cpu_seq_lens.tolist() == [1, 2, 3, 4]
    assert metadata.cpu_block_table[:, :1].tolist() == [[5], [5], [5], [5]]
    assert metadata.scatter_block_ids is not None
    assert metadata.scatter_block_offsets is not None
    assert metadata.scatter_block_ids.tolist() == [5, 5, 5, 5]
    assert metadata.scatter_block_offsets.tolist() == [0, 1, 2, 3]

    key = (
        torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 2, 8).to(torch.bfloat16)
    )
    value = (key + 100).to(torch.bfloat16)
    cots_hybrid_kv_cache_update(key, value, metadata)

    for offset in range(4):
        torch.testing.assert_close(metadata.cpu_key_cache[5, :, offset, :], key[offset])
        torch.testing.assert_close(
            metadata.cpu_value_cache[5, :, offset, :], value[offset]
        )


def test_cots_hybrid_store_overflow_prefill_clears_live_count_override() -> None:
    store = _make_store(num_layers=1, max_num_reqs=2)
    live_counts: list[tuple[int, int]] = []

    class RecordingRunner:
        kind = "recording"

        def set_runtime_counts(self, num_tokens: int, scatter_count: int) -> None:
            live_counts.append((num_tokens, scatter_count))

    store._suffix_attention_runner = RecordingRunner()  # type: ignore[assignment]
    store._last_live_counts = None
    store.on_dispatch(
        SimpleNamespace(num_tokens_unpadded=5),
        positions_cpu=[8, 9, 10, 11, 12],
    )
    assert live_counts[-1] == (-1, -1)

    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["overflow"],
        seq_lens_cpu=torch.tensor([13], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([13], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([True]),
        max_query_len=5,
        num_actual_tokens=5,
        cpu_block_ids_by_req=_cpu_block_ids([5, 6]),
        req_indices_cpu=[0, 0, 0, 0, 0],
        positions_cpu=[8, 9, 10, 11, 12],
    )

    assert metadata is not None
    assert (
        metadata.suffix_attention_task_id
        == store._suffix_attention_overflow_task_ids["layer.0.attn"]
    )
    assert metadata.cpu_seq_lens.tolist() == [1, 2, 3, 4, 5]
    assert metadata.scatter_block_offsets is not None
    assert metadata.scatter_block_offsets.tolist() == [0, 1, 2, 3, 0]
    assert live_counts[-1] == (-1, -1)


def test_cots_hybrid_store_splits_chunks_that_cross_split() -> None:
    store = _make_store(num_layers=1)

    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["mixed"],
        seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([10], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([True]),
        max_query_len=4,
        num_actual_tokens=4,
        cpu_block_ids_by_req=_cpu_block_ids([5]),
        req_indices_cpu=[0, 0, 0, 0],
        positions_cpu=[6, 7, 8, 9],
    )

    assert metadata is not None
    assert metadata.cpu_seq_lens.tolist() == [1, 2]
    assert metadata.scatter_source_indices is not None
    assert metadata.scatter_source_indices.tolist() == [2, 3]
    assert metadata.prefix_source_indices is not None
    assert metadata.prefix_source_indices.tolist() == [0, 1]
    assert metadata.prefix_seq_lens_cpu is not None
    assert metadata.prefix_seq_lens_cpu.tolist() == [7, 8]
    assert metadata.scatter_block_offsets is not None
    assert metadata.scatter_block_offsets.tolist() == [0, 1]


def test_cots_hybrid_store_free_request_clears_cached_metadata() -> None:
    store = _make_store()
    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a"],
        seq_lens_cpu=torch.tensor([17], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False]),
        max_query_len=1,
        num_actual_tokens=1,
        cpu_block_ids_by_req=_cpu_block_ids([3, 4, 5]),
    )

    assert metadata is not None
    assert store._common_cache_key is not None
    store.free_request("a")
    assert store._common_cache_key is None


def test_cots_hybrid_store_refreshes_cache_when_cpu_block_ids_change() -> None:
    store = _make_store(num_layers=1)
    kwargs = dict(
        layer_name="layer.0.attn",
        req_ids=["a"],
        seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False]),
        max_query_len=1,
        num_actual_tokens=1,
    )

    metadata_a = store.build_decode_metadata(
        **kwargs,
        cpu_block_ids_by_req=_cpu_block_ids([5]),
    )
    assert metadata_a is not None
    assert metadata_a.cpu_block_table[:, :1].tolist() == [[5]]

    metadata_b = store.build_decode_metadata(
        **kwargs,
        cpu_block_ids_by_req=_cpu_block_ids([6]),
    )
    assert metadata_b is not None
    assert metadata_b.cpu_block_table[:, :1].tolist() == [[6]]


def test_cots_hybrid_store_metadata_carries_suffix_runner() -> None:
    store = _make_store(num_layers=2)

    metadata_l0 = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a"],
        seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False]),
        max_query_len=1,
        num_actual_tokens=1,
        cpu_block_ids_by_req=_cpu_block_ids([5]),
    )
    metadata_l1 = store.build_decode_metadata(
        layer_name="layer.1.attn",
        req_ids=["a"],
        seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False]),
        max_query_len=1,
        num_actual_tokens=1,
        cpu_block_ids_by_req=_cpu_block_ids([5]),
    )

    assert metadata_l0 is not None
    assert metadata_l1 is not None
    assert metadata_l0.suffix_attention_runner is not None
    assert metadata_l0.suffix_attention_runner.kind == "native_prepared"
    # Task IDs are partitioned by layer, staging slot, then captured batch
    # size. This gives eager decode Phase-1-style leased staging slots while
    # preserving fixed prepared task IDs for each bucket.
    assert (
        metadata_l0.suffix_attention_task_id
        == store._suffix_attention_task_base_ids["layer.0.attn"]
    )
    assert (
        metadata_l1.suffix_attention_task_id
        == store._suffix_attention_task_base_ids["layer.1.attn"]
    )
    assert metadata_l0.suffix_attention_runner is metadata_l1.suffix_attention_runner

    metadata_l0_b2 = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a", "b"],
        seq_lens_cpu=torch.tensor([9, 9], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, False]),
        max_query_len=1,
        num_actual_tokens=2,
        cpu_block_ids_by_req=_cpu_block_ids([5], [6]),
    )
    assert metadata_l0_b2 is not None
    assert metadata_l0_b2.suffix_attention_task_id == (
        store._suffix_attention_task_base_ids["layer.0.attn"] + store._max_num_reqs + 1
    )


def test_cots_hybrid_kv_cache_update_writes_latest_suffix_token() -> None:
    store = _make_store(num_layers=1)
    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a", "b"],
        seq_lens_cpu=torch.tensor([8, 10], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, False]),
        max_query_len=1,
        num_actual_tokens=2,
        cpu_block_ids_by_req=_cpu_block_ids([], [7]),
    )
    assert metadata is not None

    key = (
        torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8).to(torch.bfloat16)
    )
    value = (key + 100).to(torch.bfloat16)
    cots_hybrid_kv_cache_update(key, value, metadata)

    # Request "a" is exactly at the split, so it has no CPU suffix token yet.
    b_block = int(metadata.cpu_block_table[1, 0].item())
    torch.testing.assert_close(metadata.cpu_key_cache[b_block, :, 1, :], key[1])
    torch.testing.assert_close(metadata.cpu_value_cache[b_block, :, 1, :], value[1])


def test_cots_hybrid_store_extracts_suffix_rows_from_mixed_batch() -> None:
    store = _make_store(num_layers=1)

    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["suffix", "prefill"],
        seq_lens_cpu=torch.tensor([9, 4], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, True]),
        max_query_len=4,
        num_actual_tokens=5,
        cpu_block_ids_by_req=_cpu_block_ids([8], None),
        req_indices_cpu=[0, 1, 1, 1, 1],
        positions_cpu=[8, 0, 1, 2, 3],
    )

    assert metadata is not None
    assert metadata.cpu_seq_lens.tolist() == [1]
    assert metadata.cpu_block_table[:, :1].tolist() == [[8]]
    assert metadata.scatter_source_indices is not None
    assert metadata.scatter_source_indices.tolist() == [0]

    key = (
        torch.arange(5 * 2 * 8, dtype=torch.float32).reshape(5, 2, 8).to(torch.bfloat16)
    )
    value = (key + 100).to(torch.bfloat16)
    cots_hybrid_kv_cache_update(key, value, metadata)

    block = int(metadata.cpu_block_table[0, 0].item())
    torch.testing.assert_close(metadata.cpu_key_cache[block, :, 0, :], key[0])
    torch.testing.assert_close(metadata.cpu_value_cache[block, :, 0, :], value[0])


def test_cots_hybrid_store_uses_scheduler_cpu_block_ids() -> None:
    store = _make_store(num_layers=1)

    metadata = store.build_decode_metadata(
        layer_name="layer.0.attn",
        req_ids=["a", "b"],
        seq_lens_cpu=torch.tensor([9, 10], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, False]),
        max_query_len=1,
        num_actual_tokens=2,
        cpu_block_ids_by_req=[([5],), ([6],)],
    )

    assert metadata is not None
    assert metadata.cpu_block_table[:, :1].tolist() == [[5], [6]]


def test_cots_hybrid_store_rejects_missing_scheduler_cpu_blocks() -> None:
    store = _make_store(num_layers=1)

    with pytest.raises(RuntimeError, match="scheduler/worker CPU block mismatch"):
        store.build_decode_metadata(
            layer_name="layer.0.attn",
            req_ids=["a"],
            seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
            prompt_lens_cpu=torch.tensor([4], dtype=torch.int32),
            is_prefilling_cpu=torch.tensor([False]),
            max_query_len=1,
            num_actual_tokens=1,
            cpu_block_ids_by_req=[None],
        )


@pytest.mark.parametrize("block_id", [-1, 64])
def test_cots_hybrid_store_rejects_out_of_range_scheduler_cpu_blocks(
    block_id: int,
) -> None:
    store = _make_store(num_layers=1)

    with pytest.raises(RuntimeError, match="out-of-range CPU suffix block id"):
        store.build_decode_metadata(
            layer_name="layer.0.attn",
            req_ids=["a"],
            seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
            prompt_lens_cpu=torch.tensor([4], dtype=torch.int32),
            is_prefilling_cpu=torch.tensor([False]),
            max_query_len=1,
            num_actual_tokens=1,
            cpu_block_ids_by_req=_cpu_block_ids([block_id]),
        )


def test_cots_hybrid_store_uses_stable_scatter_metadata_buffers() -> None:
    store = _make_store(num_layers=1)
    common = dict(
        layer_name="layer.0.attn",
        req_ids=["a", "b"],
        prompt_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, False]),
        max_query_len=1,
        num_actual_tokens=2,
    )

    metadata_a = store.build_decode_metadata(
        **common,
        seq_lens_cpu=torch.tensor([9, 10], dtype=torch.int32),
        cpu_block_ids_by_req=_cpu_block_ids([5], [6]),
    )
    assert metadata_a is not None
    assert metadata_a.scatter_block_ids is not None
    assert metadata_a.scatter_block_offsets is not None
    ids_ptr = metadata_a.scatter_block_ids.data_ptr()
    offsets_ptr = metadata_a.scatter_block_offsets.data_ptr()
    assert metadata_a.scatter_block_ids.tolist() == [5, 6]
    assert metadata_a.scatter_block_offsets.tolist() == [0, 1]

    metadata_b = store.build_decode_metadata(
        **common,
        seq_lens_cpu=torch.tensor([11, 12], dtype=torch.int32),
        cpu_block_ids_by_req=_cpu_block_ids([7], [8]),
    )
    assert metadata_b is not None
    assert metadata_b.scatter_block_ids is not None
    assert metadata_b.scatter_block_offsets is not None
    assert metadata_b.scatter_block_ids.data_ptr() == ids_ptr
    assert metadata_b.scatter_block_offsets.data_ptr() == offsets_ptr
    assert metadata_b.scatter_block_ids.tolist() == [7, 8]
    assert metadata_b.scatter_block_offsets.tolist() == [2, 3]

    # Captured prepared tasks keep the old pointers, so they must observe the
    # in-place refresh too.
    assert metadata_a.scatter_block_ids.tolist() == [7, 8]
    assert metadata_a.scatter_block_offsets.tolist() == [2, 3]
