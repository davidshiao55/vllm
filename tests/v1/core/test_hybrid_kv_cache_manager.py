# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_kv_cache_manager import (
    CPUKVBlockPool,
    HybridKVAccounting,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.metrics.stats import CotsHybridKVStats
from vllm.v1.request import Request


def test_hybrid_kv_accounting_splits_at_block_boundary():
    accounting = HybridKVAccounting(block_sizes=[16], split_blocks=4)

    assert accounting.gpu_blocks_for_tokens(0) == [0]
    assert accounting.cpu_blocks_for_tokens(0) == [0]
    assert accounting.gpu_blocks_for_tokens(64) == [4]
    assert accounting.cpu_blocks_for_tokens(64) == [0]
    assert accounting.gpu_blocks_for_tokens(65) == [4]
    assert accounting.cpu_blocks_for_tokens(65) == [1]
    assert accounting.gpu_blocks_for_tokens(160) == [4]
    assert accounting.cpu_blocks_for_tokens(160) == [6]
    assert accounting.suffix_local_position(64) == 0
    assert accounting.suffix_local_position(65) == 1


def test_cpu_kv_block_pool_extends_and_frees_requests():
    pool = CPUKVBlockPool([4])

    assert pool.extend_to("req-a", [2]) == ([2, 3],)
    assert pool.used_blocks == 2
    assert pool.get_block_ids("req-a") == ([2, 3],)

    assert pool.extend_to("req-a", [3]) == ([1],)
    assert pool.used_blocks == 3
    assert pool.get_block_ids("req-a") == ([2, 3, 1],)

    assert pool.extend_to("req-b", [2]) is None
    pool.free("req-a")
    assert pool.used_blocks == 0
    assert pool.extend_to("req-b", [2]) == ([3, 2],)


def test_cpu_kv_block_pool_evicts_zero_ref_cached_blocks():
    block_size = 4
    split_blocks = 2
    pool = CPUKVBlockPool([1])
    init_none_hash(sha256)
    req_a = _make_request("req-a", prompt_len=12, block_size=block_size)

    assert pool.extend_to("req-a", [1]) == ([0],)
    pool.cache_blocks(
        "req-a",
        req_a.block_hashes,
        num_tokens=12,
        block_size=block_size,
        split_blocks=split_blocks,
    )
    pool.free("req-a")
    assert pool.used_blocks == 1

    hit_ids, hit_tokens = pool.find_longest_cache_hit(
        req_a.block_hashes,
        max_cache_hit_length=12,
        block_size=block_size,
        split_blocks=split_blocks,
    )
    assert hit_ids == ([0],)
    assert hit_tokens == 12

    assert pool.extend_to("req-b", [1]) == ([0],)
    hit_ids, hit_tokens = pool.find_longest_cache_hit(
        req_a.block_hashes,
        max_cache_hit_length=12,
        block_size=block_size,
        split_blocks=split_blocks,
    )
    assert hit_ids == ([],)
    assert hit_tokens == 8
    assert pool.used_blocks == 1


def test_cpu_kv_block_pool_does_not_evict_live_cached_blocks():
    block_size = 4
    split_blocks = 2
    pool = CPUKVBlockPool([1])
    init_none_hash(sha256)
    req_a = _make_request("req-a", prompt_len=12, block_size=block_size)

    assert pool.extend_to("req-a", [1]) == ([0],)
    pool.cache_blocks(
        "req-a",
        req_a.block_hashes,
        num_tokens=12,
        block_size=block_size,
        split_blocks=split_blocks,
    )
    pool.free("req-a")
    hit_ids, hit_tokens = pool.find_longest_cache_hit(
        req_a.block_hashes,
        max_cache_hit_length=12,
        block_size=block_size,
        split_blocks=split_blocks,
    )
    assert hit_ids == ([0],)
    assert hit_tokens == 12

    pool.allocate_new_computed_blocks("req-hit", hit_ids)
    assert not pool.can_extend("req-new", [1])
    assert pool.extend_to("req-new", [1]) is None

    pool.free("req-hit")
    assert pool.extend_to("req-new", [1]) == ([0],)


def _make_request(request_id: str, prompt_len: int, block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=16)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _make_kv_cache_manager(
    *,
    block_size: int = 4,
    num_gpu_blocks: int = 8,
    num_cpu_blocks: int = 4,
    split_blocks: int = 2,
    max_model_len: int = 16,
    enable_caching: bool = False,
) -> KVCacheManager:
    init_none_hash(sha256)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_gpu_blocks * spec.real_page_size_bytes,
                shared_by=["layer"],
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=max_model_len,
        hash_block_size=block_size,
        enable_caching=enable_caching,
        cots_kv_split_blocks=split_blocks,
        cots_kv_cpu_pool_bytes=num_cpu_blocks * spec.real_page_size_bytes,
    )


def test_hybrid_kv_manager_caps_gpu_blocks_and_allocates_cpu_suffix():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=3,
        num_cpu_blocks=4,
        max_model_len=20,
    )
    request = _make_request("req-a", prompt_len=8, block_size=block_size)

    prompt_blocks = manager.allocate_slots(request, num_new_tokens=8)
    assert prompt_blocks is not None
    assert len(prompt_blocks.get_block_ids()[0]) == 2
    assert prompt_blocks.get_cpu_block_ids(allow_none=True) is None

    request.num_computed_tokens = 8
    request.append_output_token_ids(1)
    decode_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert decode_blocks is not None
    assert decode_blocks.get_block_ids(allow_none=True) is None
    assert len(decode_blocks.get_cpu_block_ids()[0]) == 1

    all_blocks = manager.get_blocks("req-a")
    assert len(all_blocks.get_block_ids()[0]) == 2
    assert len(all_blocks.get_cpu_block_ids()[0]) == 1


def test_hybrid_cpu_suffix_appears_only_after_crossing_split():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=12,
        num_cpu_blocks=2,
        split_blocks=2,
        max_model_len=16,
    )
    request = _make_request("req-a", prompt_len=7, block_size=block_size)

    prompt_blocks = manager.allocate_slots(request, num_new_tokens=7)
    assert prompt_blocks is not None
    assert len(manager.get_blocks(request.request_id).get_block_ids()[0]) == 2
    assert (
        manager.get_blocks(request.request_id).get_cpu_block_ids(allow_none=True)
        is None
    )

    request.num_computed_tokens = 7
    request.append_output_token_ids(7)
    split_edge_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert split_edge_blocks is not None
    assert split_edge_blocks.get_cpu_block_ids(allow_none=True) is None
    assert len(manager.get_blocks(request.request_id).get_block_ids()[0]) == 2

    request.num_computed_tokens = 8
    request.append_output_token_ids(8)
    suffix_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert suffix_blocks is not None
    assert suffix_blocks.get_block_ids(allow_none=True) is None
    assert len(suffix_blocks.get_cpu_block_ids()[0]) == 1


def test_hybrid_kv_manager_cpu_failure_returns_none_before_mutation():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=5,
        num_cpu_blocks=1,
        max_model_len=24,
    )

    req_a = _make_request("req-a", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_a, num_new_tokens=8) is not None
    req_a.num_computed_tokens = 8
    req_a.append_output_token_ids(1)
    assert manager.allocate_slots(req_a, num_new_tokens=1) is not None

    req_b = _make_request("req-b", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_b, num_new_tokens=8) is not None
    req_b.num_computed_tokens = 8
    req_b.append_output_token_ids(1)

    assert manager.allocate_slots(req_b, num_new_tokens=1) is None
    assert manager.get_blocks("req-b").get_cpu_block_ids(allow_none=True) is None

    manager.free(req_a)
    retry_blocks = manager.allocate_slots(req_b, num_new_tokens=1)
    assert retry_blocks is not None
    assert len(retry_blocks.get_cpu_block_ids()[0]) == 1


def test_hybrid_cache_commit_is_split_between_gpu_prefix_and_cpu_suffix():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=12,
        num_cpu_blocks=2,
        split_blocks=2,
        max_model_len=16,
        enable_caching=True,
    )
    req_a = _make_request("req-a", prompt_len=13, block_size=block_size)
    allocated_blocks = manager.allocate_slots(req_a, num_new_tokens=13)

    assert allocated_blocks is not None
    assert len(manager.get_blocks(req_a.request_id).get_block_ids()[0]) == 2
    assert len(manager.get_blocks(req_a.request_id).get_cpu_block_ids()[0]) == 2

    manager.free(req_a)

    req_b = _make_request("req-b", prompt_len=13, block_size=block_size)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req_b)

    assert num_computed_tokens == 12
    assert len(computed_blocks.get_block_ids()[0]) == 2
    assert len(computed_blocks.get_cpu_block_ids()[0]) == 1


def test_hybrid_uses_cpu_suffix_by_position_even_when_gpu_has_room():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=12,
        num_cpu_blocks=2,
        split_blocks=2,
        max_model_len=16,
        enable_caching=False,
    )

    request = _make_request("req-a", prompt_len=8, block_size=block_size)
    prompt_blocks = manager.allocate_slots(request, num_new_tokens=8)
    assert prompt_blocks is not None
    assert len(manager.get_blocks(request.request_id).get_block_ids()[0]) == 2
    assert (
        manager.get_blocks(request.request_id).get_cpu_block_ids(allow_none=True)
        is None
    )

    request.num_computed_tokens = 8
    request.append_output_token_ids(1)
    decode_blocks = manager.allocate_slots(request, num_new_tokens=1)

    assert decode_blocks is not None
    assert decode_blocks.get_block_ids(allow_none=True) is None
    assert len(decode_blocks.get_cpu_block_ids()[0]) == 1
    assert len(manager.get_blocks(request.request_id).get_block_ids()[0]) == 2


def test_hybrid_kv_manager_does_not_evict_live_cached_cpu_suffix():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=5,
        num_cpu_blocks=1,
        max_model_len=24,
        enable_caching=True,
    )

    req_a = _make_request("req-a", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_a, num_new_tokens=8) is not None
    req_a.num_computed_tokens = 8
    req_a.append_output_token_ids([8, 9, 10, 11])
    assert manager.allocate_slots(req_a, num_new_tokens=4) is not None
    assert manager.cots_cpu_block_pool is not None
    assert manager.cots_cpu_block_pool.used_blocks == 1

    req_b = _make_request("req-b", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_b, num_new_tokens=8) is not None
    req_b.num_computed_tokens = 8
    req_b.append_output_token_ids([20, 21, 22, 23])

    assert manager.allocate_slots(req_b, num_new_tokens=4) is None
    assert manager.get_blocks("req-b").get_cpu_block_ids(allow_none=True) is None

    manager.free(req_a)
    retry_blocks = manager.allocate_slots(req_b, num_new_tokens=4)
    assert retry_blocks is not None
    assert retry_blocks.get_cpu_block_ids()[0] == [0]
    assert manager.cots_cpu_block_pool.used_blocks == 1


def test_hybrid_kv_manager_evicts_zero_ref_cached_cpu_suffix_on_pressure():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=5,
        num_cpu_blocks=2,
        max_model_len=24,
        enable_caching=True,
    )

    req_a = _make_request("req-a", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_a, num_new_tokens=8) is not None
    req_a.num_computed_tokens = 8
    req_a.append_output_token_ids([8, 9, 10, 11])
    assert manager.allocate_slots(req_a, num_new_tokens=4) is not None
    manager.free(req_a)
    assert manager.cots_cpu_block_pool is not None
    assert manager.cots_cpu_block_pool.used_blocks == 1

    req_b = _make_request("req-b", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_b, num_new_tokens=8) is not None
    req_b.num_computed_tokens = 8
    req_b.append_output_token_ids([20, 21, 22, 23, 24])
    assert manager.allocate_slots(req_b, num_new_tokens=5) is not None
    assert manager.get_blocks("req-b").get_cpu_block_ids()[0] == [0, 1]

    req_a_again = _make_request("req-a-again", prompt_len=13, block_size=block_size)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req_a_again)
    assert num_computed_tokens == 8
    assert computed_blocks.get_cpu_block_ids(allow_none=True) is None
    assert manager.cots_cpu_block_pool.used_blocks == 2


def test_hybrid_kv_manager_reuses_cached_cpu_suffix_blocks():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=5,
        num_cpu_blocks=4,
        max_model_len=24,
        enable_caching=True,
    )

    req_a = _make_request("req-a", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_a, num_new_tokens=8) is not None
    req_a.num_computed_tokens = 8
    req_a.append_output_token_ids([8, 9, 10, 11])
    assert manager.allocate_slots(req_a, num_new_tokens=4) is not None
    cached_cpu_id = manager.get_blocks("req-a").get_cpu_block_ids()[0][0]

    manager.free(req_a)
    assert manager.cots_cpu_block_pool is not None
    assert manager.cots_cpu_block_pool.used_blocks == 1

    req_b = _make_request("req-b", prompt_len=13, block_size=block_size)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req_b)

    assert num_computed_tokens == 12
    assert len(computed_blocks.get_block_ids()[0]) == 2
    assert computed_blocks.get_cpu_block_ids()[0] == [cached_cpu_id]
    assert manager.can_fit_full_sequence(
        req_b,
        num_new_computed_tokens=num_computed_tokens,
        new_computed_blocks=computed_blocks,
    )

    new_blocks = manager.allocate_slots(
        req_b,
        num_new_tokens=1,
        num_new_computed_tokens=num_computed_tokens,
        new_computed_blocks=computed_blocks,
    )

    assert new_blocks is not None
    assert len(new_blocks.get_cpu_block_ids()[0]) == 1
    assert manager.get_blocks("req-b").get_cpu_block_ids()[0][0] == cached_cpu_id
    assert manager.cots_cpu_block_pool.used_blocks == 2


def test_hybrid_kv_manager_reset_prefix_cache_clears_cpu_suffix_cache():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=5,
        num_cpu_blocks=2,
        max_model_len=24,
        enable_caching=True,
    )

    req_a = _make_request("req-a", prompt_len=8, block_size=block_size)
    assert manager.allocate_slots(req_a, num_new_tokens=8) is not None
    req_a.num_computed_tokens = 8
    req_a.append_output_token_ids([8, 9, 10, 11])
    assert manager.allocate_slots(req_a, num_new_tokens=4) is not None

    assert not manager.reset_prefix_cache()
    manager.free(req_a)
    assert manager.cots_cpu_block_pool is not None
    assert manager.cots_cpu_block_pool.used_blocks == 1

    assert manager.reset_prefix_cache()
    assert manager.cots_cpu_block_pool.used_blocks == 0

    req_b = _make_request("req-b", prompt_len=13, block_size=block_size)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req_b)
    assert num_computed_tokens == 0
    assert computed_blocks is manager.empty_kv_cache_blocks


def test_hybrid_kv_manager_reports_and_resets_metrics():
    block_size = 4
    manager = _make_kv_cache_manager(
        block_size=block_size,
        num_gpu_blocks=3,
        num_cpu_blocks=4,
        max_model_len=20,
    )
    request = _make_request("req-a", prompt_len=8, block_size=block_size)

    assert manager.allocate_slots(request, num_new_tokens=8) is not None
    request.num_computed_tokens = 8
    request.append_output_token_ids(1)
    assert manager.allocate_slots(request, num_new_tokens=1) is not None
    request.num_computed_tokens = 9

    manager.record_cots_hybrid_preemption(request)
    stats = manager.make_cots_hybrid_kv_stats(
        CotsHybridKVStats(
            cpu_suffix_attn_ms=2.0,
            cpu_suffix_read_wait_ms=3.0,
            q_d2h_bytes=256,
            kv_d2h_bytes=128,
            kv_uva_h2d_bytes=64,
        )
    )

    assert stats is not None
    assert stats.hybrid_gpu_kv_blocks_used == 2
    assert stats.hybrid_cpu_kv_blocks_used == 1
    assert stats.hybrid_cpu_kv_blocks_total == 4
    assert stats.hybrid_preemptions == 1
    assert stats.hybrid_recomputed_cpu_suffix_tokens == 1
    assert stats.cpu_suffix_attn_ms == 2.0
    assert stats.cpu_suffix_read_wait_ms == 3.0
    assert stats.q_d2h_bytes == 256
    assert stats.kv_d2h_bytes == 128
    assert stats.kv_uva_h2d_bytes == 64

    next_stats = manager.make_cots_hybrid_kv_stats()
    assert next_stats is not None
    assert next_stats.hybrid_preemptions == 0
    assert next_stats.hybrid_recomputed_cpu_suffix_tokens == 0
