# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 2 COTS hybrid KV accounting primitives.

This module intentionally contains only tier/accounting logic. The first
end-to-end integration still needs the worker-side split block-table and
attention routing path before this can replace the homogeneous KV manager in
the scheduler.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True)
class HybridKVUsage:
    gpu_blocks_used: int
    cpu_blocks_used: int
    cpu_blocks_total: int
    hybrid_preemptions: int = 0
    hybrid_recomputed_cpu_suffix_tokens: int = 0

    @property
    def cpu_usage(self) -> float:
        if self.cpu_blocks_total == 0:
            return 0.0
        return self.cpu_blocks_used / self.cpu_blocks_total


@dataclass(frozen=True)
class HybridKVBlockIDs:
    """Block IDs allocated for one scheduler step.

    `gpu_block_ids` follows the existing vLLM group layout. `cpu_block_ids`
    uses the same group layout, but block indices are suffix-local:
    CPU block 0 covers token positions `[x_tokens, x_tokens + block_size)`.
    """

    gpu_block_ids: tuple[list[int], ...] | None
    cpu_block_ids: tuple[list[int], ...]

    @property
    def has_cpu_blocks(self) -> bool:
        return any(self.cpu_block_ids)


@dataclass
class CPUKVCacheBlock:
    """CPU-resident suffix KV cache block metadata."""

    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHash | None = None


class CPUKVBlockPool:
    """CPU suffix block pool with prefix-cache-like reuse.

    The pool keeps CPU suffix KV resident after a request is freed when the
    block has a committed block hash. Later requests can attach to those
    cached blocks by hash. If fresh capacity is needed, zero-ref cached blocks
    are evicted using a simple LRU over block hashes. This intentionally avoids
    vLLM's native kv_offload load/store path: COTS suffix KV stays on CPU and
    is never bulk-reloaded to GPU.
    """

    def __init__(self, num_blocks_per_group: list[int]) -> None:
        if not num_blocks_per_group:
            raise ValueError("num_blocks_per_group must be non-empty")
        if any(n < 0 for n in num_blocks_per_group):
            raise ValueError(
                f"num_blocks_per_group must be non-negative, got {num_blocks_per_group}"
            )
        self._blocks: list[list[CPUKVCacheBlock]] = [
            [CPUKVCacheBlock(block_id=i) for i in range(n)]
            for n in num_blocks_per_group
        ]
        self._free: list[list[CPUKVCacheBlock]] = [
            list(group) for group in self._blocks
        ]
        self._req_to_blocks: dict[str, tuple[list[CPUKVCacheBlock], ...]] = {}
        self._cached: list[OrderedDict[BlockHash, list[CPUKVCacheBlock]]] = [
            OrderedDict() for _ in num_blocks_per_group
        ]
        self._total = sum(num_blocks_per_group)

    @property
    def total_blocks(self) -> int:
        return self._total

    @property
    def used_blocks(self) -> int:
        # Cached zero-ref blocks are resident and count as used until evicted or
        # reset. This matches the cache-extension view of CPU KV capacity.
        return self._total - sum(len(group) for group in self._free)

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        blocks = self._req_to_blocks.get(request_id)
        if blocks is None:
            return tuple([] for _ in self._free)
        return tuple([block.block_id for block in group] for group in blocks)

    def can_extend(self, request_id: str, target_blocks: list[int]) -> bool:
        return self.can_extend_with_computed(request_id, None, target_blocks)

    def can_extend_with_computed(
        self,
        request_id: str,
        computed_block_ids: tuple[list[int], ...] | None,
        target_blocks: list[int],
    ) -> bool:
        if len(target_blocks) != len(self._free):
            raise ValueError(
                "target_blocks has "
                f"{len(target_blocks)} groups, expected {len(self._free)}"
            )
        computed_blocks = self._blocks_from_ids(computed_block_ids)
        current = self._req_to_blocks.get(request_id)
        protected_ids = {block.block_id for group in computed_blocks for block in group}
        if current is None:
            current_lens = [len(group) for group in computed_blocks]
        else:
            current_lens = [len(group) for group in current]
            protected_ids.update(block.block_id for group in current for block in group)
            if any(computed_blocks):
                return False

        for group_idx, (target, current_len) in enumerate(
            zip(target_blocks, current_lens)
        ):
            if target < current_len:
                return False
            needed = target - current_len
            if needed > self._available_for_allocation(group_idx, protected_ids):
                return False
        return True

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        computed_block_ids: tuple[list[int], ...] | None,
    ) -> None:
        computed_blocks = self._blocks_from_ids(computed_block_ids)
        if not any(computed_blocks):
            if request_id not in self._req_to_blocks:
                self._req_to_blocks[request_id] = tuple([] for _ in self._free)
            return
        if request_id in self._req_to_blocks:
            raise RuntimeError(
                f"Cannot attach computed CPU KV blocks to existing request {request_id}"
            )
        for group_idx, group in enumerate(computed_blocks):
            for block in group:
                if block.block_hash is None:
                    raise RuntimeError(f"CPU KV block {block.block_id} is not cached")
                block.ref_cnt += 1
                self._touch_cached_block(group_idx, block)
        self._req_to_blocks[request_id] = computed_blocks

    def extend_to(
        self, request_id: str, target_blocks: list[int]
    ) -> tuple[list[int], ...] | None:
        if len(target_blocks) != len(self._free):
            raise ValueError(
                "target_blocks has "
                f"{len(target_blocks)} groups, expected {len(self._free)}"
            )
        current = self._req_to_blocks.setdefault(
            request_id, tuple([] for _ in self._free)
        )
        if not self.can_extend(request_id, target_blocks):
            return None

        new_blocks: list[list[int]] = []
        protected_ids = {block.block_id for group in current for block in group}
        for group_idx, target in enumerate(target_blocks):
            group = current[group_idx]
            needed = target - len(group)
            if needed == 0:
                new_blocks.append([])
                continue
            if not self._ensure_free_blocks(group_idx, needed, protected_ids):
                return None
            allocated = self._free[group_idx][-needed:]
            del self._free[group_idx][-needed:]
            for block in allocated:
                if block.ref_cnt != 0 or block.block_hash is not None:
                    raise RuntimeError(
                        f"CPU KV free block {block.block_id} is still cached "
                        "or referenced"
                    )
                block.ref_cnt = 1
                protected_ids.add(block.block_id)
            group.extend(allocated)
            new_blocks.append([block.block_id for block in allocated])
        return tuple(new_blocks)

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[BlockHash],
        max_cache_hit_length: int,
        block_size: int,
        split_blocks: int,
    ) -> tuple[tuple[list[int], ...] | None, int]:
        max_full_blocks = max(max_cache_hit_length, 0) // block_size
        if max_full_blocks <= split_blocks:
            return None, split_blocks * block_size

        hit_blocks: list[list[int]] = [[] for _ in self._free]
        for global_block_idx in range(split_blocks, max_full_blocks):
            if global_block_idx >= len(block_hashes):
                break
            block_hash = block_hashes[global_block_idx]
            per_group: list[CPUKVCacheBlock] = []
            for group_idx in range(len(self._free)):
                block = self._get_cached_block(group_idx, block_hash)
                if block is None:
                    return tuple(hit_blocks), (
                        split_blocks + len(hit_blocks[0])
                    ) * block_size
                per_group.append(block)
            for group_idx, block in enumerate(per_group):
                hit_blocks[group_idx].append(block.block_id)
        return tuple(hit_blocks), (split_blocks + len(hit_blocks[0])) * block_size

    def cache_blocks(
        self,
        request_id: str,
        block_hashes: Sequence[BlockHash],
        num_tokens: int,
        block_size: int,
        split_blocks: int,
    ) -> None:
        blocks = self._req_to_blocks.get(request_id)
        if blocks is None:
            return
        num_full_blocks = min(max(num_tokens, 0) // block_size, len(block_hashes))
        for global_block_idx in range(split_blocks, num_full_blocks):
            suffix_idx = global_block_idx - split_blocks
            block_hash = block_hashes[global_block_idx]
            for group_idx, group in enumerate(blocks):
                if suffix_idx >= len(group):
                    continue
                block = group[suffix_idx]
                if block.block_hash is not None:
                    continue
                block.block_hash = block_hash
                self._cached[group_idx].setdefault(block_hash, []).append(block)
                self._cached[group_idx].move_to_end(block_hash)

    def free(self, request_id: str) -> None:
        blocks = self._req_to_blocks.pop(request_id, None)
        if blocks is None:
            return
        for group_idx, block_group in enumerate(blocks):
            for block in reversed(block_group):
                if block.ref_cnt <= 0:
                    raise RuntimeError(
                        f"CPU KV block {block.block_id} ref_cnt underflow"
                    )
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    if block.block_hash is None:
                        self._free[group_idx].append(block)
                    else:
                        self._touch_cached_block(group_idx, block)

    def can_reset_prefix_cache(self) -> bool:
        return all(block.ref_cnt == 0 for group in self._blocks for block in group)

    def reset_prefix_cache(self) -> bool:
        if not self.can_reset_prefix_cache():
            return False
        self._req_to_blocks.clear()
        for group_idx, group in enumerate(self._blocks):
            self._cached[group_idx].clear()
            self._free[group_idx] = list(group)
            for block in group:
                block.block_hash = None
                block.ref_cnt = 0
        return True

    def _blocks_from_ids(
        self, block_ids: tuple[list[int], ...] | None
    ) -> tuple[list[CPUKVCacheBlock], ...]:
        if block_ids is None:
            return tuple([] for _ in self._free)
        if len(block_ids) != len(self._free):
            raise ValueError(
                f"block_ids has {len(block_ids)} groups, expected {len(self._free)}"
            )
        out: list[list[CPUKVCacheBlock]] = []
        for group_idx, ids in enumerate(block_ids):
            group_blocks = self._blocks[group_idx]
            converted: list[CPUKVCacheBlock] = []
            for block_id in ids:
                if block_id < 0 or block_id >= len(group_blocks):
                    raise ValueError(
                        f"CPU KV block id {block_id} is out of range for "
                        f"group {group_idx}"
                    )
                converted.append(group_blocks[block_id])
            out.append(converted)
        return tuple(out)

    def _available_for_allocation(self, group_idx: int, protected_ids: set[int]) -> int:
        available = len(self._free[group_idx])
        seen: set[int] = set()
        for blocks in self._cached[group_idx].values():
            for block in blocks:
                if block.block_id in seen:
                    continue
                seen.add(block.block_id)
                if block.ref_cnt == 0 and block.block_id not in protected_ids:
                    available += 1
        return available

    def _ensure_free_blocks(
        self, group_idx: int, needed: int, protected_ids: set[int]
    ) -> bool:
        while len(self._free[group_idx]) < needed:
            if not self._evict_one(group_idx, protected_ids):
                return False
        return True

    def _evict_one(self, group_idx: int, protected_ids: set[int]) -> bool:
        cached = self._cached[group_idx]
        for block_hash, blocks in list(cached.items()):
            for block in list(blocks):
                if block.ref_cnt != 0 or block.block_id in protected_ids:
                    continue
                blocks.remove(block)
                if not blocks:
                    del cached[block_hash]
                block.block_hash = None
                self._free[group_idx].append(block)
                return True
            if not blocks and block_hash in cached:
                del cached[block_hash]
        return False

    def _get_cached_block(
        self, group_idx: int, block_hash: BlockHash
    ) -> CPUKVCacheBlock | None:
        blocks = self._cached[group_idx].get(block_hash)
        if not blocks:
            return None
        self._cached[group_idx].move_to_end(block_hash)
        # Prefer an already referenced block to improve sharing, but any cached
        # physical block with this hash is equivalent.
        return max(blocks, key=lambda block: block.ref_cnt)

    def _touch_cached_block(self, group_idx: int, block: CPUKVCacheBlock) -> None:
        if block.block_hash is not None and block.block_hash in self._cached[group_idx]:
            self._cached[group_idx].move_to_end(block.block_hash)


class HybridKVAccounting:
    """Position-split accounting shared by planner tests and scheduler wiring."""

    def __init__(self, *, block_sizes: list[int], split_blocks: int) -> None:
        if split_blocks < 0:
            raise ValueError(f"split_blocks must be non-negative, got {split_blocks}")
        if not block_sizes:
            raise ValueError("block_sizes must be non-empty")
        self.block_sizes = list(block_sizes)
        self.split_blocks = int(split_blocks)

    def gpu_blocks_for_tokens(self, num_tokens: int) -> list[int]:
        return [
            min(cdiv(max(num_tokens, 0), block_size), self.split_blocks)
            for block_size in self.block_sizes
        ]

    def cpu_blocks_for_tokens(self, num_tokens: int) -> list[int]:
        return [
            max(cdiv(max(num_tokens, 0), block_size) - self.split_blocks, 0)
            for block_size in self.block_sizes
        ]

    def suffix_local_position(self, position: int, group_idx: int = 0) -> int:
        x_tokens = self.split_blocks * self.block_sizes[group_idx]
        return int(position) - x_tokens
