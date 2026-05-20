# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.hybrid_kv_cache_manager import CPUKVBlockPool, HybridKVAccounting
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.metrics.stats import CotsHybridKVStats, PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    cpu_block_ids: tuple[list[int], ...] | None = None
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        cpu_block_ids = self._add_cpu_block_ids(self.cpu_block_ids, other.cpu_block_ids)
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            ),
            cpu_block_ids,
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    @staticmethod
    def _add_cpu_block_ids(
        left: tuple[list[int], ...] | None,
        right: tuple[list[int], ...] | None,
    ) -> tuple[list[int], ...] | None:
        if left is None:
            return right
        if right is None:
            return left
        return tuple(
            list(itertools.chain(left_group, right_group))
            for left_group, right_group in zip(left, right)
        )

    @overload
    def get_cpu_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_cpu_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_cpu_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        if self.cpu_block_ids is None:
            if allow_none:
                return None
            return tuple([] for _ in range(len(self.blocks)))
        if allow_none and all(len(group) == 0 for group in self.cpu_block_ids):
            return None
        return tuple(list(group) for group in self.cpu_block_ids)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        cots_kv_split_blocks: int = 0,
        cots_kv_cpu_pool_bytes: int = 0,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        self.cots_kv_split_blocks = int(cots_kv_split_blocks)
        self.cots_kv_cpu_pool_bytes = int(cots_kv_cpu_pool_bytes)
        self.cots_hybrid_kv_enabled = (
            self.cots_kv_split_blocks > 0 and self.cots_kv_cpu_pool_bytes > 0
        )
        self.cots_kv_split_tokens = 0
        self.cots_hybrid_preemptions = 0
        self.cots_hybrid_recomputed_cpu_suffix_tokens = 0
        # Hybrid KV keeps a request fully GPU-resident when its final context
        # fits. Requests that would overrun GPU KV use the block-aligned CPU
        # suffix tier from their first post-split allocation.
        self.cots_hybrid_full_gpu_req_ids: set[str] = set()
        self.cots_hybrid_cpu_suffix_req_ids: set[str] = set()
        self.cots_hybrid_accounting: HybridKVAccounting | None = None
        self.cots_cpu_block_pool: CPUKVBlockPool | None = None
        if self.cots_hybrid_kv_enabled:
            self._init_cots_hybrid_kv(kv_cache_config)

    def _init_cots_hybrid_kv(self, kv_cache_config: KVCacheConfig) -> None:
        if self.num_kv_cache_groups != 1:
            raise ValueError(
                "COTS hybrid KV Phase 2 supports exactly one decoder KV cache group"
            )

        layer_names: list[str] = []
        attention_spec: AttentionSpec | None = None
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_specs = kv_cache_spec.kv_cache_specs
            else:
                layer_specs = {
                    layer_name: kv_cache_spec
                    for layer_name in kv_cache_group.layer_names
                }

            for layer_name, layer_spec in layer_specs.items():
                if isinstance(layer_spec, EncoderOnlyAttentionSpec):
                    continue
                if not isinstance(layer_spec, FullAttentionSpec):
                    raise ValueError(
                        "COTS hybrid KV Phase 2 supports only decoder full attention; "
                        f"got {type(layer_spec).__name__} for {layer_name}"
                    )
                if (
                    layer_spec.sliding_window is not None
                    or layer_spec.attention_chunk_size is not None
                ):
                    raise ValueError(
                        "COTS hybrid KV Phase 2 does not support sliding-window or "
                        "chunked-local attention"
                    )
                if attention_spec is None:
                    attention_spec = layer_spec
                elif layer_spec != attention_spec:
                    raise ValueError(
                        "COTS hybrid KV Phase 2 requires uniform attention KV specs"
                    )
                layer_names.append(layer_name)

        if attention_spec is None or not layer_names:
            raise ValueError("COTS hybrid KV enabled but no attention layers exist")

        split_tokens = self.cots_kv_split_blocks * attention_spec.block_size
        num_cpu_blocks = self.cots_kv_cpu_pool_bytes // (
            attention_spec.real_page_size_bytes * len(layer_names)
        )
        if num_cpu_blocks <= 0:
            raise ValueError(
                "COTS hybrid KV CPU pool is too small for one suffix block per "
                "layer: "
                f"pool_bytes={self.cots_kv_cpu_pool_bytes}, "
                f"bytes_per_block_per_layer={attention_spec.real_page_size_bytes}, "
                f"num_layers={len(layer_names)}"
            )

        self.cots_hybrid_accounting = HybridKVAccounting(
            block_sizes=[attention_spec.block_size],
            split_blocks=self.cots_kv_split_blocks,
        )
        self.cots_kv_split_tokens = split_tokens
        self.cots_cpu_block_pool = CPUKVBlockPool([int(num_cpu_blocks)])

    def _cots_gpu_num_tokens(self, num_tokens: int) -> int:
        if not self.cots_hybrid_kv_enabled:
            return num_tokens
        return min(num_tokens, self.cots_kv_split_tokens)

    def _cots_gpu_num_tokens_for_request(self, request_id: str, num_tokens: int) -> int:
        if not self.cots_hybrid_kv_enabled:
            return num_tokens
        if request_id not in self.cots_hybrid_cpu_suffix_req_ids:
            return num_tokens
        return min(num_tokens, self.cots_kv_split_tokens)

    def _cots_final_target_num_tokens(self, request: Request, floor_tokens: int) -> int:
        target_tokens = request.num_prompt_tokens + request.max_tokens
        return min(max(floor_tokens, target_tokens), self.max_model_len)

    def _cots_cpu_blocks_for_tokens(self, num_tokens: int) -> list[int]:
        if self.cots_hybrid_accounting is None:
            return []
        return self.cots_hybrid_accounting.cpu_blocks_for_tokens(num_tokens)

    def _cots_cpu_blocks_for_request_tokens(
        self, request_id: str, num_tokens: int
    ) -> list[int]:
        if self.cots_hybrid_accounting is None:
            return []
        if request_id not in self.cots_hybrid_cpu_suffix_req_ids:
            return [0 for _ in self.cots_hybrid_accounting.block_sizes]
        return self.cots_hybrid_accounting.cpu_blocks_for_tokens(num_tokens)

    def _cots_can_allocate_cpu_blocks(
        self,
        request_id: str,
        num_tokens: int,
        computed_cpu_block_ids: tuple[list[int], ...] | None = None,
    ) -> bool:
        if self.cots_cpu_block_pool is None:
            return True
        return self.cots_cpu_block_pool.can_extend_with_computed(
            request_id,
            computed_cpu_block_ids,
            self._cots_cpu_blocks_for_request_tokens(request_id, num_tokens),
        )

    def _cots_allocate_new_computed_cpu_blocks(
        self,
        request_id: str,
        computed_cpu_block_ids: tuple[list[int], ...] | None,
    ) -> None:
        if self.cots_cpu_block_pool is None:
            return
        self.cots_cpu_block_pool.allocate_new_computed_blocks(
            request_id, computed_cpu_block_ids
        )

    def _cots_allocate_cpu_blocks(
        self, request_id: str, num_tokens: int
    ) -> tuple[list[int], ...] | None:
        if self.cots_cpu_block_pool is None:
            return None
        target_blocks = self._cots_cpu_blocks_for_request_tokens(request_id, num_tokens)
        current_blocks = self.cots_cpu_block_pool.get_block_ids(request_id)
        if len(current_blocks) == len(target_blocks) and all(
            len(current) >= target
            for current, target in zip(current_blocks, target_blocks)
        ):
            return tuple([] for _ in target_blocks)
        allocated = self.cots_cpu_block_pool.extend_to(request_id, target_blocks)
        if allocated is None:
            raise RuntimeError(
                "COTS hybrid KV CPU allocation failed after can_extend succeeded"
            )
        return allocated

    def _cots_cache_cpu_blocks(self, request: Request, num_tokens: int) -> None:
        if self.cots_cpu_block_pool is None or self.cots_hybrid_accounting is None:
            return
        self.cots_cpu_block_pool.cache_blocks(
            request_id=request.request_id,
            block_hashes=request.block_hashes,
            num_tokens=num_tokens,
            block_size=self.cots_hybrid_accounting.block_sizes[0],
            split_blocks=self.cots_hybrid_accounting.split_blocks,
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def record_cots_hybrid_preemption(self, request: Request) -> None:
        if not self.cots_hybrid_kv_enabled:
            return
        self.cots_hybrid_preemptions += 1
        self.cots_hybrid_recomputed_cpu_suffix_tokens += max(
            min(request.num_computed_tokens, self.max_model_len)
            - self.cots_kv_split_tokens,
            0,
        )

    def make_cots_hybrid_kv_stats(
        self,
        worker_stats: CotsHybridKVStats | None = None,
    ) -> CotsHybridKVStats | None:
        if not self.cots_hybrid_kv_enabled:
            if worker_stats and worker_stats.has_worker_activity():
                return worker_stats
            return None

        total_gpu_blocks = max(self.block_pool.num_gpu_blocks - 1, 0)
        gpu_blocks_used = max(
            self.block_pool.num_gpu_blocks - self.block_pool.get_num_free_blocks() - 1,
            0,
        )
        cpu_blocks_used = (
            self.cots_cpu_block_pool.used_blocks
            if self.cots_cpu_block_pool is not None
            else 0
        )
        cpu_blocks_total = (
            self.cots_cpu_block_pool.total_blocks
            if self.cots_cpu_block_pool is not None
            else 0
        )
        stats = CotsHybridKVStats(
            hybrid_gpu_kv_blocks_used=min(gpu_blocks_used, total_gpu_blocks),
            hybrid_cpu_kv_blocks_used=cpu_blocks_used,
            hybrid_cpu_kv_blocks_total=cpu_blocks_total,
            hybrid_preemptions=self.cots_hybrid_preemptions,
            hybrid_recomputed_cpu_suffix_tokens=(
                self.cots_hybrid_recomputed_cpu_suffix_tokens
            ),
        )
        if worker_stats is not None:
            stats.merge_worker_stats(worker_stats)
        self.cots_hybrid_preemptions = 0
        self.cots_hybrid_recomputed_cpu_suffix_tokens = 0
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        gpu_max_cache_hit_length = self._cots_gpu_num_tokens_for_request(
            request.request_id, max_cache_hit_length
        )
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, gpu_max_cache_hit_length
            )
        )

        cpu_block_ids: tuple[list[int], ...] | None = None
        if (
            self.cots_hybrid_kv_enabled
            and self.cots_cpu_block_pool is not None
            and self.cots_hybrid_accounting is not None
            and num_new_computed_tokens >= self.cots_kv_split_tokens
            and max_cache_hit_length > self.cots_kv_split_tokens
        ):
            cpu_block_ids, cpu_hit_tokens = (
                self.cots_cpu_block_pool.find_longest_cache_hit(
                    request.block_hashes,
                    max_cache_hit_length,
                    self.cots_hybrid_accounting.block_sizes[0],
                    self.cots_hybrid_accounting.split_blocks,
                )
            )
            if cpu_block_ids is not None and any(cpu_block_ids):
                num_new_computed_tokens = max(num_new_computed_tokens, cpu_hit_tokens)
            else:
                cpu_block_ids = None

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return (
            self.create_kv_cache_blocks(computed_blocks, cpu_block_ids),
            num_new_computed_tokens,
        )

    def can_fit_full_sequence(
        self,
        request: Request,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
    ) -> bool:
        """Check if the KV cache has enough free blocks to hold the full
        sequence, accounting for prefix cache hits and sliding window.

        This is used as an admission gate to prevent over-admitting requests
        when chunked prefill would otherwise only check the first chunk.
        """
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
            new_computed_cpu_block_ids = new_computed_blocks.get_cpu_block_ids(
                allow_none=True
            )
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks
            new_computed_cpu_block_ids = None

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        final_target_num_tokens = self._cots_final_target_num_tokens(
            request, full_num_tokens
        )

        if (
            self.cots_hybrid_kv_enabled
            and final_target_num_tokens > self.cots_kv_split_tokens
            and request.request_id not in self.cots_hybrid_full_gpu_req_ids
            and request.request_id not in self.cots_hybrid_cpu_suffix_req_ids
        ):
            full_gpu_blocks = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=final_target_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=final_target_num_tokens,
            )
            if full_gpu_blocks <= self.block_pool.get_num_free_blocks():
                return True
            split_tokens = min(final_target_num_tokens, self.cots_kv_split_tokens)
            split_computed = min(total_computed_tokens, self.cots_kv_split_tokens)
            split_gpu_blocks = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=split_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=split_computed,
                num_tokens_main_model=split_tokens,
            )
            if split_gpu_blocks > self.block_pool.get_num_free_blocks():
                return False
            if self.cots_cpu_block_pool is None:
                return True
            return self.cots_cpu_block_pool.can_extend_with_computed(
                request.request_id,
                new_computed_cpu_block_ids,
                self._cots_cpu_blocks_for_tokens(final_target_num_tokens),
            )

        gpu_full_num_tokens = self._cots_gpu_num_tokens_for_request(
            request.request_id, full_num_tokens
        )
        gpu_total_computed_tokens = self._cots_gpu_num_tokens_for_request(
            request.request_id, total_computed_tokens
        )
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=gpu_full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=gpu_total_computed_tokens,
            num_tokens_main_model=gpu_full_num_tokens,
        )

        gpu_fits = num_blocks_to_allocate <= self.block_pool.get_num_free_blocks()
        cpu_fits = self._cots_can_allocate_cpu_blocks(
            request.request_id, full_num_tokens, new_computed_cpu_block_ids
        )
        return gpu_fits and cpu_fits

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
            new_computed_cpu_block_ids = new_computed_blocks.get_cpu_block_ids(
                allow_none=True
            )
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks
            new_computed_cpu_block_ids = None

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        gpu_total_computed_tokens = self._cots_gpu_num_tokens_for_request(
            request.request_id, total_computed_tokens
        )
        gpu_num_tokens_main_model = self._cots_gpu_num_tokens_for_request(
            request.request_id, num_tokens_main_model
        )
        gpu_num_tokens_need_slot = self._cots_gpu_num_tokens_for_request(
            request.request_id, num_tokens_need_slot
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, gpu_total_computed_tokens
        )

        final_target_num_tokens = self._cots_final_target_num_tokens(
            request, num_tokens_need_slot
        )

        if (
            self.cots_hybrid_kv_enabled
            and final_target_num_tokens > self.cots_kv_split_tokens
            and request.request_id not in self.cots_hybrid_full_gpu_req_ids
            and request.request_id not in self.cots_hybrid_cpu_suffix_req_ids
        ):
            full_gpu_blocks = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=final_target_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
            )
            if (
                new_computed_cpu_block_ids is None
                and full_gpu_blocks <= self.block_pool.get_num_free_blocks()
            ):
                self.cots_hybrid_full_gpu_req_ids.add(request.request_id)
                gpu_total_computed_tokens = total_computed_tokens
                gpu_num_tokens_main_model = num_tokens_main_model
                gpu_num_tokens_need_slot = final_target_num_tokens
                num_blocks_to_allocate = full_gpu_blocks
            else:
                split_total_computed_tokens = min(
                    total_computed_tokens, self.cots_kv_split_tokens
                )
                split_num_tokens_main_model = min(
                    num_tokens_main_model, self.cots_kv_split_tokens
                )
                split_num_tokens_need_slot = min(
                    num_tokens_need_slot, self.cots_kv_split_tokens
                )
                split_gpu_blocks = self.coordinator.get_num_blocks_to_allocate(
                    request_id=request.request_id,
                    num_tokens=split_num_tokens_need_slot,
                    new_computed_blocks=new_computed_block_list,
                    num_encoder_tokens=num_encoder_tokens,
                    total_computed_tokens=split_total_computed_tokens,
                    num_tokens_main_model=split_num_tokens_main_model,
                )
                if split_gpu_blocks > self.block_pool.get_num_free_blocks():
                    return None
                if self.cots_cpu_block_pool is not None and not (
                    self.cots_cpu_block_pool.can_extend_with_computed(
                        request.request_id,
                        new_computed_cpu_block_ids,
                        self._cots_cpu_blocks_for_tokens(num_tokens_need_slot),
                    )
                ):
                    return None
                self.cots_hybrid_cpu_suffix_req_ids.add(request.request_id)
                gpu_total_computed_tokens = split_total_computed_tokens
                gpu_num_tokens_main_model = split_num_tokens_main_model
                gpu_num_tokens_need_slot = split_num_tokens_need_slot
                num_blocks_to_allocate = split_gpu_blocks
        else:
            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=gpu_num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=gpu_total_computed_tokens,
                num_tokens_main_model=gpu_num_tokens_main_model,
            )

            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                # Cannot allocate new GPU prefix blocks.
                return None
            if not self._cots_can_allocate_cpu_blocks(
                request.request_id, num_tokens_need_slot, new_computed_cpu_block_ids
            ):
                # Cannot allocate new CPU suffix blocks. Report failure before
                # mutating either tier so the scheduler can preempt/recompute.
                return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or new_computed_cpu_block_ids is not None
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=self._cots_gpu_num_tokens_for_request(
                    request.request_id, num_local_computed_tokens
                ),
                num_external_computed_tokens=0
                if self.cots_hybrid_kv_enabled
                else num_external_computed_tokens,
            )
            self._cots_allocate_new_computed_cpu_blocks(
                request.request_id, new_computed_cpu_block_ids
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            gpu_num_tokens_need_slot,
            gpu_num_tokens_main_model,
            num_encoder_tokens,
        )
        new_cpu_block_ids = self._cots_allocate_cpu_blocks(
            request.request_id, num_tokens_need_slot
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks, new_cpu_block_ids)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(
            request,
            self._cots_gpu_num_tokens_for_request(
                request.request_id, num_tokens_to_cache
            ),
        )
        self._cots_cache_cpu_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks, new_cpu_block_ids)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)
        self.cots_hybrid_full_gpu_req_ids.discard(request.request_id)
        self.cots_hybrid_cpu_suffix_req_ids.discard(request.request_id)
        if self.cots_cpu_block_pool is not None:
            self.cots_cpu_block_pool.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if (
            self.cots_cpu_block_pool is not None
            and not self.cots_cpu_block_pool.can_reset_prefix_cache()
        ):
            return False
        if not self.block_pool.reset_prefix_cache():
            return False
        self.cots_hybrid_full_gpu_req_ids.clear()
        self.cots_hybrid_cpu_suffix_req_ids.clear()
        if (
            self.cots_cpu_block_pool is not None
            and not self.cots_cpu_block_pool.reset_prefix_cache()
        ):
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        cpu_block_ids = (
            self.cots_cpu_block_pool.get_block_ids(request_id)
            if self.cots_cpu_block_pool is not None
            else None
        )
        return self.create_kv_cache_blocks(
            self.coordinator.get_blocks(request_id), cpu_block_ids
        )

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(
                request,
                self._cots_gpu_num_tokens_for_request(
                    request.request_id, num_computed_tokens
                ),
            )
            self._cots_cache_cpu_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self,
        blocks: tuple[list[KVCacheBlock], ...],
        cpu_block_ids: tuple[list[int], ...] | None = None,
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks.
        has_cpu_blocks = cpu_block_ids is not None and any(cpu_block_ids)
        if any(blocks) or has_cpu_blocks:
            return KVCacheBlocks(blocks, cpu_block_ids)
        return self.empty_kv_cache_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        self.coordinator.new_step_starts()
