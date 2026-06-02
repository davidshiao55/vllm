# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import OffloadConfig
    from vllm.forward_context import BatchDescriptor


@dataclass(frozen=True)
class ForwardDispatchInfo:
    """All per-forward dispatch state the offloader needs, pushed OOG.

    Single boundary between vLLM's model runner and the offloader so
    future per-forward state can land here without another vLLM-side
    call site. Pushed by `GPUModelRunner._publish_forward_dispatch`
    on the active runner path, and by legacy cudagraph utilities when
    they drive forwards directly.

    - `batch_descriptor.num_tokens` is the authoritative dispatched
      bucket (padded). Replaces the in-graph pre-hook's
      `anchor.shape[0]` inference, which saturated to the persistent
      input-buffer size under FULL CUDA Graph capture and made
      per-bucket Planner outputs ineffective at runtime.
    - `num_tokens_unpadded` is the live row count. Graph/slab sizes
      remain bucket-capacity sized; offloaders may use this value to
      avoid doing CPU work for padded rows.
    - `batch_descriptor.cots_dispatch_bucket`, when present, is the
      planner route bucket associated with the graph/compile variant.
      Offloaders may use it instead of deriving route selection from
      `batch_descriptor.num_tokens`.
    """

    batch_descriptor: "BatchDescriptor"
    num_tokens_unpadded: int


logger = init_logger(__name__)


"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: PrefetchOffloader
  * implemented by: PrefetchDeferOffloader
    * uses: _ModuleOffloader
        * uses: _BaseParamOffloader (ABC)
            * implemented by: _CpuParamOffloader
"""


class BaseOffloader(ABC):
    """Base class for model parameter offloading strategies.

    Offloaders control how model parameters are stored and loaded during
    inference. Different strategies trade memory for compute/transfer time.
    """

    @abstractmethod
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.

        Returns:
            List of modules, potentially with offloading hooks installed.
        """
        pass

    def post_init(self):
        """Called after model construction completes.

        Offloaders can use this to:
        - Finalize parameter storage
        - Start initial prefetching
        - Allocate shared resources
        """
        return

    def prepare_before_forward(self, num_tokens: int) -> None:  # noqa: B027
        """Prepare offloader state for a model forward.

        Called from eager pre-hooks and outside FULL CUDA graph capture/replay
        so offloaders can repair runtime state that depends on the active
        token bucket.
        """
        pass

    def set_live_num_tokens(self, live_num_tokens: int) -> None:  # noqa: B027
        """Push the live (unpadded) token count to the offloader.

        A CUDA graph bucket is a capacity, not a guarantee that every
        row in the bucket is semantically live. Offloaders that execute
        CPU work may use this value as a live-row cap while keeping
        graph capture, slab allocation, and buffer sizing bucket-based.
        """
        pass

    def on_dispatch(self, info: ForwardDispatchInfo) -> None:
        """Single OOG entry point for all per-forward dispatch state.

        Called before scheduler, profile dummy, warmup, CUDA Graph
        capture, or CUDA Graph replay forwards. Default impl delegates to the
        per-piece hooks so existing offloaders keep working unchanged.
        """
        self.prepare_before_forward(info.batch_descriptor.num_tokens)
        self.sync_prev_onload()
        self.set_live_num_tokens(info.num_tokens_unpadded)

    def decorate_batch_descriptor(self, batch_descriptor: "BatchDescriptor"):
        """Add offloader-specific graph/compile route identity.

        CUDAGraph dispatchers call this when they construct a runtime/capture
        `BatchDescriptor`. Most offloaders leave it unchanged. COTS uses it to
        attach the planner dispatch bucket and compile-visible route signature
        so CUDA graph and torch.compile variants are selected by the same
        structural route that the runtime will execute.
        """
        return batch_descriptor

    def post_cudagraph_capture(self) -> None:  # noqa: B027
        """One-shot hook fired by `cudagraph_utils.CudaGraphManager.capture`
        AFTER all bucket graphs are captured but BEFORE any measured
        replay. Default is no-op; offloaders may override to reset
        instrumentation counters so per-generate diagnostics
        isolate replay-time activity (see `phase1c_findings.md §1c.22`).
        """
        pass

    def shutdown(self) -> None:  # noqa: B027
        """Called from worker shutdown so offloaders can drain native
        resources deterministically. Default is no-op."""
        pass

    def sync_prev_onload(self) -> None:  # noqa: B027
        """Sync previous onload operations. Override in subclasses."""
        pass

    def join_after_forward(self) -> None:  # noqa: B027
        """Join streams after forward. Override in subclasses."""
        pass

    def _wait_for_layer(self, layer_idx: int) -> None:  # noqa: B027
        """Wait for layer prefetch. Override in subclasses."""
        pass

    def _start_prefetch(self, layer_idx: int) -> None:  # noqa: B027
        """Start layer prefetch. Override in subclasses."""
        pass

    def _start_deferred_prefetch(self) -> None:  # noqa: B027
        """Start the static deferred wraparound prefetch. Override in
        backends that emit one (e.g., `PrefetchDeferOffloader`)."""
        pass


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance
    if isinstance(instance, NoopOffloader):
        logger.debug_once(
            "Offloader set to NoopOffloader (no offloading).", scope="local"
        )
    else:
        logger.info_once("Offloader set to %s", type(instance).__name__, scope="local")


def create_offloader(offload_config: "OffloadConfig") -> BaseOffloader:
    """Create an offloader based on the offload configuration.

    Uses the explicit ``offload_backend`` selector.  When set to ``"auto"``,
    selects prefetch if ``offload_group_size > 0``, UVA if
    ``cpu_offload_gb > 0``, otherwise noop. ``"cots"`` is never auto-selected
    and must be requested explicitly.
    """
    from vllm.model_executor.offloader.cots import CotsOffloader
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from vllm.model_executor.offloader.prefetch_defer import PrefetchDeferOffloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch
    cots = offload_config.cots

    if backend == "auto":
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
        elif uva.cpu_offload_gb > 0:
            backend = "uva"
        else:
            return NoopOffloader()

    if backend == "prefetch":
        return PrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    elif backend == "prefetch_defer":
        return PrefetchDeferOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            dry_run=prefetch.dry_run,
            mode="cpu",
        )
    elif backend == "uva":
        return UVAOffloader(
            cpu_offload_max_bytes=int(uva.cpu_offload_gb * 1024**3),
            cpu_offload_params=uva.cpu_offload_params,
        )
    elif backend == "cots":
        # Phase 2-only hybrid KV uses offload_backend="cots" to select the
        # hybrid KV runtime, but with zero weight placement there is no Phase 1
        # weight work to install or dispatch. Keep that path as a true no-op so
        # every forward does not pay CotsOffloader bucket bookkeeping.
        if cots.f_cpu_store == 0.0 and cots.f_prefetch == 0.0:
            return NoopOffloader()

        dispatch_table_factory = None
        if cots.dispatch_table is not None:
            configured_table = {
                int(bucket): (float(pair[0]), float(pair[1]))
                for bucket, pair in cots.dispatch_table.items()
            }

            def dispatch_table_factory(dispatch_buckets):
                missing = sorted(set(dispatch_buckets) - set(configured_table))
                if missing:
                    raise ValueError(
                        f"cots.dispatch_table is missing dispatch buckets: {missing}"
                    )
                return {
                    int(bucket): configured_table[int(bucket)]
                    for bucket in dispatch_buckets
                }

        module_dispatch_table_factory = None
        if cots.dispatch_table_by_module is not None:
            configured_by_module = {
                str(module): {
                    int(bucket): (float(pair[0]), float(pair[1]))
                    for bucket, pair in table.items()
                }
                for module, table in cots.dispatch_table_by_module.items()
            }

            def module_dispatch_table_factory(dispatch_buckets):
                resolved = {}
                for module, table in configured_by_module.items():
                    missing = sorted(set(dispatch_buckets) - set(table))
                    if missing:
                        raise ValueError(
                            "cots.dispatch_table_by_module "
                            f"{module!r} is missing dispatch buckets: {missing}"
                        )
                    resolved[module] = {
                        int(bucket): table[int(bucket)] for bucket in dispatch_buckets
                    }
                return resolved

        return CotsOffloader(
            config=cots,
            dispatch_table_factory=dispatch_table_factory,
            module_dispatch_table_factory=module_dispatch_table_factory,
        )
    else:
        return NoopOffloader()
