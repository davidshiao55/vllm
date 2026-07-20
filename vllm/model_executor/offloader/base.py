# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import OffloadConfig
    from vllm.forward_context import BatchDescriptor

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

    def on_dispatch(  # noqa: B027
        self,
        batch_descriptor: "BatchDescriptor",
        num_tokens_unpadded: int,
    ) -> None:
        """Publish per-forward state to offloaders that require it."""
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
    ``cpu_offload_gb > 0``, otherwise noop. ``"hybrid"`` is never auto-selected
    and must be requested explicitly.
    """
    from vllm.model_executor.offloader.hybrid import HybridOffloader
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from vllm.model_executor.offloader.prefetch_defer import PrefetchDeferOffloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch
    hybrid = offload_config.hybrid

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
    elif backend == "hybrid":
        # With zero weight placement there is no Hybrid weight work to install or
        # dispatch. Keep that path as a true no-op so every forward does not pay
        # HybridOffloader bucket bookkeeping.
        if hybrid.f_cpu_store == 0.0 and hybrid.f_prefetch == 0.0:
            return NoopOffloader()

        dispatch_table_factory = None
        if hybrid.dispatch_table is not None:
            configured_table = {
                int(bucket): (float(pair[0]), float(pair[1]))
                for bucket, pair in hybrid.dispatch_table.items()
            }

            def dispatch_table_factory(dispatch_buckets):
                missing = sorted(set(dispatch_buckets) - set(configured_table))
                if missing:
                    raise ValueError(
                        f"hybrid.dispatch_table is missing dispatch buckets: {missing}"
                    )
                return {
                    int(bucket): configured_table[int(bucket)]
                    for bucket in dispatch_buckets
                }

        return HybridOffloader(
            config=hybrid,
            dispatch_table_factory=dispatch_table_factory,
        )
    else:
        return NoopOffloader()
