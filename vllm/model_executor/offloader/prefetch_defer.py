# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thesis deferred-wraparound variant of the native prefetch offloader."""

from collections.abc import Generator

import torch
import torch.nn as nn

# Import prefetch_defer_ops to register custom ops at module load time.
import vllm.model_executor.offloader.prefetch_defer_ops  # noqa: F401
from vllm.logger import init_logger
from vllm.model_executor.offloader.prefetch import (
    ParamInfo,
    PrefetchOffloader,
    StaticBufferPool,
    _ModuleOffloader,
)
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)


def _start_deferred_prefetch_pre_hook(module: nn.Module, args, kwargs) -> None:
    """Graph-safe first-decoder hook for deferred wrap-around prefetch."""
    del module
    anchor = args[0] if args else kwargs.get("hidden_states")
    if anchor is None and kwargs:
        anchor = next(iter(kwargs.values()))
    if anchor is not None:
        torch.ops.vllm.start_deferred_prefetch(anchor)


class PrefetchDeferOffloader(PrefetchOffloader):
    """Thesis native-prefetch baseline with static deferred wrap-around.

    This keeps the stock `PrefetchOffloader` behavior clean while preserving
    the CE0-contention optimization used by the thesis benchmarks.
    """

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        dry_run: bool = False,
        mode: str = "cpu",
    ):
        super().__init__(
            group_size=group_size,
            num_in_group=num_in_group,
            prefetch_step=prefetch_step,
            offload_params=offload_params,
            mode=mode,
        )
        self.dry_run = dry_run
        self.first_decoder_module: nn.Module | None = None
        self.deferred_wraparound_index: int | None = None

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules and install deferred-wraparound scheduling."""
        assert len(self.module_offloaders) == 0, (
            "wrap_modules should only be called once"
        )

        all_modules = []
        offload_modules = []
        module_offloader_cls = (
            _DryRunModuleOffloader if self.dry_run else _ModuleOffloader
        )

        for module_index, module in enumerate(modules_generator):
            all_modules.append(module)

            if module_index % self.group_size >= self.group_size - self.num_in_group:
                if self.offload_params:
                    whitelist = [
                        name
                        for name, _ in module.named_parameters()
                        if any(f".{p}." in f".{name}." for p in self.offload_params)
                    ]
                else:
                    whitelist = [name for name, _ in module.named_parameters()]

                if not whitelist:
                    continue

                offload_modules.append(module)
                self.module_offloaders.append(
                    module_offloader_cls(
                        mode=self.mode,
                        module=module,
                        copy_stream=self.copy_stream,
                        whitelist_param_names=whitelist,
                        layer_idx=len(self.module_offloaders),
                    )
                )

        for index, module in enumerate(offload_modules):
            self._hook_module_forward(index, module)

        if self.module_offloaders and self.prefetch_step > 0:
            n_offloaders = len(self.module_offloaders)
            self.deferred_wraparound_index = (
                n_offloaders - 1 + self.prefetch_step
            ) % n_offloaders

        # First-decoder pre-hook fires at the start of each model forward,
        # after vLLM input-prep H2Ds have queued. The custom op keeps this
        # path CUDA-graph replayable.
        if self.deferred_wraparound_index is not None and all_modules:
            self.first_decoder_module = all_modules[0]
            self.first_decoder_module.register_forward_pre_hook(
                _start_deferred_prefetch_pre_hook,
                with_kwargs=True,
            )

        return all_modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        """Hook module's forward with last-wraparound deferral."""
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward

            input_tensor = args[0] if args else kwargs.get("hidden_states")
            torch.ops.vllm.wait_prefetch(input_tensor, index)

            output = original_forward(*args, **kwargs)

            n_offloaders = len(self.module_offloaders)
            raw_next = index + self.prefetch_step
            next_index = raw_next % n_offloaders
            # Defer only the final wrap-around copy. Earlier wraparounds at
            # K > 1 still overlap with tail compute in this iteration.
            if raw_next >= n_offloaders and index == n_offloaders - 1:
                pass
            elif isinstance(output, tuple):
                torch.ops.vllm.start_prefetch(output[0], next_index)
            else:
                torch.ops.vllm.start_prefetch(output, next_index)

            module.forward = forward
            return output

        module.forward = forward

    def _start_deferred_prefetch(self):
        """Called by custom op - start the static deferred wrap-around copy."""
        if self.deferred_wraparound_index is not None:
            self._start_prefetch(self.deferred_wraparound_index)

    def post_init(self):
        """Allocate buffers and start initial prefetches except deferred target."""
        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()

        param_infos: list[ParamInfo] = []
        device: torch.device | None = None

        for offloader in self.module_offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device

        if device is None:
            return

        self.buffer_pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )

        for idx, offloader in enumerate(self.module_offloaders):
            slot_idx = idx % self.prefetch_step
            offloader.assign_buffer_slot(self.buffer_pool, slot_idx)

        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        logger.info_once(
            f"[PrefetchDeferOffloader] Initialized "
            f"{len(self.module_offloaders)} modules. "
            f"Total GPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static buffer pool: {self.buffer_pool.total_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode}, "
            f"deferred_wraparound_index={self.deferred_wraparound_index}, "
            f"dry_run={self.dry_run})"
        )

        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            if i == self.deferred_wraparound_index:
                continue
            self.module_offloaders[i].start_onload_to_static()


class _DryRunModuleOffloader(_ModuleOffloader):
    """Diagnostic offloader that keeps stream/event cost but skips H2D."""

    def start_onload_to_static(self):
        """Start async bookkeeping without copying CPU storage to GPU."""
        assert self._buffer_pool is not None, "Buffer pool not assigned"

        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()

        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(self.copy_stream):
            for name, offloader in self._param_offloaders.items():
                cpu_storage = offloader._cpu_storage
                gpu_buffer = offloader._gpu_buffer
                assert cpu_storage is not None, "CPU storage not initialized"
                assert gpu_buffer is not None, "GPU buffer not assigned"
                assert not is_pin_memory_available() or cpu_storage.is_pinned(), (
                    f"CPU storage for {name} is not pinned! "
                    "non_blocking=True H2D copy from non-pinned memory "
                    "causes stream synchronization that breaks "
                    "event-based fork synchronization."
                )

        self._copy_done_event.record(self.copy_stream)
        self._event_valid_for_eager = not torch.cuda.is_current_stream_capturing()
