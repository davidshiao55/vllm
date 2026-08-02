# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, field_validator, model_validator

from vllm.config.hybrid import (
    DEFAULT_HYBRID_WEIGHT_MODULES,
    HybridWeightModule,
    normalize_hybrid_weight_modules,
)
from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch", "hybrid"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class HybridOffloadConfig:
    """Configuration for Hybrid collaborative CPU-GPU offloading (thesis backend).

    Splits selected decoder sub-modules along their Hybrid policy axis so a
    fraction `f_cpu_store` of the bytes lives in pinned CPU memory and is
    GEMM'd on the CPU each forward pass, in parallel with the GPU's compute on
    the GPU-resident slice. Activation returns from CPU use an SM-issued UVA
    copy kernel that bypasses the H2D copy engine. The production module set
    covers WQKV, MLP, and WO with a single storage fraction and fixed snapping
    policy.

    The full system design and evaluation contract live in the parent PTT
    artifact's retained thesis and evaluation documentation.
    """

    f_cpu_store: float = Field(default=0.0, ge=0.0, le=1.0)
    """Fraction of enabled Hybrid weight bytes resident on CPU. Single uniform
    scalar applied to the selected module set (default WQKV / MLP1 / MLP2 /
    WO). The matched-index invariant between MLP1 col-parallel and
    MLP2 row-parallel is automatic under uniform dispatch. Default 0.0 means no
    offload. A typical measured thesis value at 7B B=1 decode is ~0.09."""

    f_prefetch: float = Field(default=0.0, ge=0.0, le=1.0)
    """Manual fallback for the layer-ahead weight-prefetch fraction. Only
    consulted when the offloader is constructed without a
    `dispatch_table`/factory; the Planner's table output overrides this
    value entirely. Constraint: f_prefetch <= f_cpu_store (prefetch consumes
    CPU-stored bytes; the f_cpu_compute = f_cpu_store - f_prefetch portion is
    CPU-computed). Default 0.0 keeps the CPU-stored slice CPU-computed."""

    dispatch_table: dict[int, tuple[float, float]] | None = Field(default=None)
    """Optional engine-local Hybrid compute dispatch table emitted by the
    Planner. Keys are vLLM `BatchDescriptor.num_tokens` bucket values. Values
    are `(f_cpu_compute, f_prefetch_compute)` for that bucket. When set, this
    table overrides the uniform `f_prefetch` fallback above. Runtime validates
    that all Hybrid dispatch buckets are present before installing slabs. In
    graph mode, every CUDA graph capture bucket must also have a matching
    dispatch row.

    This is an engine-local interface: FastTTS/global planning decides the
    storage budget and may provide a table; vLLM still owns snapping, bucket
    validation, and tensor geometry."""

    weight_modules: set[HybridWeightModule | str] = Field(
        default_factory=lambda: set(DEFAULT_HYBRID_WEIGHT_MODULES)
    )
    """Hybrid weight sub-modules to offload. Valid entries:
    * `"qkv"`: WQKV output split using the WQKV-specific KV-biased picker.
    * `"mlp"`: fused MLP block (`gate_up_proj` col split + `down_proj` row
      split) with matched intermediate indices.
    * `"wo"`: WO (`o_proj`) dense output split using a larger production snap
      quantum so the first CPU-compute slice amortizes WO's extra per-layer
      task/sync cost."""

    cpu_dtype: Literal["bfloat16"] = "bfloat16"
    """CPU weight dtype. Locked to BF16 because the evaluated oneDNN F.linear
    path is optimized for BF16, while the corresponding torch.mm path does not
    provide a viable small-batch fallback on the target CPU."""

    cpu_num_threads: int = Field(default=16, ge=1)
    """PyTorch CPU intra-op thread count. Scalar fallback when
    `cpu_num_threads_by_bucket` is unset. See `phase1a_findings.md §1.13b`."""

    cpu_num_threads_by_bucket: dict[int, int] | None = Field(default=None)
    """Per-`BatchDescriptor` thread count for the native CPU
    GEMM worker. Keys are Hybrid dispatch bucket values; values are >= 1. When
    unset, every bucket uses the scalar `cpu_num_threads`.

    This is a runtime policy hook, not an additional Planner search axis.
    The Profiler/Planner may provide the map explicitly, or derive it from
    the chosen weight dispatch table using a calibrated work-score policy
    such as `bucket * f_cpu_compute`. The Planner should model costs after
    this policy is applied rather than sweep thread count independently.

    The native C++ worker applies this per slab, so thread policy stays local
    to Hybrid weight work instead of mutating process-wide PyTorch state."""

    cpu_worker_affinity: list[int] | None = Field(default=None)
    """Optional CPU affinity mask for the native
    runner's TaskQueue worker thread. List of CPU IDs to pin the
    worker to, or None (default) for no opinion. The C++ implementation
    intersects this with the process's existing `sched_getaffinity` mask
    and warns-and-skips on empty intersection.

    Recommended on i9-14900KF: P-cores 1..7 (i.e., `[1, 2, 3, 4, 5, 6,
    7]`) — keeps the worker off P-core 0 where the main thread / CUDA
    dispatch / kernel tend to land. Hardware-specific; left as None by
    default so we don't bake i9-14900KF assumptions into the config.

    Consulted by the native Hybrid worker."""

    dry_run: bool = Field(default=False)
    """Diagnostic: install Hybrid wrappers and preserve bucket/slot/graph
    control flow, but skip active offloaded work from both paths. CPU-compute
    contribution is omitted; prefetch H2D and prefetched-slice GPU compute are
    omitted. Token output is garbage; tensor shapes remain valid. Permanent
    diagnostic for measuring the Hybrid control-plane floor."""

    @field_validator("weight_modules", mode="before")
    @classmethod
    def normalize_weight_modules_field(cls, value: object) -> set[str]:
        return normalize_hybrid_weight_modules(value)

    @model_validator(mode="after")
    def validate_hybrid_config(self) -> "HybridOffloadConfig":
        """Validate Hybrid storage/dispatch invariants."""
        self.weight_modules = normalize_hybrid_weight_modules(self.weight_modules)
        if self.f_prefetch > self.f_cpu_store:
            raise ValueError(
                f"hybrid.f_prefetch ({self.f_prefetch}) must be <= "
                f"hybrid.f_cpu_store ({self.f_cpu_store}); prefetch consumes "
                "CPU-stored bytes."
            )

        def validate_table(label: str, table: dict[int, tuple[float, float]]) -> None:
            for bucket, entry in table.items():
                if bucket <= 0:
                    raise ValueError(
                        f"{label} bucket keys must be positive; got {bucket}"
                    )
                if len(entry) != 2:
                    raise ValueError(
                        f"{label} values must be (f_cpu_compute, f_prefetch_compute)"
                    )
                f_cpu_compute, f_prefetch_compute = entry
                if f_cpu_compute < 0 or f_prefetch_compute < 0:
                    raise ValueError(
                        f"{label} fractions must be non-negative; "
                        f"got {entry} for bucket {bucket}"
                    )
                total = f_cpu_compute + f_prefetch_compute
                if total > self.f_cpu_store + 1e-9:
                    raise ValueError(
                        f"{label} entry exceeds f_cpu_store: "
                        f"bucket={bucket}, entry={entry}, "
                        f"f_cpu_store={self.f_cpu_store}"
                    )
                if total < self.f_cpu_store - 1e-9:
                    raise ValueError(
                        f"{label} entry must sum to f_cpu_store: "
                        f"bucket={bucket}, entry={entry}, "
                        f"f_cpu_store={self.f_cpu_store}"
                    )

        if self.dispatch_table is not None:
            validate_table("hybrid.dispatch_table", self.dispatch_table)

        return self


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Stock async prefetch with group-based layer offloading.
    - "hybrid": Collaborative CPU-GPU offloading (thesis backend). Must be set
      explicitly; "auto" never selects "hybrid".
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    hybrid: HybridOffloadConfig = Field(default_factory=HybridOffloadConfig)
    """Parameters for Hybrid collaborative CPU-GPU offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        prefetch_backend = self.offload_backend == "prefetch"
        if prefetch_backend or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        hybrid_active = self.hybrid.f_cpu_store > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif prefetch_backend and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend="
                f"'{self.offload_backend}'. UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        if self.offload_backend == "hybrid":
            if uva_active:
                raise ValueError(
                    "offload_backend='hybrid' is incompatible with non-zero "
                    "uva.cpu_offload_gb. Disable UVA when using hybrid."
                )
            if prefetch_active:
                raise ValueError(
                    "offload_backend='hybrid' is incompatible with non-zero "
                    "prefetch.offload_group_size. Disable prefetch offload "
                    "when using hybrid."
                )
            if self.hybrid.f_prefetch > self.hybrid.f_cpu_store:
                raise ValueError(
                    f"hybrid.f_prefetch ({self.hybrid.f_prefetch}) must be <= "
                    f"hybrid.f_cpu_store ({self.hybrid.f_cpu_store}); prefetch "
                    f"consumes CPU-stored bytes."
                )
        elif hybrid_active and self.offload_backend != "hybrid":
            warnings.warn(
                "Hybrid settings are set but offload_backend is "
                f"'{self.offload_backend}', not 'hybrid'. hybrid settings will "
                "be ignored. Pass --offload-backend hybrid to enable.",
                stacklevel=2,
            )
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
