"""Device residency of Moondream2 stage weights (resident or swapped)."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal

from adaptive_vlm.timing import perf_counter_ns

WeightResidency = Literal["resident", "swap"]


WEIGHT_RESIDENCIES: tuple[WeightResidency, ...] = ("resident", "swap")


_SWAP_STAGES = ("vision", "text")


class _StageSwapper:
    """Keep read-only stage weights in host memory with one stage on the device.

    Each parameter's host tensor is its master copy. Activating a stage copies its
    weights to the device; evicting simply points the parameter back at the
    master, so no device-to-host copy is needed and arithmetic is unchanged.
    """

    def __init__(
        self,
        stages: Mapping[str, Sequence[Any]],
        *,
        load: Callable[[Any], Any],
        synchronize: Callable[[], None],
        context: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        if set(stages) != set(_SWAP_STAGES):
            raise ValueError(f"swap stages must be exactly {_SWAP_STAGES}")
        self._stages = {
            name: tuple((parameter, parameter.data) for parameter in parameters)
            for name, parameters in stages.items()
        }
        self._load = load
        self._synchronize = synchronize
        self._context = context
        self._active: str | None = None

    @property
    def active_stage(self) -> str | None:
        return self._active

    def activate(self, name: str) -> float:
        """Make one stage device-resident; return the elapsed swap milliseconds."""

        if name not in self._stages:
            raise ValueError(f"unknown swap stage: {name}")
        if self._active == name:
            return 0.0
        started_ns = perf_counter_ns()
        with self._context():
            if self._active is not None:
                self._restore(self._active)
                self._active = None
            try:
                for parameter, master in self._stages[name]:
                    parameter.data = self._load(master)
                self._synchronize()
            except BaseException:
                self._restore(name)
                raise
        self._active = name
        return (perf_counter_ns() - started_ns) / 1_000_000

    def _restore(self, name: str) -> None:
        for parameter, master in self._stages[name]:
            parameter.data = master


def _validate_weight_residency(
    residency: str, components: str, vision_device: str, text_device: str
) -> None:
    if residency not in WEIGHT_RESIDENCIES:
        raise ValueError(f"unsupported weight residency: {residency}")
    if residency == "resident":
        return
    if components != "full":
        raise ValueError("swap residency requires the full model")
    if vision_device != text_device:
        raise ValueError("swap residency requires one shared vision/text device")
    if not text_device.startswith("cuda"):
        raise ValueError("swap residency requires a CUDA device")


def _enable_swap_residency(core: Any, torch: Any, device: str) -> _StageSwapper:
    """Pin stage weights on the host and keep only small state on the device."""

    resolved = torch.device(device)
    # Expandable segments let freed vision blocks be reused for text weights;
    # without them fragmentation pushes the text stage past dedicated VRAM.
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    # pos_emb defines the pinned model's device for every stage, so it stays.
    pos_emb = core.vision.pos_emb
    pos_emb.data = pos_emb.data.to(resolved)
    stages: dict[str, list[Any]] = {}
    for name in _SWAP_STAGES:
        module = getattr(core, name)
        parameters = [p for p in module.parameters() if p is not pos_emb]
        for parameter in parameters:
            parameter.data = parameter.data.pin_memory()
        stages[name] = parameters
        # KV caches and rotary tables are small and must persist across stages.
        for submodule in module.modules():
            for buffer_name, buffer in list(submodule._buffers.items()):
                if buffer is not None:
                    submodule._buffers[buffer_name] = buffer.to(resolved)
    core.attn_mask = core.attn_mask.to(resolved)
    gc.collect()
    return _StageSwapper(
        stages,
        load=lambda tensor: tensor.to(resolved, non_blocking=True),
        synchronize=lambda: torch.cuda.synchronize(resolved),
        context=lambda: torch.inference_mode(False),
    )
