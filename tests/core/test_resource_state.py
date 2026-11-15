"""NVML reader, host memory, Windows counter sampler and cached sampler."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.resource_state import (
    CachedSampler,
    NvmlReader,
    NvmlUnavailable,
    WindowsGpuCounterSampler,
    combine,
    host_memory_snapshot,
    select_cuda_adapter,
)


class FakeNvml:
    """ctypes-style NVML stand-in: functions fill byref() outputs, 0 = success."""

    def __init__(self, *, unsupported: set[str] | None = None, init: int = 0) -> None:
        self.unsupported = unsupported or set()
        self.init = init

    def nvmlInit_v2(self) -> int:
        return self.init

    def nvmlDeviceGetHandleByIndex_v2(self, index: Any, handle: Any) -> int:
        return 0 if index.value == 0 else 1

    def _set(self, name: str, out: Any, value: Any) -> int:
        if name in self.unsupported:
            return 3
        out._obj.value = value
        return 0

    def nvmlDeviceGetTemperature(self, _: Any, sensor: Any, out: Any) -> int:
        return self._set("temperature", out, 61)

    def nvmlDeviceGetPowerUsage(self, _: Any, out: Any) -> int:
        return self._set("power", out, 42500)

    def nvmlDeviceGetClockInfo(self, _: Any, kind: Any, out: Any) -> int:
        return self._set("clock", out, {0: 1800, 1: 1800, 2: 6001}[kind.value])

    def nvmlDeviceGetUtilizationRates(self, _: Any, out: Any) -> int:
        if "utilization" in self.unsupported:
            return 3
        out._obj.gpu, out._obj.memory = 97, 40
        return 0

    def nvmlDeviceGetMemoryInfo(self, _: Any, out: Any) -> int:
        out._obj.total, out._obj.free, out._obj.used = 4096 << 20, 800 << 20, 3296 << 20
        return 0

    def nvmlDeviceGetPerformanceState(self, _: Any, out: Any) -> int:
        return self._set("pstate", out, 0)

    def nvmlDeviceGetCurrentClocksEventReasons(self, _: Any, out: Any) -> int:
        return self._set("reasons", out, 4)


def test_nvml_reader_reports_every_field() -> None:
    state = NvmlReader(library=FakeNvml()).snapshot()
    assert state == {
        "gpu_index": 0,
        "gpu_temperature_c": 61,
        "gpu_power_w": 42.5,
        "gpu_clock_graphics_mhz": 1800,
        "gpu_clock_sm_mhz": 1800,
        "gpu_clock_memory_mhz": 6001,
        "gpu_utilization_percent": 97,
        "gpu_memory_utilization_percent": 40,
        "gpu_memory_used_mib": 3296.0,
        "gpu_memory_free_mib": 800.0,
        "gpu_performance_state": 0,
        "gpu_clock_event_reasons": 4,
    }


def test_nvml_reader_records_unsupported_queries_as_none() -> None:
    state = NvmlReader(
        library=FakeNvml(unsupported={"power", "utilization", "reasons"})
    ).snapshot()
    assert state["gpu_power_w"] is None
    assert state["gpu_utilization_percent"] is None
    assert state["gpu_clock_event_reasons"] is None
    assert state["gpu_temperature_c"] == 61


def test_nvml_reader_fails_loudly_without_a_device() -> None:
    with pytest.raises(NvmlUnavailable, match="nvmlInit"):
        NvmlReader(library=FakeNvml(init=1))
    with pytest.raises(NvmlUnavailable, match="index 1"):
        NvmlReader(1, library=FakeNvml())


def test_host_memory_snapshot(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 8000000 kB\nMemAvailable: 6144000 kB\n"
        "SwapTotal: 2048000 kB\nSwapFree: 1024000 kB\nHugePages_Total: 0\n"
    )
    assert host_memory_snapshot(str(meminfo)) == {
        "host_memory_available_mib": 6000.0,
        "host_swap_used_mib": 1000.0,
    }
    assert host_memory_snapshot(str(tmp_path / "missing")) == {}


def test_cuda_adapter_is_the_one_with_dedicated_usage() -> None:
    adapters = {
        "igpu": {"dedicated_bytes": 0.0, "shared_bytes": 4.6e8},
        "nvidia": {"dedicated_bytes": 3.2e9, "shared_bytes": 2.6e5},
    }
    assert select_cuda_adapter(adapters) == "nvidia"
    assert select_cuda_adapter({"igpu": {"dedicated_bytes": 0.0}}) is None


def test_windows_sampler_reads_streamed_samples(tmp_path: Path) -> None:
    script = tmp_path / "emit.py"
    sample = {
        "unix_s": 1000.0,
        "adapters": {
            "igpu": {"dedicated_bytes": 0, "shared_bytes": 465_772_544},
            "nvidia": {"dedicated_bytes": 3_355_443_200, "shared_bytes": 838_860_800},
        },
    }
    script.write_text(
        "import json, sys, time\n"
        "print('not json', flush=True)\n"
        f"print(json.dumps({sample!r}), flush=True)\n"
        "time.sleep(30)\n"
    )
    sampler = WindowsGpuCounterSampler(
        command=[sys.executable, str(script)], clock=lambda: 1002.5
    )
    assert sampler.snapshot() == {"windows_gpu_sample_age_s": None}
    sampler.start()
    try:
        assert sampler.wait_for_sample(timeout=20)
        assert sampler.snapshot() == {
            "windows_gpu_sample_age_s": 2.5,
            "windows_gpu_adapter": "nvidia",
            "windows_gpu_dedicated_mib": 3200.0,
            "windows_gpu_shared_mib": 800.0,
        }
        assert sampler.samples == 1
    finally:
        sampler.stop()


def test_cached_sampler_returns_latest_with_age_and_survives_errors() -> None:
    now = [10.0]
    values = iter([{"a": 1}, RuntimeError("probe broke")])

    def read() -> dict[str, Any]:
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    sampler = CachedSampler(read, period_s=60, clock=lambda: now[0])
    assert sampler.latest() == {"sample_age_s": None}
    sampler.poll_once()
    now[0] = 10.25
    assert sampler.latest() == {"a": 1, "sample_age_s": 0.25}
    sampler.poll_once()
    assert sampler.latest() == {"sampler_error": "RuntimeError", "sample_age_s": 0.0}
    with pytest.raises(ValueError, match="positive"):
        CachedSampler(read, period_s=0)


def test_cached_sampler_thread_starts_and_stops() -> None:
    sampler = CachedSampler(lambda: {"x": 1}, period_s=0.01)
    sampler.start()
    sampler.stop()
    assert sampler.latest()["x"] == 1


def test_combine_merges_sources_under_a_prefix() -> None:
    probe = combine(lambda: {"a": 1}, lambda: {"b": None}, prefix="edge_")
    assert probe() == {"edge_a": 1, "edge_b": None}
    assert json.dumps(probe())
