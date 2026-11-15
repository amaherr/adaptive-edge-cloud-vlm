"""Per-request resource-state snapshots for G2 v2 (protocol section 7).

Three sources, all read outside the timed region:

* ``NvmlReader``: GPU clocks, temperature, power, utilization, memory and
  clock-event reasons through the NVIDIA management library (ctypes, no extra
  dependency). Unsupported queries (common under WSL) are recorded as None.
* ``WindowsGpuCounterSampler``: the Windows ``GPU Adapter Memory`` counters,
  sampled continuously by ``powershell.exe`` because one read takes ~1.4 s.
  Shared usage on the CUDA adapter is what reveals a spill into system memory.
* ``CachedSampler``: polls a snapshot function in a background thread so a
  server can attach the latest state to each response without adding latency.

Snapshots are flat ``{name: scalar}`` mappings so they fit request metrics.
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from adaptive_vlm.contracts import MetricValue

Snapshot = dict[str, MetricValue]
_MIB = 1024 * 1024


class NvmlUnavailable(RuntimeError):
    """The NVIDIA management library could not be loaded or initialised."""


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlReader:
    """Read one GPU's state through NVML; each field is None if unsupported."""

    def __init__(self, index: int = 0, library: Any = None) -> None:
        if library is None:
            try:
                library = ctypes.CDLL("libnvidia-ml.so.1")
            except OSError as exc:
                raise NvmlUnavailable(f"cannot load NVML: {exc}") from exc
        self._nvml = library
        if self._nvml.nvmlInit_v2() != 0:
            raise NvmlUnavailable("nvmlInit_v2 failed")
        self._handle = ctypes.c_void_p()
        if self._nvml.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(index), ctypes.byref(self._handle)
        ):
            raise NvmlUnavailable(f"no NVML device at index {index}")
        self._index = index

    def _uint(self, function: str, *arguments: Any) -> int | None:
        value = ctypes.c_uint()
        call = getattr(self._nvml, function, None)
        if call is None or call(self._handle, *arguments, ctypes.byref(value)):
            return None
        return int(value.value)

    def snapshot(self) -> Snapshot:
        nvml = self._nvml
        power_mw = self._uint("nvmlDeviceGetPowerUsage")
        state: Snapshot = {
            "gpu_index": self._index,
            "gpu_temperature_c": self._uint(
                "nvmlDeviceGetTemperature", ctypes.c_int(0)
            ),
            "gpu_power_w": None if power_mw is None else power_mw / 1000,
            "gpu_clock_graphics_mhz": self._uint(
                "nvmlDeviceGetClockInfo", ctypes.c_int(0)
            ),
            "gpu_clock_sm_mhz": self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(1)),
            "gpu_clock_memory_mhz": self._uint(
                "nvmlDeviceGetClockInfo", ctypes.c_int(2)
            ),
        }
        utilization = _Utilization()
        call = getattr(nvml, "nvmlDeviceGetUtilizationRates", None)
        supported = call is not None and not call(
            self._handle, ctypes.byref(utilization)
        )
        state["gpu_utilization_percent"] = utilization.gpu if supported else None
        state["gpu_memory_utilization_percent"] = (
            utilization.memory if supported else None
        )
        memory = _Memory()
        call = getattr(nvml, "nvmlDeviceGetMemoryInfo", None)
        supported = call is not None and not call(self._handle, ctypes.byref(memory))
        state["gpu_memory_used_mib"] = memory.used / _MIB if supported else None
        state["gpu_memory_free_mib"] = memory.free / _MIB if supported else None
        performance = ctypes.c_int()
        call = getattr(nvml, "nvmlDeviceGetPerformanceState", None)
        state["gpu_performance_state"] = (
            int(performance.value)
            if call is not None and not call(self._handle, ctypes.byref(performance))
            else None
        )
        reasons = ctypes.c_ulonglong()
        call = getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or (
            getattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        )
        state["gpu_clock_event_reasons"] = (
            int(reasons.value)
            if call is not None and not call(self._handle, ctypes.byref(reasons))
            else None
        )
        return state


def host_memory_snapshot(path: str = "/proc/meminfo") -> Snapshot:
    """Linux available memory and swap use; empty where /proc is missing."""

    try:
        with open(path, encoding="utf-8") as handle:
            fields = {
                line.split(":")[0]: int(line.split()[1])
                for line in handle
                if line.split()[1:2] and line.split()[1].isdigit()
            }
    except OSError:
        return {}
    swap_used = fields.get("SwapTotal", 0) - fields.get("SwapFree", 0)
    return {
        "host_memory_available_mib": fields.get("MemAvailable", 0) / 1024,
        "host_swap_used_mib": swap_used / 1024,
    }


def select_cuda_adapter(adapters: Mapping[str, Mapping[str, float]]) -> str | None:
    """The CUDA adapter is the only one with dedicated usage (iGPU has none)."""

    in_use = {
        name: values.get("dedicated_bytes", 0.0)
        for name, values in adapters.items()
        if values.get("dedicated_bytes", 0.0) > 0
    }
    if not in_use:
        return None
    return max(in_use, key=lambda name: in_use[name])


_COUNTER_SCRIPT = r"""
$counters = @(
  '\GPU Adapter Memory(*)\Dedicated Usage',
  '\GPU Adapter Memory(*)\Shared Usage'
)
Get-Counter -Counter $counters -SampleInterval 1 -Continuous `
  -ErrorAction SilentlyContinue |
  ForEach-Object {
    $adapters = @{}
    foreach ($c in $_.CounterSamples) {
      $kind = 'shared_bytes'
      if ($c.Path -like '*dedicated usage') { $kind = 'dedicated_bytes' }
      $name = $c.InstanceName
      if (-not $adapters.ContainsKey($name)) { $adapters[$name] = @{} }
      $adapters[$name][$kind] = $c.CookedValue
    }
    $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $sample = @{ unix_s = $t; adapters = $adapters }
    [Console]::Out.WriteLine(($sample | ConvertTo-Json -Compress -Depth 4))
    [Console]::Out.Flush()
  }
"""


class WindowsGpuCounterSampler:
    """Continuously sample Windows GPU adapter memory through powershell.exe."""

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if command is None:
            executable = shutil.which("powershell.exe")
            if executable is None:
                raise FileNotFoundError("powershell.exe is not reachable from here")
            command = [executable, "-NoProfile", "-Command", _COUNTER_SCRIPT]
        self._command = command
        self._clock = clock
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self.samples = 0

    def start(self) -> None:
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample, dict) and isinstance(sample.get("adapters"), dict):
                with self._lock:
                    self._latest = sample
                    self.samples += 1

    def wait_for_sample(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> Snapshot:
        with self._lock:
            sample = self._latest
        if sample is None:
            return {"windows_gpu_sample_age_s": None}
        adapters: dict[str, dict[str, float]] = {
            str(name): {
                kind: float(value)
                for kind, value in values.items()
                if isinstance(value, (int, float))
            }
            for name, values in sample["adapters"].items()
            if isinstance(values, dict)
        }
        chosen = select_cuda_adapter(adapters)
        values = adapters.get(chosen, {}) if chosen else {}
        return {
            "windows_gpu_sample_age_s": self._clock() - float(sample["unix_s"]),
            "windows_gpu_adapter": chosen,
            "windows_gpu_dedicated_mib": (
                values["dedicated_bytes"] / _MIB
                if "dedicated_bytes" in values
                else None
            ),
            "windows_gpu_shared_mib": (
                values["shared_bytes"] / _MIB if "shared_bytes" in values else None
            ),
        }

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)


class CachedSampler:
    """Poll a snapshot function in the background; read the latest instantly."""

    def __init__(
        self,
        read: Callable[[], Snapshot],
        *,
        period_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if period_s <= 0:
            raise ValueError("sampling period must be positive")
        self._read = read
        self._period = period_s
        self._clock = clock
        self._latest: tuple[float, Snapshot] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def poll_once(self) -> None:
        try:
            value = self._read()
        except Exception as exc:  # A failing probe must never stop serving.
            value = {"sampler_error": type(exc).__name__}
        with self._lock:
            self._latest = (self._clock(), value)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self._period)

    def start(self) -> None:
        self.poll_once()
        self._thread.start()

    def latest(self) -> Snapshot:
        with self._lock:
            latest = self._latest
        if latest is None:
            return {"sample_age_s": None}
        taken, value = latest
        return {**value, "sample_age_s": self._clock() - taken}

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def combine(
    *sources: Callable[[], Snapshot], prefix: str = ""
) -> Callable[[], Snapshot]:
    """One probe that merges several snapshot sources under a name prefix."""

    def probe() -> Snapshot:
        merged: Snapshot = {}
        for source in sources:
            merged.update(source())
        return {f"{prefix}{key}": value for key, value in merged.items()}

    return probe
