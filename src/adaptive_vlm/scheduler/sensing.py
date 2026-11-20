"""Live state sensing for the G5 adaptive policy (amendment A4).

The adaptive policy may not read the experiment's load labels. Before each
decision it senses:

- **cloud load** from the server's own GPU telemetry, which the health
  response carries (the server's cached background sample): C2 when the GPU
  power is at or above a threshold fixed on training data;
- **edge load** from a short timing probe on the edge GPU: L1 when the probe
  takes at least a threshold fixed by the pilot rule;
- **network** uplink goodput and RTT from the latest periodic active probe
  (the pre-segment measurement), never from the experiment's nominal values.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.timing import perf_counter_ns


@dataclass(frozen=True)
class SensingConfig:
    cloud_power_threshold_w: float
    edge_probe_threshold_ms: float
    edge_probe_matrix: int = 2048
    edge_probe_repetitions: int = 5
    # Amendment A4a: synchronise after every multiply, so each repetition can
    # wait behind a competing kernel the way the model's many short kernels do.
    edge_probe_synchronize_each: bool = False
    # Amendment A4b: cloud state comes from a background sampler, never from a
    # probe on the decision path.
    cloud_poll_interval_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("cloud_power_threshold_w", "edge_probe_threshold_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.cloud_poll_interval_s)
            or self.cloud_poll_interval_s <= 0
        ):
            raise ValueError("cloud poll interval must be finite and positive")
        if self.edge_probe_matrix < 16 or self.edge_probe_repetitions < 1:
            raise ValueError("edge probe size and repetitions must be positive")

    @classmethod
    def load(cls, path: Path) -> tuple[SensingConfig, str]:
        content = path.read_bytes()
        raw = json.loads(content)
        if raw.get("schema") != "g5-sensing":
            raise ValueError("not a G5 sensing configuration")
        fields = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**fields), hashlib.sha256(content).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensedState:
    edge_load: str
    cloud_load: str
    edge_probe_ms: float
    cloud_power_w: float | None
    cloud_utilization_percent: float | None
    cloud_state_age_s: float | None
    cloud_state_source: str
    uplink_bps: float
    rtt_ms: float
    sensing_ms: float

    def to_metrics(self) -> dict[str, Any]:
        return {f"adaptive_sensed_{k}": v for k, v in asdict(self).items()}


def classify_cloud(
    state: Mapping[str, Any] | None, config: SensingConfig
) -> tuple[str, float | None, float | None, float | None, str]:
    """C2 iff the reported GPU power reaches the threshold.

    Without a usable power reading the cloud is taken as unloaded (C0) and the
    source says so; the analysis counts these cases.
    """

    if not state:
        return "C0", None, None, None, "missing"
    power = state.get("gpu_power_w")
    utilization = state.get("gpu_utilization_percent")
    age = state.get("sample_age_s")
    number = (int, float)
    power_value = float(power) if isinstance(power, number) else None
    util_value = float(utilization) if isinstance(utilization, number) else None
    age_value = float(age) if isinstance(age, number) else None
    if power_value is None or not math.isfinite(power_value):
        return "C0", None, util_value, age_value, "no_power_reading"
    level = "C2" if power_value >= config.cloud_power_threshold_w else "C0"
    return level, power_value, util_value, age_value, "health_server_state"


def classify_edge(probe_ms: float, config: SensingConfig) -> str:
    return "L1" if probe_ms >= config.edge_probe_threshold_ms else "L0"


class Sensor:
    """One fresh reading of every input before a decision."""

    def __init__(
        self,
        config: SensingConfig,
        *,
        edge_probe: Callable[[], float],
        cloud_state: Callable[[], Mapping[str, Any] | None],
        network: tuple[float, float],
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        uplink, rtt = network
        if not (
            math.isfinite(uplink) and uplink > 0 and math.isfinite(rtt) and rtt >= 0
        ):
            raise ValueError("network probe values must be finite and positive")
        self.config = config
        self.edge_probe = edge_probe
        self.cloud_state = cloud_state
        self.network = (float(uplink), float(rtt))
        self.clock_ns = clock_ns

    def sense(self) -> SensedState:
        started = self.clock_ns()
        probe_ms = float(self.edge_probe())
        cloud, power, utilization, age, source = classify_cloud(
            self.cloud_state(), self.config
        )
        return SensedState(
            edge_load=classify_edge(probe_ms, self.config),
            cloud_load=cloud,
            edge_probe_ms=probe_ms,
            cloud_power_w=power,
            cloud_utilization_percent=utilization,
            cloud_state_age_s=age,
            cloud_state_source=source,
            uplink_bps=self.network[0],
            rtt_ms=self.network[1],
            sensing_ms=(self.clock_ns() - started) / 1_000_000,
        )


class BackgroundCloudState:
    """The newest cloud state, kept off the decision path (amendment A4b).

    A daemon thread polls ``poll`` (the server's health response) every
    ``interval_s``; ``observe`` also takes the ``server_state_*`` fields that
    every remote response carries. ``latest`` returns the newer of the two with
    its age: the server's own sample age plus the time since it arrived.
    """

    def __init__(
        self,
        poll: Callable[[], Mapping[str, Any] | None],
        interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.poll = poll
        self.interval_s = interval_s
        self.clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._received = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.poll_errors = 0

    def _store(self, state: Mapping[str, Any] | None) -> None:
        if state:
            with self._lock:
                self._state, self._received = dict(state), self.clock()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._store(self.poll())
            except Exception:  # a failed poll keeps the last sample, ageing
                self.poll_errors += 1
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._store(self.poll())  # one synchronous sample before the first request
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def observe(self, metrics: Mapping[str, Any]) -> None:
        state = {
            key.removeprefix("server_state_"): value
            for key, value in metrics.items()
            if key.startswith("server_state_")
        }
        self._store(state)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if self._state is None:
                return None
            state = dict(self._state)
            waited = self.clock() - self._received
        base = state.get("sample_age_s")
        state["sample_age_s"] = (
            float(base) + waited if isinstance(base, (int, float)) else waited
        )
        return state


def torch_edge_probe(
    device: str, matrix: int, repetitions: int, synchronize_each: bool = False
) -> Callable[[], float]:
    """A fixed bf16 matrix-multiply burst on the edge GPU, timed with syncs.

    It shares the GPU with any edge load, so contention lengthens it. With
    ``synchronize_each`` every multiply is followed by a device synchronise.
    """

    torch: Any = importlib.import_module("torch")

    a = torch.randn(matrix, matrix, device=device, dtype=torch.bfloat16)
    b = torch.randn(matrix, matrix, device=device, dtype=torch.bfloat16)

    def probe() -> float:
        torch.cuda.synchronize(device)
        started = perf_counter_ns()
        for _ in range(repetitions):
            torch.matmul(a, b)
            if synchronize_each:
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        return (perf_counter_ns() - started) / 1_000_000

    probe()  # warm the kernels once, outside any measurement
    return probe


def network_from_probe(path: Path) -> tuple[float, float]:
    """Uplink goodput (bit/s) and echo RTT (ms) from a periodic probe file."""

    probe = json.loads(path.read_text(encoding="utf-8"))
    return float(probe["edge_to_cloud_receiver_bps"]), float(
        probe["echo_median_rtt_ms"]
    )
