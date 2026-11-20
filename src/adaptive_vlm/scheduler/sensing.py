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

    def __post_init__(self) -> None:
        for name in ("cloud_power_threshold_w", "edge_probe_threshold_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
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
