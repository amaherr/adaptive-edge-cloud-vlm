"""Loading and validation for reproducible experiment configurations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


class ExecutionMode(StrEnum):
    EDGE = "edge"
    CLOUD = "cloud"
    DISTRIBUTED = "distributed"


class Precision(StrEnum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass(frozen=True)
class ModelConfig:
    adapter: str
    model_id: str
    release_tag: str | None
    revision: str
    artifact_sha256: str
    status: str
    precision: Precision
    preprocessing_backend: str
    trust_remote_code: bool
    max_new_tokens: int
    do_sample: bool
    temperature: float
    batch_size: int
    variant: str | None


@dataclass(frozen=True)
class HardwareConfig:
    host_id: str
    role: str
    status: str
    operating_system: str
    cpu: str
    ram_gib: float
    gpu: str | None
    gpu_vram_mib: int | None
    notes: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    manifest_path: Path
    revision: str


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    warmup_requests: int
    measured_repetitions: int


@dataclass(frozen=True)
class TransportConfig:
    host: str
    port: int
    timeout_seconds: float
    protocol_version: int


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    mode: ExecutionMode
    source_path: Path
    model: ModelConfig
    edge: HardwareConfig
    cloud: HardwareConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    transport: TransportConfig
    artifacts_dir: Path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved representation."""

        return _as_json_dict(asdict(self))


def model_contract_fingerprint(model: ModelConfig) -> str:
    """Hash every model setting that can affect preprocessing or generation."""

    contract = {
        "schema_version": 1,
        "adapter": model.adapter,
        "model_id": model.model_id,
        "release_tag": model.release_tag,
        "revision": model.revision,
        "artifact_sha256": model.artifact_sha256,
        "precision": model.precision.value,
        "preprocessing_backend": model.preprocessing_backend,
        "trust_remote_code": model.trust_remote_code,
        "generation": {
            "max_new_tokens": model.max_new_tokens,
            "do_sample": model.do_sample,
            "temperature": model.temperature,
            "batch_size": model.batch_size,
            "variant": model.variant,
        },
    }
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=StrEnum)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file and resolve all referenced configurations."""

    source_path = Path(path).expanduser().resolve()
    root = _load_mapping(source_path)
    _reject_unknown(
        root,
        {
            "schema_version",
            "name",
            "mode",
            "model_config",
            "edge_hardware_config",
            "cloud_hardware_config",
            "dataset",
            "runtime",
            "transport",
            "artifacts_dir",
        },
        "experiment",
    )

    schema_version = _required(root, "schema_version", int, "experiment")
    if schema_version != 1:
        raise ConfigError(f"Unsupported schema_version: {schema_version}")

    base_dir = source_path.parent
    model_path = _resolve_reference(
        base_dir, _required(root, "model_config", str, "experiment")
    )
    edge_path = _resolve_reference(
        base_dir, _required(root, "edge_hardware_config", str, "experiment")
    )
    cloud_path = _resolve_reference(
        base_dir, _required(root, "cloud_hardware_config", str, "experiment")
    )

    return ExperimentConfig(
        schema_version=schema_version,
        name=_non_empty(_required(root, "name", str, "experiment"), "name"),
        mode=_enum_value(
            ExecutionMode,
            _required(root, "mode", str, "experiment"),
            "mode",
        ),
        source_path=source_path,
        model=_parse_model(_load_mapping(model_path)),
        edge=_parse_hardware(_load_mapping(edge_path), "edge"),
        cloud=_parse_hardware(_load_mapping(cloud_path), "cloud"),
        dataset=_parse_dataset(
            _required_mapping(root, "dataset", "experiment"), base_dir
        ),
        runtime=_parse_runtime(_required_mapping(root, "runtime", "experiment")),
        transport=_parse_transport(
            _required_mapping(root, "transport", "experiment")
        ),
        artifacts_dir=_resolve_reference(
            base_dir, _required(root, "artifacts_dir", str, "experiment")
        ),
    )


def _parse_model(data: Mapping[str, Any]) -> ModelConfig:
    context = "model"
    _reject_unknown(
        data,
        {
            "adapter",
            "model_id",
            "release_tag",
            "revision",
            "artifact_sha256",
            "status",
            "precision",
            "preprocessing_backend",
            "trust_remote_code",
            "generation",
        },
        context,
    )
    generation = _required_mapping(data, "generation", context)
    _reject_unknown(
        generation,
        {"max_new_tokens", "do_sample", "temperature", "batch_size", "variant"},
        "model.generation",
    )

    do_sample = _required(generation, "do_sample", bool, "model.generation")
    temperature = _number(generation, "temperature", "model.generation")
    batch_size = _positive_int(generation, "batch_size", "model.generation")
    if batch_size != 1:
        raise ConfigError("model.generation.batch_size must be 1 for the baseline")
    if do_sample:
        raise ConfigError("model.generation.do_sample must be false for the baseline")
    if temperature != 0.0:
        raise ConfigError("model.generation.temperature must be 0 for greedy decoding")

    revision = _non_empty(
        _required(data, "revision", str, context), "model.revision"
    )
    if revision.lower() in {"main", "master", "latest"}:
        raise ConfigError("model.revision must be pinned, not a moving reference")
    artifact_sha256 = _required(data, "artifact_sha256", str, context).lower()
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise ConfigError("model.artifact_sha256 must be a 64-character hex digest")

    status = _required(data, "status", str, context)
    if status not in {"candidate", "approved", "test"}:
        raise ConfigError("model.status must be candidate, approved, or test")

    return ModelConfig(
        adapter=_non_empty(_required(data, "adapter", str, context), "model.adapter"),
        model_id=_non_empty(
            _required(data, "model_id", str, context), "model.model_id"
        ),
        release_tag=_optional(data, "release_tag", str, context),
        revision=revision,
        artifact_sha256=artifact_sha256,
        status=status,
        precision=_enum_value(
            Precision,
            _required(data, "precision", str, context),
            "model.precision",
        ),
        preprocessing_backend=_non_empty(
            _required(data, "preprocessing_backend", str, context),
            "model.preprocessing_backend",
        ),
        trust_remote_code=_required(data, "trust_remote_code", bool, context),
        max_new_tokens=_positive_int(
            generation, "max_new_tokens", "model.generation"
        ),
        do_sample=do_sample,
        temperature=temperature,
        batch_size=batch_size,
        variant=_optional(generation, "variant", str, "model.generation"),
    )


def _parse_hardware(data: Mapping[str, Any], expected_role: str) -> HardwareConfig:
    context = f"hardware[{expected_role}]"
    _reject_unknown(
        data,
        {
            "host_id",
            "role",
            "status",
            "operating_system",
            "cpu",
            "ram_gib",
            "gpu",
            "gpu_vram_mib",
            "notes",
        },
        context,
    )
    role = _required(data, "role", str, context)
    if role != expected_role:
        raise ConfigError(f"{context}.role must be {expected_role!r}, got {role!r}")
    status = _required(data, "status", str, context)
    if status not in {"pending", "declared", "verified"}:
        raise ConfigError(f"{context}.status must be pending, declared, or verified")

    gpu = _optional(data, "gpu", str, context)
    gpu_vram = _optional(data, "gpu_vram_mib", int, context)
    if (gpu is None) != (gpu_vram is None):
        raise ConfigError(f"{context}.gpu and gpu_vram_mib must be set together")
    if gpu_vram is not None and gpu_vram <= 0:
        raise ConfigError(f"{context}.gpu_vram_mib must be positive")

    ram_gib = _number(data, "ram_gib", context)
    if ram_gib <= 0:
        raise ConfigError(f"{context}.ram_gib must be positive")

    return HardwareConfig(
        host_id=_non_empty(
            _required(data, "host_id", str, context), f"{context}.host_id"
        ),
        role=role,
        status=status,
        operating_system=_non_empty(
            _required(data, "operating_system", str, context),
            f"{context}.operating_system",
        ),
        cpu=_non_empty(_required(data, "cpu", str, context), f"{context}.cpu"),
        ram_gib=ram_gib,
        gpu=gpu,
        gpu_vram_mib=gpu_vram,
        notes=_optional(data, "notes", str, context) or "",
    )


def _parse_dataset(data: Mapping[str, Any], base_dir: Path) -> DatasetConfig:
    context = "experiment.dataset"
    _reject_unknown(data, {"manifest_path", "revision"}, context)
    return DatasetConfig(
        manifest_path=_resolve_reference(
            base_dir, _required(data, "manifest_path", str, context)
        ),
        revision=_non_empty(
            _required(data, "revision", str, context), f"{context}.revision"
        ),
    )


def _parse_runtime(data: Mapping[str, Any]) -> RuntimeConfig:
    context = "experiment.runtime"
    _reject_unknown(
        data, {"seed", "warmup_requests", "measured_repetitions"}, context
    )
    seed = _required(data, "seed", int, context)
    if seed < 0:
        raise ConfigError(f"{context}.seed must be non-negative")
    return RuntimeConfig(
        seed=seed,
        warmup_requests=_non_negative_int(data, "warmup_requests", context),
        measured_repetitions=_positive_int(
            data, "measured_repetitions", context
        ),
    )


def _parse_transport(data: Mapping[str, Any]) -> TransportConfig:
    context = "experiment.transport"
    _reject_unknown(
        data, {"host", "port", "timeout_seconds", "protocol_version"}, context
    )
    port = _positive_int(data, "port", context)
    if port > 65535:
        raise ConfigError(f"{context}.port must be at most 65535")
    timeout = _number(data, "timeout_seconds", context)
    if timeout <= 0:
        raise ConfigError(f"{context}.timeout_seconds must be positive")
    return TransportConfig(
        host=_non_empty(_required(data, "host", str, context), f"{context}.host"),
        port=port,
        timeout_seconds=timeout,
        protocol_version=_positive_int(data, "protocol_version", context),
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return cast(Mapping[str, Any], parsed)


def _required(
    data: Mapping[str, Any], key: str, expected_type: type[T], context: str
) -> T:
    if key not in data:
        raise ConfigError(f"Missing required field: {context}.{key}")
    value = data[key]
    if isinstance(value, bool) and expected_type in {int, float}:
        raise ConfigError(f"{context}.{key} must be {expected_type.__name__}")
    if not isinstance(value, expected_type):
        raise ConfigError(f"{context}.{key} must be {expected_type.__name__}")
    return value


def _optional(
    data: Mapping[str, Any], key: str, expected_type: type[T], context: str
) -> T | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, expected_type):
        raise ConfigError(f"{context}.{key} must be {expected_type.__name__} or null")
    return value


def _required_mapping(
    data: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context}.{key} must be a mapping")
    return cast(Mapping[str, Any], value)


def _number(data: Mapping[str, Any], key: str, context: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}.{key} must be a number")
    return float(value)


def _positive_int(data: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(data, key, int, context)
    if value <= 0:
        raise ConfigError(f"{context}.{key} must be positive")
    return value


def _non_negative_int(data: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(data, key, int, context)
    if value < 0:
        raise ConfigError(f"{context}.{key} must be non-negative")
    return value


def _non_empty(value: str, field: str) -> str:
    if not value.strip():
        raise ConfigError(f"{field} must not be empty")
    return value


def _enum_value(enum_type: type[EnumT], value: str, field: str) -> EnumT:
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ConfigError(f"{field} must be one of: {choices}") from exc


def _reject_unknown(
    data: Mapping[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown field(s) in {context}: {', '.join(unknown)}")


def _resolve_reference(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _as_json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_compatible(value))
