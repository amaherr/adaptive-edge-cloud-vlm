"""Shared request, result, and experiment artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

from adaptive_vlm.config import ExecutionMode

MetricValue: TypeAlias = int | float | str | bool | None


class InferenceStatus(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    OOM = "oom"
    PROTOCOL_ERROR = "protocol_error"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    image_path: Path
    prompt: str

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    output_text: str
    output_token_ids: tuple[int, ...]
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if any(token_id < 0 for token_id in self.output_token_ids):
            raise ValueError("output_token_ids must be non-negative")
        _validate_metric_mapping(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return _as_json_dict(asdict(self))


@dataclass(frozen=True)
class RequestRecord:
    schema_version: int
    run_id: str
    request_id: str
    mode: ExecutionMode
    status: InferenceStatus
    output_text: str | None
    output_token_ids: tuple[int, ...]
    metrics: Mapping[str, MetricValue]
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("RequestRecord schema_version must be 1")
        if not self.run_id.strip() or not self.request_id.strip():
            raise ValueError("run_id and request_id must not be empty")
        if self.status is InferenceStatus.OK and self.error_type is not None:
            raise ValueError("successful records cannot contain an error_type")
        if self.status is not InferenceStatus.OK and not self.error_type:
            raise ValueError("failed records must contain an error_type")
        _validate_metric_mapping(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return _as_json_dict(asdict(self))

    @classmethod
    def from_result(
        cls,
        *,
        run_id: str,
        mode: ExecutionMode,
        result: InferenceResult,
    ) -> RequestRecord:
        return cls(
            schema_version=1,
            run_id=run_id,
            request_id=result.request_id,
            mode=mode,
            status=InferenceStatus.OK,
            output_text=result.output_text,
            output_token_ids=result.output_token_ids,
            metrics=result.metrics,
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    created_at_utc: str
    git_commit: str | None
    git_dirty: bool | None
    config: Mapping[str, Any]
    environment: Mapping[str, Any]
    dataset_manifest_sha256: str | None = None
    dependency_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("RunManifest schema_version must be 1")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.created_at_utc.endswith("Z"):
            raise ValueError("created_at_utc must be a UTC timestamp ending in Z")
        _validate_json_mapping(self.config, "config")
        _validate_json_mapping(self.environment, "environment")

    def to_dict(self) -> dict[str, Any]:
        return _as_json_dict(asdict(self))


def _validate_metric_mapping(metrics: Mapping[str, MetricValue]) -> None:
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metric names must be non-empty strings")
        if not isinstance(value, (int, float, str, bool, type(None))):
            raise TypeError(f"metric {key!r} has an unsupported value")


def _validate_json_mapping(value: Mapping[str, Any], field_name: str) -> None:
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    _json_compatible(value)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _as_json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_compatible(value))

