"""Per-request interleaving of placements with untimed edge state preparation.

G2 v2 measures edge/full, cloud and distributed (G2 v2+ also the prompt-first
and int8 distributed variants) for the same request back to
back in a seeded random order, so slow drift (temperature, clocks, background
load) affects every placement alike instead of being confounded with one of
them. Edge/full and the edge half of distributed share one edge adapter. Before
each timed request the adapter is put, outside the timer, into the stage that a
run of same-placement requests would leave active (text before edge/full,
vision before distributed); that switch cost is recorded separately.

Each placement keeps its own run directory in the fixed-policy record format
(``plan.json``, ``run_manifest.json``, ``requests.jsonl``, ``summary.json``),
extended with an ``interleave`` block, so the profile builder and later stages
read it like any other run. The global order is in ``interleave.json``.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adaptive_vlm.artifacts import append_jsonl, write_json_atomic
from adaptive_vlm.benchmark.runner import (
    Trial,
    _quality,
    _read_rows,
    _status,
    recover_journal_tail,
    semantic_config_sha256,
    summarize,
)
from adaptive_vlm.config import (
    ExecutionMode,
    ExperimentConfig,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceStatus, RunManifest
from adaptive_vlm.data.manifest import FrozenManifest
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import RemoteInferenceError
from adaptive_vlm.process import exclusive_lock
from adaptive_vlm.scheduler.configuration import ALL_CONFIGURATIONS, SPEC_BY_PLACEMENT
from adaptive_vlm.timing import perf_counter_ns
from adaptive_vlm.transport.protocol import ProtocolError

if TYPE_CHECKING:
    from adaptive_vlm.prediction.r2.live import AdaptiveRunner

# G2 v2's three placements (the default); G2 v2+ adds the prompt-first and
# int8 variants of the distributed split. A placement names a run directory.
PLACEMENTS = ("edge", "cloud", "distributed")
ALL_PLACEMENTS = tuple(spec.placement for spec in ALL_CONFIGURATIONS)
# G5 (amendment A4): the live adaptive policy runs as one more interleaved
# placement; it chooses one of the fixed configurations per request.
ADAPTIVE_PLACEMENT = "adaptive"
INTERLEAVED_PLACEMENTS = (*ALL_PLACEMENTS, ADAPTIVE_PLACEMENT)
PREPARED_STAGE: Mapping[str, str | None] = {
    **{spec.placement: spec.prepared_stage for spec in ALL_CONFIGURATIONS},
    ADAPTIVE_PLACEMENT: None,
}
DEVICE_RESTART_EXIT_CODE = 75
ENVIRONMENT_RESTART_EXIT_CODE = 76
RESTARTABLE_EXIT_CODES = frozenset(
    {DEVICE_RESTART_EXIT_CODE, ENVIRONMENT_RESTART_EXIT_CODE}
)
StagePreparer = Callable[[str], float]
StateProbe = Callable[[], Mapping[str, Any]]


class DeviceContextFailure(RuntimeError):
    """The local accelerator context is unusable; restart the process and resume."""


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def is_device_context_failure(exc: BaseException) -> bool:
    """A local CUDA error poisons the process; remote errors do not."""

    if isinstance(exc, (RemoteInferenceError, ProtocolError, TimeoutError, OSError)):
        return False
    return "CUDA error" in str(exc)


def make_interleaved_order(
    request_ids: Sequence[str],
    *,
    repetitions: Sequence[int],
    warmups: int,
    placements: Sequence[str],
    seed: int,
) -> tuple[tuple[str, Trial], ...]:
    """Seeded global order of (placement, trial) attempts.

    Warmups come first, each placement once per warmup in a shuffled order.
    Then, for each repetition, request IDs are shuffled and every request runs
    once per placement, in a fresh shuffled placement order.
    """

    ids = list(request_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("request IDs must be non-empty and unique")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not repetitions or len(set(repetitions)) != len(repetitions):
        raise ValueError("repetition indices must be non-empty and unique")
    if any(repetition < 0 for repetition in repetitions):
        raise ValueError("repetition indices must be non-negative")
    if not placements or len(set(placements)) != len(placements):
        raise ValueError("placements must be non-empty and unique")
    unknown = set(placements) - set(INTERLEAVED_PLACEMENTS)
    if unknown:
        raise ValueError(f"unknown placements: {sorted(unknown)}")
    rng = random.Random(seed)
    order: list[tuple[str, Trial]] = []
    for index in range(warmups):
        trial = Trial(f"w{index:06d}", ids[index % len(ids)], True, -1)
        order.extend(
            (placement, trial) for placement in rng.sample(placements, len(placements))
        )
    for repetition in repetitions:
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        for position, request_id in enumerate(shuffled):
            trial = Trial(
                f"r{repetition:04d}-{position:06d}", request_id, False, repetition
            )
            order.extend(
                (placement, trial)
                for placement in rng.sample(placements, len(placements))
            )
    return tuple(order)


def interleave_parameters(
    *,
    request_ids: Sequence[str],
    repetitions: Sequence[int],
    warmups: int,
    placements: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    order = make_interleaved_order(
        request_ids,
        repetitions=repetitions,
        warmups=warmups,
        placements=placements,
        seed=seed,
    )
    return {
        "schema_version": 1,
        "seed": seed,
        "placements": list(placements),
        "request_ids": list(request_ids),
        "repetitions": list(repetitions),
        "warmups": warmups,
        "prepared_stage": {p: PREPARED_STAGE[p] for p in placements},
        "order_sha256": _digest(
            {"order": [[placement, trial.trial_id] for placement, trial in order]}
        ),
    }


def placement_schedule(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The per-placement trial list implied by recorded interleave parameters."""

    order = make_interleaved_order(
        parameters["request_ids"],
        repetitions=parameters["repetitions"],
        warmups=int(parameters["warmups"]),
        placements=parameters["placements"],
        seed=int(parameters["seed"]),
    )
    first = parameters["placements"][0]
    return [trial.to_dict() for placement, trial in order if placement == first]


def verify_interleave_parameters(parameters: Mapping[str, Any]) -> None:
    rebuilt = interleave_parameters(
        request_ids=parameters["request_ids"],
        repetitions=parameters["repetitions"],
        warmups=int(parameters["warmups"]),
        placements=parameters["placements"],
        seed=int(parameters["seed"]),
    )
    if rebuilt != dict(parameters):
        raise ValueError("interleave parameters do not reproduce their recorded order")


def cloud_state(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ``server_state_*`` a successful remote response carried, if any."""

    if row.get("status") != InferenceStatus.OK.value:
        return None
    metrics = row.get("metrics") or {}
    state = {
        key.removeprefix("server_state_"): value
        for key, value in metrics.items()
        if key.startswith("server_state_")
    }
    return state or None


def run_interleaved_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Run every placement for each request in a seeded interleaved order.

    One writer per run directory: a second process (for example a benchmark
    orphaned by a killed supervisor) is refused instead of interleaving its
    appends with ours. See ``_run_interleaved_benchmark`` for the arguments.
    """

    run_dir = Path(kwargs["run_dir"])
    with exclusive_lock(run_dir.parent / f".{run_dir.name}.writer.lock", "run writer"):
        return _run_interleaved_benchmark(**kwargs)


def _run_interleaved_benchmark(
    *,
    config: ExperimentConfig,
    manifest: FrozenManifest,
    policies: Mapping[str, ExecutionPolicy],
    run_manifests: Mapping[str, RunManifest],
    run_dir: Path,
    request_ids: Sequence[str],
    repetitions: Sequence[int],
    warmups: int,
    seed: int,
    prepare_stage: StagePreparer | None = None,
    state_probe: StateProbe | None = None,
    precondition: Callable[[], None] | None = None,
    conditions: Mapping[str, Any] | None = None,
    resume: bool = False,
    max_consecutive_failures: int = 0,
    adaptive: AdaptiveRunner | None = None,
) -> dict[str, Any]:
    """Run every placement for each request in a seeded interleaved order.

    ``adaptive`` adds the G5 adaptive placement, run like the fixed ones.
    ``conditions`` (for example the edge-load level) is recorded in every
    placement plan, so a resume under different conditions is refused.
    ``precondition`` runs before each attempt; if it raises, nothing is
    recorded for that attempt and the exception propagates (the caller exits
    for a restart). ``state_probe`` runs after stage preparation, outside the
    timer, and its snapshot is stored as ``resource_state``.
    """

    if manifest.revision != config.dataset.revision:
        raise ValueError("request manifest revision differs from experiment config")
    if max_consecutive_failures < 0:
        raise ValueError("max_consecutive_failures must be non-negative")
    fixed = [p for p in ALL_PLACEMENTS if p in policies]
    placements = fixed + ([ADAPTIVE_PLACEMENT] if adaptive is not None else [])
    if set(policies) != set(fixed) or set(run_manifests) != set(placements):
        raise ValueError("policies and run manifests must name the same placements")
    specs = {p: SPEC_BY_PLACEMENT[p] for p in fixed}
    for placement, policy in policies.items():
        spec = specs[placement]
        if policy.mode is not spec.mode:
            raise ValueError(f"{placement} policy reports mode {policy.mode.value}")
        for name, wanted in (
            ("pipeline", spec.pipeline),
            ("payload_format", spec.payload),
        ):
            actual = getattr(policy, name, wanted)
            if wanted is not None and actual != wanted:
                raise ValueError(f"{placement} policy has {name} {actual}")
    items = {item.request_id: item for item in manifest.items}
    missing = [rid for rid in request_ids if rid not in items]
    if missing:
        raise ValueError(f"request IDs not in manifest: {missing[:3]}")

    parameters = interleave_parameters(
        request_ids=request_ids,
        repetitions=repetitions,
        warmups=warmups,
        placements=placements,
        seed=seed,
    )
    order = make_interleaved_order(
        request_ids,
        repetitions=repetitions,
        warmups=warmups,
        placements=placements,
        seed=seed,
    )
    schedule = [trial.to_dict() for p, trial in order if p == placements[0]]
    plans: dict[str, dict[str, Any]] = {}
    for placement in placements:
        placement_spec = specs.get(placement)
        plan: dict[str, Any] = {
            "schema_version": 1,
            "config": config.to_dict(),
            "config_sha256": semantic_config_sha256(config.to_dict()),
            "model_contract_sha256": model_contract_fingerprint(config.model),
            "dataset_manifest_sha256": manifest.sha256,
            "mode": placement_spec.mode.value if placement_spec else ADAPTIVE_PLACEMENT,
            "pipeline": (placement_spec.pipeline or "synchronous")
            if placement_spec
            else None,
            "payload": placement_spec.payload if placement_spec else None,
            "placement": placement,
            "configuration_id": placement_spec.id.value
            if placement_spec
            else ADAPTIVE_PLACEMENT,
            "schedule": schedule,
            "interleave": parameters,
            "conditions": dict(conditions or {}),
        }
        if placement_spec is None and adaptive is not None:
            plan["adaptive"] = adaptive.record.to_dict()
        plan["plan_sha256"] = _digest(plan)
        plans[placement] = plan
    index = {
        "schema_version": 1,
        "interleave": parameters,
        "placement_plan_sha256": {p: plans[p]["plan_sha256"] for p in placements},
        "order": [[placement, trial.trial_id] for placement, trial in order],
    }

    run_ids: dict[str, str] = {}
    if resume:
        if (
            json.loads((run_dir / "interleave.json").read_text(encoding="utf-8"))
            != index
        ):
            raise ValueError("cannot resume: interleaved plan differs")
        for placement in placements:
            directory = run_dir / placement
            saved_plan = json.loads(
                (directory / "plan.json").read_text(encoding="utf-8")
            )
            if saved_plan != plans[placement]:
                raise ValueError(f"cannot resume: {placement} plan differs")
            saved = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            if (
                saved.get("dataset_manifest_sha256") != manifest.sha256
                or saved.get("config") != config.to_dict()
            ):
                raise ValueError(f"cannot resume: {placement} run manifest differs")
            if saved.get("environment", {}).get("duration_clock") != (
                run_manifests[placement].environment.get("duration_clock")
            ):
                raise ValueError("cannot resume: duration clock differs")
            run_ids[placement] = str(saved["run_id"])
            recover_journal_tail(directory / "requests.jsonl")
    else:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        for placement in placements:
            directory = run_dir / placement
            directory.mkdir()
            write_json_atomic(directory / "plan.json", plans[placement])
            write_json_atomic(
                directory / "run_manifest.json", run_manifests[placement].to_dict()
            )
            run_ids[placement] = run_manifests[placement].run_id
        write_json_atomic(run_dir / "interleave.json", index)

    expected = {trial["trial_id"]: trial for trial in schedule}
    rows: dict[str, list[dict[str, Any]]] = {}
    completed: set[tuple[str, str]] = set()
    for placement in placements:
        plan = plans[placement]
        rows[placement] = _read_rows(run_dir / placement / "requests.jsonl")
        for saved_row in rows[placement]:
            trial_id = saved_row.get("trial_id")
            if trial_id not in expected or (placement, trial_id) in completed:
                raise ValueError(f"unknown or duplicate {placement} trial: {trial_id}")
            planned = expected[trial_id]
            if (
                saved_row.get("request_id") != planned["request_id"]
                or saved_row.get("warmup") != planned["warmup"]
                or saved_row.get("repetition") != planned["repetition"]
                or saved_row.get("plan_sha256") != plan["plan_sha256"]
                or saved_row.get("run_id") != run_ids[placement]
                or saved_row.get("mode") != plan["mode"]
                or saved_row.get("placement") != placement
                or saved_row.get("dataset_manifest_sha256") != manifest.sha256
            ):
                raise ValueError(f"{placement} result does not match plan: {trial_id}")
            completed.add((placement, str(trial_id)))

    # The previous response's cloud state (amendment A2a rule 9): an input
    # observed before the timer, lagged by one remote response, None at first.
    previous_cloud: dict[str, Any] | None = None
    for saved in sorted(
        (row for placement in placements for row in rows[placement]),
        key=lambda row: int(row["interleave_position"]),
    ):
        previous_cloud = cloud_state(saved) or previous_cloud
    consecutive_failures = 0
    for position, (placement, trial) in enumerate(order, 1):
        if (placement, trial.trial_id) in completed:
            continue
        if precondition is not None:
            precondition()
        item = items[trial.request_id]
        plan = plans[placement]
        print(
            f"[{position}/{len(order)}] {placement} "
            f"{trial.trial_id} {trial.request_id}",
            flush=True,
        )
        row: dict[str, Any] = {
            **trial.to_dict(),
            "started_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "plan_sha256": plan["plan_sha256"],
            "config_sha256": plan["config_sha256"],
            "model_contract_sha256": plan["model_contract_sha256"],
            "dataset_manifest_sha256": manifest.sha256,
            "run_id": run_ids[placement],
            "mode": plan["mode"],
            "placement": placement,
            "configuration_id": plan["configuration_id"],
            "image_sha256": item.image_sha256,
            "scoring": item.scoring,
            "interleave_position": position,
            "prepared_stage": PREPARED_STAGE[placement],
            "stage_prepare_ms": None,
            "resource_state": None,
            "previous_cloud_state": previous_cloud,
        }
        failure: Exception | None = None
        started = perf_counter_ns()
        try:
            stage = PREPARED_STAGE[placement]
            if stage is not None and prepare_stage is not None:
                row["stage_prepare_ms"] = prepare_stage(stage)
            if state_probe is not None:
                row["resource_state"] = dict(state_probe())
            started = perf_counter_ns()
            infer = (
                adaptive.infer
                if placement == ADAPTIVE_PLACEMENT and adaptive is not None
                else policies[placement].infer
            )
            result = infer(item.to_request())
            if result.request_id != item.request_id:
                raise ValueError("policy returned a different request_id")
            row.update(
                status=InferenceStatus.OK.value,
                output_text=result.output_text,
                output_token_ids=list(result.output_token_ids),
                metrics=dict(result.metrics),
                quality_score=_quality(item, result.output_text),
            )
        except Exception as exc:
            failure = exc
            row.update(
                status=_status(exc).value,
                output_text=None,
                output_token_ids=[],
                metrics={},
                quality_score=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                device_context_failure=is_device_context_failure(exc),
            )
        row["observed_end_to_end_ms"] = (perf_counter_ns() - started) / 1_000_000
        append_jsonl(run_dir / placement / "requests.jsonl", row)
        rows[placement].append(row)
        previous_cloud = cloud_state(row) or previous_cloud
        if failure is not None and is_device_context_failure(failure):
            raise DeviceContextFailure(
                f"local device context failed at {placement} {trial.trial_id}; "
                "attempt saved; restart the process with --resume"
            ) from failure
        consecutive_failures = consecutive_failures + 1 if failure is not None else 0
        if (
            max_consecutive_failures
            and consecutive_failures >= max_consecutive_failures
        ):
            raise RuntimeError(
                f"stopping after {consecutive_failures} consecutive failures; "
                "attempts are saved; investigate before resuming"
            )

    summaries: dict[str, Any] = {}
    for placement in placements:
        mode = plans[placement]["mode"]
        summary = summarize(
            rows[placement],
            manifest,
            mode if mode == ADAPTIVE_PLACEMENT else ExecutionMode(mode),
            plans[placement]["plan_sha256"],
        )
        write_json_atomic(run_dir / placement / "summary.json", summary)
        summaries[placement] = summary
    overall = {
        "schema_version": 1,
        "interleave": parameters,
        "attempts": len(order),
        "placements": summaries,
    }
    write_json_atomic(run_dir / "summary.json", overall)
    return overall


def run_with_device_restarts(
    command: Sequence[str],
    *,
    max_restarts: int,
    run: Callable[[list[str]], int] | None = None,
) -> tuple[int, int]:
    """Run an interleaved benchmark, resuming after device-context failures.

    Returns ``(exit_code, restarts)``. Exit codes 75 (local CUDA context
    failure) and 76 (an environment check such as the edge load generator
    failed) trigger a restart; ``--resume`` is appended to every restart that
    does not already carry it.
    """

    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")
    execute = run or (lambda argv: subprocess.run(argv, check=False).returncode)
    argv = list(command)
    restarts = 0
    while True:
        code = execute(argv)
        if code not in RESTARTABLE_EXIT_CODES or restarts >= max_restarts:
            return code, restarts
        restarts += 1
        if "--resume" not in argv:
            argv.append("--resume")
