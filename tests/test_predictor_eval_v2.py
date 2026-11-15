from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_predictor_v2 import synthetic_training

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate_v2 import fit_models
from adaptive_vlm.benchmark.predictor_eval_v2 import (
    choose_variant,
    evaluate_validation,
    scores,
    seal_validation_predictions,
)
from adaptive_vlm.benchmark.predictor_v2_data import digest
from adaptive_vlm.scheduler.predictor import transfer_ms
from adaptive_vlm.scheduler.predictor_v2 import PredictorV2


@pytest.fixture()
def validation_case(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    training, split, contract = synthetic_training()
    models = fit_models(training, split, contract)
    fit_dir = tmp_path / "fits"
    fit_dir.mkdir()
    for variant, model in models.items():
        write_json_atomic(fit_dir / f"calibration_{variant}.json", model.to_dict())
    write_json_atomic(
        fit_dir / "source_manifest.json", {"code_git_commit": "synthetic-fixture"}
    )
    write_json_atomic(
        fit_dir / "file_hashes.json", {p.name: digest(p) for p in fit_dir.iterdir()}
    )
    bundle = tmp_path / "data"
    bundle.mkdir()
    root = Path(__file__).resolve().parents[1]
    (bundle / "split.json").write_bytes(
        (root / "configs/evaluation/physical_vqa_split_v1.json").read_bytes()
    )
    rows = []
    targets = []
    for j, rid in enumerate(split["validation_request_ids"]):
        template_id = split["training_request_ids"][j]
        for block, rate, rtt in [("N2", 5e6, 20), ("N7", 10e6, 5)]:
            for source in [
                r
                for r in training
                if r["inputs"]["request_id"] == template_id
                and r["inputs"]["block"] == "N1"
            ]:
                before = copy.deepcopy(source["inputs"])
                before.update(
                    schema_version=2,
                    request_id=rid,
                    block=block,
                    trial_id=f"r{before['repetition']}-{j}",
                )
                before["network"].update(
                    network_bucket_id=block,
                    uplink_bps=rate,
                    downlink_bps=rate,
                    rtt_ms=rtt,
                    probe_scope="controlled_block_preflight",
                )
                target = copy.deepcopy(source["target"])
                target.update(request_id=rid, repetition=before["repetition"])
                target["network_state"]["bucket_id"] = block
                f = target["fields"]
                mode = before["configuration_id"].split("/")[0]
                if mode != "edge":
                    f["transport_residual_ms"] = (
                        5
                        + 0.5 * rtt
                        + transfer_ms(f["request_wire_bytes"], rate)
                        + transfer_ms(f["response_wire_bytes"], rate)
                    )
                    f["network_exchange_ms"] = (
                        f["server_processing_ms"] + f["transport_residual_ms"]
                    )
                    target["observed_end_to_end_ms"] = (
                        (2 + f["network_exchange_ms"])
                        if mode == "cloud"
                        else f["edge_total_ms"] + 0.5 + f["network_exchange_ms"] + 1
                    )
                rows.append(before)
                targets.append(target)
    (bundle / "validation_inputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    protected = tmp_path / "sealed-validation.jsonl"
    data = "".join(json.dumps(r) + "\n" for r in targets).encode()
    protected.write_bytes(data)
    protected.chmod(0o000)
    write_json_atomic(
        bundle / "partition_audit.json",
        {
            "sealed_validation_path": str(protected),
            "split_sha256": digest(bundle / "split.json"),
            "archive_sha256": "synthetic-not-real",
            "files": {
                "validation_inputs.jsonl": digest(bundle / "validation_inputs.jsonl"),
                "validation_targets.jsonl": hashlib.sha256(data).hexdigest(),
            },
        },
    )
    return bundle, fit_dir, tmp_path / "evaluation", protected


def test_predictions_are_sealed_before_any_outcome_read(
    validation_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, fit_dir, output, protected = validation_case
    original = Path.read_bytes

    def no_targets(path: Path) -> bytes:
        if path == protected:
            raise AssertionError("outcomes read before seal")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", no_targets)
        seal = seal_validation_predictions(bundle, fit_dir, output)
    assert seal["prediction_rows"] == 1080
    assert seal["validation_outcomes_read"] is False
    assert protected.stat().st_mode & 0o777 == 0
    result = evaluate_validation(bundle, fit_dir, output)
    assert result["gate"] in {"G3 PASSED", "G3 BLOCKED", "G3 FAILED"}
    assert protected.stat().st_mode & 0o777 == 0
    report = json.loads((output / "evaluation_validation.json").read_text())
    assert report["validation_attempts"] == 360
    assert report["final_outcomes_read"] is False
    assert (
        report["models"]["v2_a"]["ranking_request_primary"][
            "scheduled_request_block_conditions"
        ]
        == 40
    )
    assert (
        report["models"]["v1_recalibrated"]["ranking_by_block"]["N7"][
            "ranked_conditions"
        ]
        == 0
    )
    assert report["models"]["v2_a"]["by_block_configuration"]["N7|cloud/raw_image"][
        "repeat_secondary"
    ]["coverage_labels"] == {"physics_extrapolation": 60}
    assert not list(output.glob("*final*"))
    with pytest.raises(FileExistsError, match="one validation"):
        evaluate_validation(bundle, fit_dir, output)


def test_unsealed_or_changed_predictions_never_unlock_outcomes(
    validation_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, fit_dir, output, protected = validation_case
    original = Path.read_bytes

    def no_targets(path: Path) -> bytes:
        if path == protected:
            raise AssertionError("should not unlock outcome")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", no_targets)
        with pytest.raises(FileNotFoundError):
            evaluate_validation(bundle, fit_dir, output)
        seal_validation_predictions(bundle, fit_dir, output)
        p = output / "predictions_validation.jsonl"
        p.write_bytes(p.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="changed after sealing"):
            evaluate_validation(bundle, fit_dir, output)
    assert protected.stat().st_mode & 0o777 == 0


def test_changed_fit_and_leaking_input_fail_before_outcomes(
    validation_case: Any,
) -> None:
    bundle, fit_dir, output, protected = validation_case
    seal_validation_predictions(bundle, fit_dir, output)
    path = fit_dir / "calibration_v2_a.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="calibration file changed"):
        evaluate_validation(bundle, fit_dir, output)
    assert protected.stat().st_mode & 0o777 == 0


def test_all_invalid_predictions_cannot_pass(
    validation_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, fit_dir, output, _ = validation_case

    def invalid(self: PredictorV2, spec: Any, state: Any, task: Any) -> Any:
        return self._invalid(spec.id.value, "synthetic_unavailable")

    monkeypatch.setattr(PredictorV2, "predict", invalid)
    seal_validation_predictions(bundle, fit_dir, output)
    result = evaluate_validation(bundle, fit_dir, output)
    assert result["gate"] == "G3 BLOCKED" and result["selected_variant"] is None
    assert not (output / "model_freeze_marker.json").exists()
    assert scores([])["median_ape"] is None
    assert scores([])["prediction_coverage"] == 0


def test_model_selection_prefers_simplicity_for_unstable_small_gain() -> None:
    conditions = [
        {"request_id": f"r{i}", "block": b, "matched": True}
        for i in range(20)
        for b in ["N2", "N7"]
    ]
    report = {
        "models": {
            v: {"ranking_request_primary": {"conditions": conditions, "matches": 40}}
            for v in ["v1_recalibrated", "v2_a"]
        }
    }
    rows = []
    for v, error in [("v1_recalibrated", 10), ("v2_a", 9.9)]:
        for i in range(20):
            rows.append(
                {
                    "variant": v,
                    "status": "ok",
                    "before": {"request_id": f"r{i}"},
                    "observed_ms": 100,
                    "prediction": {"valid": True, "total_ms": 100 + error},
                }
            )
    chosen, evidence = choose_variant(["v1_recalibrated", "v2_a"], report, rows)
    assert chosen == "v1_recalibrated" and evidence["relative_mae_gain"] < 0.05
    for row in rows:
        if row["variant"] == "v2_a":
            row["prediction"]["total_ms"] = 105
    chosen, _ = choose_variant(["v1_recalibrated", "v2_a"], report, rows)
    assert chosen == "v2_a"
