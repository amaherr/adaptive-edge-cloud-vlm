from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_predictor_v2 import ROOT, synthetic_training

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate_v2 import fit_models, train_bundle
from adaptive_vlm.benchmark.predictor_eval_v2 import review_training
from adaptive_vlm.benchmark.predictor_v2_data import digest


def test_full_synthetic_training_and_review_never_read_sealed_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, _split, contract = synthetic_training()
    bundle = tmp_path / "source"
    bundle.mkdir()
    (bundle / "training.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (bundle / "split.json").write_bytes(
        (ROOT / "configs/evaluation/physical_vqa_split_v1.json").read_bytes()
    )
    write_json_atomic(bundle / "contract.json", contract)
    protected = tmp_path / "never-read-final.json"
    protected.write_text('{"observed_end_to_end_ms":1e200}')
    protected.chmod(0o000)
    write_json_atomic(
        bundle / "partition_audit.json",
        {"files": {"training.jsonl": digest(bundle / "training.jsonl")}},
    )
    original = Path.read_bytes

    def checked(path: Path) -> bytes:
        if path == protected or path.name == "validation_targets.jsonl":
            raise AssertionError("training accessed protected outcomes")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", checked)
    output = tmp_path / "fit"
    diagnostics = train_bundle(bundle, output)
    assert diagnostics["training_attempts"] == 2700
    assert diagnostics["unique_training_requests"] == 60
    assert diagnostics["oof_predictions"] == 8100
    assert diagnostics["lobo_predictions"] == 8100
    assert diagnostics["validation_outcomes_read"] is False
    hashes = {p.name: digest(p) for p in output.iterdir()}
    write_json_atomic(output / "file_hashes.json", hashes)
    review = tmp_path / "review"
    result = review_training(bundle, output, review)
    assert result["cross_validation_manifests"] == 10
    folds = json.loads((review / "training_cv_manifests.json").read_text())["folds"]
    for fold in folds:
        assert not set(fold["training_requests"]) & set(
            fold["held_training_request_ids"]
        )
        assert not set(fold["training_blocks"]) & set(fold["held_training_blocks"])
        assert all(len(ids) == 1 for ids in fold["fold_calibration_ids"].values())
    assert all(digest(output / name) == value for name, value in hashes.items())
    with pytest.raises(FileExistsError):
        train_bundle(bundle, output)
    with pytest.raises(FileExistsError):
        review_training(bundle, output, review)
    assert protected.stat().st_mode & 0o777 == 0
    assert not list(output.glob("*final*"))


@pytest.mark.parametrize("change", ["missing", "duplicate", "identity", "alignment"])
def test_training_rejects_incomplete_or_mixed_source(change: str) -> None:
    rows, split, contract = synthetic_training()
    rows = copy.deepcopy(rows)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "identity":
        rows[0]["target"]["model_contract_sha256"] = "wrong-model"
    else:
        rows[0]["target"]["warmup"] = True
    with pytest.raises(ValueError):
        fit_models(rows, split, contract)


def test_insufficient_successful_requests_are_not_imputed() -> None:
    rows, split, contract = synthetic_training()
    for row in rows:
        if row["inputs"]["configuration_id"] == "edge/full":
            row["target"]["status"] = "oom"
    with pytest.raises(ValueError, match="independent successful"):
        fit_models(rows, split, contract)
