"""Analyze official, provisional and explicitly separate diagnostic G2 scores."""

# Generated scientific Markdown retains Unicode and long prose lines.
# ruff: noqa: E501, RUF001
from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/g2-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
REPO = HERE.parents[2]
MODES = ("edge", "cloud", "distributed")
PREFIX = "artifacts/unattended/site-preflight-20261110/campaign-heterogeneous-v2"


def read(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    (HERE / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ci(values):
    a = np.asarray(values)
    rng = np.random.default_rng(20261115)
    index = rng.integers(0, len(a), size=(10000, len(a)))
    return [float(v) for v in np.quantile(a[index].mean(axis=1), [0.025, 0.975])]


def provisional(answer, references):
    normalized = answer.strip().casefold()
    refs = [r.strip().casefold() for r in references]
    return (
        sum(min(1, (refs.count(normalized) - int(r == normalized)) / 3) for r in refs)
        / 10
    )


def supporting_span(forms):
    text = " " + forms["prediction_unconditionally_normalized_for_diagnosis"] + " "
    refs = forms["references_unconditionally_normalized_for_diagnosis"]
    return any(ref and (" " + ref + " ") in text for ref in refs)


def main():
    archive = RESULTS / "evidence.tar.gz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == (RESULTS / "SHA256SUMS").read_text().split()[0]
    )
    manifest_path = REPO / "data/manifests/physical_vqa_v1.jsonl"
    manifest = {
        int(r["request_id"].split("-")[-1]): r
        for r in map(json.loads, manifest_path.read_text().splitlines())
    }
    ids = sorted(manifest)
    forms = {m: read(HERE / f"{m}-answer-forms.json")["questions"] for m in MODES}
    diagnostic = read(HERE / "format_diagnostic.json")
    index = read(HERE / "evaluation_index.json")
    assert len(index) == 81
    score = {m: [] for m in MODES}
    baseline = {}
    run_rows = []
    paired_runs = []
    answer_types = {}
    # Official exports contain question/answer type aggregate results. The source
    # annotations stay ignored; the question-type labels are read from predictions'
    # original question-ID match in the locally prepared subset for this analysis.
    subset = read(REPO / "artifacts/official-vqa-g2/scoring/subset/annotations.json")
    annotation_by_id = {a["question_id"]: a for a in subset["annotations"]}
    answer_types = {qid: annotation_by_id[qid]["answer_type"] for qid in ids}
    for qid in ids:
        assert Counter(
            a["answer"] for a in annotation_by_id[qid]["answers"]
        ) == Counter(manifest[qid]["references"])
    raw_lookup = {}
    with tarfile.open(archive, "r:gz") as tar:
        for block_i in range(1, 10):
            block = f"N{block_i}"
            for mode in MODES:
                raw_rows = [
                    json.loads(line)
                    for line in tar.extractfile(
                        f"{PREFIX}/{block}/{mode}/requests.jsonl"
                    )
                ]
                measured = [r for r in raw_rows if not r["warmup"]]
                assert len(measured) == 300 and all(
                    r["status"] == "ok" for r in measured
                )
                for rep in range(3):
                    rs = {
                        int(r["request_id"].split("-")[-1]): r
                        for r in measured
                        if r["repetition"] == rep
                    }
                    assert set(rs) == set(ids)
                    predictions_path = (
                        HERE / "predictions" / block / f"{mode}-r{rep}.json"
                    )
                    predictions = {
                        r["question_id"]: r["answer"] for r in read(predictions_path)
                    }
                    output_path = (
                        HERE / "official_outputs" / block / f"{mode}-r{rep}.json"
                    )
                    evaluation = read(output_path)
                    assert (
                        evaluation["input_sha256"]["predictions"]
                        == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
                    )
                    entry = next(
                        e
                        for e in index
                        if e["block"] == block
                        and e["mode"] == mode
                        and e["repetition"] == rep
                    )
                    assert (
                        entry["official_output_sha256"]
                        == hashlib.sha256(output_path.read_bytes()).hexdigest()
                    )
                    values = np.array(
                        [
                            evaluation["per_question_accuracy_percent"][str(qid)]
                            for qid in ids
                        ]
                    )
                    assert (
                        abs(
                            float(values.mean())
                            - evaluation["accuracy_percent"]["overall"]
                        )
                        < 0.0051
                    )
                    score[mode].append(values)
                    raw_lookup[block, mode, rep] = rs
                    for qid in ids:
                        assert predictions[qid] == rs[qid]["output_text"]
                        assert (
                            abs(
                                provisional(
                                    rs[qid]["output_text"], manifest[qid]["references"]
                                )
                                - rs[qid]["quality_score"]
                            )
                            < 1e-12
                        )
                        if (mode, qid) not in baseline:
                            baseline[mode, qid] = rs[qid]
                        assert (
                            rs[qid]["output_text"] == baseline[mode, qid]["output_text"]
                        )
                        assert (
                            rs[qid]["output_token_ids"]
                            == baseline[mode, qid]["output_token_ids"]
                        )
                    acc = evaluation["accuracy_percent"]
                    run_rows.append(
                        {
                            "block": block,
                            "mode": mode,
                            "repetition": rep,
                            "questions": 100,
                            "official_raw_accuracy_percent": acc["overall"],
                            "provisional_accuracy_percent": 100
                            * float(np.mean([r["quality_score"] for r in rs.values()])),
                            **{
                                f"official_{k}_percent": v
                                for k, v in acc["perAnswerType"].items()
                            },
                        }
                    )
    average = {m: np.array(score[m]).mean(axis=0) for m in MODES}
    assert all(np.all(np.array(score[m]) == average[m]) for m in MODES)
    summaries = []
    for mode in MODES:
        rows = [baseline[mode, qid] for qid in ids]
        candidates = sum(
            average[mode][j] == 0 and supporting_span(forms[mode][str(qid)])
            for j, qid in enumerate(ids)
        )
        summaries.append(
            {
                "placement": mode,
                "evaluated_attempts": 2700,
                "independent_question_ids": 100,
                "official_raw_accuracy_percent": float(average[mode].mean()),
                "request_bootstrap_ci95_percent": ci(average[mode]),
                "provisional_accuracy_percent": 100
                * float(np.mean([r["quality_score"] for r in rows])),
                "one_word_answers": sum(
                    len(r["output_text"].split()) == 1 for r in rows
                ),
                "answers_at_32_token_cap": sum(
                    len(r["output_token_ids"]) == 32 for r in rows
                ),
                "zero_score_answers_containing_reference_span": int(candidates),
                "leading_yes_no_diagnostic_percent": diagnostic[mode][
                    "accuracy_percent"
                ]["overall"],
                "lowercase_only_diagnostic_percent": diagnostic[mode][
                    "lowercase_only_diagnostic"
                ]["accuracy_percent"]["overall"],
            }
        )
    pairs = []
    for left, right in combinations(MODES, 2):
        delta = average[left] - average[right]
        p = {
            "left": left,
            "right": right,
            "difference_definition": "left minus right, percentage points",
            "paired_questions": 100,
            "mean_difference_pp": float(delta.mean()),
            "ci95_low_pp": ci(delta)[0],
            "ci95_high_pp": ci(delta)[1],
            "left_higher_questions": int(np.sum(delta > 0)),
            "right_higher_questions": int(np.sum(delta < 0)),
            "tied_score_questions": int(np.sum(delta == 0)),
            "different_raw_answer_text_questions": sum(
                baseline[left, q]["output_text"] != baseline[right, q]["output_text"]
                for q in ids
            ),
            "different_token_sequence_questions": sum(
                baseline[left, q]["output_token_ids"]
                != baseline[right, q]["output_token_ids"]
                for q in ids
            ),
        }
        pairs.append(p)
        for i in range(9):
            for rep in range(3):
                d = score[left][i * 3 + rep] - score[right][i * 3 + rep]
                paired_runs.append(
                    {
                        "block": f"N{i + 1}",
                        "repetition": rep,
                        "left": left,
                        "right": right,
                        "paired_questions": 100,
                        "mean_difference_pp": float(np.mean(d)),
                        "left_higher_questions": int(np.sum(d > 0)),
                        "right_higher_questions": int(np.sum(d < 0)),
                    }
                )
    question_rows = []
    for j, qid in enumerate(ids):
        r = {
            "question_id": qid,
            "question": manifest[qid]["prompt"],
            "answer_type": answer_types[qid],
            "references": json.dumps(manifest[qid]["references"]),
        }
        for mode in MODES:
            b = baseline[mode, qid]
            f = forms[mode][str(qid)]
            r.update(
                {
                    f"{mode}_answer": b["output_text"],
                    f"{mode}_official_percent": float(average[mode][j]),
                    f"{mode}_provisional_percent": b["quality_score"] * 100,
                    f"{mode}_token_count": len(b["output_token_ids"]),
                    f"{mode}_official_normalization_branch_active": f[
                        "official_normalization_branch_active"
                    ],
                    f"{mode}_normalized_actual_official_answer": f[
                        "prediction_after_actual_official_branch"
                    ],
                    f"{mode}_contains_reference_span_diagnostic": supporting_span(f),
                }
            )
        question_rows.append(r)
    write_csv("per_question_quality.csv", question_rows)
    write_csv("run_quality_summary.csv", run_rows)
    write_csv("paired_quality_differences.csv", pairs)
    write_csv("paired_quality_by_run.csv", paired_runs)
    dump(
        "quality_summary.json",
        {
            "placements": summaries,
            "paired": pairs,
            "answer_type_counts": dict(Counter(answer_types.values())),
            "scope": "Official raw-answer scores on 100 frozen VQA v2 validation questions, repeated across 9 states and 3 repetitions. Not full-validation benchmark accuracy.",
            "uncertainty": "10000 paired request-ID cluster bootstrap replicates, seed20261115; average repetitions/blocks within ID. Zero-width paired intervals do not prove population equivalence.",
        },
    )
    dump(
        "quality_validation.json",
        {
            "saved_attempts_verified": 8100,
            "official_evaluations": 81,
            "question_ids_per_evaluation": 100,
            "warmups_excluded": 27,
            "predictions_match_saved_raw_text": True,
            "within_placement_text_and_tokens_stable_across_all_runs": True,
            "official_score_vectors_stable_within_placement": True,
            "provisional_formula_reproduced_for_every_attempt": True,
            "question_and_reference_alignment": True,
            "python3_analyzer_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    make_plot(
        summaries,
        read(HERE / "official_outputs/N1/edge-r0.json")["accuracy_percent"],
        diagnostic["edge"]["accuracy_percent"],
    )
    write_report(summaries, pairs, question_rows)
    print(json.dumps({"placements": summaries, "pairs": pairs}, indent=2))


def make_plot(summaries, raw, diagnostic):
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "official-vqa-g2"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    x = np.arange(3)
    for offset, key, label, color in [
        (
            -0.25,
            "official_raw_accuracy_percent",
            "Official, saved raw answers",
            "#4477AA",
        ),
        (0, "provisional_accuracy_percent", "Provisional metric", "#999999"),
        (
            0.25,
            "leading_yes_no_diagnostic_percent",
            "Leading yes/no: post-hoc diagnostic",
            "#EE7733",
        ),
    ]:
        bars = axes[0].bar(
            x + offset,
            [r[key] for r in summaries],
            width=0.24,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.0f", fontsize=8)
    axes[0].set_xticks(x, MODES)
    axes[0].set_ylabel("Subset score (%)")
    axes[0].set_ylim(0, 45)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Raw baseline stays 2%; diagnostics are separate")
    types = ["yes/no", "number", "other"]
    for offset, values, label, color in [
        (-0.18, raw["perAnswerType"], "Official raw", "#4477AA"),
        (0.18, diagnostic["perAnswerType"], "Leading yes/no diagnostic", "#EE7733"),
    ]:
        bars = axes[1].bar(
            x + offset, [values[k] for k in types], width=0.35, label=label, color=color
        )
        axes[1].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[1].set_xticks(x, ["yes/no (n=37)", "number (n=13)", "other (n=50)"])
    axes[1].set_ylabel("Answer-type score (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Formatting diagnostic: identical scores for all placements")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "svg"):
        path = HERE / f"quality_diagnosis.{extension}"
        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
            metadata={"Date": None}
            if extension == "svg"
            else {"Software": "G2 quality analysis"},
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def write_report(summaries, pairs, question_rows):
    lines = [
        "# Official VQA scoring and low-score diagnosis",
        "",
        "The saved G2 predictions were scored with the **unmodified GT-Vision-Lab/VQA evaluator**, commit `a013f0043c1e2cdc995922dfe257f7149aa9af06`, using the repository's official-evaluation adapter under checksum-verified PyPy 2.7.18 (7.3.15). No inference was rerun. The frozen manifest, questions, ten human references and image IDs were cross-checked against the original official VQA v2 validation sources. Downloaded source hashes match the dataset freeze.",
        "",
        "## Primary official result",
        "",
        "**Every placement scores 2.0% on the 100-question frozen cohort.** All nine blocks and all three repetitions give that same result: 81 official evaluator invocations cover 8,100 measured predictions. This is a subset score, not the model's accuracy on the full VQA v2 validation set. The 27 warmups are excluded, and no failed prediction was dropped (there were none).",
        "",
        "| Placement | Evaluated attempts | Unique questions | Official raw score | Request-cluster 95% bootstrap interval | Provisional score |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in summaries:
        lines.append(
            f"| {r['placement']} | 2700 | 100 | {r['official_raw_accuracy_percent']:.1f}% | [{r['request_bootstrap_ci95_percent'][0]:.1f}, {r['request_bootstrap_ci95_percent'][1]:.1f}]% | {r['provisional_accuracy_percent']:.1f}% |"
        )
    lines += [
        "",
        "Only two question IDs receive any official credit (both 100%): `480379000` → `2`, and `31984003` → `1`. The 13 number questions average 15.38%; all 37 yes/no and all 50 other questions score 0% with the raw saved sentences.",
        "",
        "## Paired placement differences",
        "",
        "| Paired comparison (left − right) | Mean difference | Paired 95% interval | Left better / right better / tied | Different answer texts |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for p in pairs:
        lines.append(
            f"| {p['left']} − {p['right']} | {p['mean_difference_pp']:.1f} pp | [{p['ci95_low_pp']:.1f}, {p['ci95_high_pp']:.1f}] pp | {p['left_higher_questions']} / {p['right_higher_questions']} / {p['tied_score_questions']} | {p['different_raw_answer_text_questions']}/100 |"
        )
    lines += [
        "",
        "Scores are paired by question ID, averaging the 27 observations per placement within each ID before resampling 100 IDs (10,000 bootstrap replicates, fixed seed). Per-question score differences are zero in all 27 block/repetition pairs too. Repetitions and network blocks **do not provide 2,700 independent questions**. Zero-width bootstrap intervals occur because every observed pair ties; they do not prove population equivalence or exclude degradation on unseen questions. With 98/100 questions at the floor, this score has little power to discriminate the 26 question IDs with any cross-placement output disagreement (the union across pairs). Cloud and distributed differ on 16 question IDs despite identical score vectors. The approved implementation/transport correctness evidence remains separate from semantic answer quality.",
        "",
        "## Why the provisional score is low",
        "",
        "### 1. Answer-format mismatch is directly observed",
        "",
        "The runner scores the **whole** stripped/case-folded output against each short human answer. Of 100 saved answers per placement, only three are single-word answers and those are the only three receiving provisional credit. The other 97 are multi-word outputs and receive zero provisional credit. A full sentence is not reduced to its answer-bearing phrase by the official evaluator either. For example:",
        "",
        "| Question | Saved output (edge; remote is identical for these examples) | Human answer(s) | Raw official score |",
        "| --- | --- | --- | ---: |",
        "| How many people in the photo? | There are two people in the photo. | 2 (10/10) | 0% |",
        "| Are they on a river? | Yes, the couple is on a riverboat or canal boat. | yes (10/10) | 0% |",
        "| Are there white clouds in the sky? | No, there are no white clouds in the sky. The sky is clear and cloudless. | no (10/10) | 0% |",
        "",
        "The importer froze the original question verbatim; it did not add an instruction to answer with a short VQA label. The adapter invokes the pinned query template, so free-form sentences are being compared with short-answer references. The double suffix in the adapter is also present in the pinned non-reasoning upstream query path; this inspection does not identify it as a new adapter-only bug. A prompt/decoder change would be a new experiment contract, not a correction to these saved results.",
        "",
    ]
    for r in summaries:
        lines.append(
            f"For {r['placement']}, **{r['zero_score_answers_containing_reference_span']} zero-credit answers** contain at least one reference phrase after unconditional upstream normalization. This is a formatting candidate flag, **not semantic accuracy**: negations, conflicting words and short incidental references can create false positives."
        )
    lines += [
        "",
        "### 2. The official/provisional one-point difference is a case-handling edge case",
        "",
        "Question `50431001` predicts `Hotel`, while all ten references are `hotel`. The provisional case-folding awards 100%. In the pinned official `VQAEval.evaluate`, punctuation/digit/article normalization (including lowercasing) is guarded by `len(set(gtAnswers)) > 1`; unanimous reference answers skip that branch. `Hotel` therefore does not match `hotel`, losing one point in the 100-question mean. This is documented evaluator behavior at the pinned revision, not an annotation-ID mismatch or a lost inference. A **lowercase-only, post-hoc diagnostic** scores 3.0% for every placement. That diagnostic is kept separate from the official score of the untouched saved answers.",
        "",
        "### 3. Formatting is not the only error",
        "",
        "The girl/container question predicts a camera while the references mostly say bottle; the bike/basket question predicts yes while all ten references say no. Such errors cannot be repaired by standard text normalization. Eight answers per placement reach the 32-token cap; this is a possible truncation risk, not a logged proof of stop reason or the main demonstrated cause. A score recalculation cannot establish the quality a new short-answer prompt would achieve.",
        "",
        "## Controlled post-hoc formatting diagnostic",
        "",
        "A reference-free rule takes a leading standalone `yes` or `no` from a saved answer and otherwise leaves the answer unchanged. It changes 37 answers per placement. The same **unmodified official evaluator** scores those separate diagnostic files:",
        "",
        "| Placement | Untouched raw baseline | Lowercase-only diagnostic | Leading yes/no diagnostic | Yes/no diagnostic subgroup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summaries:
        lines.append(
            f"| {r['placement']} | 2.0% | {r['lowercase_only_diagnostic_percent']:.1f}% | {r['leading_yes_no_diagnostic_percent']:.1f}% | 81.08% (37 questions) |"
        )
    lines += [
        "",
        "The 30 percentage-point lift from 2% to 32% demonstrates an answer-format contribution using a simple deterministic transform; it is **not** a new raw-G2 score, a generation run, official full-benchmark accuracy, or a guarantee about new prompts. It only addresses yes/no phrasing and leaves numerical/other verbose answers unconverted. The rule was selected post hoc and operates only on prediction text; no reference answer decides an individual transformation. Keep its predictions, scores and provenance separate from the primary raw result. All three placements also tie under this diagnostic.",
        "",
        "![Quality diagnosis](quality_diagnosis.png)",
        "",
        "## Consequences for the thesis and next experiment",
        "",
        "G2 establishes successful placement execution and latency tradeoffs on the frozen contract. Its 2% raw official score is too low to support a claim of useful VQA task performance, even though paired placement differences are zero. The earlier hardware-aware gate's equal provisional scores remain historically true; official scoring does not retroactively change the gate or establish semantic equivalence of different sentences.",
        "",
        "Before a quality-constrained adaptive evaluation, predeclare a revised short-answer prompt/output contract, verify it on a calibration set, use the official evaluator with a pinned normalization/export policy, then evaluate on held-out question IDs. Re-run placement parity and timing under the new contract: shorter outputs alter decode time and may alter the adaptive crossover. Do not splice new short-answer quality with old sentence-generation latency. Keep current G2 as the archived baseline and report it alongside the revised workload. The scheduler's quality constraint needs a meaningful calibrated quality floor; always-wrong ties do not validate that floor.",
        "",
        "## Evidence and reproduction",
        "",
        "- `official_outputs/`: 81 untouched upstream evaluator results, with source and input hashes.\n- `predictions/`: official-format raw answers plus exporter provenance, one file per run/repetition.\n- `run_quality_summary.csv`: all 81 raw official and provisional scores.\n- `per_question_quality.csv`: all 100 paired raw answers, official/provisional scores, reference answers and formatting flags.\n- `paired_quality_differences.csv` and `paired_quality_by_run.csv`: aggregate and block/repetition paired differences.\n- `diagnostic_predictions/` and `diagnostic_outputs/`: explicitly separate lowercase and leading-yes/no transformations.\n- `evaluator_provenance.json`, `subset_provenance.json`, `evaluation_index.json`, `quality_validation.json`: pins, SHA-256s, alignment and denominator checks.\n- `*-answer-forms.json`: inspection of the actual conditional upstream normalization branch and unconditional diagnostic forms.",
        "",
        "The official source JSONs remain under ignored `data/raw/` or `artifacts/`; no full VQA dataset is redistributed. Re-fetch the exact pinned upstream checkout and official inputs if absent. See [REPRODUCE.md](REPRODUCE.md) for commands. No GPU or cloud device is required.",
        "",
        "Sources: [official VQA evaluation documentation](https://visualqa.org/evaluation.html), [pinned upstream evaluator](https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py), [official annotations download](https://visualqa.org/download.html), [frozen dataset protocol](../../../docs/physical_dataset_protocol_v1.md), [provisional runner](../../../src/adaptive_vlm/benchmark/runner.py), and [pinned upstream checksum source for PyPy](https://pypy.org/checksums.html).",
    ]
    # Both quality and analysis are direct children of the run folder.
    (HERE / "quality_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
