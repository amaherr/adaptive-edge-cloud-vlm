"""Independent checks on published scores, pairing and diagnostic separation."""

import csv
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("edge", "cloud", "distributed")


def read(path):
    return json.loads(path.read_text())


class QualityResultsTests(unittest.TestCase):
    def test_all_81_official_outputs_keep_full_denominator(self):
        records = read(HERE / "evaluation_index.json")
        self.assertEqual(len(records), 81)
        self.assertEqual(sum(r["question_count"] for r in records), 8100)
        for r in records:
            self.assertEqual(r["failed_answers_exported_empty"], 0)
            self.assertEqual(r["accuracy_percent"]["overall"], 2.0)
            scores = read(
                HERE
                / "official_outputs"
                / r["block"]
                / f"{r['mode']}-r{r['repetition']}.json"
            )["per_question_accuracy_percent"]
            self.assertEqual(len(scores), 100)
            self.assertEqual(sum(v == 100 for v in scores.values()), 2)
            self.assertEqual(sum(v == 0 for v in scores.values()), 98)

    def test_case_discrepancy_is_specific_and_preserved(self):
        with (HERE / "per_question_quality.csv").open() as stream:
            row = next(
                r for r in csv.DictReader(stream) if r["question_id"] == "50431001"
            )
        for mode in MODES:
            self.assertEqual(row[f"{mode}_answer"], "Hotel")
            self.assertEqual(float(row[f"{mode}_provisional_percent"]), 100)
            self.assertEqual(float(row[f"{mode}_official_percent"]), 0)
            self.assertEqual(
                row[f"{mode}_official_normalization_branch_active"], "False"
            )

    def test_pairing_does_not_confuse_equal_scores_with_equal_text(self):
        with (HERE / "paired_quality_differences.csv").open() as stream:
            pairs = list(csv.DictReader(stream))
        expected = {
            ("edge", "cloud"): 19,
            ("edge", "distributed"): 18,
            ("cloud", "distributed"): 16,
        }
        for row in pairs:
            self.assertEqual(float(row["mean_difference_pp"]), 0)
            self.assertEqual(int(row["tied_score_questions"]), 100)
            self.assertEqual(
                int(row["different_raw_answer_text_questions"]),
                expected[row["left"], row["right"]],
            )
        with (HERE / "per_question_quality.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            sum(len({r[f"{m}_answer"] for m in MODES}) > 1 for r in rows), 26
        )

    def test_diagnostics_cannot_overwrite_raw_answers_or_choose_gold(self):
        for mode in MODES:
            raw = {
                r["question_id"]: r["answer"]
                for r in read(HERE / "predictions/N1" / f"{mode}-r0.json")
            }
            leading = {
                r["question_id"]: r["answer"]
                for r in read(
                    HERE / "diagnostic_predictions" / f"{mode}-leading-yes-no.json"
                )
            }
            self.assertEqual(raw[50431001], "Hotel")
            self.assertEqual(leading[386313002], "yes")  # Gold is "no"; no oracle fix.
            diagnostic = read(
                HERE / "diagnostic_outputs" / f"{mode}-leading-yes-no.json"
            )
            self.assertEqual(
                diagnostic["per_question_accuracy_percent"]["386313002"], 0
            )
            self.assertEqual(diagnostic["accuracy_percent"]["overall"], 32.0)
            self.assertEqual(
                read(HERE / "official_outputs/N1" / f"{mode}-r0.json")[
                    "accuracy_percent"
                ]["overall"],
                2.0,
            )

    def test_provisional_loss_and_diagnostics_are_consistent(self):
        summary = read(HERE / "quality_summary.json")
        for placement in summary["placements"]:
            self.assertEqual(placement["official_raw_accuracy_percent"], 2.0)
            self.assertEqual(placement["provisional_accuracy_percent"], 3.0)
            self.assertEqual(placement["one_word_answers"], 3)
            self.assertEqual(placement["leading_yes_no_diagnostic_percent"], 32.0)
            self.assertEqual(placement["lowercase_only_diagnostic_percent"], 3.0)


if __name__ == "__main__":
    unittest.main()
