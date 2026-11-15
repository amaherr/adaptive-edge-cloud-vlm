"""Independent accounting fixtures and comparison with archived run summaries.

Run with: /usr/bin/python3 -m unittest discover -s PATH/analysis -v
"""

import csv
import json
import tarfile
import unittest
from pathlib import Path

import analyze_stages as analysis


class AttributionTests(unittest.TestCase):
    def test_split_parent_timers_are_not_double_counted(self):
        row = {
            "trial_id": "fixture",
            "observed_end_to_end_ms": 152,
            "metrics": {
                "vision_encoder_ms": 20,
                "projection_ms": 5,
                "edge_vision_end_to_end_ms": 30,
                "tensor_serialization_ms": 2,
                "network_round_trip_ms": 120,
                "server_total_ms": 80,
                "server_model_tensor_deserialization_ms": 2,
                "server_model_feature_transfer_ms": 1,
                "server_model_image_prefill_ms": 20,
                "server_model_prompt_prepare_ms": 1,
                "server_model_prompt_prefill_ms": 6,
                "server_model_decode_ms": 40,
                # The receive wait contains server processing and is not additive.
                "network_receive_ms": 100,
                "server_model_language_generation_ms": 48,
            },
        }
        result = analysis.decompose(row, "distributed")
        self.assertEqual(result["transport_residual_ms"], 40)
        self.assertEqual(result["vision_other_ms"], 5)
        self.assertEqual(result["server_other_ms"], 10)
        self.assertEqual(result["client_or_local_other_ms"], 0)
        self.assertEqual(sum(result.values()), 152)
        row["metrics"]["server_total_ms"] = 130
        with self.assertRaises(AssertionError):
            analysis.decompose(row, "distributed")

    def test_cloud_combined_stage_remains_combined(self):
        row = {
            "trial_id": "fixture",
            "observed_end_to_end_ms": 200,
            "metrics": {
                "image_read_ms": 4,
                "network_round_trip_ms": 190,
                "server_total_ms": 150,
                "server_model_image_encode_and_prefill_ms": 100,
                "server_model_prompt_prepare_ms": 1,
                "server_model_prompt_prefill_ms": 10,
                "server_model_decode_ms": 30,
            },
        }
        result = analysis.decompose(row, "cloud")
        self.assertEqual(result["cloud_image_encode_and_prefill_ms"], 100)
        self.assertEqual(result["transport_residual_ms"], 40)
        self.assertEqual(result["server_other_ms"], 9)
        self.assertEqual(result["client_or_local_other_ms"], 10)
        self.assertEqual(result["vision_encoder_ms"], 0)  # Bookkeeping only.
        self.assertEqual(sum(result.values()), 200)

    def test_local_accounting_preserves_unscoped_work(self):
        row = {
            "trial_id": "fixture",
            "observed_end_to_end_ms": 100,
            "metrics": {
                "vision_encoder_ms": 30,
                "projection_ms": 5,
                "vision_other_ms": 2,
                "feature_transfer_ms": 1,
                "image_prefill_ms": 20,
                "prompt_prepare_ms": 1,
                "prompt_prefill_ms": 5,
                "decode_ms": 30,
            },
        }
        result = analysis.decompose(row, "edge")
        self.assertEqual(result["client_or_local_other_ms"], 6)
        self.assertEqual(result["transport_residual_ms"], 0)
        self.assertEqual(sum(result.values()), 100)

    def test_bootstrap_preserves_constant_paired_difference(self):
        self.assertEqual(analysis.paired_ci([7] * 100), [7, 7])

    def test_output_matches_independent_original_summaries(self):
        root = Path(__file__).resolve().parent
        with (root / "stage_summary.csv").open() as stream:
            stages = list(csv.DictReader(stream))
        self.assertEqual(len(stages), 27)
        with tarfile.open(root.parent / "evidence.tar.gz", "r:gz") as archive:
            for record in stages:
                original = json.load(
                    archive.extractfile(
                        f"{analysis.PREFIX}/{record['block']}/{record['mode']}/summary.json"
                    )
                )
                self.assertAlmostEqual(
                    float(record["mean_latency_ms"]),
                    original["latency_ms_mean_success"],
                    places=8,
                )
                self.assertAlmostEqual(
                    sum(float(record[stage]) for stage in analysis.STAGES),
                    original["latency_ms_mean_success"],
                    places=8,
                )
        with (root / "placement_comparison.csv").open() as stream:
            comparisons = list(csv.DictReader(stream))
        winners = {r["block"]: r["best_mean_configuration"] for r in comparisons}
        self.assertEqual(
            winners,
            {
                b: "distributed" if b in {"N5", "N6"} else "cloud"
                for b in analysis.BLOCKS
            },
        )


if __name__ == "__main__":
    unittest.main()
