"""Reproduce G2 stage attribution from the immutable evidence archive.

Requires Python >=3.10, NumPy and Matplotlib. No model, GPU or cloud access.
Run: MPLCONFIGDIR=/tmp/g2-matplotlib /usr/bin/python3 PATH/analyze_stages.py
"""

# Generated Markdown tables/prose retain long lines and scientific Unicode.
# ruff: noqa: E501, RUF001
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tarfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/g2-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = "artifacts/unattended/site-preflight-20261110/campaign-heterogeneous-v2"
MODES = ("edge", "cloud", "distributed")
BLOCKS = tuple(f"N{i}" for i in range(1, 10))
SEED = 20261115
BOOTSTRAPS = 4000
STAGES = (
    "vision_encoder_ms",
    "projection_ms",
    "vision_other_ms",
    "serialization_ms",
    "transport_residual_ms",
    "deserialization_ms",
    "feature_transfer_ms",
    "image_prefill_ms",
    "cloud_image_encode_and_prefill_ms",
    "prompt_prepare_ms",
    "prompt_prefill_ms",
    "decode_ms",
    "server_other_ms",
    "client_or_local_other_ms",
)
LABELS = (
    "Edge vision encoder",
    "Edge projection",
    "Edge vision preparation/other",
    "Tensor serialization",
    "Transport/remote residual",
    "Tensor deserialization",
    "Feature transfer",
    "Image prefill",
    "Cloud image encoding + prefill (combined)",
    "Prompt preparation",
    "Prompt prefill",
    "Decoding",
    "Server other",
    "Client/local other",
)
COLORS = (
    "#4477AA",
    "#66CCEE",
    "#A8C7DF",
    "#AA3377",
    "#EE7733",
    "#CCBB44",
    "#BBBBBB",
    "#228833",
    "#66AA55",
    "#DDCC77",
    "#44AA99",
    "#884488",
    "#999933",
    "#AAAAAA",
)


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(name, records):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def mean(values):
    return float(np.mean(values))


def paired_ci(values):
    """Resample request IDs, preserving each ID's repeated observations."""
    rng = np.random.default_rng(SEED)
    a = np.asarray(values)
    index = rng.integers(0, len(a), size=(BOOTSTRAPS, len(a)))
    distribution = a[index].mean(axis=1)
    return [float(v) for v in np.quantile(distribution, [0.025, 0.975])]


def ranks(values):
    a = np.asarray(values)
    return np.array([np.sum(a < x) + (np.sum(a == x) - 1) / 2 for x in a])


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def decompose(row, mode):
    """Exclusive additive partition. Parent timers are reported separately."""
    m = row["metrics"]
    s = dict.fromkeys(STAGES, 0.0)
    remote = mode != "edge"
    key = (lambda name: "server_model_" + name) if remote else (lambda name: name)
    for name in ("prompt_prepare_ms", "prompt_prefill_ms", "decode_ms"):
        s[name] = float(m[key(name)])
    if mode != "cloud":
        for name in ("vision_encoder_ms", "projection_ms"):
            s[name] = float(m[name])
        s["vision_other_ms"] = (
            m["edge_vision_end_to_end_ms"] - s["vision_encoder_ms"] - s["projection_ms"]
            if mode == "distributed"
            else m["vision_other_ms"]
        )
        s["image_prefill_ms"] = float(m[key("image_prefill_ms")])
        s["feature_transfer_ms"] = float(m[key("feature_transfer_ms")])
    if mode == "cloud":
        s["cloud_image_encode_and_prefill_ms"] = float(
            m[key("image_encode_and_prefill_ms")]
        )
    if remote:
        s["transport_residual_ms"] = m["network_round_trip_ms"] - m["server_total_ms"]
        server_parts = (
            "image_prefill_ms",
            "cloud_image_encode_and_prefill_ms",
            "prompt_prepare_ms",
            "prompt_prefill_ms",
            "decode_ms",
            "deserialization_ms",
            "feature_transfer_ms",
        )
        if mode == "distributed":
            s["serialization_ms"] = m["tensor_serialization_ms"]
            s["deserialization_ms"] = m[key("tensor_deserialization_ms")]
        s["server_other_ms"] = m["server_total_ms"] - sum(s[k] for k in server_parts)
    s["client_or_local_other_ms"] = row["observed_end_to_end_ms"] - sum(s.values())
    assert min(s.values()) >= -1e-6, (row["trial_id"], mode, s)
    assert abs(sum(s.values()) - row["observed_end_to_end_ms"]) < 1e-6
    return s


def save_plot(fig, name):
    for extension in ("svg", "png"):
        metadata = {"Date": None} if extension == "svg" else {"Software": "G2 analysis"}
        fig.savefig(
            HERE / f"{name}.{extension}",
            dpi=180,
            bbox_inches="tight",
            metadata=metadata,
        )
        if extension == "svg":
            path = HERE / f"{name}.svg"
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def main():
    archive = ROOT / "evidence.tar.gz"
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_sha == (ROOT / "SHA256SUMS").read_text().split()[0]
    inventory = json.loads((ROOT / "archive_inventory.json").read_text())
    with tarfile.open(archive, "r:gz") as tar:
        blobs = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
    assert set(blobs) == set(inventory)
    for name, blob in blobs.items():
        assert len(blob) == inventory[name]["bytes"]
        assert hashlib.sha256(blob).hexdigest() == inventory[name]["sha256"]

    def read(path):
        return json.loads(blobs[f"{PREFIX}/{path}"])

    samples, stats, stages_long, conditions, memories = [], {}, [], [], []
    raw = {}
    requests = None
    cluster = {}
    repeat = {}
    maximum_closure_error = 0.0
    for block in BLOCKS:
        plan = read(f"{block}/plan.json")
        validation = read(f"{block}/validation_report.json")
        profile = read(f"{block}/profile/summary.json")
        assert profile == validation["profile_summary"]
        assert profile["controlled_network_gate_passed"] and profile["row_count"] == 903
        for relative, digest in validation["artifact_hashes"].items():
            assert (
                hashlib.sha256(blobs[f"{PREFIX}/{block}/{relative}"]).hexdigest()
                == digest
            )
        for relative, field in (
            ("network_pre.json", "network_pre_sha256"),
            ("network_post.json", "network_post_sha256"),
            ("profile/profile.jsonl", "profile_sha256"),
            ("edge_gpu_samples.csv", "edge_gpu_samples_sha256"),
        ):
            assert (
                hashlib.sha256(blobs[f"{PREFIX}/{block}/{relative}"]).hexdigest()
                == validation[field]
            )
        network = read(f"{block}/profile_specification.json")["network_state"]
        condition = {
            "block": block,
            "configured_rate_mbps": plan["configured_rate_mbps"],
            "configured_added_rtt_ms": plan["configured_added_rtt_ms"],
            "policy_order": ",".join(plan["policy_order"]),
            "measured_uplink_mbps": network["uplink_bps"] / 1e6,
            "measured_downlink_mbps": network["downlink_bps"] / 1e6,
            "measured_rtt_ms": network["rtt_ms"],
        }
        conditions.append(condition)
        for mode in MODES:
            all_rows = [
                json.loads(line)
                for line in blobs[
                    f"{PREFIX}/{block}/{mode}/requests.jsonl"
                ].splitlines()
            ]
            assert len(all_rows) == 301 and sum(r["warmup"] for r in all_rows) == 1
            rs = [r for r in all_rows if not r["warmup"]]
            assert len(rs) == 300 and Counter(r["status"] for r in rs) == {"ok": 300}
            assert len({(r["request_id"], r["repetition"]) for r in rs}) == 300
            ids = sorted({r["request_id"] for r in rs})
            if requests is None:
                requests = ids
            assert requests == ids
            raw[block, mode] = rs
            grouped = {rid: [r for r in rs if r["request_id"] == rid] for rid in ids}
            assert all(len(group) == 3 for group in grouped.values())
            cluster[block, mode] = np.array(
                [
                    mean([r["observed_end_to_end_ms"] for r in grouped[rid]])
                    for rid in ids
                ]
            )
            repeat[block, mode] = np.array(
                [
                    [
                        next(
                            r["observed_end_to_end_ms"]
                            for r in grouped[rid]
                            if r["repetition"] == rep
                        )
                        for rep in range(3)
                    ]
                    for rid in ids
                ]
            )
            decomposed = [decompose(r, mode) for r in rs]
            mean_stages = {s: mean([row[s] for row in decomposed]) for s in STAGES}
            latency = mean([r["observed_end_to_end_ms"] for r in rs])
            dominant = max(mean_stages, key=mean_stages.get)
            macro = (
                {
                    "vision_path_ms": mean(
                        [r["metrics"]["edge_vision_end_to_end_ms"] for r in rs]
                    ),
                    "cloud_processing_ms": mean(
                        [r["metrics"]["server_total_ms"] for r in rs]
                    ),
                    "transport_residual_ms": mean_stages["transport_residual_ms"],
                }
                if mode == "distributed"
                else {"vision_encoder_ms": mean_stages["vision_encoder_ms"]}
                if mode == "edge"
                else {
                    "cloud_processing_ms": mean(
                        [r["metrics"]["server_total_ms"] for r in rs]
                    ),
                    "transport_residual_ms": mean_stages["transport_residual_ms"],
                }
            )
            metrics = {
                name: mean([r["metrics"][name] for r in rs])
                for name in rs[0]["metrics"]
                if name.endswith("_ms")
            }
            stats[block, mode] = {
                "block": block,
                "mode": mode,
                "n": 300,
                "mean_latency_ms": latency,
                "median_latency_ms": float(
                    np.median([r["observed_end_to_end_ms"] for r in rs])
                ),
                "p95_latency_ms": float(
                    np.quantile([r["observed_end_to_end_ms"] for r in rs], 0.95)
                ),
                "dominant_atomic_stage": dominant,
                "dominant_atomic_share_percent": 100 * mean_stages[dominant] / latency,
                "dominant_macro_stage": max(macro, key=macro.get),
                "mean_server_processing_ms": metrics.get("server_total_ms", 0),
                "mean_exchange_ms": metrics.get("network_round_trip_ms", 0),
                **mean_stages,
            }
            for stage in STAGES:
                stages_long.append(
                    {
                        "block": block,
                        "mode": mode,
                        "stage": stage,
                        "measurement_status": (
                            "not_separately_instrumented_combined"
                            if mode == "cloud"
                            and stage
                            in {
                                "vision_encoder_ms",
                                "projection_ms",
                                "image_prefill_ms",
                            }
                            else "structural_zero_not_applicable"
                            if all(s[stage] == 0 for s in decomposed)
                            else "attributed_timer_or_residual"
                        ),
                        "mean_ms": mean_stages[stage],
                        "p95_ms": float(
                            np.quantile([s[stage] for s in decomposed], 0.95)
                        ),
                        "share_of_mean_latency_percent": 100
                        * mean_stages[stage]
                        / latency,
                    }
                )
            for row, stage in zip(rs, decomposed, strict=True):
                maximum_closure_error = max(
                    maximum_closure_error,
                    abs(sum(stage.values()) - row["observed_end_to_end_ms"]),
                )
                samples.append(
                    {
                        "block": block,
                        "mode": mode,
                        "request_id": row["request_id"],
                        "repetition": row["repetition"],
                        "observed_ms": row["observed_end_to_end_ms"],
                        **stage,
                    }
                )
            prefix = "server_model_" if mode == "cloud" else ""
            allocated = [
                r["metrics"][prefix + "vision_cuda_peak_allocated_mib"] for r in rs
            ]
            reserved = [
                r["metrics"][prefix + "vision_cuda_peak_reserved_mib"] for r in rs
            ]
            memories.append(
                {
                    "block": block,
                    "mode": mode,
                    "edge_or_cloud": "cloud" if mode == "cloud" else "edge",
                    "mean_peak_allocated_mib": mean(allocated),
                    "max_peak_allocated_mib": max(allocated),
                    "max_peak_reserved_mib": max(reserved),
                    "mean_vision_encoder_ms": mean_stages["vision_encoder_ms"]
                    if mode != "cloud"
                    else None,
                }
            )

    comparisons = []
    for block in BLOCKS:
        matrix = np.array([cluster[block, m] for m in MODES])
        delta = matrix[2] - matrix[1]
        ci = paired_ci(delta)
        avg = matrix.mean(axis=1)
        winner = MODES[int(np.argmin(avg))]
        win_counts = Counter(MODES[j] for j in np.argmin(matrix, axis=0))
        comparisons.append(
            {
                **conditions[BLOCKS.index(block)],
                **{f"{m}_mean_ms": float(avg[i]) for i, m in enumerate(MODES)},
                "best_mean_configuration": winner,
                "distributed_minus_cloud_ms": mean(delta),
                "paired_request_ci95_low_ms": ci[0],
                "paired_request_ci95_high_ms": ci[1],
                "distributed_saving_vs_cloud_percent": 100 * (avg[1] - avg[2]) / avg[1],
                **{f"request_mean_winner_{m}_count": win_counts[m] for m in MODES},
                "distributed_wins_matched_repetition_percent": 100
                * mean((repeat[block, "distributed"] < repeat[block, "cloud"]).ravel()),
                "retrospective_request_oracle_ms": mean(matrix.min(axis=0)),
            }
        )
    table("request_stage_samples.csv", samples)
    table("stage_summary.csv", list(stats.values()))
    table("stage_shares.csv", stages_long)
    table("network_conditions.csv", conditions)
    table("placement_comparison.csv", comparisons)
    table("memory_summary.csv", memories)

    # Descriptive equal-delay bandwidth fit, explicitly not a validated selector.
    rates = np.array([c["measured_uplink_mbps"] for c in conditions[:6]])
    y = np.array([c["distributed_minus_cloud_ms"] / 1000 for c in comparisons[:6]])
    design = np.column_stack([np.ones(6), 1 / rates])
    fit = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ fit
    crossovers = []
    for omitted in range(6):
        keep = np.arange(6) != omitted
        a, b = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        crossovers.append(float(-b / a))
    model = {
        "scope": "Descriptive N1-N6 fit only; compute state changes across blocks",
        "formula_seconds": "T_distributed - T_cloud = a + b / uplink_Mbps",
        "a_seconds": float(fit[0]),
        "b_seconds_mbps": float(fit[1]),
        "fitted_crossover_mbps": float(-fit[1] / fit[0]),
        "leave_one_condition_out_crossover_range_mbps": [
            min(crossovers),
            max(crossovers),
        ],
        "in_sample_rmse_ms": float(np.sqrt(mean((prediction - y) ** 2)) * 1000),
        "observed_nominal_crossover_bracket_mbps": [25, 50],
        "observed_measured_crossover_bracket_mbps": [float(rates[3]), float(rates[4])],
        "not_a_confidence_interval": "Leave-one-condition-out range is a sensitivity check, not CI or independent validation",
    }
    dump("descriptive_crossover.json", model)

    transfers = []
    for block in BLOCKS:
        c = conditions[BLOCKS.index(block)]
        for mode in ("cloud", "distributed"):
            rs = raw[block, mode]
            up = mean([r["metrics"]["request_wire_bytes"] for r in rs])
            down = mean([r["metrics"]["response_wire_bytes"] for r in rs])
            ideal = (
                up * 8 / (c["measured_uplink_mbps"] * 1e6) * 1000
                + down * 8 / (c["measured_downlink_mbps"] * 1e6) * 1000
            )
            residual = stats[block, mode]["transport_residual_ms"]
            transfers.append(
                {
                    "block": block,
                    "mode": mode,
                    "mean_request_wire_bytes": up,
                    "mean_response_wire_bytes": down,
                    "mean_payload_bytes": mean(
                        [
                            r["metrics"][
                                "image_bytes"
                                if mode == "cloud"
                                else "projected_payload_bytes"
                            ]
                            for r in rs
                        ]
                    ),
                    "transport_residual_ms": residual,
                    "serialization_only_transfer_reference_ms": ideal,
                    "residual_minus_reference_ms": residual - ideal,
                    "mean_connect_ms": mean(
                        [r["metrics"]["network_connect_ms"] for r in rs]
                    ),
                    "mean_socket_send_ms": mean(
                        [r["metrics"]["network_send_ms"] for r in rs]
                    ),
                    "mean_socket_receive_ms": mean(
                        [r["metrics"]["network_receive_ms"] for r in rs]
                    ),
                }
            )
    table("transfer_accounting.csv", transfers)

    # Each bootstrap resamples the same 100 IDs across all nine conditions.
    overall = np.array(
        [np.mean([cluster[b, m] for b in BLOCKS], axis=0) for m in MODES]
    )
    static_index = int(np.argmin(overall.mean(axis=1)))
    best_block = np.mean(
        [
            cluster[b, comparisons[j]["best_mean_configuration"]]
            for j, b in enumerate(BLOCKS)
        ],
        axis=0,
    )
    retrospective = np.mean(
        [np.array([cluster[b, m] for m in MODES]).min(axis=0) for b in BLOCKS], axis=0
    )
    adaptive = {
        "weighting": "Equal weight per condition and request; 3 repeats averaged per request",
        "best_static": MODES[static_index],
        "best_static_mean_ms": mean(overall[static_index]),
        "retrospective_condition_choice_mean_ms": mean(best_block),
        "retrospective_condition_saving_ms": mean(overall[static_index] - best_block),
        "retrospective_condition_saving_percent": 100
        * mean(overall[static_index] - best_block)
        / mean(overall[static_index]),
        "conditional_fixed_choice_saving_ci95_ms": paired_ci(
            overall[static_index] - best_block
        ),
        "retrospective_request_oracle_mean_ms": mean(retrospective),
        "retrospective_request_oracle_saving_percent": 100
        * mean(overall[static_index] - retrospective)
        / mean(overall[static_index]),
        "caution": "Both choices use observed candidate outcomes: optimistic in-sample references, not adaptive runtime results. CI conditions on selected block policies.",
    }
    dump("adaptive_implications.json", adaptive)

    feature_rows = []
    for mode in MODES:
        per_id = {
            rid: [r for b in BLOCKS for r in raw[b, mode] if r["request_id"] == rid]
            for rid in requests
        }
        decode = [
            mean(
                [
                    r["metrics"][
                        ("server_model_" if mode != "edge" else "") + "decode_ms"
                    ]
                    for r in per_id[rid]
                ]
            )
            for rid in requests
        ]
        tokens = [
            mean(
                [
                    r["metrics"][
                        ("server_model_" if mode != "edge" else "") + "decode_steps"
                    ]
                    for r in per_id[rid]
                ]
            )
            for rid in requests
        ]
        areas = [
            per_id[rid][0]["metrics"][
                ("server_model_" if mode == "cloud" else "") + "image_width"
            ]
            * per_id[rid][0]["metrics"][
                ("server_model_" if mode == "cloud" else "") + "image_height"
            ]
            for rid in requests
        ]
        visual = [
            mean(
                [
                    r["metrics"][
                        "server_model_image_encode_and_prefill_ms"
                        if mode == "cloud"
                        else "vision_encoder_ms"
                    ]
                    for r in per_id[rid]
                ]
            )
            for rid in requests
        ]
        feature_rows.append(
            {
                "mode": mode,
                "request_ids": 100,
                "decode_steps_vs_decode_ms_spearman": spearman(tokens, decode),
                "image_area_vs_visual_stage_spearman": spearman(areas, visual),
                "decode_ms_per_step_ratio_of_totals": sum(decode) / sum(tokens),
                "mean_decode_steps": mean(tokens),
                "mean_visual_stage_ms": mean(visual),
            }
        )
    table("task_feature_associations.csv", feature_rows)

    # Replicate-level compute drift is evidence against a bandwidth-only cost model.
    drift = []
    for block in BLOCKS:
        for mode in MODES:
            for rep in range(3):
                rs = [r for r in raw[block, mode] if r["repetition"] == rep]
                pref = "server_model_" if mode != "edge" else ""
                drift.append(
                    {
                        "block": block,
                        "mode": mode,
                        "repetition": rep,
                        "mean_latency_ms": mean(
                            [r["observed_end_to_end_ms"] for r in rs]
                        ),
                        "mean_encoder_ms": mean(
                            [r["metrics"].get("vision_encoder_ms", 0) for r in rs]
                        ),
                        "mean_image_prefill_or_combined_ms": mean(
                            [
                                r["metrics"][
                                    pref
                                    + (
                                        "image_encode_and_prefill_ms"
                                        if mode == "cloud"
                                        else "image_prefill_ms"
                                    )
                                ]
                                for r in rs
                            ]
                        ),
                    }
                )
    table("replicate_drift.csv", drift)
    validation = {
        "archive_sha256": archive_sha,
        "archive_files_verified": len(blobs),
        "measured_rows": len(samples),
        "warmup_rows_excluded": 27,
        "requests_per_block_mode": 300,
        "unique_request_ids": len(requests),
        "additive_closure_max_absolute_error_ms": maximum_closure_error,
        "nonnegative_components": True,
        "all_block_artifact_hashes_verified": True,
        "all_controlled_gates_passed": True,
        "ci_method": "Paired request-ID cluster percentile bootstrap; 3 repetitions averaged within ID",
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    dump("analysis_validation.json", validation)
    make_figures(stats, comparisons, memories, transfers, rates, model)
    write_report(stats, comparisons, memories, transfers, model, adaptive, feature_rows)
    print(
        json.dumps(
            {
                "verified_rows": len(samples),
                "archive_sha256": archive_sha,
                "crossover_mbps": model["fitted_crossover_mbps"],
                "adaptive": adaptive,
            },
            indent=2,
        )
    )


def make_figures(stats, comparisons, memories, transfers, rates, model):
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "physical-g2-20261110"})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for mode, ax in zip(MODES, axes, strict=True):
        base = np.zeros(9)
        for key, label, color in zip(STAGES, LABELS, COLORS, strict=True):
            values = np.array([stats[b, mode][key] / 1000 for b in BLOCKS])
            ax.bar(BLOCKS, values, bottom=base, label=label, color=color)
            base += values
        ax.set_ylabel("Mean latency (s)")
        ax.set_title(mode.capitalize(), loc="left")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Exclusive stage attribution: 300 measured requests per condition and placement"
    )
    fig.subplots_adjust(bottom=0.16, hspace=0.35)
    save_plot(fig, "stage_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = {"edge": "#4477AA", "cloud": "#228833", "distributed": "#EE7733"}
    for mode in MODES:
        axes[0].plot(
            rates,
            [stats[b, mode]["mean_latency_ms"] / 1000 for b in BLOCKS[:6]],
            "o-",
            color=palette[mode],
            label=mode,
        )
    axes[0].axvline(
        model["fitted_crossover_mbps"],
        color="gray",
        ls=":",
        label="Descriptive crossing",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Measured uplink (Mb/s); N1–N6, +20 ms configured RTT")
    axes[0].set_ylabel("Mean latency (s)")
    axes[0].legend()
    for mode in ("cloud", "distributed"):
        selected = [comparisons[i] for i in (6, 2, 7, 8)]
        axes[1].plot(
            [r["measured_rtt_ms"] for r in selected],
            [r[f"{mode}_mean_ms"] / 1000 for r in selected],
            "o-",
            color=palette[mode],
            label=mode,
        )
    axes[1].set_xlabel("Measured echo RTT (ms); N7, N3, N8, N9 at 10 Mb/s")
    axes[1].set_ylabel("Mean latency (s)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Placement choice changes with bandwidth; RTT sweep does not reverse its ranking"
    )
    fig.tight_layout()
    save_plot(fig, "placement_latency")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [r["block"] for r in comparisons]
    delta = np.array([r["distributed_minus_cloud_ms"] / 1000 for r in comparisons])
    low = np.array([r["paired_request_ci95_low_ms"] / 1000 for r in comparisons])
    high = np.array([r["paired_request_ci95_high_ms"] / 1000 for r in comparisons])
    axes[0].errorbar(
        labels,
        delta,
        yerr=[delta - low, high - delta],
        fmt="o",
        capsize=4,
        color="#4477AA",
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Distributed − cloud mean latency (s)")
    axes[0].set_title("Paired 95% request-cluster bootstrap intervals")
    axes[0].grid(axis="y", alpha=0.2)
    for mode in ("cloud", "distributed"):
        rs = [r for r in transfers if r["mode"] == mode][:6]
        axes[1].plot(
            rates,
            [r["transport_residual_ms"] / 1000 for r in rs],
            "o-",
            color=palette[mode],
            label=f"{mode}: measured residual",
        )
        axes[1].plot(
            rates,
            [r["serialization_only_transfer_reference_ms"] / 1000 for r in rs],
            "--",
            color=palette[mode],
            label=f"{mode}: bytes/rate reference",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Measured uplink (Mb/s); N1–N6")
    axes[1].set_ylabel("Transfer-related time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    save_plot(fig, "transfer_and_uncertainty")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode in ("edge", "distributed"):
        rs = [r for r in memories if r["mode"] == mode]
        axes[0].plot(
            BLOCKS,
            [r["max_peak_allocated_mib"] for r in rs],
            "o-",
            color=palette[mode],
            label=mode,
        )
        axes[1].plot(
            BLOCKS,
            [r["mean_vision_encoder_ms"] / 1000 for r in rs],
            "o-",
            color=palette[mode],
            label=mode,
        )
    axes[0].axhline(
        4096, color="black", ls="--", label="Edge dedicated VRAM (4096 MiB)"
    )
    axes[0].set_ylabel("Maximum per-request peak allocated (MiB)")
    axes[0].set_title("Process-wide request peak, not encoder-only allocation")
    axes[1].set_ylabel("Mean edge vision encoder time (s)")
    axes[1].set_title("Full-model and vision-only residency differ")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_plot(fig, "memory_and_vision")


def write_report(stats, comparisons, memories, transfers, model, adaptive, features):
    lines = [
        "# G2 stage-level results and implications for adaptive placement",
        "",
        "Analysis of the completed 10–11 November 2026 physical run. All 8,100 measured rows are used; 27 warmups are excluded. Each of nine conditions has 100 frozen request IDs × three repetitions × three placements. All archived hashes and controlled profile gates were rechecked. No new inference was performed.",
        "",
        "## Main result",
        "",
        "**Cloud-only is the best mean-latency placement in N1–N4 and N7–N9; the projected split wins N5–N6. Full edge is never the mean-latency winner on this host pair.** The result supports a state-dependent placement choice, not a claim that splitting is always faster. The always-cloud baseline is already strong.",
        "",
        "| Condition | Rate, nominal (Mb/s) | Uplink, measured (Mb/s) | Echo RTT (ms) | Edge (s) | Cloud (s) | Split (s) | Best mean | Split − cloud, paired 95% CI (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['block']} | {r['configured_rate_mbps']} | {r['measured_uplink_mbps']:.2f} | {r['measured_rtt_ms']:.2f} | {r['edge_mean_ms'] / 1000:.3f} | {r['cloud_mean_ms'] / 1000:.3f} | {r['distributed_mean_ms'] / 1000:.3f} | {r['best_mean_configuration']} | {r['distributed_minus_cloud_ms']:.0f} [{r['paired_request_ci95_low_ms']:.0f}, {r['paired_request_ci95_high_ms']:.0f}] |"
        )
    lines += [
        "",
        "Positive differences favor cloud. Intervals resample 100 paired request IDs, averaging their three repetitions first (4,000 bootstrap replicates, fixed seed). They describe request mix uncertainty conditional on this run; they do not cover day-to-day host/network variation or establish causal network effects. Samples are repeated workloads, not 300 independent request types.",
        "",
        "![Placement latency](placement_latency.png)",
        "",
        "## Dominant bottleneck in every condition",
        "",
        "A bottleneck here is the largest mean additive latency component, not a measured throughput saturation point. The macro view groups cloud timers; the fine view separates prefill and decoding. Parent timers are never stacked with their children.",
        "",
        "| Condition | Edge: largest fine component | Cloud: largest identifiable component | Split: largest fine component | Split: largest macro component |",
        "| --- | --- | --- | --- | --- |",
    ]
    for b in BLOCKS:
        text = []
        for m in MODES:
            s = stats[b, m]
            name = s["dominant_atomic_stage"]
            text.append(
                f"{name.removesuffix('_ms')} {s[name] / 1000:.3f}s ({s['dominant_atomic_share_percent']:.1f}%)"
            )
        lines.append(
            f"| {b} | {text[0]} | {text[1]} | {text[2]} | {stats[b, 'distributed']['dominant_macro_stage'].removesuffix('_ms')} |"
        )
    lines += [
        "",
        "### Split-stage budgets (mean milliseconds per request)",
        "",
        "| Condition | Encoder | Projection | Tensor serialize | Transport residual | Cloud total (parent) | Image prefill | Prompt prefill | Decode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in BLOCKS:
        v = stats[b, "distributed"]
        keys = [
            "vision_encoder_ms",
            "projection_ms",
            "serialization_ms",
            "transport_residual_ms",
            "mean_server_processing_ms",
            "image_prefill_ms",
            "prompt_prefill_ms",
            "decode_ms",
        ]
        lines.append("| " + b + " | " + " | ".join(f"{v[k]:.2f}" for k in keys) + " |")
    lines += [
        "",
        "Cloud total is a parent timer and must not be added to its image/prompt/decode children. Tensor deserialization is about 1–2 ms and serialization about 1.2 ms; projected serialization and projection are small relative to the dominant stages. Exact means/p95s and residuals are in the CSVs. A serializer-only optimization cannot remove the bandwidth penalty.",
        "",
        "### Edge and cloud comparison of prefill/decoding (mean ms)",
        "",
        "| Condition | Edge encoder | Edge projection | Edge image prefill | Edge prompt prefill | Edge decode | Cloud combined image stage | Cloud prompt prefill | Cloud decode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in BLOCKS:
        e, c = stats[b, "edge"], stats[b, "cloud"]
        vals = [
            e[k]
            for k in [
                "vision_encoder_ms",
                "projection_ms",
                "image_prefill_ms",
                "prompt_prefill_ms",
                "decode_ms",
            ]
        ] + [
            c[k]
            for k in [
                "cloud_image_encode_and_prefill_ms",
                "prompt_prefill_ms",
                "decode_ms",
            ]
        ]
        lines.append("| " + b + " | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    n1 = stats["N1", "distributed"]
    n6 = stats["N6", "distributed"]
    lines += [
        "",
        f"For the split, transport/remote residual falls from **{n1['transport_residual_ms'] / 1000:.3f}s ({100 * n1['transport_residual_ms'] / n1['mean_latency_ms']:.1f}%) in N1** to **{n6['transport_residual_ms'] / 1000:.3f}s ({100 * n6['transport_residual_ms'] / n6['mean_latency_ms']:.1f}%) in N6**. Macro transport dominates N1–N4 and N7–N9; cloud processing dominates N5–N6. At finer granularity, residual transfer remains the largest single component in N5, while image prefill is largest in N6. Those statements use different aggregation levels and are compatible.",
        "",
        "![Exclusive stage decomposition](stage_decomposition.png)",
        "",
        "## Timing definitions and what cannot be recovered",
        "",
        "The decomposition is reconstructed per request, then averaged. `stage_summary.csv` contains the 27 exclusive mean budgets plus parent server/exchange timers; `stage_shares.csv` contains per-component means, p95s and shares. Percentages are ratios of means. Zero-valued absent categories in the exclusive-budget CSVs are bookkeeping zeros, not proof that uninstrumented work takes zero time; `stage_shares.csv` labels the unavailable cloud sub-stages explicitly. Component p95s do **not** sum to end-to-end p95.",
        "",
        "| Requested stage | Evidence and scope |",
        "| --- | --- |",
        "| Vision encoding | `vision_encoder_ms`: synchronized `_vis_enc` calls on edge/full and split/vision-only paths. Not separately timed on cloud-only. |",
        "| Projection | `projection_ms`: synchronized `_vis_proj` calls on edge/full and split. Cloud-only projection is inside a combined timer. |",
        "| Serialization | `tensor_serialization_ms`: projected-tensor conversion/serialization before the TCP exchange. Frame/JSON/checksum work is outside this timer. Cloud image upload uses `image_read_ms`, not this tensor stage. |",
        "| Network transfer | `network_round_trip_ms − server_total_ms`: **transport/remote residual**, not isolated wire time. It includes connection, upload/download, TCP effects, framing/checksums and remote work outside the server timer (including raw-image temporary-file staging). No packet capture, one-way timestamping or isolated server queue timer exists. |",
        "| Cloud processing | `server_total_ms`, a parent timer for the inference function. Split: deserialization + feature transfer + image prefill + prompt preparation/prefill + decode + server residual. Cloud-only: combined image encoding/prefill + prompt preparation/prefill + decode + server residual. |",
        "| Image prefill | `image_prefill_ms` is separate for edge and split. Cloud-only `server_model_image_encode_and_prefill_ms` combines image preparation, vision encoding, projection and image prefill. **These cloud-only sub-stages are not identifiable from this run.** |",
        "| Prompt prefill | `prompt_prefill_ms` includes the initial text forward pass producing the first next-token candidate. |",
        "| Decoding | `decode_ms` times the subsequent autoregressive loop and synchronization; `decode_steps` records its iterations. This is not complete time-to-first-token or a streaming token-arrival measurement. |",
        "| Other | Explicit residuals close the additive budget. Local image open/preparation and unscoped local/model overhead are retained, not relabeled as network or GPU kernels. |",
        "",
        "For the synchronous split: `T = edge_vision_total + tensor_serialization + transport_residual + server_total + client_other`. The vision parent is split into encoder, projection and vision-other. Server processing is split as above. Cloud-only: `T = image_read + transport_residual + server_total + client_other` (image read is retained in the client/local-other bin of the exclusive plot). For local edge: measured vision, projection, image prefill, prompt, decode and explicit local remainder sum to observed latency. Every row closes within floating-point precision; no negative component was accepted.",
        "",
        "**Do not add `network_receive_ms` to server processing.** The receive wait includes the inference execution while the client waits for a reply. `network_send_ms` measures blocking writes/socket behavior, not when the complete payload reaches the server. For small raw images the kernel can buffer the write almost immediately despite a slow link; the rest appears in receive wait/residual. Large split tensors make the send call block. These socket timers are diagnostics, not independent physical transfer stages.",
        "",
        "Sources: [adapter timers](../../../src/adaptive_vlm/models/moondream.py), [split executor](../../../src/adaptive_vlm/execution/distributed.py), [cloud executor](../../../src/adaptive_vlm/execution/cloud.py), [socket exchange](../../../src/adaptive_vlm/transport/client.py), [server timing scope](../../../src/adaptive_vlm/transport/server.py), and [profile construction](../../../src/adaptive_vlm/benchmark/profile.py). Paths are relative to the repository root via the results folder; see the corrected source links below if viewing this nested report.",
        "",
        "## Payload size explains the bandwidth penalty",
        "",
    ]
    payload_cloud = mean(
        [r["mean_payload_bytes"] for r in transfers if r["mode"] == "cloud"]
    )
    payload_split = mean(
        [r["mean_payload_bytes"] for r in transfers if r["mode"] == "distributed"]
    )
    lines += [
        f"Projected features are fixed at **729 × 2048 × 2 = {payload_split:,.0f} bytes**. The mean raw compressed image is **{payload_cloud:,.0f} bytes**, so the split payload is **{payload_split / payload_cloud:.1f}× larger** on average. Splitting after projection is not a compression win over JPEG on this workload. Response framing and request metadata are retained in `transfer_accounting.csv`.",
        "",
        "The `bytes / measured rate` curves are serialization-only transfer references, not packet-level predictions or strict bounds. Measured rate comes from separate bidirectional iperf probes; inference uses different socket/payload sizes and times. RTT, TCP behavior and remote staging remain in the difference. Never subtract this reference and call the remainder pure queue delay.",
        "",
        "![Transfer and paired uncertainty](transfer_and_uncertainty.png)",
        "",
        "## Crossover and delay sensitivity",
        "",
        f"For the +20 ms configured RTT bandwidth sweep N1–N6, the observed crossing is bracketed by **25 and 50 Mb/s nominal** ({model['observed_measured_crossover_bracket_mbps'][0]:.2f}–{model['observed_measured_crossover_bracket_mbps'][1]:.2f} Mb/s measured). A descriptive fit to the six condition means gives:",
        "",
        f"`T_split − T_cloud ≈ {model['a_seconds']:.3f} + {model['b_seconds_mbps']:.3f}/B` seconds, with `B` in measured uplink Mb/s. Its crossing is **{model['fitted_crossover_mbps']:.1f} Mb/s**. Omitting one condition at a time gives **{model['leave_one_condition_out_crossover_range_mbps'][0]:.1f}–{model['leave_one_condition_out_crossover_range_mbps'][1]:.1f} Mb/s**; that is a sensitivity range, not a confidence interval. See `descriptive_crossover.json`.",
        "",
        "This is a calibration hypothesis, **not a validated universal switching threshold**. Split vision and cloud image-prefill times also change across this sequential sweep; the fitted intercept is therefore not a clean compute-only constant. Test intermediate bandwidths with randomized/interleaved placement and thermal/memory telemetry before deploying a threshold.",
        "",
        "At fixed 10 Mb/s, N7/N3/N8/N9 change configured added RTT from 5 to 20 to 50 to 100 ms. Cloud remains the best mean placement in all four. This supports retaining RTT in prediction but does not justify claiming a separately identified linear RTT coefficient; TCP buffering, scheduling, and state drift can contribute.",
        "",
        "## System state: full-model residency changes vision cost",
        "",
    ]
    edge_encoder = mean([stats[b, "edge"]["vision_encoder_ms"] for b in BLOCKS])
    split_encoder = mean([stats[b, "distributed"]["vision_encoder_ms"] for b in BLOCKS])
    edge_memory = max(
        r["max_peak_allocated_mib"] for r in memories if r["mode"] == "edge"
    )
    split_memory = max(
        r["max_peak_allocated_mib"] for r in memories if r["mode"] == "distributed"
    )
    lines += [
        f"Mean edge vision-encoder time is **{edge_encoder / 1000:.3f}s with the full model** versus **{split_encoder / 1000:.3f}s with only vision resident**: **{edge_encoder / split_encoder:.1f}×** difference on the same edge GPU. The maximum recorded per-request allocated peaks are **{edge_memory:.0f} MiB** and **{split_memory:.0f} MiB**, respectively, versus **4096 MiB dedicated edge VRAM**. Reserved peaks are also recorded in `memory_summary.csv`.",
        "",
        "The full-model baseline uses WDDM shared memory. The latency difference is consistent with residency/memory-pressure effects, but these traces do not isolate page faults, PCIe migration, clock state or thermal causality. Do not attribute the complete gap to one cause or assume the full-model vision cost transfers to a vision-only executor. Peak counters are process-wide request peaks, not isolated per-stage allocations.",
        "",
        "The split encoder falls across the bandwidth sweep and cloud image-prefill falls too, despite unchanged frozen requests. `replicate_drift.csv` exposes within-block repeated means. Netem rate does not by itself prove it speeds GPU computation: elapsed-time placement ordering, GPU idle intervals, clocks/thermals and memory state are plausible confounders. The N1–N6 bandwidth sweep occurs over time rather than being repeatedly randomized. The frozen rotation changes placement order per block, not per request. Three request repetitions are not three independent days of network experiments.",
        "",
        "![Memory and vision costs](memory_and_vision.png)",
        "",
        "## Prefill, decoding and task features",
        "",
        "Image prefill is materially larger than prompt prefill in the split. At fast N6, image prefill is the largest individual stage; simply improving serializer speed or a small projector cannot remove this floor. Cloud-only's combined image stage dominates its identifiable budget, so its independent encoder/projector/image-prefill ranking remains unknown.",
        "",
        "| Placement | Mean decode steps | Decode time per step, ratio of totals (ms) | Spearman: decode steps vs decode time | Spearman: image area vs visual timer |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for f in features:
        lines.append(
            f"| {f['mode']} | {f['mean_decode_steps']:.2f} | {f['decode_ms_per_step_ratio_of_totals']:.2f} | {f['decode_steps_vs_decode_ms_spearman']:.3f} | {f['image_area_vs_visual_stage_spearman']:.3f} |"
        )
    lines += [
        "",
        "These descriptive correlations use 100 request-level averages over blocks/repeats. For cloud the visual timer is the inseparable combined timer. Image dimensions/area are proxies: actual crop count is not logged here. Output decode steps are known only **after** generation; a live selector must use a prior token estimate, prompt/task features and the configured cap rather than leaking observed output length into its choice. Different GPU outputs can also have different lengths under the approved criterion, so unconditional decode averages are not equal-work speed ratios.",
        "",
        "## What this means for the thesis",
        "",
        "The framework should select among feasible configurations using **both system state and network/task features**. This run supplies a physical calibration set and an observed rank reversal. It does not yet demonstrate an online adaptive policy, its switching overhead or held-out generalization.",
        "",
        "A useful decomposed prediction is:",
        "",
        "```text\nT_edge  = vision_full(edge_state, image) + image_prefill_edge\n          + prompt_prefill_edge(prompt) + decode_edge(estimated_tokens) + local_other\nT_cloud = image_read + raw_bytes/up + response_bytes/down + transport_overhead(RTT)\n          + cloud_image_combined(cloud_state, image) + prompt_prefill_cloud\n          + decode_cloud(estimated_tokens) + cloud_other\nT_split = vision_only(edge_state, image) + projection + tensor_serialization\n          + projected_bytes/up + response_bytes/down + transport_overhead(RTT)\n          + deserialization + image_prefill_cloud + prompt_prefill_cloud\n          + decode_cloud(estimated_tokens) + cloud_other\n```",
        "",
        "Rates above are bytes/second in the cost equations, with times converted consistently. Do not add separately estimated queue time if it is already captured by a calibrated residual; add a queue term only when independently measured. Preserve the cloud combined timer until new instrumentation separates its children. Calibrate separate full-model and vision-only edge costs.",
        "",
        "| Selector feature/constraint | Evidence from G2 | Consequence for the framework |",
        "| --- | --- | --- |",
        "| Effective bidirectional rate, RTT, payload/framing bytes | Split tensor is much larger than raw image; winners switch at higher bandwidth | Predict transfer using the actual candidate payload, not a label such as offload or split. Use rate uncertainty and an abstention margin near crossings. |",
        "| Edge residency, dedicated/shared GPU memory, available host RAM | Full edge vision cost and memory exceed the vision-only path | Feasibility filtering must precede latency ranking; model shared-memory execution separately and penalize risky headroom. |",
        "| Cloud health/load, recent prefill/decode service times | High-bandwidth split becomes compute dominated | Update compute estimates online; cloud queue/load is unmeasured in this baseline and requires new telemetry/experiments. |",
        "| Image bytes/dimensions/actual tile count when available | Raw JPEG sizes vary; projected tensor is fixed | Image-aware transfer/vision estimates can distinguish requests; dimensions alone are a proxy. |",
        "| Prompt length and predicted output length | Prompt prefill and decoding are distinct timers | Condition language costs on workload; never use realized decode steps at selection time. |",
        "| Resident configurations, load/eviction/switch cost | Baseline loads each placement for a block; no live per-request switching trial | Compare predicted savings against switch cost/hysteresis and evaluate resident-policy feasibility before claiming adaptive gains. |",
        "",
        f"With equal weight on these nine conditions, best-static **{adaptive['best_static']}** averages **{adaptive['best_static_mean_ms'] / 1000:.3f}s**. A retrospective condition-wise choice averages **{adaptive['retrospective_condition_choice_mean_ms'] / 1000:.3f}s**, saving **{adaptive['retrospective_condition_saving_percent']:.2f}%** before sensing, model switching and scheduling costs. A retrospective per-request minimum averages **{adaptive['retrospective_request_oracle_mean_ms'] / 1000:.3f}s** (optimistic **{adaptive['retrospective_request_oracle_saving_percent']:.2f}%** improvement). These are **in-sample references selected using observed outcomes**, not measured adaptive performance; the minima have selection/noise optimism. See `adaptive_implications.json` for the conditional paired interval. Reweighting the operating-condition distribution changes the opportunity.",
        "",
        "For a defensible thesis evaluation: keep always-edge, always-cloud and always-split controls; separate request-ID and network-state calibration/holdout sets; hold out intermediate rates and new days; preserve image/prompt/model contracts; record selection/sensing/load/queue/decision overhead; measure feasibility exclusions and fallback; report paired latency/quality/regret/SLA results and condition-wise choice distribution. Use a held-out measured oracle for evaluation only, never as the online decision. Fresh counterfactual executions are needed to quantify live policy benefit. This complements the repository's [scheduler plan](../../../../docs/dynamic_scheduler_implementation_plan.md).",
        "",
        "## Scientific limits and correctness",
        "",
        "All inference attempts succeeded and all approved hardware-aware gates passed, but 26/100 request IDs have cross-GPU output differences; legacy strict identity remains false. Matching provisional per-request VQA scores does not establish official VQA accuracy or semantic equivalence. The observed mean provisional score is 0.03 for all placements; investigate the scoring contract and run the official evaluator before thesis accuracy claims. No energy, packet-level one-way delay, concurrent-cloud load, live model-switch overhead or independently randomized system-state intervention was measured. Full-edge success does not establish dedicated-VRAM fit. N1–N9 describe one host pair, one frozen workload and one campaign, not universal operating envelopes.",
        "",
        "## Reproduction and files",
        "",
        "From the repository root, using Python with NumPy and Matplotlib installed:",
        "",
        "```bash\nMPLCONFIGDIR=/tmp/g2-matplotlib /usr/bin/python3 \\\n  results/physical-g2-20261110/analysis/analyze_stages.py\n```",
        "",
        "The script reads the committed archive directly without extraction, verifies its SHA-256 plus all 330 member hashes, rechecks each block's bound artifacts, excludes warmups, enforces unique request/repetition pairs, reconstructs all 8,100 additive budgets, and regenerates CSV/JSON tables and SVG/PNG scientific figures. No model/cloud/GPU is needed. `analysis_validation.json` records versions, seeds, script/archive hashes and closure checks. The report itself is regenerated. Original evidence and the existing summary remain unchanged.",
        "",
        "- `request_stage_samples.csv`: per-request exclusive stages for audit/reuse.\n- `stage_summary.csv`, `stage_shares.csv`: mean/p95 stage budgets and dominant components.\n- `placement_comparison.csv`: every condition, paired uncertainty and request winner counts.\n- `network_conditions.csv`, `transfer_accounting.csv`: actual versus configured networks, exact application bytes and transfer references.\n- `memory_summary.csv`, `replicate_drift.csv`, `task_feature_associations.csv`: system/task diagnostics.\n- `descriptive_crossover.json`, `adaptive_implications.json`: model and retrospective-policy calculations.\n- Four figures in both SVG (editable vector) and PNG formats.",
    ]
    # The report lives one directory deeper than the run README.
    text = "\n".join(lines).replace("../../../../", "../../../")
    text = text.replace(
        " Paths are relative to the repository root via the results folder; see the corrected source links below if viewing this nested report.",
        "",
    )
    (HERE / "stage_analysis.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
