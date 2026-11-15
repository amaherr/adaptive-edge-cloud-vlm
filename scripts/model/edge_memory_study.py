#!/usr/bin/env python3
"""Diagnose edge/full GPU memory residency on the 4 GiB edge GPU.

Diagnostic study only. It does not change the frozen campaign, the predictor or
any gate. Variants:

- vision_only:   vision encoder/projector resident (distributed edge stage).
- full_resident: complete BF16 model resident, as in the G2 edge/full policy.
- full_swap:     identical BF16 arithmetic; vision and text weights live in pinned
                 host memory and are copied to the GPU only for their own stage.
- int8_text:     text-model linear layers stored as per-channel INT8 weights
                 (a different numerical contract); vision stays BF16.

Any variant can carry --ballast-mib of idle GPU memory to emulate competing load.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.data.manifest import RequestItem, load_frozen_manifest
from adaptive_vlm.models.moondream.adapter import MoondreamAdapter
from adaptive_vlm.prediction.v2.diagnosis.components import crop_count

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/physical_vqa_v1.yaml"
ARTIFACT_MANIFEST = ROOT / "configs/model/moondream2_artifact.yaml"
DEV = "cuda:0"
MIB = 1024**2
VARIANTS = ("vision_only", "full_resident", "full_swap", "int8_text")


def now_ms() -> float:
    return time.perf_counter_ns() / 1e6


def sync() -> None:
    torch.cuda.synchronize(DEV)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    start = now_ms()
    value = fn()
    sync()
    return value, now_ms() - start


def free_mib() -> float:
    return torch.cuda.mem_get_info(DEV)[0] / MIB


def select_requests(per_group: int) -> list[tuple[int, RequestItem]]:
    config = load_experiment_config(CONFIG)
    manifest = load_frozen_manifest(
        config.dataset.manifest_path, expected_revision=config.dataset.revision
    )
    groups: dict[int, list[RequestItem]] = {}
    for item in manifest.items:
        with Image.open(item.image_path) as image:
            width, height = image.size
        groups.setdefault(crop_count(height, width), []).append(item)
    return [
        (crops, item) for crops in sorted(groups) for item in groups[crops][:per_group]
    ]


def load(vision_device: str, text_device: str, components: str) -> MoondreamAdapter:
    config = load_experiment_config(CONFIG)
    return MoondreamAdapter.from_pretrained(
        config.model,
        cache_dir=ROOT / ".model-cache",
        artifact_manifest=ARTIFACT_MANIFEST,
        vision_device=vision_device,
        text_device=text_device,
        offline=True,
        components=components,  # type: ignore[arg-type]
    )


def module_mib(module: torch.nn.Module) -> dict[str, float]:
    params = sum(p.numel() * p.element_size() for p in module.parameters())
    buffers = sum(b.numel() * b.element_size() for b in module.buffers())
    return {"parameters_mib": params / MIB, "buffers_mib": buffers / MIB}


def move_buffers(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        for name, buffer in list(submodule._buffers.items()):
            if buffer is not None:
                submodule._buffers[name] = buffer.to(DEV)


def retarget(adapter: MoondreamAdapter) -> None:
    adapter._vision_device = DEV
    adapter._text_device = DEV


class Swapper:
    """Keep stage weights in pinned host memory; copy them in only when needed."""

    def __init__(self, adapter: MoondreamAdapter) -> None:
        core = adapter._core
        # pos_emb defines core.device for every stage, so it stays resident.
        core.vision.pos_emb.data = core.vision.pos_emb.data.to(DEV)
        keep = {id(core.vision.pos_emb)}
        self.groups: dict[str, list[tuple[torch.nn.Parameter, torch.Tensor]]] = {}
        for name, module in (("vision", core.vision), ("text", core.text)):
            pairs = []
            for parameter in module.parameters():
                if id(parameter) in keep:
                    continue
                master = parameter.data.pin_memory()
                parameter.data = master
                pairs.append((parameter, master))
            self.groups[name] = pairs
        move_buffers(core.vision)
        move_buffers(core.text)  # KV caches, rotary tables: permanently resident
        core.attn_mask = core.attn_mask.to(DEV)
        self.resident = {"vision": False, "text": False}
        gc.collect()

    def load(self, name: str) -> None:
        if self.resident[name]:
            return
        for parameter, master in self.groups[name]:
            parameter.data = master.to(DEV, non_blocking=True)
        self.resident[name] = True

    def evict(self, name: str) -> None:
        if not self.resident[name]:
            return
        for parameter, master in self.groups[name]:
            parameter.data = master  # read-only weights: no copy back
        self.resident[name] = False


_INT8_MM: bool | None = None


def int8_mm_available() -> bool:
    global _INT8_MM
    if _INT8_MM is None:
        try:
            x = torch.randn(4, 64, device=DEV, dtype=torch.bfloat16)
            w = torch.randint(-127, 127, (32, 64), device=DEV, dtype=torch.int8)
            s = torch.rand(32, device=DEV, dtype=torch.bfloat16)
            got = torch._weight_int8pack_mm(x, w, s)
            ref = F.linear(x, w.to(torch.bfloat16)) * s
            _INT8_MM = bool(torch.allclose(got, ref, rtol=0.05, atol=0.5))
        except (RuntimeError, AttributeError, NotImplementedError):
            _INT8_MM = False
    return _INT8_MM


class Int8Linear(torch.nn.Module):
    """Weight-only symmetric per-output-channel INT8 (round to nearest)."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.data.float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer(
            "qweight", torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scale", scale.squeeze(1).to(torch.bfloat16))
        bias = linear.bias
        self.register_buffer(
            "bias", None if bias is None else bias.data.to(torch.bfloat16).clone()
        )
        self.use_pack = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pack:
            flat = x.reshape(-1, x.shape[-1])
            out = torch._weight_int8pack_mm(flat, self.qweight, self.scale)
            out = out.reshape(*x.shape[:-1], -1)
        else:
            out = F.linear(x, self.qweight.to(x.dtype)) * self.scale
        if self.bias is not None:
            out = out + self.bias
        return out


def quantize_text(adapter: MoondreamAdapter) -> int:
    text = adapter._core.text
    replaced = 0
    for block in text["blocks"]:
        for group, names in (("attn", ("qkv", "proj")), ("mlp", ("fc1", "fc2"))):
            for name in names:
                block[group][name] = Int8Linear(block[group][name])
                replaced += 1
    text["lm_head"] = Int8Linear(text["lm_head"])
    return replaced + 1


def prepare(variant: str) -> tuple[MoondreamAdapter, Swapper | None, dict[str, Any]]:
    info: dict[str, Any] = {"free_before_load_mib": free_mib()}
    swapper = None
    if variant == "vision_only":
        adapter = load(DEV, "cpu", "vision")
    elif variant == "full_resident":
        adapter = load(DEV, DEV, "full")
    else:
        adapter = load("cpu", "cpu", "full")
        core = adapter._core
        info["vision_cpu_sizes"] = module_mib(core.vision)
        info["text_cpu_sizes_before"] = module_mib(core.text)
        info["region_cpu_sizes"] = module_mib(core.region)
        if variant == "full_swap":
            swapper = Swapper(adapter)
        else:
            info["int8_layers"] = quantize_text(adapter)
            info["text_cpu_sizes_after"] = module_mib(core.text)
            core.vision.to(DEV)
            core.text.to(DEV)
            core.attn_mask = core.attn_mask.to(DEV)
            info["int8_pack_kernel"] = int8_mm_available()
            for module in core.text.modules():
                if isinstance(module, Int8Linear):
                    module.use_pack = info["int8_pack_kernel"]
        retarget(adapter)
        gc.collect()
    sync()
    info["free_after_load_mib"] = free_mib()
    info["allocated_after_load_mib"] = torch.cuda.memory_allocated(DEV) / MIB
    return adapter, swapper, info


def run_request(
    adapter: MoondreamAdapter,
    swapper: Swapper | None,
    item: RequestItem,
    vision_only: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(DEV)
    sync()
    started = now_ms()
    image = adapter._open_image(item.image_path)
    if swapper is not None:
        _, metrics["swap_to_vision_ms"] = timed(
            lambda: (swapper.evict("text"), swapper.load("vision"))
        )
    with torch.inference_mode():
        (projected, vision_metrics), _ = timed(lambda: adapter._encode_projected(image))
    metrics.update(vision_metrics)
    if not vision_only:
        if swapper is not None:
            _, metrics["swap_to_text_ms"] = timed(
                lambda: (swapper.evict("vision"), swapper.load("text"))
            )
        with torch.inference_mode():
            features = projected.tensor.to(DEV)
            position, metrics["image_prefill_ms"] = timed(
                lambda: adapter._prefill_projected(features)
            )
            trace, _ = timed(lambda: adapter._generate_token_ids(item.prompt, position))
        metrics.update(trace.metrics())
        metrics["output_token_ids"] = list(trace.token_ids)
        metrics["output_text"] = adapter._core.tokenizer.decode(list(trace.token_ids))
    sync()
    metrics["end_to_end_ms"] = now_ms() - started
    metrics["peak_allocated_mib"] = torch.cuda.max_memory_allocated(DEV) / MIB
    metrics["peak_reserved_mib"] = torch.cuda.max_memory_reserved(DEV) / MIB
    metrics["free_after_request_mib"] = free_mib()
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "end_to_end_ms",
        "vision_encoder_ms",
        "vision_total_ms",
        "image_prefill_ms",
        "prompt_prefill_ms",
        "decode_ms",
        "swap_to_vision_ms",
        "swap_to_text_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    summary: dict[str, Any] = {}
    for crops in [*sorted({row["crops"] for row in rows}), "all"]:
        subset = [r for r in rows if crops == "all" or r["crops"] == crops]
        entry: dict[str, Any] = {"n": len(subset)}
        for key in keys:
            values = [r[key] for r in subset if r.get(key) is not None]
            if values:
                entry[f"median_{key}"] = statistics.median(values)
        decode = [
            r["decode_ms"] / r["decode_steps"] for r in subset if r.get("decode_steps")
        ]
        if decode:
            entry["median_decode_ms_per_token"] = statistics.median(decode)
        summary[str(crops)] = entry
    return summary


def cmd_run(args: argparse.Namespace) -> None:
    requests = select_requests(args.per_group)
    adapter, swapper, info = prepare(args.variant)
    ballast = None
    if args.ballast_mib:
        ballast = torch.empty(args.ballast_mib * MIB, dtype=torch.uint8, device=DEV)
        ballast.fill_(1)
        sync()
        info["free_after_ballast_mib"] = free_mib()
    print(json.dumps({"variant": args.variant, **info}), flush=True)
    vision_only = args.variant == "vision_only"
    _, warm_item = requests[-1]
    run_request(adapter, swapper, warm_item, vision_only)  # excluded warmup
    rows = []
    for repetition in range(args.reps):
        for crops, item in requests:
            row = run_request(adapter, swapper, item, vision_only)
            row.update(
                {
                    "variant": args.variant,
                    "ballast_mib": args.ballast_mib,
                    "repetition": repetition,
                    "request_id": item.request_id,
                    "crops": crops,
                    "wall_clock_utc": time.time(),
                }
            )
            rows.append(row)
            print(
                f"{args.variant} r{repetition} {item.request_id} crops={crops} "
                f"e2e={row['end_to_end_ms']:.0f}ms "
                f"vision={row['vision_encoder_ms']:.0f}ms "
                f"peak_alloc={row['peak_allocated_mib']:.0f}MiB "
                f"free={row['free_after_request_mib']:.0f}MiB",
                flush=True,
            )
    del ballast
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "variant": args.variant,
        "ballast_mib": args.ballast_mib,
        "per_group": args.per_group,
        "reps": args.reps,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(DEV),
        "total_vram_mib": torch.cuda.get_device_properties(DEV).total_memory / MIB,
        "info": info,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Vision-only latency while idle ballast grows in fixed steps."""

    requests = select_requests(1)
    nine = next(item for crops, item in requests if crops == 9)
    adapter, _, info = prepare("vision_only")
    run_request(adapter, None, nine, True)
    ballast: list[torch.Tensor] = []
    rows = []
    step = 0
    while step <= args.max_mib:
        if step:
            chunk = torch.empty(args.step_mib * MIB, dtype=torch.uint8, device=DEV)
            chunk.fill_(1)
            ballast.append(chunk)
        sync()
        times = []
        for _ in range(args.reps):
            row = run_request(adapter, None, nine, True)
            times.append(row["vision_encoder_ms"])
        record = {
            "ballast_mib": step,
            "free_mib": free_mib(),
            "allocated_mib": torch.cuda.memory_allocated(DEV) / MIB,
            "peak_allocated_mib": row["peak_allocated_mib"],
            "vision_encoder_ms": times,
            "median_vision_encoder_ms": statistics.median(times),
            "wall_clock_utc": time.time(),
        }
        rows.append(record)
        print(
            f"ballast={step}MiB free={record['free_mib']:.0f}MiB "
            f"alloc={record['allocated_mib']:.0f}MiB "
            f"vision_median={record['median_vision_encoder_ms']:.0f}ms",
            flush=True,
        )
        step += args.step_mib
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"diagnostic_only": True, "info": info, "rows": rows}, indent=2)
    )


def cmd_compare(args: argparse.Namespace) -> None:
    runs = {path.stem: json.loads(path.read_text()) for path in args.inputs}
    reference = runs[args.reference]
    ref_tokens = {
        (r["request_id"], r["repetition"]): r["output_token_ids"]
        for r in reference["rows"]
    }
    for name, run in runs.items():
        rows = [r for r in run["rows"] if "output_token_ids" in r]
        if not rows:
            continue
        same = sum(
            r["output_token_ids"] == ref_tokens.get((r["request_id"], r["repetition"]))
            for r in rows
        )
        print(f"{name}: {same}/{len(rows)} token sequences match {args.reference}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--per-group", type=int, default=4)
    run.add_argument("--reps", type=int, default=2)
    run.add_argument("--ballast-mib", type=int, default=0)
    run.add_argument("--output", type=Path, required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--step-mib", type=int, default=256)
    sweep.add_argument("--max-mib", type=int, default=3072)
    sweep.add_argument("--reps", type=int, default=3)
    sweep.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    {"run": cmd_run, "sweep": cmd_sweep, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
