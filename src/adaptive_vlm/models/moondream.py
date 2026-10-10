"""Pinned Moondream2 adapter with an explicit vision/projector boundary."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Self, cast

import yaml

from adaptive_vlm.config import ModelConfig, Precision
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, MetricValue
from adaptive_vlm.measurement import measure_call
from adaptive_vlm.model_artifact import sha256_file
from adaptive_vlm.models.base import ModelAdapter, ModelMetadata
from adaptive_vlm.models.features import ProjectedVisionFeatures


@dataclass(frozen=True)
class MoondreamRun:
    """One inference result and the projected tensor that produced it."""

    result: InferenceResult
    projected: ProjectedVisionFeatures


@dataclass(frozen=True)
class _RuntimeModules:
    torch: Any
    image: Any
    auto_model: Any
    tokenizer: Any
    hf_hub_download: Any
    psutil: Any


class MoondreamAdapter(ModelAdapter):
    """Execute pinned Moondream2 through monolithic or decomposed paths."""

    def __init__(
        self,
        *,
        model: Any,
        config: ModelConfig,
        runtime: _RuntimeModules,
        vision_device: str,
        text_device: str,
    ) -> None:
        self._model = model
        self._core = model.model
        self._config = config
        self._runtime = runtime
        self._vision_device = str(runtime.torch.device(vision_device))
        self._text_device = str(runtime.torch.device(text_device))
        self._process = runtime.psutil.Process()
        self._lock = threading.RLock()
        self._metadata = ModelMetadata(
            model_id=config.model_id,
            revision=config.revision,
            fingerprint=config.artifact_sha256,
        )

    @classmethod
    def from_pretrained(
        cls,
        config: ModelConfig,
        *,
        cache_dir: str | Path,
        artifact_manifest: str | Path,
        vision_device: str = "cpu",
        text_device: str = "cpu",
        offline: bool = True,
    ) -> Self:
        """Load and verify the exact model and tokenizer artifacts."""

        _validate_model_config(config)
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        manifest = _load_artifact_manifest(artifact_manifest)
        _validate_manifest(config, manifest)
        runtime = _load_runtime_modules()
        hub_cache = cache_path / "hub"

        weights = cast(Mapping[str, Any], manifest["weights"])
        weights_path = Path(
            runtime.hf_hub_download(
                repo_id=config.model_id,
                filename=str(weights["filename"]),
                revision=config.revision,
                cache_dir=hub_cache,
                local_files_only=offline,
            )
        )
        _verify_file(weights_path, str(weights["sha256"]), "model weights")

        tokenizer_manifest = cast(
            Mapping[str, Any], manifest["tokenizer_dependency"]
        )
        tokenizer_path = Path(
            runtime.hf_hub_download(
                repo_id=str(tokenizer_manifest["repository"]),
                filename=str(tokenizer_manifest["filename"]),
                revision=str(tokenizer_manifest["revision"]),
                cache_dir=hub_cache,
                local_files_only=offline,
            )
        )
        _verify_file(
            tokenizer_path,
            str(tokenizer_manifest["sha256"]),
            "tokenizer",
        )

        torch = runtime.torch
        model = runtime.auto_model.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map={"": str(torch.device(text_device))},
            low_cpu_mem_usage=True,
            cache_dir=hub_cache,
            local_files_only=offline,
        )
        model.model.tokenizer = runtime.tokenizer.from_file(str(tokenizer_path))
        model.eval()

        # Caches must follow the text model, while the vision module may later
        # move independently to the edge GPU.
        model._setup_caches()
        model.model.vision.to(torch.device(vision_device))

        return cls(
            model=model,
            config=config,
            runtime=runtime,
            vision_device=vision_device,
            text_device=text_device,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def vision_device(self) -> str:
        return self._vision_device

    @property
    def text_device(self) -> str:
        return self._text_device

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Run the decomposed local path used by the all-edge policy."""

        return self.run_decomposed(request).result

    def run_decomposed(self, request: InferenceRequest) -> MoondreamRun:
        """Encode projected tokens, transfer them, then run the text model."""

        with self._lock, self._runtime.torch.inference_mode():
            self._validate_request(request)
            self._reset_memory_stats()
            rss_before = self._rss_mib()
            self._synchronize_all()
            started_ns = perf_counter_ns()
            image = self._open_image(request.image_path)

            projected, metrics = self._encode_projected(image)
            transferred = measure_call(
                lambda: projected.tensor.to(self._text_device),
                synchronize=self._synchronize_all,
            )
            text_features = ProjectedVisionFeatures.from_tensor(transferred.value)
            metrics["feature_transfer_ms"] = transferred.elapsed_ms

            image_prefill = measure_call(
                lambda: self._prefill_projected(text_features.tensor),
                synchronize=self._synchronizer(self._text_device),
            )
            metrics["image_prefill_ms"] = image_prefill.elapsed_ms

            generation = measure_call(
                lambda: self._generate_token_ids(
                    request.prompt,
                    image_prefill.value,
                ),
                synchronize=self._synchronizer(self._text_device),
            )
            token_ids = generation.value
            metrics["language_generation_ms"] = generation.elapsed_ms

            self._synchronize_all()
            metrics["end_to_end_ms"] = (perf_counter_ns() - started_ns) / 1_000_000
            metrics.update(
                {
                    "projected_visual_tokens": text_features.shape[0],
                    "projected_hidden_size": text_features.shape[1],
                    "projected_dtype": text_features.dtype,
                    "projected_source_device": projected.device,
                    "projected_target_device": text_features.device,
                    "projected_nbytes": text_features.nbytes,
                }
            )
            metrics["output_tokens"] = len(token_ids)
            metrics.update(self._memory_metrics(rss_before))

            result = InferenceResult(
                request_id=request.request_id,
                output_text=self._core.tokenizer.decode(list(token_ids)),
                output_token_ids=token_ids,
                metrics=metrics,
            )
            return MoondreamRun(result=result, projected=projected)

    def run_monolithic(self, request: InferenceRequest) -> MoondreamRun:
        """Run the pinned model's original image encoding and text prefill path."""

        if self._vision_device != self._text_device:
            raise ValueError("monolithic execution requires one shared device")

        with self._lock, self._runtime.torch.inference_mode():
            self._validate_request(request)
            self._reset_memory_stats()
            rss_before = self._rss_mib()
            self._synchronize_all()
            started_ns = perf_counter_ns()
            image = self._open_image(request.image_path)
            captured: list[Any] = []
            original = self._core._run_vision_encoder

            def capture_projected(source_image: Any) -> Any:
                tensor = original(source_image)
                captured.append(tensor)
                return tensor

            self._core._run_vision_encoder = capture_projected
            try:
                encoded = measure_call(
                    lambda: self._core.encode_image(
                        image,
                        {"variant": self._config.variant},
                    ),
                    synchronize=self._synchronizer(self._text_device),
                )
            finally:
                del self._core._run_vision_encoder

            if len(captured) != 1:
                raise RuntimeError(
                    "monolithic path did not expose one projected tensor"
                )
            projected = ProjectedVisionFeatures.from_tensor(captured[0])
            self._core.load_encoded_image(encoded.value)
            generation = measure_call(
                lambda: self._generate_token_ids(request.prompt, encoded.value.pos),
                synchronize=self._synchronizer(self._text_device),
            )
            token_ids = generation.value
            self._synchronize_all()
            metrics: dict[str, MetricValue] = {
                "image_encode_and_prefill_ms": encoded.elapsed_ms,
                "language_generation_ms": generation.elapsed_ms,
                "end_to_end_ms": (perf_counter_ns() - started_ns) / 1_000_000,
                "output_tokens": len(token_ids),
                "projected_visual_tokens": projected.shape[0],
                "projected_hidden_size": projected.shape[1],
                "projected_dtype": projected.dtype,
                "projected_source_device": projected.device,
                "projected_target_device": projected.device,
                "projected_nbytes": projected.nbytes,
                **self._memory_metrics(rss_before),
            }
            result = InferenceResult(
                request_id=request.request_id,
                output_text=self._core.tokenizer.decode(list(token_ids)),
                output_token_ids=token_ids,
                metrics=metrics,
            )
            return MoondreamRun(result=result, projected=projected)

    def compare_runs(
        self, reference: MoondreamRun, candidate: MoondreamRun
    ) -> dict[str, bool | float]:
        """Compare split-boundary tensors and generated outputs."""

        torch = self._runtime.torch
        reference_tensor = reference.projected.tensor.to("cpu").float()
        candidate_tensor = candidate.projected.tensor.to("cpu").float()
        max_difference = float(
            torch.max(torch.abs(reference_tensor - candidate_tensor)).item()
        )
        return {
            "projected_tensor_equal": bool(
                torch.equal(reference_tensor, candidate_tensor)
            ),
            "projected_tensor_max_abs_difference": max_difference,
            "token_ids_equal": (
                reference.result.output_token_ids
                == candidate.result.output_token_ids
            ),
            "text_equal": reference.result.output_text == candidate.result.output_text,
        }

    def _encode_projected(
        self, image: Any
    ) -> tuple[ProjectedVisionFeatures, dict[str, MetricValue]]:
        encoder_ms = 0.0
        projector_ms = 0.0
        original_encoder = self._core._vis_enc
        original_projector = self._core._vis_proj
        synchronize = self._synchronizer(self._vision_device)

        def timed_encoder(tensor: Any) -> Any:
            nonlocal encoder_ms
            measured = measure_call(
                lambda: original_encoder(tensor), synchronize=synchronize
            )
            encoder_ms += measured.elapsed_ms
            return measured.value

        def timed_projector(global_features: Any, reconstructed: Any) -> Any:
            nonlocal projector_ms
            measured = measure_call(
                lambda: original_projector(global_features, reconstructed),
                synchronize=synchronize,
            )
            projector_ms += measured.elapsed_ms
            return measured.value

        self._core._vis_enc = timed_encoder
        self._core._vis_proj = timed_projector
        try:
            total = measure_call(
                lambda: self._core._run_vision_encoder(image),
                synchronize=synchronize,
            )
        finally:
            del self._core._vis_enc
            del self._core._vis_proj

        features = ProjectedVisionFeatures.from_tensor(total.value)
        metrics: dict[str, MetricValue] = {
            "vision_total_ms": total.elapsed_ms,
            "vision_encoder_ms": encoder_ms,
            "projection_ms": projector_ms,
            "vision_other_ms": max(
                0.0, total.elapsed_ms - encoder_ms - projector_ms
            ),
        }
        return features, metrics

    def _prefill_projected(self, projected: Any) -> int:
        torch = self._runtime.torch
        bos = torch.tensor(
            [[self._core.config.tokenizer.bos_id]],
            device=self._text_device,
            dtype=torch.long,
        )
        bos_embedding = torch.nn.functional.embedding(bos, self._core.text.wte)
        inputs = torch.cat([bos_embedding, projected.unsqueeze(0)], dim=1)
        sequence_length = int(inputs.size(1))
        mask = self._core.attn_mask[:, :, :sequence_length, :]
        positions = torch.arange(
            sequence_length,
            device=self._text_device,
            dtype=torch.long,
        )
        self._core._prefill(inputs, mask, positions, None)
        return sequence_length

    def _generate_token_ids(self, prompt: str, position: int) -> tuple[int, ...]:
        torch = self._runtime.torch
        tokenizer_config = self._core.config.tokenizer
        query_template = tokenizer_config.templates["query"]
        if query_template is None:
            raise RuntimeError("pinned model does not define a query template")

        # This intentionally mirrors the pinned query implementation, including
        # its second suffix append in the non-reasoning path.
        prompt_ids = (
            list(query_template["prefix"])
            + self._core.tokenizer.encode(prompt).ids
            + list(query_template["suffix"])
            + list(query_template["suffix"])
        )
        prompt_tensor = torch.tensor(
            [prompt_ids], device=self._text_device, dtype=torch.long
        )
        _, _, next_token, position = self._core._prefill_prompt(
            prompt_tensor,
            position,
            0.0,
            1.0,
            None,
            attn_mask=self._core.attn_mask,
            lora=None,
        )

        mask = torch.zeros(
            1,
            1,
            self._core.config.text.max_context,
            device=self._text_device,
            dtype=torch.bool,
        )
        mask[:, :, :position] = 1
        position_ids = torch.tensor(
            [position], device=self._text_device, dtype=torch.long
        )
        generated: list[int] = []
        eos_id = tokenizer_config.eos_id

        while (
            int(next_token.item()) != eos_id
            and len(generated) < self._config.max_new_tokens
        ):
            generated.append(int(next_token.item()))
            next_embedding = torch.nn.functional.embedding(
                next_token, self._core.text.wte
            )
            mask[:, :, position] = 1
            position_ids[0] = position
            logits, _ = self._core._decode_one_tok(
                next_embedding,
                mask,
                position_ids,
                None,
            )
            logits[:, tokenizer_config.answer_id] = float("-inf")
            position += 1
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        return tuple(generated)

    def _validate_request(self, request: InferenceRequest) -> None:
        if not request.image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {request.image_path}")

    def _open_image(self, path: Path) -> Any:
        with self._runtime.image.open(path) as source:
            return source.convert("RGB")

    def _synchronizer(self, device: str) -> Any:
        torch = self._runtime.torch
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return None
        return lambda: torch.cuda.synchronize(resolved)

    def _synchronize_all(self) -> None:
        for device in dict.fromkeys((self._vision_device, self._text_device)):
            synchronize = self._synchronizer(device)
            if synchronize is not None:
                synchronize()

    def _reset_memory_stats(self) -> None:
        torch = self._runtime.torch
        for device in dict.fromkeys((self._vision_device, self._text_device)):
            resolved = torch.device(device)
            if resolved.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved)

    def _memory_metrics(self, rss_before: float) -> dict[str, MetricValue]:
        torch = self._runtime.torch
        rss_after = self._rss_mib()
        metrics: dict[str, MetricValue] = {
            "rss_before_mib": rss_before,
            "rss_after_mib": rss_after,
            "rss_delta_mib": rss_after - rss_before,
        }
        for role, device in (
            ("vision", self._vision_device),
            ("text", self._text_device),
        ):
            resolved = torch.device(device)
            if resolved.type == "cuda":
                metrics[f"{role}_cuda_peak_allocated_mib"] = (
                    torch.cuda.max_memory_allocated(resolved) / 1024**2
                )
                metrics[f"{role}_cuda_peak_reserved_mib"] = (
                    torch.cuda.max_memory_reserved(resolved) / 1024**2
                )
        return metrics

    def _rss_mib(self) -> float:
        return float(self._process.memory_info().rss / 1024**2)


def _validate_model_config(config: ModelConfig) -> None:
    if config.adapter != "moondream2":
        raise ValueError("MoondreamAdapter requires adapter 'moondream2'")
    if config.precision is not Precision.BF16:
        raise ValueError("the pinned Moondream baseline requires BF16")
    if config.preprocessing_backend != "pillow":
        raise ValueError("the pinned Moondream baseline requires Pillow")
    if config.do_sample or config.temperature != 0.0:
        raise ValueError("the pinned Moondream baseline requires greedy decoding")
    if config.variant is not None:
        raise ValueError("the baseline does not permit a LoRA variant")


def _load_artifact_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    parsed = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("model artifact manifest must be a mapping")
    return cast(Mapping[str, Any], parsed)


def _validate_manifest(config: ModelConfig, manifest: Mapping[str, Any]) -> None:
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("artifact manifest is missing weights")
    expected = {
        "repository": config.model_id,
        "revision": config.revision,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"artifact manifest {key} does not match model config")
    if weights.get("sha256") != config.artifact_sha256:
        raise ValueError("artifact manifest weight hash does not match model config")


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _load_runtime_modules() -> _RuntimeModules:
    try:
        import psutil  # type: ignore[import-untyped]
        import torch  # type: ignore[import-not-found]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Moondream dependencies are not installed; use the model lock file"
        ) from exc
    return _RuntimeModules(
        torch=torch,
        image=Image,
        auto_model=AutoModelForCausalLM,
        tokenizer=Tokenizer,
        hf_hub_download=hf_hub_download,
        psutil=psutil,
    )
