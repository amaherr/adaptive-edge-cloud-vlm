"""Load Phase 1 policies once so manual routing needs no model reload."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from adaptive_vlm.config import ExperimentConfig, model_contract_fingerprint
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.scheduler.configuration import ConfigurationId

EdgeComponents = Literal["full", "vision", "none"]


def load_phase1_policies(
    config: ExperimentConfig,
    *,
    artifact_manifest: Path | None,
    cache_dir: Path,
    offline: bool,
    vision_device: str,
    text_device: str,
    edge_components: EdgeComponents,
) -> dict[ConfigurationId, ExecutionPolicy]:
    """Keep one edge model resident and share its vision path if full."""

    model = ModelMetadata(
        config.model.model_id,
        config.model.revision,
        model_contract_fingerprint(config.model),
    )
    policies: dict[ConfigurationId, ExecutionPolicy] = {
        ConfigurationId.CLOUD_RAW_IMAGE: CloudExecutionPolicy(
            transport=config.transport, model=model
        ),
    }
    if edge_components == "none":
        return policies
    if config.model.adapter != "moondream2":
        raise ValueError("adaptive edge policies require the Moondream2 adapter")
    if artifact_manifest is None:
        raise ValueError("adaptive edge policies require --artifact-manifest")
    adapter = MoondreamAdapter.from_pretrained(
        config.model,
        cache_dir=cache_dir,
        artifact_manifest=artifact_manifest,
        vision_device=vision_device,
        text_device=text_device if edge_components == "full" else "cpu",
        offline=offline,
        components=edge_components,
    )
    if edge_components == "full":
        policies[ConfigurationId.EDGE_FULL] = LocalExecutionPolicy(adapter)
    policies[ConfigurationId.DISTRIBUTED_PROJECTED] = DistributedExecutionPolicy(
        encoder=adapter,
        transport=config.transport,
        model=model,
        pipeline="synchronous",
    )
    return policies
