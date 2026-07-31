"""Dataiku recipe entry points kept intentionally thin."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from defect_curation_core.errors import ConfigurationError
from defect_curation_core.pipeline import DefectClusteringPipeline, PipelineRunResult
from defect_curation_core.provenance import read_git_head

from defect_curation_plugin.dataiku_io import (
    DataikuManagedFolderArtifactStore,
    DataikuManagedFolderImageSource,
    DataikuReviewDatasetSink,
)
from defect_curation_plugin.plugin_settings import build_pipeline_config

LOGGER = logging.getLogger(__name__)


def _one(values: Sequence[str], *, role: str) -> str:
    if len(values) != 1:
        raise ConfigurationError(f"Recipe role {role!r} requires exactly one object; received {len(values)}")
    return str(values[0])


def _repository_root() -> Path:
    # python-lib/defect_curation_plugin/recipe_entrypoints.py -> repository root
    return Path(__file__).resolve().parents[2]


def _plugin_version(root: Path) -> str:
    try:
        payload = json.loads((root / "plugin.json").read_text(encoding="utf-8"))
        return str(payload["version"])
    except Exception:
        return "unknown"


def _dss_version(dataiku_module: Any) -> str | None:
    for name in ("get_dss_version", "dss_version"):
        candidate = getattr(dataiku_module, name, None)
        if callable(candidate):
            try:
                return str(candidate())
            except Exception:
                continue
    return None


def run_cluster_defect_instances() -> PipelineRunResult:
    try:
        import dataiku
        import dataiku.customrecipe as customrecipe
    except ImportError as exc:  # pragma: no cover - exercised only outside DSS.
        raise RuntimeError("This entry point must run inside a Dataiku DSS plugin code environment") from exc

    input_folder_id = _one(
        customrecipe.get_input_names_for_role("images_folder"),
        role="images_folder",
    )
    review_dataset_name = _one(
        customrecipe.get_output_names_for_role("review_dataset"),
        role="review_dataset",
    )
    artifact_folder_id = _one(
        customrecipe.get_output_names_for_role("artifact_bundle"),
        role="artifact_bundle",
    )
    if input_folder_id == artifact_folder_id:
        raise ConfigurationError(
            "The source image folder and artifact output folder must be different managed folders"
        )

    recipe_config = customrecipe.get_recipe_config()
    plugin_config = customrecipe.get_plugin_config()
    root = _repository_root()
    cfg = build_pipeline_config(
        plugin_config=plugin_config,
        recipe_config=recipe_config,
        config_root=root / "python-lib" / "Configs",
    )

    image_folder = dataiku.Folder(input_folder_id)
    artifact_folder = dataiku.Folder(artifact_folder_id)
    review_dataset = dataiku.Dataset(review_dataset_name)

    pipeline = DefectClusteringPipeline(
        image_source=DataikuManagedFolderImageSource(
            image_folder,
            identifier=input_folder_id,
            read_chunk_size=int(cfg.runtime.input_read_chunk_size),
        ),
        artifact_store=DataikuManagedFolderArtifactStore(
            artifact_folder,
            identifier=artifact_folder_id,
        ),
        review_sink=DataikuReviewDatasetSink(
            review_dataset,
            identifier=review_dataset_name,
        ),
        cfg=cfg,
        plugin_version=_plugin_version(root),
        plugin_source_commit=read_git_head(str(root)),
        dss_version=_dss_version(dataiku),
    )
    result = pipeline.run()
    LOGGER.info(
        "Defect clustering run %s published %d review rows and %d embedded instances",
        result.run_id,
        result.review_row_count,
        result.embedded_instance_count,
    )
    return result
