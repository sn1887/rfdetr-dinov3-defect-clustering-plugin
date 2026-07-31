"""Versioned cache-signature construction."""

from __future__ import annotations

from typing import Any

from defect_curation_core.hashing import sha256_json

DETECTION_CACHE_SCHEMA_VERSION = 2
EMBEDDING_CACHE_SCHEMA_VERSION = 2
POSTPROCESS_SCHEMA_VERSION = 1
POOLING_SCHEMA_VERSION = 1


def build_detector_signature(
    *,
    checkpoint_sha256: str,
    rfdetr_package_version: str,
    model_class: str,
    preprocessing_spec: str,
    detection_threshold: float,
    max_detections_per_image: int,
    device: str,
    clip_to_image: bool,
    drop_degenerate: bool,
    loader_implementation_version: str,
) -> str:
    return sha256_json(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "rfdetr_package_version": rfdetr_package_version,
            "model_class": model_class,
            "preprocessing_spec": preprocessing_spec,
            "detection_threshold": float(detection_threshold),
            "max_detections_per_image": int(max_detections_per_image),
            "device": device,
            "clip_to_image": bool(clip_to_image),
            "drop_degenerate": bool(drop_degenerate),
            "loader_implementation_version": loader_implementation_version,
            "postprocess_schema_version": POSTPROCESS_SCHEMA_VERSION,
        }
    )


def build_basis_input_signature(
    *,
    artifact_sha256: str,
    artifact_revision: str,
    model_id: str,
    timm_version: str,
    loader_implementation_version: str,
    input_size: int,
    patch_size: int,
    components: int,
    basis_image: str,
    device: str,
    inference_precision: str,
) -> str:
    return sha256_json(
        {
            "artifact_sha256": artifact_sha256,
            "artifact_revision": artifact_revision,
            "model_id": model_id,
            "timm_version": timm_version,
            "loader_implementation_version": loader_implementation_version,
            "input_size": int(input_size),
            "patch_size": int(patch_size),
            "components": int(components),
            "basis_image": basis_image,
            "device": device,
            "inference_precision": inference_precision,
            "basis_schema_version": 2,
        }
    )


def build_embedding_signature(
    *,
    artifact_sha256: str,
    artifact_revision: str,
    model_id: str,
    timm_version: str,
    loader_implementation_version: str,
    input_size: int,
    patch_size: int,
    padding_fraction: float,
    letterbox_spec: str,
    patch_layer: str,
    token_normalization: str,
    basis_sha256: str,
    pooling_spec: dict[str, Any],
    device: str,
    inference_precision: str,
) -> str:
    return sha256_json(
        {
            "artifact_sha256": artifact_sha256,
            "artifact_revision": artifact_revision,
            "model_id": model_id,
            "timm_version": timm_version,
            "loader_implementation_version": loader_implementation_version,
            "input_size": int(input_size),
            "patch_size": int(patch_size),
            "padding_fraction": float(padding_fraction),
            "letterbox_spec": letterbox_spec,
            "patch_layer": patch_layer,
            "token_normalization": token_normalization,
            "basis_sha256": basis_sha256,
            "pooling_spec": pooling_spec,
            "device": device,
            "inference_precision": inference_precision,
            "embedding_schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "pooling_schema_version": POOLING_SCHEMA_VERSION,
        }
    )
