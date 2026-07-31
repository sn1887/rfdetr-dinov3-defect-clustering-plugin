from __future__ import annotations

from defect_curation_core.artifacts.signatures import (
    build_detector_signature,
    build_embedding_signature,
)


def detector_signature(**changes):
    values = dict(
        checkpoint_sha256="a" * 64,
        rfdetr_package_version="1.5.2",
        model_class="RFDETRSmall",
        preprocessing_spec="default-v1",
        detection_threshold=0.35,
        max_detections_per_image=20,
        device="cuda",
        clip_to_image=True,
        drop_degenerate=True,
        loader_implementation_version="rfdetr-loader-v1",
    )
    values.update(changes)
    return build_detector_signature(**values)


def embedding_signature(**changes):
    values = dict(
        artifact_sha256="b" * 64,
        artifact_revision="c" * 40,
        model_id="timm/vit_base_patch16_dinov3.lvd1689m",
        timm_version="1.0.20",
        loader_implementation_version="timm-loader-v1",
        input_size=512,
        patch_size=16,
        padding_fraction=0.15,
        letterbox_spec="letterbox-v1",
        patch_layer="final",
        token_normalization="l2",
        basis_sha256="d" * 64,
        pooling_spec={"method": "fractional_box_mean", "minimum": 4.0},
        device="cuda",
        inference_precision="bf16",
    )
    values.update(changes)
    return build_embedding_signature(**values)


def test_signatures_are_stable() -> None:
    assert detector_signature() == detector_signature()
    assert embedding_signature() == embedding_signature()


def test_detector_signature_invalidates_relevant_changes() -> None:
    baseline = detector_signature()
    assert detector_signature(detection_threshold=0.2) != baseline
    assert detector_signature(checkpoint_sha256="f" * 64) != baseline
    assert detector_signature(max_detections_per_image=40) != baseline
    assert detector_signature(device="cpu") != baseline
    assert detector_signature(clip_to_image=False) != baseline
    assert detector_signature(model_class="RFDETRMedium") != baseline
    assert detector_signature(rfdetr_package_version="1.5.3") != baseline
    assert detector_signature(loader_implementation_version="rfdetr-loader-v2") != baseline


def test_embedding_signature_invalidates_representation_changes_not_k() -> None:
    baseline = embedding_signature()
    assert embedding_signature(padding_fraction=0.2) != baseline
    assert embedding_signature(input_size=768) != baseline
    assert embedding_signature(basis_sha256="e" * 64) != baseline
    assert embedding_signature(device="cpu", inference_precision="float32") != baseline
    assert embedding_signature(timm_version="1.0.21") != baseline
    assert embedding_signature(artifact_sha256="e" * 64) != baseline
    assert embedding_signature(loader_implementation_version="timm-loader-v2") != baseline
