from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch
from PIL import Image

from defect_curation_core.config import compose_config, format_override
from defect_curation_core.errors import ConfigurationError
from defect_curation_core.io.local import LocalArtifactStore, LocalImageSource, MemoryReviewDatasetSink
from defect_curation_core.pipeline import DefectClusteringPipeline, FitReference, load_fit_reference
from defect_curation_core.types import BBoxXYXY, Detection


class FakeDetector:
    def __init__(self, counter: dict[str, int]) -> None:
        self.counter = counter

    def predict(self, images: Sequence[Image.Image], *, threshold: float):
        del threshold
        self.counter["detector_batches"] = self.counter.get("detector_batches", 0) + 1
        output = []
        for image in images:
            red, green, blue = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            if abs(red - green) < 2 and abs(green - blue) < 2:
                output.append([])
                continue
            width, height = image.size
            output.append(
                [
                    Detection(
                        BBoxXYXY(width * 0.15, height * 0.15, width * 0.85, height * 0.85),
                        0.95,
                        0,
                    )
                ]
            )
        return output

    def close(self) -> None:
        self.counter["detector_closed"] = self.counter.get("detector_closed", 0) + 1


class FakeEmbedder:
    def __init__(self, counter: dict[str, int]) -> None:
        self.counter = counter

    def extract_patch_maps(self, images: Sequence[Image.Image]):
        self.counter["embedder_batches"] = self.counter.get("embedder_batches", 0) + 1
        values = torch.zeros((len(images), 768, 32, 32), dtype=torch.float32)
        for index, image in enumerate(images):
            means = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            channel = int(np.argmax(means))
            values[index, channel] = 1.0
            values[index, 3 + channel] = 0.05
        return values

    def normalized_zero_basis_features(self):
        raise AssertionError("The synthetic config disables positional debiasing")

    def close(self) -> None:
        self.counter["embedder_closed"] = self.counter.get("embedder_closed", 0) + 1


def build_config(
    tmp_path: Path,
    *,
    k: int,
    max_failure_fraction: float = 0.25,
    embedding_granularity: str = "object",
):
    checkpoint = tmp_path / "models" / "rfdetr.pth"
    weights = tmp_path / "models" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-rfdetr")
    weights.write_bytes(b"synthetic-dinov3")
    return compose_config(
        [
            format_override("detector.checkpoint_path", str(checkpoint.resolve())),
            format_override("detector.package_compatibility", "==1.5.2"),
            format_override("detector.device", "cpu"),
            format_override("detector.batch_size", 2),
            format_override("embedding.artifact_path", str(weights.resolve())),
            format_override("embedding.artifact_revision", "synthetic-revision"),
            format_override("embedding.device", "cpu"),
            format_override("embedding.granularity", embedding_granularity),
            format_override("embedding.batch_size", 2),
            format_override("embedding.positional_debias.enabled", False),
            format_override("embedding.inference_precision", "float32"),
            format_override("clustering.k", k),
            format_override("clustering.backend", "numpy_reference"),
            format_override("clustering.nredo", 2),
            format_override("runtime.max_failure_fraction", max_failure_fraction),
        ]
    )


def make_images(root: Path) -> None:
    colors = {
        "red/1.png": (230, 20, 20),
        "red/2.png": (210, 30, 20),
        "green/1.png": (20, 230, 20),
        "green/2.png": (30, 210, 30),
        "blue/1.png": (20, 20, 230),
        "blue/2.png": (30, 30, 210),
        "normal.png": (128, 128, 128),
    }
    for relative, color in colors.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 64), color=color).save(path)


def run_pipeline(
    *,
    image_root: Path,
    artifact_root: Path,
    cfg,
    counter: dict[str, int],
    review_sink: MemoryReviewDatasetSink,
    detector_factory=None,
    embedder_factory=None,
):
    return DefectClusteringPipeline(
        image_source=LocalImageSource(image_root),
        artifact_store=LocalArtifactStore(artifact_root),
        review_sink=review_sink,
        cfg=cfg,
        plugin_version="test",
        detector_factory=detector_factory or (lambda _cfg: FakeDetector(counter)),
        embedder_factory=embedder_factory or (lambda _cfg: FakeEmbedder(counter)),
        rfdetr_package_version="1.5.2",
    ).run()


def test_full_pipeline_and_cache_only_recluster(
    tmp_path: Path,
    patch_parquet_writer,
) -> None:
    del patch_parquet_writer
    image_root = tmp_path / "images"
    artifacts = tmp_path / "artifacts"
    make_images(image_root)

    first_counter: dict[str, int] = {}
    first_sink = MemoryReviewDatasetSink()
    first = run_pipeline(
        image_root=image_root,
        artifact_root=artifacts,
        cfg=build_config(tmp_path, k=3),
        counter=first_counter,
        review_sink=first_sink,
    )
    assert first.image_count == 7
    assert first.instance_count == 6
    assert first.embedded_instance_count == 6
    assert len(first_sink.rows) == 7
    assert sum(row["image_status"] == "NO_DETECTION" for row in first_sink.rows) == 1
    assert all(row["run_id"] == first.run_id for row in first_sink.rows)
    assert first_counter["detector_batches"] >= 1
    assert first_counter["embedder_batches"] >= 1

    latest = json.loads((artifacts / "LATEST.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == first.run_id
    first_manifest = json.loads(
        (artifacts / "runs" / first.run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["cache"]["detection_misses"] == 7
    assert first_manifest["cache"]["embedding_misses"] == 6
    assert first_manifest["models"]["rfdetr"]["package_version"] == "1.5.2"
    assert first_manifest["models"]["rfdetr"]["model_class"] == "RFDETRSmall"
    assert first_manifest["models"]["dinov3"]["model_id"] == (
        "timm/vit_base_patch16_dinov3.lvd1689m"
    )
    assert first_manifest["models"]["dinov3"]["timm_version"] == "1.0.20"
    provenance = json.loads(
        (artifacts / "runs" / first.run_id / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["dinov3"]["artifact_revision"] == "synthetic-revision"
    assert "observed_git_head" not in provenance["dinov3"]
    assert "configured_code_revision" not in provenance["dinov3"]
    assert (artifacts / "runs" / first.run_id / "embeddings.f16.npy").is_file()
    assert (artifacts / "runs" / first.run_id / "centroids.f32.npy").is_file()

    schema_root = Path(__file__).resolve().parents[2] / "resources" / "schemas"
    checker = jsonschema.FormatChecker()
    for row in first_sink.rows:
        jsonschema.validate(
            row,
            json.loads((schema_root / "review_dataset.schema.json").read_text(encoding="utf-8")),
            format_checker=checker,
        )
    jsonschema.validate(
        latest,
        json.loads((schema_root / "latest_pointer.schema.json").read_text(encoding="utf-8")),
        format_checker=checker,
    )
    jsonschema.validate(
        first_manifest,
        json.loads((schema_root / "run_manifest.schema.json").read_text(encoding="utf-8")),
        format_checker=checker,
    )
    instance_schema = json.loads((schema_root / "instance_record.schema.json").read_text(encoding="utf-8"))
    for row in json.loads(
        (artifacts / "runs" / first.run_id / "instances.parquet").read_text(encoding="utf-8")
    ):
        jsonschema.validate(row, instance_schema, format_checker=checker)
    detection_schema = json.loads((schema_root / "detection_cache.schema.json").read_text(encoding="utf-8"))
    detection_cache_files = sorted((artifacts / "cache" / "detections").rglob("*.json"))
    assert len(detection_cache_files) == 7
    for cache_file in detection_cache_files:
        jsonschema.validate(
            json.loads(cache_file.read_text(encoding="utf-8")),
            detection_schema,
            format_checker=checker,
        )

    second_sink = MemoryReviewDatasetSink()

    def forbidden(_cfg):
        raise AssertionError("Compatible caches should make model construction unnecessary")

    second = run_pipeline(
        image_root=image_root,
        artifact_root=artifacts,
        cfg=build_config(tmp_path, k=2),
        counter={},
        review_sink=second_sink,
        detector_factory=forbidden,
        embedder_factory=forbidden,
    )
    assert second.detector_signature == first.detector_signature
    assert second.embedding_signature == first.embedding_signature
    second_manifest = json.loads(
        (artifacts / "runs" / second.run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert second_manifest["cache"]["detection_hits"] == 7
    assert second_manifest["cache"]["embedding_hits"] == 6
    assert second_manifest["clustering"]["k"] == 2
    assert json.loads((artifacts / "LATEST.json").read_text(encoding="utf-8"))["run_id"] == second.run_id


def test_image_level_mode_clusters_all_readable_images_and_preserves_detections(
    tmp_path: Path,
    patch_parquet_writer,
) -> None:
    del patch_parquet_writer
    image_root = tmp_path / "images"
    artifacts = tmp_path / "artifacts"
    make_images(image_root)

    sink = MemoryReviewDatasetSink()
    result = run_pipeline(
        image_root=image_root,
        artifact_root=artifacts,
        cfg=build_config(tmp_path, k=3, embedding_granularity="image"),
        counter={},
        review_sink=sink,
    )
    assert result.image_count == 7
    assert result.instance_count == 7
    assert result.embedded_instance_count == 7
    normal = next(row for row in sink.rows if row["image_path"] == "normal.png")
    assert normal["image_status"] == "NO_DETECTION"
    assert normal["num_defects"] == 0
    assert normal["primary_cluster_id"] is not None
    assert normal["review_order"] is not None
    assert normal["detection_bbox"] == "[]"
    detected = next(row for row in sink.rows if row["image_path"] == "red/1.png")
    assert json.loads(detected["detection_bbox"]) == [
        {"bbox": [14, 9, 68, 46], "category": "defect"}
    ]
    assert json.loads(detected["detection_score"]) == [0.95]
    assert json.loads(detected["detection_bbox_cluster"])[0]["category"] == str(
        detected["primary_cluster_id"]
    )
    manifest = json.loads((artifacts / "runs" / result.run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clustering"]["embedding_granularity"] == "image"


def test_score_dataset_uses_fitted_centroids(
    tmp_path: Path,
    patch_parquet_writer,
) -> None:
    del patch_parquet_writer
    fit_images = tmp_path / "fit-images"
    score_images = tmp_path / "score-images"
    fit_artifacts = tmp_path / "fit-artifacts"
    score_artifacts = tmp_path / "score-artifacts"
    make_images(fit_images)
    make_images(score_images)

    fit_sink = MemoryReviewDatasetSink()
    fit = run_pipeline(
        image_root=fit_images,
        artifact_root=fit_artifacts,
        cfg=build_config(tmp_path, k=3),
        counter={},
        review_sink=fit_sink,
    )
    fit_reference = load_fit_reference(LocalArtifactStore(fit_artifacts))

    score_sink = MemoryReviewDatasetSink()
    score = DefectClusteringPipeline(
        image_source=LocalImageSource(score_images),
        artifact_store=LocalArtifactStore(score_artifacts),
        review_sink=score_sink,
        cfg=build_config(tmp_path, k=fit_reference.cluster_count),
        plugin_version="test",
        detector_factory=lambda _cfg: FakeDetector({}),
        embedder_factory=lambda _cfg: FakeEmbedder({}),
        rfdetr_package_version="1.5.2",
    ).score(fit_reference=fit_reference)

    assert score.image_count == 7
    assert score.embedded_instance_count == 6
    assert len(score_sink.rows) == 7
    manifest = json.loads((score_artifacts / "runs" / score.run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == "rfdetr_dinov3_centroid_scoring"
    assert manifest["fit_reference"]["run_id"] == fit.run_id
    assert manifest["clustering"]["algorithm"] == "nearest_fitted_centroid"

    incompatible_reference = FitReference(
        run_id=fit_reference.run_id,
        manifest_path=fit_reference.manifest_path,
        manifest_sha256=fit_reference.manifest_sha256,
        manifest=fit_reference.manifest,
        centroids=fit_reference.centroids,
        embedding_signature=fit_reference.embedding_signature,
        embedding_granularity="image",
        box_padding_fraction=fit_reference.box_padding_fraction,
        cluster_count=fit_reference.cluster_count,
    )
    with pytest.raises(ConfigurationError, match="embedding granularity"):
        DefectClusteringPipeline(
            image_source=LocalImageSource(score_images),
            artifact_store=LocalArtifactStore(tmp_path / "bad-score-artifacts"),
            review_sink=MemoryReviewDatasetSink(),
            cfg=build_config(tmp_path, k=fit_reference.cluster_count),
            plugin_version="test",
            detector_factory=lambda _cfg: FakeDetector({}),
            embedder_factory=lambda _cfg: FakeEmbedder({}),
            rfdetr_package_version="1.5.2",
        ).score(fit_reference=incompatible_reference)


def test_corrupt_image_is_an_error_row_but_does_not_abort(
    tmp_path: Path,
    patch_parquet_writer,
) -> None:
    del patch_parquet_writer
    image_root = tmp_path / "images"
    artifacts = tmp_path / "artifacts"
    for index, color in enumerate(((220, 20, 20), (200, 30, 20), (20, 20, 220), (30, 30, 200))):
        path = image_root / f"valid-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=color).save(path)
    (image_root / "broken.png").write_bytes(b"not-an-image")

    sink = MemoryReviewDatasetSink()
    result = run_pipeline(
        image_root=image_root,
        artifact_root=artifacts,
        cfg=build_config(tmp_path, k=2, max_failure_fraction=0.25),
        counter={},
        review_sink=sink,
    )
    assert result.image_count == 5
    error = next(row for row in sink.rows if row["image_path"] == "broken.png")
    assert error["image_status"] == "ERROR"
    assert error["num_defects"] == 0
    assert error["detection_bbox"] == "[]"
    assert error["detection_score"] == "[]"
    assert error["detection_bbox_cluster"] == "[]"
    assert error["instances_json"] == "[]"
    assert error["error_code"] == "IMAGE_DECODE_FAILED"


def test_duplicate_image_bytes_have_distinct_review_ids_and_shared_cache_keys(
    tmp_path: Path,
    patch_parquet_writer,
) -> None:
    del patch_parquet_writer
    image_root = tmp_path / "images"
    artifacts = tmp_path / "artifacts"
    first = image_root / "first.png"
    copy = image_root / "copy" / "first.png"
    other = image_root / "other.png"
    first.parent.mkdir(parents=True, exist_ok=True)
    copy.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(220, 20, 20)).save(first)
    copy.write_bytes(first.read_bytes())
    Image.new("RGB", (64, 64), color=(20, 20, 220)).save(other)

    sink = MemoryReviewDatasetSink()
    result = run_pipeline(
        image_root=image_root,
        artifact_root=artifacts,
        cfg=build_config(tmp_path, k=2),
        counter={},
        review_sink=sink,
    )
    review_ids = {
        row["image_path"]: json.loads(row["instances_json"])[0]["instance_id"]
        for row in sink.rows
    }
    assert review_ids["first.png"] != review_ids["copy/first.png"]

    # The lightweight test fixture serializes Parquet rows as JSON, allowing us
    # to verify the internal content-addressed key without requiring pyarrow.
    instance_rows = json.loads(
        (artifacts / "runs" / result.run_id / "instances.parquet").read_text(encoding="utf-8")
    )
    by_path = {row["image_path"]: row for row in instance_rows}
    assert by_path["first.png"]["instance_key"] == by_path["copy/first.png"]["instance_key"]
    assert by_path["first.png"]["instance_id"] != by_path["copy/first.png"]["instance_id"]
