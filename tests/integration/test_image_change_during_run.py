from __future__ import annotations

import io
from pathlib import Path

from defect_curation_core.io.local import LocalArtifactStore, MemoryReviewDatasetSink
from defect_curation_core.pipeline import DefectClusteringPipeline
from PIL import Image

from tests.integration.test_filesystem_pipeline import FakeDetector, FakeEmbedder, build_config


def png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class ChangingSource:
    identifier = "changing-source"

    def __init__(self):
        self.values = {
            "changed.png": (png_bytes((220, 20, 20)), png_bytes((20, 220, 20))),
            "stable-a.png": (png_bytes((210, 20, 20)),) * 2,
            "stable-b.png": (png_bytes((20, 20, 220)),) * 2,
        }
        self.read_counts = {path: 0 for path in self.values}

    def list_paths(self):
        return sorted(self.values)

    def read_bytes(self, path):
        index = min(self.read_counts[path], len(self.values[path]) - 1)
        self.read_counts[path] += 1
        return self.values[path][index]

    def size(self, path):
        return len(self.values[path][0])


def test_changed_image_is_invalidated_and_review_inert(tmp_path: Path, patch_parquet_writer) -> None:
    del patch_parquet_writer
    source = ChangingSource()
    sink = MemoryReviewDatasetSink()
    counter = {}
    result = DefectClusteringPipeline(
        image_source=source,
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        review_sink=sink,
        cfg=build_config(tmp_path, k=2, max_failure_fraction=0.5),
        plugin_version="test",
        detector_factory=lambda _cfg: FakeDetector(counter),
        embedder_factory=lambda _cfg: FakeEmbedder(counter),
        rfdetr_package_version="1.5.2",
    ).run()
    assert result.embedded_instance_count == 2
    changed = next(row for row in sink.rows if row["image_path"] == "changed.png")
    assert changed["image_status"] == "ERROR"
    assert changed["error_code"] == "IMAGE_CHANGED_DURING_RUN"
    assert changed["num_defects"] == 0
    assert changed["prelabels_json"] == "[]"


class CacheOnlySecondRunSource:
    identifier = "cache-only-changing-source"

    def __init__(self):
        self.old = {
            "changed.png": png_bytes((220, 20, 20)),
            "stable-a.png": png_bytes((210, 20, 20)),
            "stable-b.png": png_bytes((20, 20, 220)),
        }
        self.changed = png_bytes((20, 220, 20))
        self.run_number = 1
        self.read_counts = {path: 0 for path in self.old}

    def begin_second_run(self) -> None:
        self.run_number = 2
        self.read_counts = {path: 0 for path in self.old}

    def list_paths(self):
        return sorted(self.old)

    def read_bytes(self, path):
        self.read_counts[path] += 1
        if self.run_number == 2 and path == "changed.png" and self.read_counts[path] >= 2:
            return self.changed
        return self.old[path]

    def size(self, path):
        return len(self.old[path])


def test_cache_only_recluster_revalidates_source_bytes(
    tmp_path: Path,
    patch_parquet_writer,
) -> None:
    del patch_parquet_writer
    source = CacheOnlySecondRunSource()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    first_sink = MemoryReviewDatasetSink()
    first_counter: dict[str, int] = {}
    first = DefectClusteringPipeline(
        image_source=source,
        artifact_store=artifacts,
        review_sink=first_sink,
        cfg=build_config(tmp_path, k=2, max_failure_fraction=0.5),
        plugin_version="test",
        detector_factory=lambda _cfg: FakeDetector(first_counter),
        embedder_factory=lambda _cfg: FakeEmbedder(first_counter),
        rfdetr_package_version="1.5.2",
    ).run()
    assert first.embedded_instance_count == 3

    source.begin_second_run()
    second_sink = MemoryReviewDatasetSink()

    def forbidden(_cfg):
        raise AssertionError("A cache-only recluster must not construct either model")

    second = DefectClusteringPipeline(
        image_source=source,
        artifact_store=artifacts,
        review_sink=second_sink,
        cfg=build_config(tmp_path, k=2, max_failure_fraction=0.5),
        plugin_version="test",
        detector_factory=forbidden,
        embedder_factory=forbidden,
        rfdetr_package_version="1.5.2",
    ).run()
    assert second.embedded_instance_count == 2
    changed = next(row for row in second_sink.rows if row["image_path"] == "changed.png")
    assert changed["image_status"] == "ERROR"
    assert changed["error_code"] == "IMAGE_CHANGED_DURING_RUN"
    assert changed["prelabels_json"] == "[]"
    assert source.read_counts == {"changed.png": 2, "stable-a.png": 2, "stable-b.png": 2}
