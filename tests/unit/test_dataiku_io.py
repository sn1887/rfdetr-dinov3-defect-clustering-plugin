from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path

import pytest
from defect_curation_core.errors import ConfigurationError
from defect_curation_plugin.dataiku_io import (
    DataikuManagedFolderArtifactStore,
    DataikuManagedFolderImageSource,
    DataikuReviewDatasetSink,
)
from defect_curation_plugin.dataiku_schema import REVIEW_DATASET_COLUMNS


class RemoteFolder:
    def __init__(self, initial=None):
        self.objects = dict(initial or {})

    def get_path(self):
        raise RuntimeError("remote")

    def list_paths_in_partition(self):
        return [f"/{key}" for key in sorted(self.objects)]

    @contextmanager
    def get_download_stream(self, path):
        yield io.BytesIO(self.objects[path.lstrip("/")])

    def get_path_details(self, path):
        key = path.lstrip("/")
        if key not in self.objects:
            raise KeyError(key)
        return {"exists": True, "size": len(self.objects[key]), "directory": False}

    def upload_data(self, path, data):
        self.objects[path.lstrip("/")] = bytes(data)

    def upload_file(self, path, local_path):
        self.objects[path.lstrip("/")] = Path(local_path).read_bytes()


class Writer:
    def __init__(self):
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write_row_dict(self, row):
        self.rows.append(dict(row))


class Dataset:
    def __init__(self):
        self.schema = None
        self.writer = Writer()

    def write_schema(self, schema):
        self.schema = schema

    def get_writer(self):
        return self.writer


def test_remote_image_and_artifact_adapters(tmp_path: Path) -> None:
    folder = RemoteFolder({"nested/a.png": b"image-bytes"})
    source = DataikuManagedFolderImageSource(folder, identifier="input")
    assert list(source.list_paths()) == ["nested/a.png"]
    assert source.read_bytes("nested/a.png") == b"image-bytes"
    assert source.size("nested/a.png") == 11

    store = DataikuManagedFolderArtifactStore(folder, identifier="output")
    store.write_bytes("runs/x/a.json", b"{}")
    assert store.exists("runs/x/a.json")
    assert store.read_bytes("runs/x/a.json") == b"{}"
    local = tmp_path / "file.bin"
    local.write_bytes(b"payload")
    store.upload_file("runs/x/file.bin", str(local))
    assert "runs/x/file.bin" in store.list_paths("runs/x")


def test_fixed_dataset_writer() -> None:
    dataset = Dataset()
    sink = DataikuReviewDatasetSink(dataset, identifier="review")
    row = {column: None for column in REVIEW_DATASET_COLUMNS}
    row.update(
        image_path="a.png",
        image_status="NO_DETECTION",
        num_defects=0,
        detection_bbox="[]",
        detection_score="[]",
        detection_bbox_cluster="[]",
        instances_json="[]",
        run_id="run",
    )
    sink.write_rows([row])
    assert dataset.schema is not None
    assert dataset.writer.rows == [row]


def test_fixed_dataset_writer_rejects_missing_columns() -> None:
    dataset = Dataset()
    sink = DataikuReviewDatasetSink(dataset, identifier="review")
    with pytest.raises(ConfigurationError, match="missing fixed-schema columns"):
        sink.write_rows([{"image_path": "a.png"}])
