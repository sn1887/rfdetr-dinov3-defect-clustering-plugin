from __future__ import annotations

import json

import numpy as np
import pytest
from defect_curation_core.artifacts.bundle import RunBundlePublisher
from defect_curation_core.io.local import LocalArtifactStore
from defect_curation_core.types import BBoxXYXY, ImageRecord, InstanceRecord


@pytest.mark.requires_pyarrow
def test_real_parquet_bundle_when_pyarrow_is_available(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    store = LocalArtifactStore(tmp_path / "store")
    image = ImageRecord(
        image_path="a.png",
        image_sha256="a" * 64,
        byte_size=10,
        source_mode="RGB",
        width=20,
        height=20,
        status="OK",
    )
    instance = InstanceRecord(
        instance_id="di_0123456789abcdef",
        instance_key="b" * 64,
        image_path="a.png",
        image_sha256="a" * 64,
        bbox=BBoxXYXY(1, 1, 10, 10),
        bbox_xywh=(1, 1, 9, 9),
        detector_score=0.9,
        embedding_row=0,
        embedding_status="COMPUTED",
        cluster_id=0,
        cosine_to_centroid=1.0,
        cluster_rank=1,
        cluster_size=1,
    )
    result = RunBundlePublisher(store).publish(
        run_id="00000000-0000-0000-0000-000000000000",
        resolved_config_yaml="schema_version: 1\n",
        provenance={},
        manifest_fields={"pipeline": "test"},
        images=[image],
        instances=[instance],
        embeddings=np.ones((1, 768), dtype=np.float16),
        centroids=np.ones((1, 768), dtype=np.float32),
        failures=[],
        write_checksums=True,
    )
    assert result.run_path.startswith("runs/")
    assert json.loads((tmp_path / "store" / "LATEST.json").read_text())["run_id"]
