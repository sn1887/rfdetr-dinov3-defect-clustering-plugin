"""Atomic, checksummed publication of a versioned run bundle."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from defect_curation_core.errors import ArtifactError, DependencyError
from defect_curation_core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from defect_curation_core.io.protocols import ArtifactStore
from defect_curation_core.types import FailureRecord, ImageRecord, InstanceRecord


@dataclass(frozen=True, slots=True)
class BundlePublishResult:
    run_path: str
    manifest_sha256: str
    artifact_hashes: dict[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DependencyError("pyarrow is required to write the versioned Parquet run bundle") from exc
    return pa, pq


def _write_parquet(path: Path, rows: list[dict[str, Any]], schema: Any) -> None:
    pa, pq = _require_pyarrow()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _input_schema():
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            ("image_path", pa.string()),
            ("image_sha256", pa.string()),
            ("byte_size", pa.int64()),
            ("source_mode", pa.string()),
            ("image_width", pa.int32()),
            ("image_height", pa.int32()),
            ("image_status", pa.string()),
            ("error_code", pa.string()),
            ("error_message", pa.string()),
            ("warning_codes", pa.list_(pa.string())),
        ]
    )


def _instance_schema():
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            ("run_id", pa.string()),
            ("instance_id", pa.string()),
            ("instance_key", pa.string()),
            ("image_path", pa.string()),
            ("image_sha256", pa.string()),
            ("bbox_xmin", pa.int32()),
            ("bbox_ymin", pa.int32()),
            ("bbox_width", pa.int32()),
            ("bbox_height", pa.int32()),
            ("bbox_xyxy_float", pa.list_(pa.float32(), 4)),
            ("detector_score", pa.float32()),
            ("raw_class_id", pa.int32()),
            ("embedding_row", pa.int64()),
            ("embedding_status", pa.string()),
            ("cluster_id", pa.int32()),
            ("cosine_to_centroid", pa.float32()),
            ("cluster_rank", pa.int32()),
            ("cluster_size", pa.int32()),
            ("warning_codes", pa.list_(pa.string())),
        ]
    )


def _failure_schema():
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            ("run_id", pa.string()),
            ("image_path", pa.string()),
            ("image_sha256", pa.string()),
            ("instance_id", pa.string()),
            ("stage", pa.string()),
            ("error_code", pa.string()),
            ("exception_type", pa.string()),
            ("sanitized_message", pa.string()),
            ("retry_count", pa.int32()),
            ("occurred_at_utc", pa.string()),
        ]
    )


def _cluster_schema():
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            ("cluster_id", pa.int32()),
            ("cluster_size", pa.int64()),
            ("representative_instance_id", pa.string()),
            ("mean_cosine_to_centroid", pa.float32()),
            ("min_cosine_to_centroid", pa.float32()),
            ("max_cosine_to_centroid", pa.float32()),
        ]
    )


def _image_rows(images: Sequence[ImageRecord]) -> list[dict[str, Any]]:
    return [
        {
            "image_path": image.image_path,
            "image_sha256": image.image_sha256,
            "byte_size": image.byte_size,
            "source_mode": image.source_mode,
            "image_width": image.width,
            "image_height": image.height,
            "image_status": image.status,
            "error_code": image.error_code,
            "error_message": image.error_message,
            "warning_codes": list(image.warning_codes),
        }
        for image in images
    ]


def _instance_rows(run_id: str, instances: Sequence[InstanceRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        xmin, ymin, width, height = instance.bbox_xywh
        rows.append(
            {
                "run_id": run_id,
                "instance_id": instance.instance_id,
                "instance_key": instance.instance_key,
                "image_path": instance.image_path,
                "image_sha256": instance.image_sha256,
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_xyxy_float": [np.float32(value) for value in instance.bbox.as_list()],
                "detector_score": np.float32(instance.detector_score),
                "raw_class_id": instance.raw_class_id,
                "embedding_row": instance.embedding_row,
                "embedding_status": instance.embedding_status,
                "cluster_id": instance.cluster_id,
                "cosine_to_centroid": (
                    None if instance.cosine_to_centroid is None else np.float32(instance.cosine_to_centroid)
                ),
                "cluster_rank": instance.cluster_rank,
                "cluster_size": instance.cluster_size,
                "warning_codes": list(instance.warning_codes),
            }
        )
    return rows


def _cluster_rows(instances: Sequence[InstanceRecord]) -> list[dict[str, Any]]:
    grouped: dict[int, list[InstanceRecord]] = {}
    for instance in instances:
        if instance.cluster_id is not None:
            grouped.setdefault(instance.cluster_id, []).append(instance)
    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(grouped):
        members = grouped[cluster_id]
        similarities = np.asarray([member.cosine_to_centroid for member in members], dtype=np.float32)
        representative = min(
            members,
            key=lambda item: (
                -float(item.cosine_to_centroid if item.cosine_to_centroid is not None else -np.inf),
                item.instance_id,
            ),
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "representative_instance_id": representative.instance_id,
                "mean_cosine_to_centroid": np.float32(similarities.mean()),
                "min_cosine_to_centroid": np.float32(similarities.min()),
                "max_cosine_to_centroid": np.float32(similarities.max()),
            }
        )
    return rows


class RunBundlePublisher:
    def __init__(self, store: ArtifactStore, *, temporary_root: str | None = None) -> None:
        self.store = store
        self.temporary_root = temporary_root

    def publish(
        self,
        *,
        run_id: str,
        resolved_config_yaml: str,
        provenance: dict[str, Any],
        manifest_fields: dict[str, Any],
        images: Sequence[ImageRecord],
        instances: Sequence[InstanceRecord],
        embeddings: np.ndarray,
        centroids: np.ndarray,
        failures: Sequence[FailureRecord],
        write_checksums: bool,
        before_latest: Callable[[], None] | None = None,
    ) -> BundlePublishResult:
        run_prefix = f"runs/{run_id}"
        with tempfile.TemporaryDirectory(prefix="defect-curation-run-", dir=self.temporary_root) as tmp:
            root = Path(tmp)
            (root / "resolved_config.yaml").write_text(resolved_config_yaml, encoding="utf-8")
            (root / "provenance.json").write_bytes(canonical_json_bytes(provenance))

            _write_parquet(root / "input_manifest.parquet", _image_rows(images), _input_schema())
            _write_parquet(root / "instances.parquet", _instance_rows(run_id, instances), _instance_schema())
            _write_parquet(
                root / "failures.parquet",
                [failure.to_dict() for failure in failures],
                _failure_schema(),
            )
            _write_parquet(root / "cluster_summary.parquet", _cluster_rows(instances), _cluster_schema())

            np.save(root / "embeddings.f16.npy", np.asarray(embeddings, dtype=np.float16), allow_pickle=False)
            np.save(root / "centroids.f32.npy", np.asarray(centroids, dtype=np.float32), allow_pickle=False)

            artifact_paths = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )
            artifact_hashes = {relative: sha256_file(root / relative) for relative in artifact_paths}
            manifest = {
                "schema_version": "1.0",
                "run_id": run_id,
                "created_at_utc": _utc_now(),
                **manifest_fields,
                "artifacts": {
                    relative: {"sha256": digest}
                    for relative, digest in artifact_hashes.items()
                },
            }
            (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            manifest_sha = sha256_file(root / "manifest.json")
            artifact_hashes["manifest.json"] = manifest_sha

            if write_checksums:
                checksum_lines = [
                    f"{digest}  {relative}\n" for relative, digest in sorted(artifact_hashes.items())
                ]
                (root / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
                artifact_hashes["checksums.sha256"] = sha256_file(root / "checksums.sha256")

            for relative in sorted(artifact_hashes):
                self.store.upload_file(f"{run_prefix}/{relative}", str(root / relative))

            uploaded_manifest = self.store.read_bytes(f"{run_prefix}/manifest.json")
            if sha256_file(root / "manifest.json") != sha256_bytes(uploaded_manifest):
                raise ArtifactError("Uploaded run manifest failed the integrity read-back check")

            if before_latest is not None:
                before_latest()

            latest = {
                "schema_version": 1,
                "run_id": run_id,
                "manifest_path": f"{run_prefix}/manifest.json",
                "manifest_sha256": manifest_sha,
                "published_at_utc": _utc_now(),
            }
            self.store.write_bytes("LATEST.json", canonical_json_bytes(latest))

        return BundlePublishResult(
            run_path=run_prefix,
            manifest_sha256=manifest_sha,
            artifact_hashes=artifact_hashes,
        )
