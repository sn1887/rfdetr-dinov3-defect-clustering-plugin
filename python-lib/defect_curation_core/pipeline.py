"""Single-path RF-DETR → DINOv3 → spherical k-means pipeline."""

from __future__ import annotations

import importlib.metadata
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from omegaconf import DictConfig
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from defect_curation_core.artifacts.bundle import BundlePublishResult, RunBundlePublisher
from defect_curation_core.artifacts.cache import CacheManager
from defect_curation_core.artifacts.signatures import (
    build_basis_input_signature,
    build_detector_signature,
    build_embedding_signature,
)
from defect_curation_core.clustering.review_order import assign_cluster_ranks, build_review_order
from defect_curation_core.clustering.spherical_kmeans import fit_spherical_kmeans
from defect_curation_core.config import config_as_dict, config_hash, resolved_yaml
from defect_curation_core.detection.postprocess import (
    build_instance_identity,
    postprocess_detections,
    quantized_bbox_key,
)
from defect_curation_core.detection.rfdetr_adapter import RFDETRAdapter
from defect_curation_core.embeddings.crop_geometry import CropSample, make_letterboxed_crop
from defect_curation_core.embeddings.dinov3_adapter import (
    DINO_LOADER_IMPLEMENTATION_VERSION,
    DinoV3Adapter,
)
from defect_curation_core.embeddings.pooling import pool_fractional_box_features
from defect_curation_core.embeddings.positional_debias import compute_positional_basis
from defect_curation_core.errors import (
    ArtifactError,
    ConfigurationError,
    DependencyError,
    FatalPipelineError,
    ModelLoadError,
    RowProcessingError,
)
from defect_curation_core.hashing import (
    normalize_relative_path,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_expected_sha256,
)
from defect_curation_core.image_io import decode_image_bytes
from defect_curation_core.io.protocols import ArtifactStore, ImageSource, ReviewDatasetSink
from defect_curation_core.labeling.dataiku_object_detection import build_review_rows
from defect_curation_core.provenance import collect_provenance
from defect_curation_core.sanitize import sanitize_message
from defect_curation_core.types import FailureRecord, ImageRecord, InstanceRecord

DetectorFactory = Callable[[DictConfig], Any]
EmbedderFactory = Callable[[DictConfig], Any]


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    run_id: str
    review_row_count: int
    image_count: int
    instance_count: int
    embedded_instance_count: int
    detector_signature: str
    embedding_signature: str
    bundle: BundlePublishResult


@dataclass(slots=True)
class _DetectionWork:
    image_index: int
    image: Any


@dataclass(slots=True)
class _EmbeddingWork:
    instance_index: int
    crop: CropSample


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_cuda_oom(exc: BaseException) -> bool:
    if "out of memory" in str(exc).lower() and "cuda" in str(exc).lower():
        return True
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:
        return False


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class DefectClusteringPipeline:
    """Orchestrates the one approved production path with injected I/O boundaries."""

    def __init__(
        self,
        *,
        image_source: ImageSource,
        artifact_store: ArtifactStore,
        review_sink: ReviewDatasetSink,
        cfg: DictConfig,
        plugin_version: str,
        plugin_source_commit: str | None = None,
        dss_version: str | None = None,
        detector_factory: DetectorFactory | None = None,
        embedder_factory: EmbedderFactory | None = None,
        rfdetr_package_version: str | None = None,
    ) -> None:
        self.image_source = image_source
        self.artifact_store = artifact_store
        self.review_sink = review_sink
        self.cfg = cfg
        self.plugin_version = plugin_version
        self.plugin_source_commit = plugin_source_commit
        self.dss_version = dss_version
        self.detector_factory = detector_factory or self._default_detector_factory
        self.embedder_factory = embedder_factory or self._default_embedder_factory
        self._rfdetr_package_version_override = rfdetr_package_version
        self.cache = CacheManager(artifact_store)

    @staticmethod
    def _default_detector_factory(cfg: DictConfig) -> RFDETRAdapter:
        return RFDETRAdapter(
            checkpoint_path=str(cfg.detector.checkpoint_path),
            model_class=str(cfg.detector.model_class),
            package_compatibility=str(cfg.detector.package_compatibility),
            device=str(cfg.detector.device),
            safe_checkpoint_loading=bool(cfg.detector.safe_checkpoint_loading),
        )

    @staticmethod
    def _default_embedder_factory(cfg: DictConfig) -> DinoV3Adapter:
        return DinoV3Adapter(
            artifact_path=str(cfg.embedding.artifact_path),
            model_id=str(cfg.embedding.model_id),
            timm_version=str(cfg.embedding.timm_version),
            device=str(cfg.embedding.device),
            input_size=int(cfg.embedding.input_size),
            patch_size=int(cfg.embedding.patch_size),
            embedding_dim=int(cfg.embedding.embedding_dim),
            expected_prefix_tokens=int(cfg.embedding.expected_prefix_tokens),
            inference_precision=str(cfg.embedding.inference_precision),
        )

    def _resolve_rfdetr_package_version(self) -> str:
        if self._rfdetr_package_version_override is not None:
            version = self._rfdetr_package_version_override
        else:
            try:
                version = importlib.metadata.version("rfdetr")
            except importlib.metadata.PackageNotFoundError as exc:
                raise DependencyError("The rfdetr package is not installed") from exc
        try:
            specifier = SpecifierSet(str(self.cfg.detector.package_compatibility))
        except InvalidSpecifier as exc:
            raise ConfigurationError(
                f"Invalid RF-DETR compatibility specifier: {self.cfg.detector.package_compatibility}"
            ) from exc
        if version not in specifier:
            raise ModelLoadError(
                f"Installed rfdetr {version} does not satisfy {self.cfg.detector.package_compatibility}"
            )
        return version

    def _eligible_image_paths(self) -> list[str]:
        extensions = {str(item) for item in self.cfg.runtime.image_extensions}
        ignored_names = {"thumbs.db", "desktop.ini"}
        normalized: list[str] = []
        for raw_path in self.image_source.list_paths():
            path = normalize_relative_path(raw_path)
            parts = PurePosixPath(path).parts
            if any(part.startswith(".") or part == "__MACOSX" for part in parts):
                continue
            if parts[-1].lower() in ignored_names:
                continue
            if PurePosixPath(path).suffix.lower() not in extensions:
                continue
            normalized.append(path)
        if bool(self.cfg.runtime.sort_paths):
            normalized.sort()
        if not normalized:
            raise FatalPipelineError("The input managed folder contains no supported JPG/JPEG/PNG images")
        return normalized

    def _new_failure(
        self,
        *,
        run_id: str,
        image_path: str,
        stage: str,
        code: str,
        exc: BaseException,
        image_sha256: str | None = None,
        instance_id: str | None = None,
        retry_count: int = 0,
        message: str | None = None,
    ) -> FailureRecord:
        return FailureRecord(
            run_id=run_id,
            image_path=image_path,
            image_sha256=image_sha256,
            instance_id=instance_id,
            stage=stage,
            error_code=code,
            exception_type=type(exc).__name__,
            sanitized_message=sanitize_message(message if message is not None else str(exc)),
            retry_count=retry_count,
        )

    def _mark_image_error(
        self,
        *,
        image: ImageRecord,
        code: str,
        message: str,
    ) -> None:
        image.status = "ERROR"
        image.error_code = code
        image.error_message = sanitize_message(message)

    def _check_failure_fraction(self, images: Sequence[ImageRecord], *, stage: str) -> None:
        if not images:
            return
        errors = sum(image.status == "ERROR" for image in images)
        fraction = errors / len(images)
        if fraction > float(self.cfg.runtime.max_failure_fraction):
            raise FatalPipelineError(
                f"Row-level error fraction {fraction:.3%} after {stage} exceeds configured ceiling "
                f"{float(self.cfg.runtime.max_failure_fraction):.3%}"
            )

    def _process_detection_batch(
        self,
        *,
        run_id: str,
        detector: Any,
        work: Sequence[_DetectionWork],
        images: list[ImageRecord],
        detector_signature: str,
        failures: list[FailureRecord],
        attempted_batch_sizes: list[int],
        depth: int = 0,
    ) -> None:
        if not work:
            return
        attempted_batch_sizes.append(len(work))
        try:
            predictions = detector.predict(
                [item.image for item in work],
                threshold=float(self.cfg.detector.threshold),
            )
            if len(predictions) != len(work):
                raise RuntimeError("Detector prediction count does not match batch size")
            for item, raw_detections in zip(work, predictions, strict=True):
                image = images[item.image_index]
                processed = postprocess_detections(
                    raw_detections,
                    image_width=int(image.width),
                    image_height=int(image.height),
                    threshold=float(self.cfg.detector.threshold),
                    max_detections=int(self.cfg.detector.max_detections_per_image),
                    clip_to_image=bool(self.cfg.detector.clip_to_image),
                    drop_degenerate=bool(self.cfg.detector.drop_degenerate),
                )
                image.detections = processed
                self.cache.save_detections(
                    detector_signature=detector_signature,
                    image_sha256=str(image.image_sha256),
                    width=int(image.width),
                    height=int(image.height),
                    detections=processed,
                )
        except (DependencyError, ModelLoadError, ConfigurationError, ArtifactError):
            raise
        except Exception as exc:
            if _is_cuda_oom(exc):
                _clear_cuda_cache()
            if len(work) > 1 and depth < int(self.cfg.detector.oom_retries):
                midpoint = len(work) // 2
                self._process_detection_batch(
                    run_id=run_id,
                    detector=detector,
                    work=work[:midpoint],
                    images=images,
                    detector_signature=detector_signature,
                    failures=failures,
                    attempted_batch_sizes=attempted_batch_sizes,
                    depth=depth + 1,
                )
                self._process_detection_batch(
                    run_id=run_id,
                    detector=detector,
                    work=work[midpoint:],
                    images=images,
                    detector_signature=detector_signature,
                    failures=failures,
                    attempted_batch_sizes=attempted_batch_sizes,
                    depth=depth + 1,
                )
                return
            for item in work:
                image = images[item.image_index]
                code = "RFDETR_CUDA_OOM" if _is_cuda_oom(exc) else "RFDETR_INFERENCE_FAILED"
                self._mark_image_error(image=image, code=code, message="RF-DETR inference failed for this image")
                failures.append(
                    self._new_failure(
                        run_id=run_id,
                        image_path=image.image_path,
                        image_sha256=image.image_sha256,
                        stage="detection",
                        code=code,
                        exc=exc,
                        retry_count=depth,
                    )
                )
                if not bool(self.cfg.runtime.continue_on_image_error):
                    raise FatalPipelineError(f"Detection failed for {image.image_path}") from exc

    def _detection_phase(
        self,
        *,
        run_id: str,
        paths: Sequence[str],
        detector_signature: str,
        failures: list[FailureRecord],
    ) -> tuple[list[ImageRecord], int, int, list[int]]:
        images: list[ImageRecord] = []
        cache_hits = 0
        cache_misses = 0
        pending: list[_DetectionWork] = []
        detector: Any | None = None
        attempted_batch_sizes: list[int] = []

        def flush() -> None:
            nonlocal detector, pending
            if not pending:
                return
            if detector is None:
                detector = self.detector_factory(self.cfg)
            batch = pending
            pending = []
            try:
                self._process_detection_batch(
                    run_id=run_id,
                    detector=detector,
                    work=batch,
                    images=images,
                    detector_signature=detector_signature,
                    failures=failures,
                    attempted_batch_sizes=attempted_batch_sizes,
                )
            finally:
                for item in batch:
                    try:
                        item.image.close()
                    except Exception:
                        pass

        try:
            for path in paths:
                record = ImageRecord(image_path=path)
                images.append(record)
                image_index = len(images) - 1
                try:
                    raw = self.image_source.read_bytes(path)
                    record.byte_size = len(raw)
                    record.image_sha256 = sha256_bytes(raw)
                    decoded = decode_image_bytes(raw)
                    record.source_mode = decoded.source_mode
                    record.width = decoded.width
                    record.height = decoded.height
                    cached = None
                    if not bool(self.cfg.runtime.force_recompute):
                        cached = self.cache.load_detections(
                            detector_signature=detector_signature,
                            image_sha256=record.image_sha256,
                            width=decoded.width,
                            height=decoded.height,
                            threshold=float(self.cfg.detector.threshold),
                            max_detections=int(self.cfg.detector.max_detections_per_image),
                        )
                    if cached is not None:
                        cache_hits += 1
                        record.detections = cached
                        decoded.image.close()
                    else:
                        cache_misses += 1
                        pending.append(_DetectionWork(image_index=image_index, image=decoded.image))
                        if len(pending) >= int(self.cfg.detector.batch_size):
                            flush()
                except RowProcessingError as exc:
                    self._mark_image_error(image=record, code=exc.code, message=str(exc))
                    failures.append(
                        self._new_failure(
                            run_id=run_id,
                            image_path=path,
                            image_sha256=record.image_sha256,
                            stage=exc.stage,
                            code=exc.code,
                            exc=exc.cause or exc,
                            message=str(exc),
                        )
                    )
                    if not bool(self.cfg.runtime.continue_on_image_error):
                        raise FatalPipelineError(f"Image processing failed for {path}") from exc
                except (ArtifactError, DependencyError, ModelLoadError, ConfigurationError):
                    raise
                except Exception as exc:
                    self._mark_image_error(image=record, code="IMAGE_READ_FAILED", message="Image could not be read")
                    failures.append(
                        self._new_failure(
                            run_id=run_id,
                            image_path=path,
                            image_sha256=record.image_sha256,
                            stage="read",
                            code="IMAGE_READ_FAILED",
                            exc=exc,
                        )
                    )
                    if not bool(self.cfg.runtime.continue_on_image_error):
                        raise FatalPipelineError(f"Image read failed for {path}") from exc
            flush()
        finally:
            for item in pending:
                try:
                    item.image.close()
                except Exception:
                    pass
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass

        for image in images:
            if image.status == "OK" and not image.detections:
                image.status = "NO_DETECTION"
        self._check_failure_fraction(images, stage="detection")
        return images, cache_hits, cache_misses, attempted_batch_sizes

    def _build_instances(
        self,
        *,
        images: Sequence[ImageRecord],
        detector_signature: str,
    ) -> list[InstanceRecord]:
        instances: list[InstanceRecord] = []
        for image in images:
            if image.status == "ERROR" or image.image_sha256 is None:
                continue
            occurrences: Counter[str] = Counter()
            for detection in image.detections:
                bbox_key = quantized_bbox_key(detection.bbox)
                occurrence = occurrences[bbox_key]
                occurrences[bbox_key] += 1
                instance_id, instance_key = build_instance_identity(
                    image_path=image.image_path,
                    image_sha256=image.image_sha256,
                    bbox=detection.bbox,
                    detector_signature=detector_signature,
                    occurrence_index=occurrence,
                )
                instances.append(
                    InstanceRecord(
                        instance_id=instance_id,
                        instance_key=instance_key,
                        image_path=image.image_path,
                        image_sha256=image.image_sha256,
                        bbox=detection.bbox,
                        bbox_xywh=detection.bbox.to_xywh_int(),
                        detector_score=detection.score,
                        raw_class_id=detection.raw_class_id,
                    )
                )
        return instances

    def _process_embedding_batch(
        self,
        *,
        run_id: str,
        embedder: Any,
        work: Sequence[_EmbeddingWork],
        instances: list[InstanceRecord],
        embedding_signature: str,
        positional_basis: np.ndarray | None,
        failures: list[FailureRecord],
        attempted_batch_sizes: list[int],
        depth: int = 0,
    ) -> None:
        if not work:
            return
        attempted_batch_sizes.append(len(work))
        try:
            feature_maps = embedder.extract_patch_maps([item.crop.image for item in work])
            if len(feature_maps) != len(work):
                raise RuntimeError("DINOv3 feature-map count does not match crop batch size")
        except (DependencyError, ModelLoadError, ConfigurationError, ArtifactError):
            raise
        except Exception as exc:
            if _is_cuda_oom(exc):
                _clear_cuda_cache()
            if len(work) > 1 and depth < int(self.cfg.embedding.oom_retries):
                midpoint = len(work) // 2
                self._process_embedding_batch(
                    run_id=run_id,
                    embedder=embedder,
                    work=work[:midpoint],
                    instances=instances,
                    embedding_signature=embedding_signature,
                    positional_basis=positional_basis,
                    failures=failures,
                    attempted_batch_sizes=attempted_batch_sizes,
                    depth=depth + 1,
                )
                self._process_embedding_batch(
                    run_id=run_id,
                    embedder=embedder,
                    work=work[midpoint:],
                    instances=instances,
                    embedding_signature=embedding_signature,
                    positional_basis=positional_basis,
                    failures=failures,
                    attempted_batch_sizes=attempted_batch_sizes,
                    depth=depth + 1,
                )
                return
            for item in work:
                instance = instances[item.instance_index]
                code = "DINOV3_CUDA_OOM" if _is_cuda_oom(exc) else "DINOV3_INFERENCE_FAILED"
                instance.embedding_status = "ERROR"
                failures.append(
                    self._new_failure(
                        run_id=run_id,
                        image_path=instance.image_path,
                        image_sha256=instance.image_sha256,
                        instance_id=instance.instance_id,
                        stage="embedding",
                        code=code,
                        exc=exc,
                        retry_count=depth,
                    )
                )
            return

        try:
            import torch
        except ImportError as exc:
            raise DependencyError("PyTorch is required for DINOv3 pooling") from exc
        basis_tensor = None if positional_basis is None else torch.from_numpy(positional_basis.astype(np.float32, copy=False))
        for item, feature_map in zip(work, feature_maps, strict=True):
            instance = instances[item.instance_index]
            try:
                tensor = feature_map if isinstance(feature_map, torch.Tensor) else torch.as_tensor(feature_map)
                pooled = pool_fractional_box_features(
                    tensor,
                    item.crop.detector_box_in_letterbox,
                    canvas_size=int(self.cfg.embedding.input_size),
                    minimum_effective_patch_weight=float(
                        self.cfg.embedding.pooling.min_effective_patch_weight
                    ),
                    positional_basis=basis_tensor,
                    expansion_bounds=item.crop.transform.content_box,
                )
                vector = pooled.vector.numpy().astype(np.float32, copy=False)
                instance.embedding = self.cache.save_embedding(
                    embedding_signature=embedding_signature,
                    instance_key=instance.instance_key,
                    vector=vector,
                )
                instance.embedding_status = "COMPUTED"
                if pooled.expanded_for_minimum:
                    instance.warning_codes.append("POOLING_BOX_EXPANDED")
            except (ArtifactError, ConfigurationError):
                raise
            except Exception as exc:
                instance.embedding_status = "ERROR"
                failures.append(
                    self._new_failure(
                        run_id=run_id,
                        image_path=instance.image_path,
                        image_sha256=instance.image_sha256,
                        instance_id=instance.instance_id,
                        stage="pooling",
                        code="EMBEDDING_POOLING_FAILED",
                        exc=exc,
                    )
                )

    def _embedding_phase(
        self,
        *,
        run_id: str,
        images: list[ImageRecord],
        instances: list[InstanceRecord],
        artifact_sha256: str,
        failures: list[FailureRecord],
    ) -> tuple[str, str, int, int, list[int]]:
        embedder: Any | None = None
        attempted_batch_sizes: list[int] = []

        def get_embedder() -> Any:
            nonlocal embedder
            if embedder is None:
                embedder = self.embedder_factory(self.cfg)
            return embedder

        positional_basis: np.ndarray | None = None
        if bool(self.cfg.embedding.positional_debias.enabled):
            basis_input_signature = build_basis_input_signature(
                artifact_sha256=artifact_sha256,
                artifact_revision=str(self.cfg.embedding.artifact_revision),
                model_id=str(self.cfg.embedding.model_id),
                timm_version=str(self.cfg.embedding.timm_version),
                loader_implementation_version=DINO_LOADER_IMPLEMENTATION_VERSION,
                input_size=int(self.cfg.embedding.input_size),
                patch_size=int(self.cfg.embedding.patch_size),
                components=int(self.cfg.embedding.positional_debias.components),
                basis_image=str(self.cfg.embedding.positional_debias.basis_image),
                device=str(self.cfg.embedding.device),
                inference_precision=str(self.cfg.embedding.inference_precision),
            )
            cached_basis = None
            if not bool(self.cfg.runtime.force_recompute):
                cached_basis = self.cache.load_basis(
                    basis_input_signature=basis_input_signature,
                    expected_shape=(
                        int(self.cfg.embedding.embedding_dim),
                        int(self.cfg.embedding.positional_debias.components),
                    ),
                )
            if cached_basis is None:
                basis_features = get_embedder().normalized_zero_basis_features()
                basis_tensor = compute_positional_basis(
                    basis_features,
                    components=int(self.cfg.embedding.positional_debias.components),
                )
                cached_basis = self.cache.save_basis(
                    basis_input_signature=basis_input_signature,
                    array=basis_tensor.cpu().numpy().astype(np.float32, copy=False),
                )
            positional_basis = cached_basis.array
            basis_sha256 = cached_basis.sha256
        else:
            basis_input_signature = "disabled"
            basis_sha256 = "disabled"

        embedding_signature = build_embedding_signature(
            artifact_sha256=artifact_sha256,
            artifact_revision=str(self.cfg.embedding.artifact_revision),
            model_id=str(self.cfg.embedding.model_id),
            timm_version=str(self.cfg.embedding.timm_version),
            loader_implementation_version=DINO_LOADER_IMPLEMENTATION_VERSION,
            input_size=int(self.cfg.embedding.input_size),
            patch_size=int(self.cfg.embedding.patch_size),
            padding_fraction=float(self.cfg.embedding.box_padding_fraction),
            letterbox_spec="aspect_preserving_bicubic_imagenet_mean_v1",
            patch_layer=str(self.cfg.embedding.layer),
            token_normalization=str(self.cfg.embedding.patch_normalization),
            basis_sha256=basis_sha256,
            pooling_spec={
                "method": str(self.cfg.embedding.pooling.method),
                "minimum_effective_patch_weight": float(
                    self.cfg.embedding.pooling.min_effective_patch_weight
                ),
                "minimum_expansion_bounds": "letterbox_content",
                "output_normalization": str(self.cfg.embedding.output_normalization),
            },
            device=str(self.cfg.embedding.device),
            inference_precision=str(self.cfg.embedding.inference_precision),
        )
        self.cache.write_signature_spec(
            signature=embedding_signature,
            kind="embedding",
            payload={
                "artifact_sha256": artifact_sha256,
                "artifact_revision": str(self.cfg.embedding.artifact_revision),
                "model_id": str(self.cfg.embedding.model_id),
                "timm_version": str(self.cfg.embedding.timm_version),
                "loader_implementation_version": DINO_LOADER_IMPLEMENTATION_VERSION,
                "basis_input_signature": basis_input_signature,
                "basis_sha256": basis_sha256,
                "configuration": config_as_dict(self.cfg, redact_paths=True)["embedding"],
            },
        )

        cache_hits = 0
        cache_misses = 0
        pending_indices_by_image: dict[str, list[int]] = defaultdict(list)
        for index, instance in enumerate(instances):
            cached = None
            if not bool(self.cfg.runtime.force_recompute):
                cached = self.cache.load_embedding(
                    embedding_signature=embedding_signature,
                    instance_key=instance.instance_key,
                    expected_dimension=int(self.cfg.embedding.embedding_dim),
                )
            if cached is None:
                cache_misses += 1
                pending_indices_by_image[instance.image_path].append(index)
            else:
                cache_hits += 1
                instance.embedding = cached
                instance.embedding_status = "CACHED"

        images_by_path = {image.image_path: image for image in images}
        all_indices_by_image: dict[str, list[int]] = defaultdict(list)
        for index, instance in enumerate(instances):
            all_indices_by_image[instance.image_path].append(index)
        pending_batch: list[_EmbeddingWork] = []

        def flush() -> None:
            nonlocal pending_batch
            if not pending_batch:
                return
            batch = pending_batch
            pending_batch = []
            try:
                self._process_embedding_batch(
                    run_id=run_id,
                    embedder=get_embedder(),
                    work=batch,
                    instances=instances,
                    embedding_signature=embedding_signature,
                    positional_basis=positional_basis,
                    failures=failures,
                    attempted_batch_sizes=attempted_batch_sizes,
                )
            finally:
                for item in batch:
                    try:
                        item.crop.image.close()
                    except Exception:
                        pass

        try:
            for image_path in [image.image_path for image in images]:
                all_indices = all_indices_by_image.get(image_path)
                if not all_indices:
                    continue
                indices = pending_indices_by_image.get(image_path, [])
                image_record = images_by_path[image_path]
                decoded = None
                try:
                    raw = self.image_source.read_bytes(image_path)
                    observed_sha = sha256_bytes(raw)
                    if observed_sha != image_record.image_sha256:
                        raise RowProcessingError(
                            "IMAGE_CHANGED_DURING_RUN",
                            "Image bytes changed between detection and embedding phases",
                            stage="embedding_read",
                        )
                    # A cache-only recluster still revalidates source bytes, but
                    # it need not decode pixels or construct a DINOv3 model.
                    if indices:
                        decoded = decode_image_bytes(raw)
                except RowProcessingError as exc:
                    # Invalidate cached and pending embeddings alike. The review
                    # path now points at bytes that no longer match the detector
                    # snapshot, so retaining any instance would be unsafe.
                    for index in all_indices_by_image[image_path]:
                        instance = instances[index]
                        instance.embedding = None
                        instance.embedding_status = "ERROR"
                        failures.append(
                            self._new_failure(
                                run_id=run_id,
                                image_path=image_path,
                                image_sha256=image_record.image_sha256,
                                instance_id=instance.instance_id,
                                stage=exc.stage,
                                code=exc.code,
                                exc=exc.cause or exc,
                                message=str(exc),
                            )
                        )
                    self._mark_image_error(image=image_record, code=exc.code, message=str(exc))
                    continue
                except Exception as exc:
                    for index in all_indices_by_image[image_path]:
                        instance = instances[index]
                        instance.embedding = None
                        instance.embedding_status = "ERROR"
                        failures.append(
                            self._new_failure(
                                run_id=run_id,
                                image_path=image_path,
                                image_sha256=image_record.image_sha256,
                                instance_id=instance.instance_id,
                                stage="embedding_read",
                                code="IMAGE_REREAD_FAILED",
                                exc=exc,
                            )
                        )
                    self._mark_image_error(
                        image=image_record,
                        code="IMAGE_REREAD_FAILED",
                        message="Image could not be read for DINOv3 embedding",
                    )
                    continue

                if not indices:
                    continue
                if decoded is None:  # Defensive invariant for static type checkers and adapters.
                    raise FatalPipelineError(f"Image decode invariant failed for {image_path}")
                try:
                    for index in indices:
                        instance = instances[index]
                        try:
                            crop = make_letterboxed_crop(
                                decoded.image,
                                instance.bbox,
                                padding_fraction=float(self.cfg.embedding.box_padding_fraction),
                                output_size=int(self.cfg.embedding.input_size),
                                fill=str(self.cfg.embedding.letterbox_fill),
                            )
                            pending_batch.append(_EmbeddingWork(instance_index=index, crop=crop))
                            if len(pending_batch) >= int(self.cfg.embedding.batch_size):
                                flush()
                        except Exception as exc:
                            instance.embedding_status = "ERROR"
                            failures.append(
                                self._new_failure(
                                    run_id=run_id,
                                    image_path=image_path,
                                    image_sha256=image_record.image_sha256,
                                    instance_id=instance.instance_id,
                                    stage="crop",
                                    code="CROP_TRANSFORM_FAILED",
                                    exc=exc,
                                )
                            )
                finally:
                    decoded.image.close()
            flush()
        finally:
            for item in pending_batch:
                try:
                    item.crop.image.close()
                except Exception:
                    pass
            if embedder is not None:
                try:
                    embedder.close()
                except Exception:
                    pass

        instances_by_image: dict[str, list[InstanceRecord]] = defaultdict(list)
        for instance in instances:
            instances_by_image[instance.image_path].append(instance)
        for image in images:
            members = instances_by_image.get(image.image_path, [])
            if not members or (image.status == "ERROR" and image.error_code not in {
                "ALL_INSTANCE_EMBEDDINGS_FAILED",
                "IMAGE_CHANGED_DURING_RUN",
                "IMAGE_REREAD_FAILED",
            }):
                continue
            successes = sum(member.embedding is not None for member in members)
            failures_count = len(members) - successes
            if successes == 0:
                self._mark_image_error(
                    image=image,
                    code=image.error_code or "ALL_INSTANCE_EMBEDDINGS_FAILED",
                    message=image.error_message or "All detected instances failed during DINOv3 embedding",
                )
            elif failures_count > 0:
                image.status = "OK"
                image.error_code = None
                image.error_message = None
                if "PARTIAL_INSTANCE_EMBEDDING_FAILURES" not in image.warning_codes:
                    image.warning_codes.append("PARTIAL_INSTANCE_EMBEDDING_FAILURES")

        self._check_failure_fraction(images, stage="embedding")
        return embedding_signature, basis_sha256, cache_hits, cache_misses, attempted_batch_sizes

    def run(self) -> PipelineRunResult:
        run_id = str(uuid.uuid4())
        started_at = _utc_now()
        failures: list[FailureRecord] = []

        checkpoint_path = Path(str(self.cfg.detector.checkpoint_path))
        artifact_path = DinoV3Adapter.resolve_artifact(str(self.cfg.embedding.artifact_path))
        if not checkpoint_path.is_file():
            raise ModelLoadError(f"RF-DETR checkpoint does not exist: {checkpoint_path}")

        checkpoint_sha256 = sha256_file(checkpoint_path)
        artifact_sha256 = sha256_file(artifact_path)
        validate_expected_sha256(
            checkpoint_sha256,
            self.cfg.detector.expected_checkpoint_sha256,
            label="RF-DETR checkpoint",
        )
        validate_expected_sha256(
            artifact_sha256,
            self.cfg.embedding.expected_artifact_sha256,
            label="DINOv3 artifact",
        )
        rfdetr_package_version = self._resolve_rfdetr_package_version()
        detector_signature = build_detector_signature(
            checkpoint_sha256=checkpoint_sha256,
            rfdetr_package_version=rfdetr_package_version,
            model_class=str(self.cfg.detector.model_class),
            preprocessing_spec=str(self.cfg.detector.preprocessing_spec),
            detection_threshold=float(self.cfg.detector.threshold),
            max_detections_per_image=int(self.cfg.detector.max_detections_per_image),
            device=str(self.cfg.detector.device),
            clip_to_image=bool(self.cfg.detector.clip_to_image),
            drop_degenerate=bool(self.cfg.detector.drop_degenerate),
            loader_implementation_version=RFDETRAdapter.LOADER_IMPLEMENTATION_VERSION,
        )
        self.cache.write_signature_spec(
            signature=detector_signature,
            kind="detector",
            payload={
                "checkpoint_sha256": checkpoint_sha256,
                "rfdetr_package_version": rfdetr_package_version,
                "model_class": str(self.cfg.detector.model_class),
                "loader_implementation_version": RFDETRAdapter.LOADER_IMPLEMENTATION_VERSION,
                "configuration": config_as_dict(self.cfg, redact_paths=True)["detector"],
            },
        )

        paths = self._eligible_image_paths()
        images, detection_hits, detection_misses, detector_batches = self._detection_phase(
            run_id=run_id,
            paths=paths,
            detector_signature=detector_signature,
            failures=failures,
        )
        observed_class_ids = sorted(
            {
                int(detection.raw_class_id)
                for image in images
                for detection in image.detections
                if detection.raw_class_id is not None
            }
        )
        if len(observed_class_ids) > 1:
            raise FatalPipelineError(
                "The provisioned RF-DETR checkpoint emitted multiple class IDs "
                f"{observed_class_ids}; the MVP requires a class-agnostic detector"
            )
        instances = self._build_instances(images=images, detector_signature=detector_signature)
        if not instances:
            raise FatalPipelineError("RF-DETR produced no valid defect instances for clustering")

        embedding_signature, basis_sha256, embedding_hits, embedding_misses, embedding_batches = (
            self._embedding_phase(
                run_id=run_id,
                images=images,
                instances=instances,
                artifact_sha256=artifact_sha256,
                failures=failures,
            )
        )

        valid_instances = [instance for instance in instances if instance.embedding is not None]
        if len(valid_instances) < int(self.cfg.runtime.min_successful_instances):
            raise FatalPipelineError(
                f"Only {len(valid_instances)} valid embeddings remain; at least "
                f"{int(self.cfg.runtime.min_successful_instances)} are required"
            )
        if int(self.cfg.clustering.k) > len(valid_instances):
            raise FatalPipelineError(
                f"cluster_count={int(self.cfg.clustering.k)} exceeds {len(valid_instances)} valid embedded instances"
            )

        with tempfile.TemporaryDirectory(
            prefix="defect-curation-work-",
            dir=(None if self.cfg.runtime.temporary_root is None else str(self.cfg.runtime.temporary_root)),
        ) as work_dir:
            matrix_path = Path(work_dir) / "embeddings.f16.npy"
            matrix_writer = np.lib.format.open_memmap(
                matrix_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(valid_instances), int(self.cfg.embedding.embedding_dim)),
            )
            for row_index, instance in enumerate(valid_instances):
                vector = np.asarray(instance.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or norm <= 1e-12:
                    raise FatalPipelineError(f"Invalid embedding for instance {instance.instance_id}")
                matrix_writer[row_index] = vector.astype(np.float16, copy=False)
                instance.embedding_row = row_index
                instance.embedding = None
            matrix_writer.flush()
            del matrix_writer
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)

            cluster_result = fit_spherical_kmeans(
                matrix,
                instance_ids=[instance.instance_id for instance in valid_instances],
                k=int(self.cfg.clustering.k),
                niter=int(self.cfg.clustering.niter),
                nredo=int(self.cfg.clustering.nredo),
                seed=int(self.cfg.clustering.seed),
                backend=str(self.cfg.clustering.backend),
                max_points_per_centroid=int(self.cfg.clustering.max_points_per_centroid),
                assignment_chunk_size=int(self.cfg.clustering.assignment_chunk_size),
                max_empty_cluster_retries=int(self.cfg.clustering.max_empty_cluster_retries),
                canonicalize_ids=bool(self.cfg.clustering.canonicalize_ids),
            )
            for row_index, instance in enumerate(valid_instances):
                instance.cluster_id = int(cluster_result.assignments[row_index])
                instance.cosine_to_centroid = float(cluster_result.similarities[row_index])
            assign_cluster_ranks(valid_instances)
            review_order = build_review_order(valid_instances)
            review_rows = build_review_rows(
                run_id=run_id,
                images=images,
                instances=instances,
                review_order=review_order,
            )

            input_manifest_hash = sha256_json(
                [
                    {
                        "image_path": image.image_path,
                        "image_sha256": image.image_sha256,
                        "byte_size": image.byte_size,
                    }
                    for image in images
                ]
            )
            counts = {
                "successful_images": sum(image.status == "OK" for image in images),
                "no_detection_images": sum(image.status == "NO_DETECTION" for image in images),
                "error_images": sum(image.status == "ERROR" for image in images),
                "valid_detector_instances": len(instances),
                "valid_embedded_instances": len(valid_instances),
                "failed_instance_embeddings": sum(
                    instance.embedding_status == "ERROR" for instance in instances
                ),
            }
            provenance = collect_provenance(
                plugin_version=self.plugin_version,
                plugin_source_commit=self.plugin_source_commit,
                dss_version=self.dss_version,
                dinov3_model_id=str(self.cfg.embedding.model_id),
                dinov3_timm_version=str(self.cfg.embedding.timm_version),
                dinov3_artifact_revision=str(self.cfg.embedding.artifact_revision),
                dinov3_artifact_sha256=artifact_sha256,
                dinov3_loader_version=DINO_LOADER_IMPLEMENTATION_VERSION,
            )
            manifest_fields: dict[str, Any] = {
                "pipeline": str(self.cfg.pipeline_name),
                "job": {
                    "started_at_utc": started_at,
                    "completed_at_utc": _utc_now(),
                },
                "input": {
                    "managed_folder_id": self.image_source.identifier,
                    "image_count": len(images),
                    "input_manifest_sha256": input_manifest_hash,
                },
                "counts": counts,
                "signatures": {
                    "detector": detector_signature,
                    "embedding": embedding_signature,
                    "positional_basis": basis_sha256,
                    "config": config_hash(self.cfg),
                },
                "models": {
                    "rfdetr": {
                        "model_class": str(self.cfg.detector.model_class),
                        "package_version": rfdetr_package_version,
                        "checkpoint_sha256": checkpoint_sha256,
                        "loader_implementation_version": RFDETRAdapter.LOADER_IMPLEMENTATION_VERSION,
                        "observed_foreground_class_ids": observed_class_ids,
                    },
                    "dinov3": {
                        "model_id": str(self.cfg.embedding.model_id),
                        "timm_version": str(self.cfg.embedding.timm_version),
                        "artifact_revision": str(self.cfg.embedding.artifact_revision),
                        "artifact_sha256": artifact_sha256,
                        "loader_implementation_version": DINO_LOADER_IMPLEMENTATION_VERSION,
                    },
                },
                "clustering": {
                    "algorithm": "spherical_kmeans",
                    "backend": str(self.cfg.clustering.backend),
                    "k": int(self.cfg.clustering.k),
                    "niter": int(self.cfg.clustering.niter),
                    "nredo": int(self.cfg.clustering.nredo),
                    "configured_seed": int(self.cfg.clustering.seed),
                    "seed_used": int(cluster_result.seed_used),
                },
                "cache": {
                    "detection_hits": detection_hits,
                    "detection_misses": detection_misses,
                    "embedding_hits": embedding_hits,
                    "embedding_misses": embedding_misses,
                },
                "effective_batch_sizes": {
                    "detector_attempts": detector_batches,
                    "dinov3_attempts": embedding_batches,
                },
            }

            publisher = RunBundlePublisher(
                self.artifact_store,
                temporary_root=(
                    None if self.cfg.runtime.temporary_root is None else str(self.cfg.runtime.temporary_root)
                ),
            )
            bundle = publisher.publish(
                run_id=run_id,
                resolved_config_yaml=resolved_yaml(self.cfg, redact_paths=True),
                provenance=provenance,
                manifest_fields=manifest_fields,
                images=images,
                instances=instances,
                embeddings=matrix,
                centroids=np.asarray(cluster_result.centroids, dtype=np.float32),
                failures=failures,
                write_checksums=bool(self.cfg.runtime.write_checksums),
                before_latest=lambda: self.review_sink.write_rows(review_rows),
            )

        return PipelineRunResult(
            run_id=run_id,
            review_row_count=len(review_rows),
            image_count=len(images),
            instance_count=len(instances),
            embedded_instance_count=len(valid_instances),
            detector_signature=detector_signature,
            embedding_signature=embedding_signature,
            bundle=bundle,
        )
