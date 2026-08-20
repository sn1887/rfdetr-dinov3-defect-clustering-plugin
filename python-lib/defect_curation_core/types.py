"""Typed records shared by detection, embedding, clustering, and outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ImageStatus = Literal["OK", "NO_DETECTION", "ERROR"]
EmbeddingGranularity = Literal["object", "image"]


@dataclass(frozen=True, slots=True)
class BBoxXYXY:
    """Floating point bounding box in half-open xyxy image coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        values = (self.xmin, self.ymin, self.xmax, self.ymax)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite bounding box: {values}")

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def is_degenerate(self) -> bool:
        return self.width <= 0.0 or self.height <= 0.0

    def clip(self, image_width: int, image_height: int) -> BBoxXYXY:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        return BBoxXYXY(
            xmin=min(max(self.xmin, 0.0), float(image_width)),
            ymin=min(max(self.ymin, 0.0), float(image_height)),
            xmax=min(max(self.xmax, 0.0), float(image_width)),
            ymax=min(max(self.ymax, 0.0), float(image_height)),
        )

    def expand(self, fraction: float, image_width: int, image_height: int) -> BBoxXYXY:
        if fraction < 0.0:
            raise ValueError("Expansion fraction must be non-negative")
        pad_x = self.width * fraction
        pad_y = self.height * fraction
        return BBoxXYXY(
            self.xmin - pad_x,
            self.ymin - pad_y,
            self.xmax + pad_x,
            self.ymax + pad_y,
        ).clip(image_width=image_width, image_height=image_height)

    def to_xywh_int(self) -> tuple[int, int, int, int]:
        """Convert to a conservative integer Dataiku bbox using floor/ceil."""

        xmin_i = math.floor(self.xmin)
        ymin_i = math.floor(self.ymin)
        xmax_i = math.ceil(self.xmax)
        ymax_i = math.ceil(self.ymax)
        return xmin_i, ymin_i, max(0, xmax_i - xmin_i), max(0, ymax_i - ymin_i)

    def as_list(self) -> list[float]:
        return [float(self.xmin), float(self.ymin), float(self.xmax), float(self.ymax)]


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBoxXYXY
    score: float
    raw_class_id: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("Detection score must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox.as_list(),
            "score": float(self.score),
            "raw_class_id": self.raw_class_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Detection:
        coords = value["bbox_xyxy"]
        return cls(
            bbox=BBoxXYXY(*map(float, coords)),
            score=float(value["score"]),
            raw_class_id=(None if value.get("raw_class_id") is None else int(value["raw_class_id"])),
        )


@dataclass(slots=True)
class ImageRecord:
    image_path: str
    image_sha256: str | None = None
    byte_size: int | None = None
    source_mode: str | None = None
    width: int | None = None
    height: int | None = None
    status: ImageStatus = "OK"
    detections: list[Detection] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    warning_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InstanceRecord:
    instance_id: str
    instance_key: str
    image_path: str
    image_sha256: str
    bbox: BBoxXYXY
    bbox_xywh: tuple[int, int, int, int]
    detector_score: float
    raw_class_id: int | None = None
    embedding_granularity: EmbeddingGranularity = "object"
    embedding: Any | None = None
    embedding_row: int | None = None
    embedding_status: str = "PENDING"
    cluster_id: int | None = None
    cosine_to_centroid: float | None = None
    cluster_rank: int | None = None
    cluster_size: int | None = None
    warning_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FailureRecord:
    run_id: str
    image_path: str
    stage: str
    error_code: str
    exception_type: str
    sanitized_message: str
    retry_count: int = 0
    image_sha256: str | None = None
    instance_id: str | None = None
    occurred_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "instance_id": self.instance_id,
            "stage": self.stage,
            "error_code": self.error_code,
            "exception_type": self.exception_type,
            "sanitized_message": self.sanitized_message,
            "retry_count": self.retry_count,
            "occurred_at_utc": self.occurred_at_utc,
        }


@dataclass(frozen=True, slots=True)
class ClusterResult:
    assignments: Any
    similarities: Any
    centroids: Any
    seed_used: int


@dataclass(frozen=True, slots=True)
class CacheStats:
    detection_hits: int = 0
    detection_misses: int = 0
    embedding_hits: int = 0
    embedding_misses: int = 0
