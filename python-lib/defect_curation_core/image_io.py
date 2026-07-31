"""Deterministic image decoding used by both pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from defect_curation_core.errors import RowProcessingError


@dataclass(frozen=True, slots=True)
class DecodedImage:
    image: Image.Image
    source_mode: str
    width: int
    height: int


def decode_image_bytes(data: bytes) -> DecodedImage:
    """Decode bytes, apply EXIF orientation, fully load, and convert to RGB."""

    if not data:
        raise RowProcessingError(
            "IMAGE_DECODE_FAILED",
            "Image file is empty",
            stage="decode",
        )
    try:
        with Image.open(BytesIO(data)) as opened:
            source_mode = opened.mode
            oriented = ImageOps.exif_transpose(opened)
            oriented.load()
            rgb = oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RowProcessingError(
            "IMAGE_DECODE_FAILED",
            "Image could not be decoded as a supported format",
            stage="decode",
            cause=exc,
        ) from exc

    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise RowProcessingError(
            "INVALID_IMAGE_DIMENSIONS",
            "Decoded image has non-positive dimensions",
            stage="decode",
        )
    return DecodedImage(image=rgb, source_mode=source_mode, width=width, height=height)
