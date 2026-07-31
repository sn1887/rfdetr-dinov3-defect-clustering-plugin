"""Aspect-preserving crop and letterbox geometry for defect instances."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from defect_curation_core.types import BBoxXYXY

IMAGENET_MEAN_RGB_8BIT = (124, 116, 104)


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    crop_box: BBoxXYXY
    output_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale_x: float
    scale_y: float

    @property
    def content_box(self) -> BBoxXYXY:
        """Bounds of real resized pixels, excluding synthetic letterbox fill."""

        return BBoxXYXY(
            float(self.pad_left),
            float(self.pad_top),
            float(self.pad_left + self.resized_width),
            float(self.pad_top + self.resized_height),
        )

    def map_box(self, source_box: BBoxXYXY) -> BBoxXYXY:
        mapped = BBoxXYXY(
            xmin=(source_box.xmin - self.crop_box.xmin) * self.scale_x + self.pad_left,
            ymin=(source_box.ymin - self.crop_box.ymin) * self.scale_y + self.pad_top,
            xmax=(source_box.xmax - self.crop_box.xmin) * self.scale_x + self.pad_left,
            ymax=(source_box.ymax - self.crop_box.ymin) * self.scale_y + self.pad_top,
        )
        return mapped.clip(self.output_size, self.output_size)


@dataclass(frozen=True, slots=True)
class CropSample:
    image: Image.Image
    detector_box_in_letterbox: BBoxXYXY
    transform: LetterboxTransform


def _integer_crop_box(expanded: BBoxXYXY, image_width: int, image_height: int) -> BBoxXYXY:
    xmin = max(0, min(image_width - 1, math.floor(expanded.xmin)))
    ymin = max(0, min(image_height - 1, math.floor(expanded.ymin)))
    xmax = max(xmin + 1, min(image_width, math.ceil(expanded.xmax)))
    ymax = max(ymin + 1, min(image_height, math.ceil(expanded.ymax)))
    return BBoxXYXY(float(xmin), float(ymin), float(xmax), float(ymax))


def make_letterboxed_crop(
    image: Image.Image,
    detector_box: BBoxXYXY,
    *,
    padding_fraction: float,
    output_size: int,
    fill: str = "imagenet_mean",
) -> CropSample:
    if image.mode != "RGB":
        raise ValueError("DINOv3 crop input must already be RGB")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    width, height = image.size
    clipped = detector_box.clip(width, height)
    if clipped.width < 1.0 or clipped.height < 1.0:
        raise ValueError("Detector box is degenerate after clipping")

    expanded = clipped.expand(padding_fraction, width, height)
    crop_box = _integer_crop_box(expanded, width, height)
    crop_width = int(crop_box.width)
    crop_height = int(crop_box.height)
    cropped = image.crop(tuple(map(int, crop_box.as_list())))
    try:
        nominal_scale = min(output_size / crop_width, output_size / crop_height)
        resized_width = max(1, min(output_size, round(crop_width * nominal_scale)))
        resized_height = max(1, min(output_size, round(crop_height * nominal_scale)))
        resized = cropped.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)
        try:
            if fill == "imagenet_mean":
                fill_value = IMAGENET_MEAN_RGB_8BIT
            else:
                raise ValueError(f"Unsupported letterbox fill: {fill}")
            canvas = Image.new("RGB", (output_size, output_size), color=fill_value)
            pad_left = (output_size - resized_width) // 2
            pad_top = (output_size - resized_height) // 2
            canvas.paste(resized, (pad_left, pad_top))
        finally:
            resized.close()
    finally:
        cropped.close()

    transform = LetterboxTransform(
        crop_box=crop_box,
        output_size=output_size,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        scale_x=resized_width / crop_width,
        scale_y=resized_height / crop_height,
    )
    return CropSample(
        image=canvas,
        detector_box_in_letterbox=transform.map_box(clipped),
        transform=transform,
    )
