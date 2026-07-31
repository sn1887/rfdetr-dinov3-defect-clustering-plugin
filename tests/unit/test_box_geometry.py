from __future__ import annotations

import math

from defect_curation_core.embeddings.crop_geometry import make_letterboxed_crop
from defect_curation_core.types import BBoxXYXY
from PIL import Image


def test_bbox_clipping_and_integer_conversion() -> None:
    box = BBoxXYXY(-2.2, 4.1, 10.2, 20.01).clip(10, 18)
    assert box == BBoxXYXY(0.0, 4.1, 10.0, 18.0)
    assert box.to_xywh_int() == (0, 4, 10, 14)


def test_letterbox_preserves_aspect_and_maps_detector_box() -> None:
    image = Image.new("RGB", (200, 100), color=(10, 20, 30))
    detector = BBoxXYXY(50.0, 20.0, 150.0, 60.0)
    sample = make_letterboxed_crop(
        image,
        detector,
        padding_fraction=0.15,
        output_size=512,
    )
    try:
        assert sample.image.size == (512, 512)
        transform = sample.transform
        source_ratio = transform.crop_box.width / transform.crop_box.height
        resized_ratio = transform.resized_width / transform.resized_height
        assert math.isclose(source_ratio, resized_ratio, rel_tol=0.01)
        mapped = sample.detector_box_in_letterbox
        assert 0 <= mapped.xmin < mapped.xmax <= 512
        assert 0 <= mapped.ymin < mapped.ymax <= 512
        assert mapped.width > mapped.height
    finally:
        sample.image.close()


def test_crop_near_boundary_is_valid() -> None:
    image = Image.new("RGB", (20, 20), color=(255, 255, 255))
    sample = make_letterboxed_crop(
        image,
        BBoxXYXY(-10.0, -4.0, 4.2, 3.2),
        padding_fraction=0.5,
        output_size=512,
    )
    try:
        assert sample.transform.crop_box.xmin == 0.0
        assert sample.transform.crop_box.ymin == 0.0
        assert sample.detector_box_in_letterbox.area > 0
    finally:
        sample.image.close()
