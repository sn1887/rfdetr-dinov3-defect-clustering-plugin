from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def patch_parquet_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise bundle publication without requiring pyarrow in lightweight CI."""

    import defect_curation_core.artifacts.bundle as bundle

    def write(path: Path, rows: list[dict[str, Any]], schema: Any) -> None:
        del schema

        def default(value: Any):
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        path.write_text(json.dumps(rows, default=default, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(bundle, "_write_parquet", write)
    monkeypatch.setattr(bundle, "_input_schema", lambda: None)
    monkeypatch.setattr(bundle, "_instance_schema", lambda: None)
    monkeypatch.setattr(bundle, "_failure_schema", lambda: None)
    monkeypatch.setattr(bundle, "_cluster_schema", lambda: None)


@pytest.fixture
def solid_image_factory():
    def create(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (96, 64)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=color).save(path, format="PNG")
        return path

    return create
