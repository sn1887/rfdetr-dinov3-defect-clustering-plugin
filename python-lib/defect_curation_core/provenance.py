"""Best-effort, deterministic runtime provenance collection."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections.abc import Iterable
from pathlib import Path


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_git_head(repo_path: str | Path) -> str | None:
    """Read a checkout's HEAD without invoking a shell or Git executable."""

    git_dir = Path(repo_path) / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            git_dir = (Path(repo_path) / text.split(":", 1)[1].strip()).resolve()
    if not git_dir.is_dir():
        return None
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head if len(head) == 40 else None
    ref_name = head.split(":", 1)[1].strip()
    ref_file = git_dir / ref_name
    if ref_file.is_file():
        return ref_file.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            commit, name = line.split(" ", 1)
            if name.strip() == ref_name:
                return commit.strip()
    return None


def collect_provenance(
    *,
    plugin_version: str,
    plugin_source_commit: str | None,
    dss_version: str | None,
    dinov3_model_id: str,
    dinov3_timm_version: str,
    dinov3_artifact_revision: str,
    dinov3_artifact_sha256: str,
    dinov3_loader_version: str,
    direct_packages: Iterable[str] = (
        "torch",
        "torchvision",
        "timm",
        "safetensors",
        "rfdetr",
        "numpy",
        "pandas",
        "pyarrow",
        "hydra-core",
        "omegaconf",
        "faiss-cpu",
        "Pillow",
        "packaging",
    ),
) -> dict[str, object]:
    packages = {name: version for name in direct_packages if (version := _version(name)) is not None}
    runtime: dict[str, object] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }

    try:
        import torch

        cuda: dict[str, object] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            cuda["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
        runtime["cuda"] = cuda
    except Exception as exc:
        runtime["cuda"] = {"available": False, "diagnostic": type(exc).__name__}

    return {
        "plugin_version": plugin_version,
        "plugin_source_commit": plugin_source_commit,
        "dss_version": dss_version,
        "runtime": runtime,
        "dinov3": {
            "model_id": dinov3_model_id,
            "timm_version": dinov3_timm_version,
            "artifact_revision": dinov3_artifact_revision,
            "artifact_sha256": dinov3_artifact_sha256,
            "loader_implementation_version": dinov3_loader_version,
        },
    }
