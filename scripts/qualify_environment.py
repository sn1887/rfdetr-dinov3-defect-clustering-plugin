#!/usr/bin/env python3
"""Preflight and optional real-forward qualification for provisioned models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python-lib"))

from defect_curation_core.detection.rfdetr_adapter import RFDETRAdapter  # noqa: E402
from defect_curation_core.embeddings.dinov3_adapter import (  # noqa: E402
    DINO_MODEL_ID,
    DINO_TIMM_VERSION,
    DinoV3Adapter,
)
from defect_curation_core.hashing import sha256_file  # noqa: E402


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--rfdetr-model-class", default="RFDETRSmall")
    parser.add_argument("--rfdetr-compatibility", default="==1.5.2")
    parser.add_argument("--dinov3-model-id", default=DINO_MODEL_ID)
    parser.add_argument("--dinov3-timm-version", default=DINO_TIMM_VERSION)
    parser.add_argument("--dinov3-artifact", type=Path, required=True)
    parser.add_argument("--dinov3-artifact-revision", required=True)
    parser.add_argument("--dinov3-artifact-sha256")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-forward", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"RF-DETR checkpoint not found: {args.checkpoint}")
    artifact = DinoV3Adapter.resolve_artifact(args.dinov3_artifact)

    checkpoint_sha = sha256_file(args.checkpoint)
    artifact_sha = sha256_file(artifact)
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256.lower():
        parser.error("RF-DETR checkpoint digest mismatch")
    if args.dinov3_artifact_sha256 and artifact_sha != args.dinov3_artifact_sha256.lower():
        parser.error("DINOv3 artifact digest mismatch")

    try:
        import torch
    except ImportError:
        parser.error("PyTorch is not installed")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if version("rfdetr") != "1.5.2":
        parser.error("qualification requires exactly rfdetr==1.5.2")
    if version("timm") != DINO_TIMM_VERSION:
        parser.error(f"qualification requires exactly timm=={DINO_TIMM_VERSION}")
    try:
        safe_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        RFDETRAdapter._extract_state_dict(safe_checkpoint, torch=torch)
    except Exception as exc:
        parser.error(f"RF-DETR checkpoint is not a safe weights-only state dictionary: {exc}")

    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "rfdetr",
                "timm",
                "safetensors",
                "faiss-cpu",
                "numpy",
                "Pillow",
                "hydra-core",
                "omegaconf",
                "pyarrow",
            )
        },
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "checkpoint_sha256": checkpoint_sha,
        "dinov3_model_id": args.dinov3_model_id,
        "dinov3_timm_version": args.dinov3_timm_version,
        "dinov3_artifact_sha256": artifact_sha,
        "dinov3_artifact_revision": args.dinov3_artifact_revision,
    }

    if args.run_forward:
        detector = RFDETRAdapter(
            checkpoint_path=str(args.checkpoint),
            model_class=args.rfdetr_model_class,
            package_compatibility=args.rfdetr_compatibility,
            device=args.device,
            safe_checkpoint_loading=True,
        )
        embedder = DinoV3Adapter(
            artifact_path=str(artifact),
            model_id=args.dinov3_model_id,
            timm_version=args.dinov3_timm_version,
            device=args.device,
            input_size=512,
            patch_size=16,
            embedding_dim=768,
            expected_prefix_tokens=5,
            inference_precision="bf16" if args.device == "cuda" else "float32",
        )
        try:
            from PIL import Image

            test_image = Image.new("RGB", (512, 512), color=(127, 127, 127))
            try:
                detections = detector.predict([test_image], threshold=0.99)
                features = embedder.extract_patch_maps([test_image])
                assert tuple(features.shape) == (1, 768, 32, 32)
                assert bool(torch.isfinite(features).all())
                report["forward_test"] = {
                    "rfdetr_prediction_sets": len(detections),
                    "rfdetr_detections": len(detections[0]),
                    "dinov3_feature_shape": list(features.shape),
                    "dinov3_finite": True,
                }
            finally:
                test_image.close()
        finally:
            detector.close()
            embedder.close()

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
