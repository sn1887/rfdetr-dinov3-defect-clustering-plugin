"""Version-isolated adapter for the provisioned RF-DETR checkpoint."""

from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from PIL import Image

from defect_curation_core.errors import DependencyError, ModelLoadError
from defect_curation_core.types import BBoxXYXY, Detection


class RFDETRAdapter:
    """Load and run RF-DETR without exposing package-specific objects downstream."""

    LOADER_IMPLEMENTATION_VERSION = "rfdetr_1_5_2_weights_only_state_dict_v1"
    SUPPORTED_MODEL_CLASSES = frozenset(
        {"RFDETRBase", "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"}
    )

    def __init__(
        self,
        *,
        checkpoint_path: str,
        model_class: str,
        package_compatibility: str,
        device: str,
        safe_checkpoint_loading: bool = True,
    ) -> None:
        if not safe_checkpoint_loading:
            raise ModelLoadError("Unsafe RF-DETR checkpoint loading is prohibited")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise ModelLoadError(f"RF-DETR checkpoint does not exist: {checkpoint}")

        try:
            installed_version = importlib.metadata.version("rfdetr")
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyError("The rfdetr package is not installed in the recipe code environment") from exc
        try:
            specifier = SpecifierSet(package_compatibility)
        except InvalidSpecifier as exc:
            raise ModelLoadError(f"Invalid RF-DETR package compatibility specifier: {package_compatibility}") from exc
        if package_compatibility != "==1.5.2":
            raise ModelLoadError("RF-DETR runtime compatibility must be exactly ==1.5.2")
        if installed_version != "1.5.2" or installed_version not in specifier:
            raise ModelLoadError(
                f"Installed rfdetr {installed_version} does not satisfy checkpoint requirement {package_compatibility}"
            )

        try:
            import rfdetr as module
            import torch

            self._model = self._load_model(
                module=module,
                torch=torch,
                checkpoint_path=str(checkpoint),
                model_class=model_class,
                device=device,
            )
        except (DependencyError, ModelLoadError):
            raise
        except Exception as exc:
            raise ModelLoadError(f"Could not load the RF-DETR checkpoint safely: {exc}") from exc

        self.package_version = installed_version
        self.device = device

    @staticmethod
    def _resolve_model_class(module: Any, model_class: str) -> type[Any]:
        if model_class == "auto":
            raise ModelLoadError(
                "RF-DETR model_class='auto' is unsupported for 1.5.2; configure a concrete qualified variant"
            )
        if model_class not in RFDETRAdapter.SUPPORTED_MODEL_CLASSES:
            raise ModelLoadError(
                f"Unsupported RF-DETR 1.5.2 model class {model_class!r}; expected one of "
                f"{sorted(RFDETRAdapter.SUPPORTED_MODEL_CLASSES)}"
            )
        resolved = getattr(module, model_class, None)
        if not isinstance(resolved, type):
            raise ModelLoadError(f"RF-DETR model_class does not resolve to a class: {model_class}")
        return resolved

    @classmethod
    def _load_model(
        cls,
        *,
        module: Any,
        torch: Any,
        checkpoint_path: str,
        model_class: str,
        device: str,
    ) -> Any:
        target = cls._resolve_model_class(module, model_class)
        original_download = getattr(target, "maybe_download_pretrain_weights", None)
        if original_download is None:
            raise ModelLoadError(f"RF-DETR class {model_class!r} is not a validated 1.5.2 variant")
        try:
            # The 1.5.2 base constructor calls this hook even for locally supplied
            # values. Suppressing it makes architecture construction network inert.
            target.maybe_download_pretrain_weights = lambda self: None
            model = target(pretrain_weights=None, device=device)
        finally:
            target.maybe_download_pretrain_weights = original_download

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelLoadError(
                "RF-DETR checkpoint is not a safe weights-only artifact; convert and qualify it "
                "administratively before recipe execution"
            ) from exc
        state = cls._extract_state_dict(checkpoint, torch=torch)
        inner = getattr(getattr(model, "model", None), "model", None)
        if inner is None or not hasattr(inner, "load_state_dict"):
            raise ModelLoadError("RF-DETR 1.5.2 variant does not expose the qualified model state target")
        bias = state.get("class_embed.bias")
        if bias is not None:
            reinitialize = getattr(getattr(model, "model", None), "reinitialize_detection_head", None)
            if reinitialize is None:
                raise ModelLoadError("RF-DETR variant cannot align its detection head to the checkpoint")
            reinitialize(int(bias.shape[0]))
            inner = model.model.model
        incompatible = inner.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ModelLoadError(
                "RF-DETR state dictionary does not match the configured variant: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        inner.eval()
        inner.requires_grad_(False)
        return model

    @staticmethod
    def _extract_state_dict(checkpoint: Any, *, torch: Any) -> Mapping[str, Any]:
        if not isinstance(checkpoint, Mapping):
            raise ModelLoadError("RF-DETR safe artifact must be a state dictionary mapping")
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, Mapping) or not state:
            raise ModelLoadError("RF-DETR safe artifact has no non-empty 'model' state dictionary")
        if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
            raise ModelLoadError("RF-DETR model state dictionary must contain only string-to-tensor entries")
        return state

    def predict(self, images: Sequence[Image.Image], *, threshold: float) -> list[list[Detection]]:
        if not images:
            return []
        predict = getattr(self._model, "predict", None)
        if predict is None:
            raise ModelLoadError("Loaded RF-DETR object has no predict() method")

        kwargs: dict[str, Any] = {"threshold": float(threshold)}
        try:
            signature = inspect.signature(predict)
            if "include_source_image" in signature.parameters:
                kwargs["include_source_image"] = False
        except (TypeError, ValueError):
            # Some decorated callables do not expose a signature. The conservative
            # path omits the optional memory-saving argument.
            pass

        raw = predict(list(images), **kwargs)
        if len(images) == 1 and not isinstance(raw, list):
            raw_items = [raw]
        else:
            raw_items = list(raw)
        if len(raw_items) != len(images):
            raise ModelLoadError(
                f"RF-DETR returned {len(raw_items)} prediction sets for {len(images)} input images"
            )
        return [self._convert_detections(item) for item in raw_items]

    @staticmethod
    def _convert_detections(value: Any) -> list[Detection]:
        xyxy = np.asarray(getattr(value, "xyxy", np.empty((0, 4))), dtype=np.float64)
        confidence = np.asarray(getattr(value, "confidence", np.empty((0,))), dtype=np.float64)
        raw_class = getattr(value, "class_id", None)
        class_ids = None if raw_class is None else np.asarray(raw_class)

        if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
            raise ModelLoadError(f"Unexpected RF-DETR xyxy shape: {xyxy.shape}")
        if confidence.shape != (xyxy.shape[0],):
            raise ModelLoadError(
                f"RF-DETR confidence shape {confidence.shape} does not match {xyxy.shape[0]} boxes"
            )
        if class_ids is not None and class_ids.shape != (xyxy.shape[0],):
            raise ModelLoadError("RF-DETR class_id shape does not match boxes")

        converted: list[Detection] = []
        for index, coordinates in enumerate(xyxy):
            converted.append(
                Detection(
                    bbox=BBoxXYXY(*map(float, coordinates.tolist())),
                    score=float(confidence[index]),
                    raw_class_id=(None if class_ids is None else int(class_ids[index])),
                )
            )
        return converted

    def close(self) -> None:
        self._model = None
        try:
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
