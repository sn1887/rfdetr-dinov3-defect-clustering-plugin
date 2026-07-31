from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from defect_curation_core.detection.rfdetr_adapter import RFDETRAdapter
from defect_curation_core.errors import ModelLoadError


class Tensor:
    shape = (2,)


class Inner:
    def __init__(self):
        self.loaded = None
        self.training = True
        self.frozen = False

    def load_state_dict(self, state, strict):
        self.loaded = (state, strict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        self.training = False

    def requires_grad_(self, value):
        self.frozen = not value


class RFDETRSmall:
    download_calls = 0

    def maybe_download_pretrain_weights(self):
        type(self).download_calls += 1

    def __init__(self, pretrain_weights, device):
        self.maybe_download_pretrain_weights()
        self.args = (pretrain_weights, device)
        inner = Inner()
        self.model = SimpleNamespace(
            model=inner,
            reinitialize_detection_head=lambda _count: None,
        )


class FakeTorch:
    def __init__(self, checkpoint=None, error=None):
        self.checkpoint = checkpoint or {"model": {"weight": Tensor()}}
        self.error = error
        self.load_calls = []

    def load(self, path, **kwargs):
        self.load_calls.append((path, kwargs))
        if self.error:
            raise self.error
        return self.checkpoint

    @staticmethod
    def is_tensor(value):
        return isinstance(value, Tensor)


def test_concrete_variant_construction_is_download_free_and_weights_only() -> None:
    torch = FakeTorch()
    model = RFDETRAdapter._load_model(
        module=SimpleNamespace(RFDETRSmall=RFDETRSmall),
        torch=torch,
        checkpoint_path="/models/checkpoint.pth",
        model_class="RFDETRSmall",
        device="cpu",
    )
    assert model.args == (None, "cpu")
    assert RFDETRSmall.download_calls == 0
    assert torch.load_calls == [
        ("/models/checkpoint.pth", {"map_location": "cpu", "weights_only": True})
    ]
    assert model.model.model.training is False
    assert model.model.model.frozen is True


def test_auto_and_unknown_classes_are_rejected_clearly() -> None:
    with pytest.raises(ModelLoadError, match=r"auto.*unsupported"):
        RFDETRAdapter._resolve_model_class(SimpleNamespace(), "auto")
    with pytest.raises(ModelLoadError, match="Unsupported RF-DETR"):
        RFDETRAdapter._resolve_model_class(SimpleNamespace(), "MadeUpModel")
    with pytest.raises(ModelLoadError, match="does not resolve"):
        RFDETRAdapter._resolve_model_class(SimpleNamespace(), "RFDETRMedium")


def test_unsafe_checkpoint_deserialization_is_not_retried() -> None:
    torch = FakeTorch(error=RuntimeError("unsupported global"))
    with pytest.raises(ModelLoadError, match="weights-only artifact"):
        RFDETRAdapter._load_model(
            module=SimpleNamespace(RFDETRSmall=RFDETRSmall),
            torch=torch,
            checkpoint_path="/models/checkpoint.pth",
            model_class="RFDETRSmall",
            device="cpu",
        )
    assert len(torch.load_calls) == 1


def test_batched_prediction_conversion_is_preserved() -> None:
    adapter = RFDETRAdapter.__new__(RFDETRAdapter)
    detection = SimpleNamespace(
        xyxy=np.array([[1, 2, 3, 4]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int64),
    )
    adapter._model = SimpleNamespace(predict=lambda images, threshold: [detection for _ in images])
    output = adapter.predict([object(), object()], threshold=0.5)
    assert len(output) == 2
    assert output[0][0].bbox.xmin == 1.0
    assert output[0][0].raw_class_id == 0
