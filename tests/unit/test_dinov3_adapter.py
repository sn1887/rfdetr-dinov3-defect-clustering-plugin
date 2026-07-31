from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
from defect_curation_core.embeddings.dinov3_adapter import DINO_MODEL_ID, DinoV3Adapter
from defect_curation_core.errors import ModelLoadError
from PIL import Image


class FakeTimmModel(torch.nn.Module):
    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.output = None
        self.feature_kwargs = None

    def forward_intermediates(self, _batch, **kwargs):
        self.feature_kwargs = kwargs
        if self.output is not None:
            return self.output
        return [(torch.ones(1, 1024, 768), torch.ones(1, 5, 768))]


def build_adapter(tmp_path, monkeypatch):
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"safe")
    model = FakeTimmModel()
    calls = []
    timm = ModuleType("timm")

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return model

    timm.create_model = create_model
    safe_root = ModuleType("safetensors")
    safe_torch = ModuleType("safetensors.torch")
    safe_torch.load_file = lambda path, device: {"weight": torch.zeros(1)}
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "safetensors", safe_root)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safe_torch)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0.20" if name == "timm" else "0")
    adapter = DinoV3Adapter(
        artifact_path=str(artifact),
        model_id=DINO_MODEL_ID,
        timm_version="1.0.20",
        device="cpu",
        input_size=512,
        patch_size=16,
        embedding_dim=768,
        expected_prefix_tokens=5,
        inference_precision="float32",
    )
    return adapter, model, calls


def test_timm_creation_is_local_frozen_and_uses_qualified_identity(tmp_path, monkeypatch) -> None:
    adapter, model, calls = build_adapter(tmp_path, monkeypatch)
    assert calls == [("vit_base_patch16_dinov3.lvd1689m", {"pretrained": False, "num_classes": 0})]
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    result = adapter.extract_patch_maps([Image.new("RGB", (512, 512))])
    assert tuple(result.shape) == (1, 768, 32, 32)
    assert model.feature_kwargs["return_prefix_tokens"] is True
    assert model.feature_kwargs["norm"] is True


@pytest.mark.parametrize(
    ("patches", "prefixes", "match"),
    [
        (torch.ones(1, 1023, 768), torch.ones(1, 5, 768), "patch-token shape"),
        (torch.ones(1, 1024, 767), torch.ones(1, 5, 768), "patch-token shape"),
        (torch.ones(1, 1024, 768), torch.ones(1, 4, 768), "prefix-token shape"),
    ],
)
def test_invalid_token_shapes_fail(tmp_path, monkeypatch, patches, prefixes, match) -> None:
    adapter, model, _ = build_adapter(tmp_path, monkeypatch)
    model.output = [(patches, prefixes)]
    with pytest.raises(ModelLoadError, match=match):
        adapter.extract_patch_maps([Image.new("RGB", (512, 512))])


def test_non_finite_features_fail(tmp_path, monkeypatch) -> None:
    adapter, model, _ = build_adapter(tmp_path, monkeypatch)
    patches = torch.ones(1, 1024, 768)
    patches[0, 0, 0] = torch.nan
    model.output = [(patches, torch.ones(1, 5, 768))]
    with pytest.raises(ModelLoadError, match="non-finite"):
        adapter.extract_patch_maps([Image.new("RGB", (512, 512))])


def test_runtime_rejects_pickle_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"pickle")
    with pytest.raises(ModelLoadError, match="safetensors"):
        DinoV3Adapter.resolve_artifact(checkpoint)
