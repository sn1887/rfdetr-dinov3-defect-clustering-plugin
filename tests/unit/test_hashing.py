from __future__ import annotations

import pytest
from defect_curation_core.errors import ConfigurationError
from defect_curation_core.hashing import normalize_relative_path, sha256_bytes, validate_expected_sha256


def test_normalize_managed_folder_paths() -> None:
    assert normalize_relative_path("/nested\\image.PNG") == "nested/image.PNG"
    with pytest.raises(ConfigurationError):
        normalize_relative_path("../secret")
    with pytest.raises(ConfigurationError):
        normalize_relative_path("")


def test_expected_digest_validation() -> None:
    digest = sha256_bytes(b"abc")
    validate_expected_sha256(digest, digest.upper(), label="file")
    with pytest.raises(ConfigurationError):
        validate_expected_sha256(digest, "0" * 64, label="file")
