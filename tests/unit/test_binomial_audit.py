from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def load_module(repo_root: Path):
    path = repo_root / "scripts" / "binomial_audit.py"
    spec = importlib.util.spec_from_file_location("binomial_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_bound_matches_design_examples(repo_root) -> None:
    module = load_module(repo_root)
    assert abs(module.one_sided_lower_bound(59, 59) - 0.95049) < 2e-5
    assert abs(module.one_sided_lower_bound(99, 100) - 0.95344) < 2e-5
    assert abs(module.one_sided_lower_bound(388, 400) - 0.95185) < 3e-5


def test_cli_rejects_zero_audited_images_without_traceback(repo_root) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "binomial_audit.py"),
            "--audited",
            "0",
            "--incorrect",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "audited must be positive" in result.stderr
    assert "Traceback" not in result.stderr
