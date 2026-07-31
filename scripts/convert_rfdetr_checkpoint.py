#!/usr/bin/env python3
"""Administrator-only conversion of a trusted RF-DETR checkpoint.

This script intentionally permits Python pickle deserialization. Run it only in
an isolated qualification environment after verifying checkpoint provenance.
It is never imported or invoked by Dataiku recipe runtime.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--i-understand-this-deserializes-untrusted-python",
        action="store_true",
        help="Required acknowledgement for administrator-side unsafe deserialization.",
    )
    args = parser.parse_args()
    if not args.i_understand_this_deserializes_untrusted_python:
        parser.error("explicit unsafe-deserialization acknowledgement is required")
    if not args.input.is_file():
        parser.error(f"input checkpoint does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    import torch

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        parser.error("checkpoint is not a mapping")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        parser.error("checkpoint has no non-empty model state dictionary")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        parser.error("model state must contain only string-to-tensor entries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dict(state)}, args.output)
    # Prove that the produced artifact is accepted by the recipe's security boundary.
    converted = torch.load(args.output, map_location="cpu", weights_only=True)
    if not isinstance(converted, Mapping) or set(converted) != {"model"}:
        parser.error("converted artifact failed weights-only verification")
    print(f"input_sha256={sha256_file(args.input)}")
    print(f"output_sha256={sha256_file(args.output)}")
    print(f"tensor_count={len(state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
