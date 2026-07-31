from __future__ import annotations

import json

import jsonschema


def test_all_json_schemas_are_valid(repo_root) -> None:
    for path in (repo_root / "resources" / "schemas").glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
