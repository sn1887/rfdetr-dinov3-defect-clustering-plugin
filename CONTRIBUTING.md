# Contributing

Changes must preserve the single-pipeline product contract unless an architecture decision explicitly replaces it.

## Required checks

```bash
PYTHONPATH=python-lib python scripts/validate_plugin.py
PYTHONPATH=python-lib pytest
```

Run real-model qualification on the deployment GPU whenever model adapters, preprocessing, PyTorch/CUDA versions, RF-DETR version, DINOv3 revision, crop geometry, positional debiasing, or pooling changes.

## Core boundaries

- `defect_curation_core` must not import `dataiku`.
- Dataiku API calls belong in `defect_curation_plugin` or the recipe bootstrap.
- Do not add runtime downloads.
- Do not expose model/algorithm selectors without an accepted architecture decision.
- Do not turn cluster IDs into label categories.
- Schema changes require a version increment, migration notes, and compatibility tests.
- Cache-signature inputs must change whenever an operation can change its output.

## Pull requests

Include the motivation, compatibility impact, tests, performance impact, cache invalidation behavior, and any licensing implications. Never commit model weights, industrial images, credentials, absolute deployment paths, or generated run bundles.
