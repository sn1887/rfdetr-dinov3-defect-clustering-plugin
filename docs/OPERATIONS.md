# Operations runbook

## Preflight

- Confirm the plugin code environment is qualified on the execution node.
- Confirm CUDA visibility when the policy is `cuda_required`.
- Verify model paths, permissions, hashes, RF-DETR concrete variant, exact package versions, and the DINOv3 artifact revision.
- Ensure the artifact managed folder has enough capacity for caches plus one staged run.
- Ensure `cluster_count` is plausible and no greater than the expected instance count.
- Bind distinct managed folders for source images and artifact output.
- Keep the source image folder immutable for the duration of a run and while its review task is active.
- Do not run concurrent builds that write the same review dataset or artifact folder; publication is atomic per run, not a multi-writer coordination protocol.

Run `scripts/qualify_environment.py` on the same node and code environment used by DSS.

## Successful publication semantics

The recipe uploads all files below `runs/<run_id>/`, writes the review dataset, and only then writes `LATEST.json`. A directory under `runs/` without a matching current pointer is incomplete or superseded and must not be treated as current.

## Cache behavior

- Detection cache key: checkpoint/package/model/preprocessing/threshold/cap plus image content hash.
- Embedding cache key: timm version, Hugging Face model identity, local artifact digest/revision, loader version, crop geometry, debias basis, pooling specification, and detector-derived instance key.
- Clustering is always rerun from valid vectors; changing K does not repeat model inference.
- `force_recompute` bypasses compatible detection and embedding entries but does not delete historical cache files.
- Cache-only reclustering still rereads and hashes every image with detector instances before publication, so changed source bytes are made review-inert rather than paired with stale boxes.

## Failure behavior

Row-level image/decode/inference failures are recorded and processing continues until `max_failure_fraction` is exceeded. Batch OOM failures are split recursively. At batch size one, the affected image or crop is marked failed; production never silently switches from CUDA to CPU.

Fatal conditions include invalid configuration, missing/incompatible model resources, output publication failure, no valid instances, insufficient embeddings, K greater than valid instances, excessive row failures, multiple detector class IDs, and artifact integrity failure.

## Retention

Do not clear the artifact folder from the recipe. Retain every run referenced by an audit or downstream process. A separate administrator process may delete unreferenced historical `runs/` and cache signatures after policy review. Never delete `LATEST.json` first.

## Recovery

1. Inspect the DSS job log and `failures.parquet` when a bundle exists.
2. Correct the model/resource/input/output issue.
3. Rerun with the same configuration; compatible completed cache entries will be reused.
4. Use `force_recompute` only when cache integrity or model semantics are in doubt.
5. Validate the resulting bundle with `scripts/validate_run_bundle.py` after exporting it to a local path.
