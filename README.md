# RF-DETR + DINOv3 Defect Clustering for Dataiku DSS

A standalone Dataiku DSS plugin that implements one deliberately narrow pipeline:

```text
managed image folder
  → class-agnostic RF-DETR boxes
  → frozen DINOv3 ViT-B/16 dense box descriptors
  → spherical k-means review groups
  → Dataiku review dataset + versioned artifact/cache folder
```

The plugin is an **annotation-assistance system**, not an autonomous label generator. Every machine box is exported with the literal category `UNVALIDATED_DEFECT`; cluster IDs are contextual review metadata and are never represented as semantic classes.

## Repository status

This repository contains the full MVP implementation, Dataiku adapter, configuration, cache and artifact formats, tests, operational scripts, and documentation. Real-model qualification is intentionally deployment-specific: model weights are not included, and the release must be validated against the provisioned safe checkpoint, offline DINOv3 artifact, GPU environment, and industrial pilot set.

## What is implemented

- One visual recipe: `cluster-defect-instances`.
- One required input: a managed folder containing JPG/JPEG/PNG images.
- Two required outputs:
  - one image-level review dataset;
  - one managed folder containing caches and versioned run artifacts.
- Local-filesystem and streamed/remote managed-folder adapters.
- Content-hash-based detector and embedding caches.
- Path-unique review instance IDs with content-addressed cache sharing for duplicate image bytes.
- Safe RF-DETR 1.5.2 weights-only adapter with configurable concrete variants.
- Frozen timm DINOv3 ViT-B/16 loader using local safetensors with no runtime network access.
- Aspect-preserving 512×512 letterbox crops with 15% box context.
- Final-layer patch features, per-patch L2 normalization, optional locked INSID3-style positional-subspace removal, fractional detector-box pooling, and final L2 normalization.
- CPU Faiss spherical k-means with deterministic cluster-ID canonicalization.
- Deterministic representative ranking and small-cluster-first round-robin review order.
- Fixed Dataiku output schema and native object-detection label JSON.
- Per-image partial failures, adaptive OOM batch splitting, integrity checks, provenance, and atomic `LATEST.json` publication.
- Unit and synthetic integration tests that do not require proprietary model weights.

## Deliberate non-goals

The MVP does not train models, ingest operator labels, run segmentation, discover proposals through anomaly models, choose K automatically, name clusters, construct a custom webapp, or expose alternative localization/clustering branches.

## Repository layout

```text
plugin.json                         Dataiku plugin metadata and admin settings
code-env/python/                    code-environment declaration and dependencies
custom-recipes/cluster-defect-instances/
                                    recipe declaration and thin bootstrap
python-lib/defect_curation_core/    Dataiku-independent pipeline implementation
python-lib/defect_curation_plugin/  Dataiku managed-folder/dataset adapters
python-lib/Configs/                 authoritative Hydra configuration groups
resources/schemas/                  machine-readable artifact/output schemas
scripts/                            validation, qualification, audit, and packaging tools
tests/                              unit and synthetic integration tests
docs/                               deployment, labeling, audit, and design documentation
```

## Installation

1. Build or use the release zip whose root contains `plugin.json`:

   ```bash
   python scripts/build_plugin_zip.py --output dist/rfdetr-dinov3-defect-clustering-plugin.zip
   ```

2. Install the zip from the Dataiku DSS plugin administration page.
3. Create or assign a qualified code environment. Read `code-env/python/spec/requirements-gpu-notes.md` before installing GPU packages.
4. Provision a qualified RF-DETR weights-only state dictionary and the DINOv3 `model.safetensors` file (or offline snapshot containing it) on paths readable by the DSS execution identity.
5. Configure the plugin-level settings, including immutable package/revision identifiers and optional expected SHA-256 hashes.
6. Add **Cluster defect instances** to a Flow, bind the image folder, create the review dataset and artifact managed folder, and supply `cluster_count`.

The recipe performs no network download. The Dataiku plugin zip does not contain model weights.

## Administrator settings

| Setting | Purpose |
|---|---|
| `rfdetr_checkpoint_path` | Absolute path to the trusted class-agnostic detector checkpoint. |
| `rfdetr_checkpoint_sha256` | Optional expected checkpoint digest; mismatch is fatal. |
| `rfdetr_model_class` | Concrete qualified 1.5.2 variant such as `RFDETRSmall` or `RFDETRMedium`; `auto` is rejected. |
| `rfdetr_package_compatibility` | Locked to `==1.5.2`. |
| `dinov3_model_id` | Locked to `timm/vit_base_patch16_dinov3.lvd1689m`. |
| `dinov3_timm_version` | Locked to the qualified `1.0.20` release. |
| `dinov3_artifact_path` | Absolute local `.safetensors` file or offline snapshot directory. |
| `dinov3_artifact_revision` | Immutable Hub snapshot or administrator artifact revision. |
| `dinov3_artifact_sha256` | Optional expected selected-artifact digest; mismatch is fatal. |
| `device_policy` | CUDA required for production, or CPU allowed for diagnostics. |
| `temporary_root` | Optional local staging directory. |
| `max_failure_fraction` | Fatal ceiling for image-level row failures. |

## Recipe parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `cluster_count` | required | Number of spherical k-means review groups. |
| `detection_threshold` | `0.35` | RF-DETR score threshold, calibrated for recall. |
| `box_padding_fraction` | `0.15` | Context added around every detector box. |
| `max_detections_per_image` | `20` | Deterministic safety cap after score sorting. |
| `force_recompute` | `false` | Ignore compatible detector/embedding caches. |

## Review dataset contract

One row is emitted for every enumerated image, including no-detection and error cases.

| Column | Type | Meaning |
|---|---|---|
| `image_path` | string | POSIX path relative to the source managed folder. |
| `image_status` | string | `OK`, `NO_DETECTION`, or `ERROR`. |
| `image_width`, `image_height` | bigint | Oriented RGB dimensions. |
| `num_defects` | bigint | Number of reviewable detector instances. |
| `primary_cluster_id` | bigint | Cluster of the highest-score successfully embedded instance. |
| `review_order` | bigint | One-based cross-cluster representative review order. |
| `prelabels_json` | string | Dataiku box JSON; category is always `UNVALIDATED_DEFECT`. |
| `instances_json` | string | Instance IDs, scores, grouping context, and warnings. |
| `error_code`, `error_message` | string | Sanitized row-level diagnostic. |
| `run_id` | string | Identifier of the versioned artifact bundle. |

`ERROR` rows deliberately contain empty prelabels and instance context. Detector evidence remains in the run bundle for audit, but the labeling dataset never offers boxes against unreadable or changed source bytes.

## Artifact folder contract

```text
LATEST.json
runs/<run_id>/
  manifest.json
  resolved_config.yaml
  provenance.json
  input_manifest.parquet
  instances.parquet
  embeddings.f16.npy
  centroids.f32.npy
  cluster_summary.parquet
  failures.parquet
  checksums.sha256
cache/detections/<detector_signature>/<image_sha256>.json
cache/embeddings/<embedding_signature>/<instance_key>.f16.npy
specs/<signature>.json
specs/positional_basis/<basis_signature>.{json,f32.npy}
```

Changing only K reuses detection and embedding caches. Changes to detector threshold/checkpoint invalidate detections and downstream vectors; changes to crop/DINOv3/debias/pooling settings reuse detections but invalidate embeddings.

The input image folder and artifact output folder must be different. Keep source images immutable during a run and throughout downstream review, and do not launch concurrent builds against the same review dataset/artifact folder pair. Even a cache-only recluster rereads and hashes source images before publishing labels.

## Native Dataiku labeling

Create an object-detection Labeling task from the review dataset, associate the original image managed folder, use `image_path` as the path column, and import `prelabels_json` as existing labels. Reviewers must replace `UNVALIDATED_DEFECT` with a real taxonomy class or delete the box. See [`docs/DATAIKU_LABELING.md`](docs/DATAIKU_LABELING.md).

## Local development

The core package contains no `dataiku` imports. Run checks from the repository root:

```bash
PYTHONPATH=python-lib python scripts/validate_plugin.py
PYTHONPATH=python-lib pytest
```

The test suite uses deterministic detector/embedder fakes and the `numpy_reference` clustering backend. Real-weight adapter tests belong on a qualified GPU node:

```bash
PYTHONPATH=python-lib python scripts/qualify_environment.py \
  --checkpoint /models/rfdetr.pth \
  --rfdetr-model-class RFDETRSmall \
  --rfdetr-compatibility '==1.5.2' \
  --dinov3-artifact /models/dinov3-snapshot \
  --dinov3-artifact-revision <immutable-snapshot-commit> \
  --device cuda --run-forward
```

## Release-quality boundary

Model confidence, centroid distance, cluster purity, and reviewer agreement are not proof of annotation correctness. A “greater than 95%” quality claim is permitted only after an independent image-level audit whose one-sided 95% exact binomial lower confidence bound is strictly above 0.95. See [`docs/QUALITY_AUDIT.md`](docs/QUALITY_AUDIT.md) and `scripts/binomial_audit.py`.

## Security and licensing

Only administrator-provisioned model files are loaded. DINOv3 requires safetensors; RF-DETR requires a weights-only state dictionary and never invokes the 1.5.2 downloader or unsafe native loader. Expected hashes should be configured in production. See [`docs/SECURITY.md`](docs/SECURITY.md).

The plugin source is Apache-2.0. Third-party dependencies and model/code assets retain their own licenses. DINOv3 and RF-DETR weights are not redistributed. See [`LICENSES/THIRD_PARTY_NOTICES.md`](LICENSES/THIRD_PARTY_NOTICES.md).
