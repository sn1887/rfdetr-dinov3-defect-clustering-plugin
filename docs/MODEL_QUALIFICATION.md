# Model and environment qualification

A plugin installation is not a qualified production system until the exact safe RF-DETR artifact/variant, DINOv3 snapshot/artifact revision, package set, hardware, and industrial pilot have passed this process.

## RF-DETR gate

On a frozen expert sample, measure candidate instance recall, image-level all-defects-found rate, multi-defect recall, box acceptability/IoU, proposals per image, and critical-stratum false-negative rates. Select the default threshold for recall subject to the agreed review-load ceiling. If the detector fails, improve it outside this plugin rather than adding a hidden fallback localizer.

## DINOv3 representation gate

Hold detector boxes, crop geometry, pooling, clustering, and sample constant. Compare no positional projection with 20-component INSID3-style projection. Ship the 20-component setting only when macro top-10 relevance improves, single-concept cluster rate improves or is statistically indistinguishable, rare-defect recall falls by no more than one percentage point, and no critical stratum falls by more than two points.

Check 512 versus 768 crops only on a declared small/thin-defect stratum when 512 is inadequate. One setting is locked for release; these are not runtime branches.

## K usability envelope

Evaluate a small declared set of K values around domain-expert expectations. Report occupancy, tiny-cluster rate, expert concept coherence, retrieval quality, seed stability, and review time. Publish an advisory range without automatically selecting K.

## Adapter qualification

The environment script should confirm:

- RF-DETR 1.5.2, concrete variant, weights-only loading, and proof that construction did not call its downloader;
- expected output coordinate convention and one foreground class ID;
- timm 1.0.20 DINOv3 local safetensors loader, `[B,768,32,32]` feature shape at 512, finite values, class/register-token exclusion, and no network access;
- bfloat16 support when configured;
- exact model ID, artifact hashes/revisions, package versions, and loader implementation versions;
- deterministic vectors within tolerance across repeated calls;
- compatible PyTorch, torchvision, CUDA, driver, and Faiss builds.

The timm migration is a new implementation qualification boundary. Compare retrieval and clustering acceptance metrics to the approved baseline; do not require numerical identity with Meta's former torch.hub implementation.

Legacy RF-DETR files containing `argparse.Namespace` or other Python objects must be converted on an isolated administrator node with `scripts/convert_rfdetr_checkpoint.py`, then scanned, hashed, and forward-qualified. Recipe runtime must receive only the converted `{ "model": <string-to-tensor state_dict> }` artifact.
