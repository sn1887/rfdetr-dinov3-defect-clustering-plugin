# Security and privacy controls

## Model supply chain

- Load only administrator-provisioned checkpoint and weight files.
- Configure expected SHA-256 digests in production.
- Pin RF-DETR exactly to 1.5.2 and select a qualified concrete exported variant.
- Pin timm exactly to 1.0.20 and record the immutable Hugging Face snapshot/artifact revision.
- Do not accept model paths from ordinary recipe users.
- RF-DETR recipe runtime accepts only a state dictionary that `torch.load(..., weights_only=True)` can read. It never calls the package's 1.5.2 native loader, which uses `weights_only=False` and may redownload on failure.
- DINOv3 recipe runtime accepts only safetensors and calls `timm.create_model(..., pretrained=False)`; it never passes a Hub URI to timm.
- Convert legacy RF-DETR checkpoints only in a quarantined administrator environment after provenance review. The conversion tool intentionally performs unsafe deserialization and must never be exposed to recipe users.

## Data and paths

Managed-folder paths are normalized to relative POSIX paths and traversal is rejected. Error messages written to datasets are sanitized; detailed stack traces remain in restricted DSS logs. Do not place secrets in folder names, model paths, plugin settings, or image metadata.

## Network

The recipe does not call a network API or download models. Enforce egress restrictions at the code-environment/execution-node level when required by policy.

## Artifact access

Embeddings and centroids can reveal properties of industrial imagery even without source pixels. Protect the artifact folder with the same access class as the input images. Apply retention, encryption, backup, and audit logging policies to caches and run bundles.

## Dependency controls

Build the code environment from approved indexes, record hashes/SBOM, scan direct and transitive dependencies, and repeat qualification after any package, driver, CUDA, model, or operating-system change.
