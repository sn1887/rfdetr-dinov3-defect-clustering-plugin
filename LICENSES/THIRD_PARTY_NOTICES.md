# Third-party notices

This repository does not redistribute RF-DETR or DINOv3 model weights. The operator is responsible for confirming the license and approved use of every provisioned artifact.

| Component | Typical upstream license/access condition | Repository use |
|---|---|---|
| PyTorch / torchvision | BSD-style | Runtime inference and tensor operations. |
| RF-DETR | License depends on exact package/model lineage; Apache-designated and separately licensed variants exist | Imported package and administrator-provisioned checkpoint. Record exact lineage before deployment. |
| timm | Apache-2.0 | Pinned DINOv3 model architecture loader; no runtime Hub access. |
| safetensors | Apache-2.0 | Non-pickle DINOv3 artifact loading. |
| DINOv3 weights | DINOv3 license and gated/approved weight access | Administrator-provisioned ViT-B/16 safetensors artifact; not bundled. |
| INSID3 | Apache-2.0 | Mathematical techniques reimplemented with attribution; full project is not vendored. |
| Faiss | MIT | CPU spherical k-means. |
| Hydra Core | MIT | Configuration composition. |
| OmegaConf | BSD-3-Clause | Structured resolved configuration. |
| NumPy | BSD-3-Clause | Array and artifact operations. |
| pandas | BSD-3-Clause | Deployment utility dependency. |
| PyArrow | Apache-2.0 | Parquet run artifacts. |
| Pillow | HPND | Deterministic image decoding and crop transforms. |
| packaging | Apache-2.0/BSD | PEP 440 compatibility checks. |

Before release, regenerate a software bill of materials from the qualified code environment, preserve upstream notices required by the exact installed versions, and review CUDA/runtime redistribution terms where applicable.
