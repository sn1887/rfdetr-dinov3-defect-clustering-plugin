# GPU environment qualification notes

The checked-in `requirements.txt` is a reproducible reference, not a universal
CUDA lock file. A production DSS administrator must qualify one environment
against the actual RF-DETR checkpoint and GPU driver stack.

1. Install the PyTorch and torchvision wheels that exactly match the node's CUDA
   runtime. Do not let an unconstrained `pip install rfdetr` replace them.
2. Install exactly `rfdetr==1.5.2`, `timm==1.0.20`, and the pinned safetensors
   dependency. Do not relax the plugin compatibility settings.
3. Install `faiss-cpu` unless a separately qualified GPU Faiss build is required.
   The MVP clustering reference backend is CPU Faiss.
4. Install the remaining packages from `requirements.txt` without upgrading
   PyTorch implicitly.
5. Run `python scripts/qualify_environment.py` with the provisioned artifact paths
   from a checkout of this repository before assigning the environment to the plugin.
6. Record the final `pip freeze`, CUDA runtime, driver, GPU model, RF-DETR
   checkpoint SHA-256, concrete RF-DETR variant, DINOv3 model ID, immutable
   artifact revision, artifact SHA-256, and loader implementation versions.

The plugin performs no runtime network downloads. Provision DINOv3 as a local
`.safetensors` file or offline snapshot. Convert legacy RF-DETR checkpoints to a
weights-only state-dictionary artifact in a quarantined administrator environment.
