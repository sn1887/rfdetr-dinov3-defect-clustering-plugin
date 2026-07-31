# Development guide

## Architecture boundaries

`defect_curation_core` is framework-neutral. I/O is expressed through small protocols and model packages are isolated behind adapters. `defect_curation_plugin` translates Dataiku roles/settings and handles managed folders and fixed dataset schemas. The recipe script only calls the entry point.

## Configuration

The authoritative tree is `python-lib/Configs`. Dataiku form values become explicit dotlist overrides. The resolved configuration is validated, made read-only, hashed, and persisted before inference. A small OmegaConf fallback composer exists only so offline developer environments can run tests when `hydra-core` is unavailable; production requirements include Hydra.

## Testing levels

- Unit tests: geometry, pooling, debiasing, signatures, clustering, ordering, labels, schemas, config, adapters.
- Synthetic integration: complete pipeline with deterministic model fakes and local or remote-style I/O.
- DSS integration: real local/remote managed folders and native labeling import.
- GPU qualification: safe RF-DETR state dictionary, DINOv3 safetensors, and exact production software/hardware.
- Scale tests: representative image and instance distributions up to the target corpus size.

## Adding a cache-affecting change

Update the relevant schema-version constant or signature payload, add an invalidation test, document the behavior, and verify that changing only clustering parameters still reuses embeddings.

Model adapter tests must mock package APIs and remain network-free. Any model-loader migration gets a new implementation-version constant even when output geometry is unchanged.

## Release

1. Run static repository validation and the test suite.
2. Run real-model qualification and performance tests.
3. Review third-party notices and weight access.
4. Update the changelog/version.
5. Build the deterministic zip.
6. Install into a staging DSS instance and run local-folder, remote-folder, labeling, cache-recluster, partial-failure, and scheduled-build smoke tests.
