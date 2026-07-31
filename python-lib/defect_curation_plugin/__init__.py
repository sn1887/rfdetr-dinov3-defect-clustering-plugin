"""Thin Dataiku DSS integration layer for the defect-curation core."""

from defect_curation_plugin.recipe_entrypoints import run_cluster_defect_instances

__all__ = ["run_cluster_defect_instances"]
