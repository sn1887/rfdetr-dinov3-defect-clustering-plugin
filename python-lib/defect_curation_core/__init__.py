"""Core implementation for RF-DETR-DINOv3 defect clustering."""

from defect_curation_core.pipeline import DefectClusteringPipeline, FitReference, PipelineRunResult, load_fit_reference

__all__ = ["DefectClusteringPipeline", "FitReference", "PipelineRunResult", "load_fit_reference"]
__version__ = "0.1.0"
