"""Stable exception types used across the plugin boundary."""

from __future__ import annotations


class DefectCurationError(RuntimeError):
    """Base class for expected plugin failures."""


class ConfigurationError(DefectCurationError):
    """Raised when the resolved configuration violates an invariant."""


class DependencyError(DefectCurationError):
    """Raised when a required runtime dependency is missing or incompatible."""


class ModelLoadError(DefectCurationError):
    """Raised when a provisioned model cannot be loaded safely."""


class ArtifactError(DefectCurationError):
    """Raised when a cache or run artifact cannot be validated or published."""


class FatalPipelineError(DefectCurationError):
    """Raised for a run-level condition that must abort the recipe."""


class RowProcessingError(DefectCurationError):
    """A recoverable image- or instance-level error with a stable code."""

    def __init__(self, code: str, message: str, *, stage: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.cause = cause
