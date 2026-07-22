"""sarva.multimodal.degraders — concrete Degrader implementations for the
registry defined in sarva.multimodal.content."""

from sarva.multimodal.degraders.image import ImageDecodeError, ImageToTextDegrader

__all__ = ["ImageDecodeError", "ImageToTextDegrader"]
