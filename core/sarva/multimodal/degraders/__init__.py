"""sarva.multimodal.degraders — concrete Degrader implementations for the
registry defined in sarva.multimodal.content."""

from sarva.multimodal.content import Degrader, Modality
from sarva.multimodal.degraders.audio import AudioToTextDegrader
from sarva.multimodal.degraders.image import ImageDecodeError, ImageToTextDegrader

__all__ = [
    "AudioToTextDegrader",
    "ImageDecodeError",
    "ImageToTextDegrader",
    "default_degraders",
]


def default_degraders() -> dict[Modality, Degrader]:
    """The degrader set every skin (CLI, server) wires into its `AgentLoop`
    by default. One place, so "what does Sarva degrade out of the box"
    never drifts between call sites — see loop.py's `degraders` parameter
    docstring for what this actually changes (an opt-in fallback, not a
    change to default routing)."""
    return {
        Modality.IMAGE: ImageToTextDegrader(),
        Modality.AUDIO: AudioToTextDegrader(),
    }
