"""Stage 2 dreaming — knowledge extraction from session transcripts."""

from openharness.services.dreaming.transcript import load_compacted_transcript, load_full_transcript
from openharness.services.dreaming.prompt import build_dream_prompt
from openharness.services.dreaming.service import DreamingExecutor

__all__ = [
    "load_compacted_transcript",
    "load_full_transcript",
    "build_dream_prompt",
    "DreamingExecutor",
]
