"""Data models for the comedy laughter detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TimelineEntry:
    """A single entry on the unified timeline (speech or laughter)."""

    entry_type: Literal["speech", "laughter"]
    start: float
    end: float
    text: str = ""

    # Laughter-specific fields
    intensity: float = 0.0
    intensity_category: Literal["big_laugh", "medium_laugh", "chuckle", ""] = ""
    duration: float = 0.0
    latency: float = 0.0

    # Detection metadata
    confidence: float = 0.0
    source: str = ""  # "yamnet", "gillick", "ensemble"
    event_id: int = 0


@dataclass
class LaughterEvent:
    """A detected laughter event with full metadata."""

    event_id: int = 0
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    intensity: float = 0.0
    intensity_category: Literal["big_laugh", "medium_laugh", "chuckle"] = "chuckle"
    confidence: float = 0.0
    source: str = "ensemble"
    latency: float = 0.0

    # Spectral validation
    spectral_valid: bool = True
    spectral_centroid: float = 0.0


@dataclass
class TranscriptSegment:
    """A segment of transcript text with timing."""

    text: str
    start: float
    end: float
    duration: float = 0.0


@dataclass
class Paragraph:
    """A context-based paragraph with inline laugh annotations."""

    paragraph_id: int
    text: str
    annotated_text: str
    start_time: float
    end_time: float
    segment_reason: str = "topic_shift"
    inline_laughs: list[dict] = field(default_factory=list)
    laugh_count: int = 0
    has_laughs: bool = False


@dataclass
class VideoMetadata:
    """Metadata for a comedy video."""

    video_id: str = ""
    comedian: str = "Unknown"
    special_name: str = "Unknown"
    year: int | None = None
    duration: float = 0.0
    transcript_type: str = "unknown"
