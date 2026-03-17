"""Data layer for the analysis viewer — no Qt dependencies."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path


LAUGH_COLORS: dict[str, str] = {
    "big_laugh": "#FF6B6B",
    "medium_laugh": "#FFA040",
    "chuckle": "#FFE066",
}


@dataclass
class LaughEvent:
    event_id: int
    start: float
    end: float
    duration: float
    intensity: float
    intensity_category: str
    confidence: float
    source: str
    spectral_valid: bool
    latency: float = 0.0


@dataclass
class ParagraphViewModel:
    paragraph_id: int
    start_time: float
    end_time: float
    plain_text: str
    html_text: str
    has_laughs: bool
    laugh_count: int
    inline_laughs: list[dict] = field(default_factory=list)


@dataclass
class PipelineResult:
    video_id: str
    comedian: str
    special_name: str
    year: int | None
    total_duration_sec: float
    transcript_type: str
    annotated_transcript: str
    paragraphs: list[ParagraphViewModel]
    laughter_events: list[LaughEvent]
    summary_stats: dict

    @classmethod
    def from_json(cls, path: Path) -> "PipelineResult":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        paragraphs = [
            ParagraphViewModel(
                paragraph_id=p["paragraph_id"],
                start_time=p["start_time"],
                end_time=p["end_time"],
                plain_text=p["text"],
                html_text=_build_html(p["text"], p.get("inline_laughs", [])),
                has_laughs=p["has_laughs"],
                laugh_count=p["laugh_count"],
                inline_laughs=p.get("inline_laughs", []),
            )
            for p in data.get("paragraphs", [])
        ]

        events = [
            LaughEvent(
                event_id=e.get("event_id", i),
                start=e["start"],
                end=e["end"],
                duration=e.get("duration", e["end"] - e["start"]),
                intensity=e.get("intensity", 0.0),
                intensity_category=e.get("intensity_category", "chuckle"),
                confidence=e.get("confidence", 0.0),
                source=e.get("source", "unknown"),
                spectral_valid=e.get("spectral_valid", True),
                latency=e.get("latency", 0.0),
            )
            for i, e in enumerate(data.get("laughter_events", []))
        ]

        return cls(
            video_id=data.get("video_id", ""),
            comedian=data.get("comedian", "Unknown"),
            special_name=data.get("special_name", ""),
            year=data.get("year"),
            total_duration_sec=data.get("total_duration_sec", 0.0),
            transcript_type=data.get("transcript_type", "unknown"),
            annotated_transcript=data.get("annotated_transcript", ""),
            paragraphs=paragraphs,
            laughter_events=events,
            summary_stats=data.get("summary_stats", {}),
        )

    def paragraph_at(self, position_sec: float) -> ParagraphViewModel | None:
        for p in self.paragraphs:
            if p.start_time <= position_sec < p.end_time:
                return p
        return None

    def laughs_near(self, position_sec: float, window: float = 1.0) -> list[LaughEvent]:
        return [e for e in self.laughter_events if abs(e.start - position_sec) <= window]


def _build_html(plain_text: str, inline_laughs: list[dict]) -> str:
    """Build an HTML fragment from plain text and inline laugh offsets."""
    if not inline_laughs:
        return f'<p style="line-height:1.6;">{_escape_newlines(html.escape(plain_text))}</p>'

    # Sort ascending so we can slice left-to-right
    sorted_laughs = sorted(inline_laughs, key=lambda x: x.get("char_offset", 0))

    parts: list[str] = []
    cursor = 0
    for laugh in sorted_laughs:
        offset = min(laugh.get("char_offset", len(plain_text)), len(plain_text))
        # Text before this laugh badge
        segment = plain_text[cursor:offset]
        parts.append(_escape_newlines(html.escape(segment)))
        # Laugh badge
        laugh_type = laugh.get("type", "chuckle")
        color = LAUGH_COLORS.get(laugh_type, "#cccccc")
        parts.append(
            f'<span style="background:{color}; color:#111; border-radius:3px;'
            f' padding:1px 5px; font-size:0.82em; font-weight:600;">'
            f"[{laugh_type}]</span>"
        )
        cursor = offset

    # Remaining text after last laugh
    if cursor < len(plain_text):
        parts.append(_escape_newlines(html.escape(plain_text[cursor:])))

    return f'<p style="line-height:1.6;">{"".join(parts)}</p>'


def _escape_newlines(text: str) -> str:
    return text.replace("\n", "<br>")
