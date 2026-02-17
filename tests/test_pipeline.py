"""Tests for the comedy laughter detection pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from comedy_pipeline.models import LaughterEvent, TimelineEntry, TranscriptSegment
from comedy_pipeline.phase4_postprocessing import filter_by_duration, merge_close_events
from comedy_pipeline.phase5_timeline import build_unified_timeline, compute_latencies
from comedy_pipeline.phase7_segmentation import (
    build_annotated_text,
    build_paragraphs,
    detect_long_pauses,
    detect_post_laugh_breaks,
    detect_transition_phrases,
    find_paragraph_breaks,
)
from comedy_pipeline.phase8_transcript import (
    compute_summary,
    export_dataset,
    generate_annotated_transcript,
    generate_detailed_transcript,
)


# === Fixtures ===


def make_speech(start: float, end: float, text: str) -> TimelineEntry:
    return TimelineEntry(entry_type="speech", start=start, end=end, text=text, duration=end - start)


def make_laugh(
    start: float, end: float, category: str = "medium_laugh", intensity: float = 0.5
) -> TimelineEntry:
    return TimelineEntry(
        entry_type="laughter",
        start=start,
        end=end,
        intensity=intensity,
        intensity_category=category,
        duration=end - start,
    )


@pytest.fixture
def sample_timeline() -> list[TimelineEntry]:
    """A timeline simulating a comedian doing 2 bits with a transition."""
    return [
        # Bit 1: barbershop story
        make_speech(0.0, 2.0, "I was in the barbershop the other day."),
        make_speech(2.1, 4.0, "My barber looks at me and starts talking about politics."),
        make_laugh(4.1, 5.5, "medium_laugh", 0.55),
        make_speech(5.6, 8.0, "I just sat there nodding. He could've told me the earth was flat."),
        make_laugh(8.1, 11.0, "big_laugh", 0.88),
        # Long pause = topic change
        # Transition / setup (no laugh)
        make_speech(15.0, 17.0, "So I leave the barbershop and I'm walking down the street."),
        make_speech(17.1, 20.0, "And I start thinking about something. You ever notice how the world changed?"),
        # Bit 2: sensitivity
        make_speech(20.5, 23.0, "People are sensitive. Everything offends somebody."),
        make_speech(23.1, 26.0, "You gotta be a person with a disclaimer."),
        make_laugh(26.1, 28.0, "medium_laugh", 0.50),
    ]


# === Phase 4 Tests ===


class TestMergeCloseEvents:
    def test_merge_adjacent(self):
        events = [
            LaughterEvent(start=1.0, end=1.5, duration=0.5),
            LaughterEvent(start=1.7, end=2.2, duration=0.5),
        ]
        merged = merge_close_events(events, max_gap=0.5)
        assert len(merged) == 1
        assert merged[0].start == 1.0
        assert merged[0].end == 2.2

    def test_no_merge_far_apart(self):
        events = [
            LaughterEvent(start=1.0, end=1.5, duration=0.5),
            LaughterEvent(start=5.0, end=5.5, duration=0.5),
        ]
        merged = merge_close_events(events, max_gap=0.5)
        assert len(merged) == 2

    def test_empty_input(self):
        assert merge_close_events([]) == []


class TestFilterByDuration:
    def test_filter_short(self):
        events = [
            LaughterEvent(start=0, end=0.1, duration=0.1),
            LaughterEvent(start=1, end=2, duration=1.0),
        ]
        filtered = filter_by_duration(events, min_duration=0.3)
        assert len(filtered) == 1
        assert filtered[0].duration == 1.0

    def test_filter_long(self):
        events = [
            LaughterEvent(start=0, end=40, duration=40.0),
            LaughterEvent(start=50, end=51, duration=1.0),
        ]
        filtered = filter_by_duration(events, max_duration=30.0)
        assert len(filtered) == 1


# === Phase 5 Tests ===


class TestUnifiedTimeline:
    def test_builds_sorted_timeline(self):
        transcript = [
            TranscriptSegment(text="Hello", start=0.0, end=1.0, duration=1.0),
            TranscriptSegment(text="World", start=2.0, end=3.0, duration=1.0),
        ]
        laughs = [LaughterEvent(start=1.2, end=1.8, duration=0.6)]
        timeline = build_unified_timeline(transcript, laughs)

        assert len(timeline) == 3
        assert timeline[0].entry_type == "speech"
        assert timeline[1].entry_type == "laughter"
        assert timeline[2].entry_type == "speech"

    def test_compute_latencies(self):
        timeline = [
            make_speech(0.0, 1.0, "setup"),
            make_laugh(1.5, 2.5, "big_laugh"),
        ]
        timeline = compute_latencies(timeline)
        assert timeline[1].latency == 0.5


# === Phase 7 Tests ===


class TestLongPauseDetection:
    def test_detects_long_pause(self, sample_timeline):
        breaks = detect_long_pauses(sample_timeline, min_pause=3.0)
        # Should detect the gap between 11.0 and 15.0 (4 seconds)
        assert len(breaks) >= 1
        assert any(abs(bp - 15.0) < 0.5 for bp in breaks)

    def test_no_pause(self):
        timeline = [
            make_speech(0, 1, "a"),
            make_speech(1.5, 2.5, "b"),
        ]
        breaks = detect_long_pauses(timeline, min_pause=3.0)
        assert len(breaks) == 0


class TestTransitionPhraseDetection:
    def test_detects_transition(self):
        timeline = [
            make_speech(0, 1, "So check this out."),
            make_speech(2, 3, "I was at the store."),
            make_speech(4, 5, "Have you ever noticed something weird?"),
        ]
        breaks = detect_transition_phrases(timeline)
        assert len(breaks) >= 2  # "So check this out" and "Have you ever noticed"

    def test_no_transition(self):
        timeline = [
            make_speech(0, 1, "He said yes."),
            make_speech(2, 3, "She said no."),
        ]
        breaks = detect_transition_phrases(timeline)
        assert len(breaks) == 0


class TestPostLaughBreaks:
    def test_detects_break_after_big_laugh(self):
        timeline = [
            make_speech(0, 1, "punchline"),
            make_laugh(1.1, 3.0, "big_laugh", 0.9),
            make_speech(6.0, 7.0, "new topic"),
        ]
        breaks = detect_post_laugh_breaks(timeline, min_post_laugh_pause=2.0)
        assert len(breaks) == 1
        assert breaks[0] == 6.0

    def test_no_break_after_medium_laugh(self):
        timeline = [
            make_speech(0, 1, "joke"),
            make_laugh(1.1, 2.0, "medium_laugh", 0.5),
            make_speech(5.0, 6.0, "next"),
        ]
        breaks = detect_post_laugh_breaks(timeline, min_post_laugh_pause=2.0)
        assert len(breaks) == 0


class TestBuildAnnotatedText:
    def test_inline_tags(self):
        speeches = [
            make_speech(0, 1, "Setup joke."),
            make_speech(3, 4, "Tag line."),
        ]
        laughs = [make_laugh(1.5, 2.5, "big_laugh")]

        text, inline_laughs = build_annotated_text(speeches, laughs)

        assert "[big_laugh]" in text
        assert "Setup joke." in text
        assert "Tag line." in text
        assert len(inline_laughs) == 1

    def test_no_laughs(self):
        speeches = [make_speech(0, 1, "Just setup.")]
        text, inline_laughs = build_annotated_text(speeches, [])

        assert text == "Just setup."
        assert len(inline_laughs) == 0


class TestBuildParagraphs:
    def test_splits_at_break_points(self, sample_timeline):
        break_points = [15.0]
        paragraphs = build_paragraphs(sample_timeline, break_points)

        assert len(paragraphs) == 2
        assert paragraphs[0]["has_laughs"] is True
        # Second paragraph may or may not have laughs depending on the break

    def test_no_break_points(self, sample_timeline):
        paragraphs = build_paragraphs(sample_timeline, [])
        assert len(paragraphs) == 1  # Everything in one paragraph


class TestFindParagraphBreaks:
    def test_finds_breaks(self, sample_timeline):
        # Disable semantic shift (requires model) by using only other signals
        breaks = find_paragraph_breaks(
            sample_timeline,
            weights={
                "long_pause": 0.5,
                "transition_phrase": 0.3,
                "semantic_shift": 0.0,
                "post_big_laugh_pause": 0.25,
            },
            vote_threshold=0.4,
        )
        # Should find at least one break around the 15.0s mark
        assert len(breaks) >= 1


# === Phase 8 Tests ===


class TestTranscriptGeneration:
    def test_clean_transcript(self):
        paragraphs = [
            {
                "paragraph_id": 1,
                "annotated_text": "Hello world.\n[big_laugh]\nGoodbye.",
                "start_time": 0.0,
                "end_time": 5.0,
                "has_laughs": True,
                "laugh_count": 1,
            },
        ]
        metadata = {"comedian": "Test", "special_name": "Special", "video_id": "x", "year": 2024}
        stats = {
            "total_paragraphs": 1, "paragraphs_with_laughs": 1,
            "paragraphs_without_laughs": 0, "total_laughs": 1,
            "big_laughs": 1, "medium_laughs": 0, "chuckles": 0,
            "avg_laugh_duration": 2.0, "avg_latency": 0.3,
            "laughs_per_minute": 1.5, "laugh_time_percentage": 10.0,
        }
        result = generate_annotated_transcript(paragraphs, metadata, stats)

        assert "[COMEDIAN] Test" in result
        assert "[big_laugh]" in result
        assert "[STATS]" in result

    def test_detailed_transcript(self):
        paragraphs = [
            {
                "paragraph_id": 1,
                "annotated_text": "Hello.",
                "start_time": 65.0,
                "end_time": 70.0,
                "has_laughs": False,
                "laugh_count": 0,
            },
        ]
        metadata = {"comedian": "Test", "special_name": "S", "video_id": "y"}
        stats = {}
        result = generate_detailed_transcript(paragraphs, metadata, stats)

        assert "[P1]" in result
        assert "01:05" in result  # 65 seconds = 01:05
        assert "(setup/transition)" in result


class TestComputeSummary:
    def test_basic_stats(self):
        paragraphs = [
            {"has_laughs": True},
            {"has_laughs": False},
        ]
        events = [
            {"duration": 2.0, "intensity": 0.8, "intensity_category": "big_laugh", "latency": 0.3},
            {"duration": 1.5, "intensity": 0.5, "intensity_category": "medium_laugh", "latency": 0.5},
        ]
        stats = compute_summary(paragraphs, events, total_duration=120.0)

        assert stats["total_paragraphs"] == 2
        assert stats["paragraphs_with_laughs"] == 1
        assert stats["total_laughs"] == 2
        assert stats["big_laughs"] == 1
        assert stats["medium_laughs"] == 1
        assert stats["laughs_per_minute"] == 1.0


class TestExport:
    def test_export_creates_files(self, tmp_path):
        paragraphs = [
            {
                "paragraph_id": 1,
                "text": "Hello world.",
                "annotated_text": "Hello world.\n[big_laugh]",
                "start_time": 0.0,
                "end_time": 5.0,
                "has_laughs": True,
                "laugh_count": 1,
                "inline_laughs": [],
            },
        ]
        metadata = {"comedian": "Test", "special_name": "S", "video_id": "test123"}
        stats = {
            "total_paragraphs": 1, "paragraphs_with_laughs": 1,
            "paragraphs_without_laughs": 0, "total_laughs": 1,
            "big_laughs": 1, "medium_laughs": 0, "chuckles": 0,
            "avg_laugh_duration": 2.0, "avg_latency": 0.3,
            "laughs_per_minute": 1.5, "laugh_time_percentage": 10.0,
        }
        events = [{"start": 2.0, "end": 4.0, "intensity_category": "big_laugh"}]

        paths = export_dataset(
            video_id="test123",
            metadata=metadata,
            paragraphs=paragraphs,
            laughter_events=events,
            summary_stats=stats,
            output_dir=str(tmp_path),
        )

        assert Path(paths["clean"]).exists()
        assert Path(paths["detailed"]).exists()
        assert Path(paths["json"]).exists()

        # Verify JSON structure
        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["video_id"] == "test123"
        assert data["comedian"] == "Test"
        assert len(data["paragraphs"]) == 1
