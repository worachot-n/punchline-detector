"""Phase 5: Unified Timeline — merge transcript + laughter on a single timeline."""

from __future__ import annotations

from .models import LaughterEvent, TimelineEntry, TranscriptSegment


def build_unified_timeline(
    transcript: list[TranscriptSegment],
    laughter_events: list[LaughterEvent],
) -> list[TimelineEntry]:
    """
    Build a unified timeline combining transcript segments and laughter events.

    Both speech and laughter entries are placed on a single sorted timeline,
    allowing us to see exactly when laughter occurs relative to the comedian's words.
    """
    timeline: list[TimelineEntry] = []

    # Add speech entries
    for seg in transcript:
        timeline.append(TimelineEntry(
            entry_type="speech",
            start=seg.start,
            end=seg.end,
            text=seg.text,
            duration=seg.duration,
        ))

    # Add laughter entries
    for evt in laughter_events:
        timeline.append(TimelineEntry(
            entry_type="laughter",
            start=evt.start,
            end=evt.end,
            intensity=evt.intensity,
            intensity_category=evt.intensity_category,
            duration=evt.duration,
            confidence=evt.confidence,
            source=evt.source,
            event_id=evt.event_id,
        ))

    # Sort by start time, with speech before laughter at the same timestamp
    timeline.sort(key=lambda e: (e.start, 0 if e.entry_type == "speech" else 1))

    print(
        f"  Timeline: {len(timeline)} entries "
        f"({sum(1 for e in timeline if e.entry_type == 'speech')} speech, "
        f"{sum(1 for e in timeline if e.entry_type == 'laughter')} laughter)"
    )

    return timeline


def compute_latencies(timeline: list[TimelineEntry]) -> list[TimelineEntry]:
    """
    Compute latency for each laughter event.

    Latency = time between the end of the preceding speech and
    the start of the laughter. This measures how quickly the
    audience reacts to the comedian's words.

    Typical values:
    - < 0.3s: very quick reaction (punchline was obvious/expected)
    - 0.3-0.8s: normal reaction time
    - > 0.8s: delayed reaction (joke needed processing)
    """
    last_speech_end = 0.0

    for entry in timeline:
        if entry.entry_type == "speech":
            last_speech_end = entry.end
        elif entry.entry_type == "laughter":
            entry.latency = max(0.0, round(entry.start - last_speech_end, 3))

    latencies = [e.latency for e in timeline if e.entry_type == "laughter" and e.latency > 0]
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  Latencies: avg={avg:.2f}s, min={min(latencies):.2f}s, max={max(latencies):.2f}s")

    return timeline
