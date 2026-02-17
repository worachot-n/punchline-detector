"""Phase 4: Post-Processing — merge, filter, intensity classification, spectral validation."""

from __future__ import annotations

import librosa
import numpy as np

from .models import LaughterEvent


def merge_close_events(
    events: list[LaughterEvent],
    max_gap: float = 0.5,
) -> list[LaughterEvent]:
    """
    Merge laughter events that are close together (< max_gap seconds apart).

    Audience laughter often comes in waves — multiple short detections
    that are really one continuous laugh. This merges them.
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.start)
    merged = [LaughterEvent(
        start=sorted_events[0].start,
        end=sorted_events[0].end,
        duration=sorted_events[0].duration,
        confidence=sorted_events[0].confidence,
        source=sorted_events[0].source,
    )]

    for event in sorted_events[1:]:
        last = merged[-1]

        if event.start - last.end <= max_gap:
            # Merge: extend the current event
            last.end = max(last.end, event.end)
            last.duration = last.end - last.start
            last.confidence = max(last.confidence, event.confidence)
            if last.source != event.source:
                last.source = "merged"
        else:
            merged.append(LaughterEvent(
                start=event.start,
                end=event.end,
                duration=event.duration,
                confidence=event.confidence,
                source=event.source,
            ))

    print(f"  Merged: {len(events)} → {len(merged)} events (gap threshold: {max_gap}s)")
    return merged


def filter_by_duration(
    events: list[LaughterEvent],
    min_duration: float = 0.3,
    max_duration: float = 30.0,
) -> list[LaughterEvent]:
    """
    Filter out events that are too short (noise) or too long (misdetection).

    - Too short (< 0.3s): likely false positives (coughs, clicks)
    - Too long (> 30s): likely misdetection (music, continuous noise)
    """
    filtered = [
        e for e in events
        if min_duration <= e.duration <= max_duration
    ]
    removed = len(events) - len(filtered)
    if removed:
        print(f"  Duration filter: removed {removed} events ({min_duration}s-{max_duration}s)")
    return filtered


def compute_intensity(
    audio_path: str,
    events: list[LaughterEvent],
    big_threshold: float = 0.7,
    medium_threshold: float = 0.4,
) -> list[LaughterEvent]:
    """
    Compute intensity for each laughter event based on RMS energy.

    Categories:
    - big_laugh:    intensity >= 0.7 (normalized)
    - medium_laugh: intensity >= 0.4
    - chuckle:      intensity < 0.4
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Compute overall RMS for normalization
    overall_rms = np.sqrt(np.mean(y ** 2))
    if overall_rms == 0:
        return events

    for event in events:
        start_sample = int(event.start * sr)
        end_sample = int(event.end * sr)
        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            event.intensity = 0.0
            event.intensity_category = "chuckle"
            continue

        # RMS of the laughter segment, normalized by overall RMS
        event_rms = np.sqrt(np.mean(segment ** 2))
        intensity = min(event_rms / overall_rms, 1.0)

        # Also consider peak amplitude
        peak = np.max(np.abs(segment))
        peak_norm = min(peak / np.max(np.abs(y)), 1.0) if np.max(np.abs(y)) > 0 else 0

        # Combined intensity (weighted average of RMS and peak)
        event.intensity = round(0.7 * intensity + 0.3 * peak_norm, 3)

        # Classify
        if event.intensity >= big_threshold:
            event.intensity_category = "big_laugh"
        elif event.intensity >= medium_threshold:
            event.intensity_category = "medium_laugh"
        else:
            event.intensity_category = "chuckle"

    counts = {
        "big_laugh": sum(1 for e in events if e.intensity_category == "big_laugh"),
        "medium_laugh": sum(1 for e in events if e.intensity_category == "medium_laugh"),
        "chuckle": sum(1 for e in events if e.intensity_category == "chuckle"),
    }
    print(f"  Intensity: {counts['big_laugh']} big, {counts['medium_laugh']} medium, {counts['chuckle']} chuckle")

    return events


def validate_laughter_spectral(
    audio_path: str,
    events: list[LaughterEvent],
    min_centroid: float = 500.0,
    max_centroid: float = 4000.0,
) -> list[LaughterEvent]:
    """
    Validate laughter events using spectral features.

    Laughter has characteristic spectral properties:
    - Spectral centroid typically 500-4000 Hz
    - Rhythmic energy pattern (ha-ha-ha)
    - Distinguishable from applause (which is more broadband)
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    validated = []
    rejected = 0

    for event in events:
        start_sample = int(event.start * sr)
        end_sample = int(event.end * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.1:  # Too short for spectral analysis
            event.spectral_valid = True
            validated.append(event)
            continue

        # Compute spectral centroid
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
        mean_centroid = float(np.mean(centroid))
        event.spectral_centroid = mean_centroid

        # Validate: laughter typically in 500-4000 Hz range
        if min_centroid <= mean_centroid <= max_centroid:
            event.spectral_valid = True
            validated.append(event)
        else:
            # Check if it could still be laughter (borderline cases)
            # Allow slightly out of range with high confidence
            if event.confidence > 0.7:
                event.spectral_valid = True
                validated.append(event)
            else:
                event.spectral_valid = False
                rejected += 1

    if rejected:
        print(f"  Spectral validation: rejected {rejected} events")

    return validated
