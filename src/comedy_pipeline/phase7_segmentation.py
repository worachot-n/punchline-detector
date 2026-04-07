"""Phase 7: Context-Based Paragraph Segmentation.

Segments the transcript into paragraphs based on comedy bit structure
(topic/context), NOT based on laughter positions. Uses 4 signals:
1. Long pause detection
2. Topic transition phrase detection
3. Semantic shift detection (sentence embeddings)
4. Post-big-laugh pause detection

These signals are combined via weighted voting to find paragraph breaks.
"""

from __future__ import annotations

import re

import numpy as np

from .models import TimelineEntry


# === Signal 1: Long Pause Detection ===

def detect_long_pauses(
    timeline: list[TimelineEntry],
    min_pause: float = 2.0,
) -> list[float]:
    """
    Find points where the comedian pauses for > min_pause seconds.

    Does not count pauses that are filled by audience laughter,
    since the comedian typically waits for laughter to subside
    before continuing the same bit.

    Returns list of timestamps where a paragraph break should occur.
    """
    speeches = [e for e in timeline if e.entry_type == "speech"]
    laughters = [e for e in timeline if e.entry_type == "laughter"]

    break_points = []

    for i in range(1, len(speeches)):
        gap_start = speeches[i - 1].end
        gap_end = speeches[i].start
        gap = gap_end - gap_start

        if gap < min_pause:
            continue

        # Check if laughter fills the gap
        laugh_in_gap = any(
            l.start >= gap_start - 0.5 and l.end <= gap_end + 0.5
            for l in laughters
        )

        if laugh_in_gap:
            # Laughter fills the gap — only break if silence remains after laugh
            laugh_dur = sum(
                l.duration for l in laughters
                if l.start >= gap_start and l.end <= gap_end
            )
            remaining_silence = gap - laugh_dur
            if remaining_silence > min_pause:
                break_points.append(speeches[i].start)
        else:
            # Pure silence = clear topic change
            break_points.append(speeches[i].start)

    return break_points


# === Signal 2: Topic Transition Phrase Detection ===

TRANSITION_PATTERNS = [
    # Clear topic changes
    r"\b(anyway|anyhow|moving on|so anyway)\b",
    r"\b(but (you know )?what|here'?s the thing)\b",
    r"\b(speaking of|talking about|that reminds me)\b",
    r"\b(let me tell you|i('ll| will) tell you)\b",
    r"\b(so (check this out|get this|listen|look))\b",
    r"\b(now[,.]?\s+(here'?s|the other))\b",

    # Opening a new topic
    r"\b(have you ever (noticed|thought|seen))\b",
    r"\b(you (ever notice|know what|wanna know))\b",
    r"\b(i was (at|in|on|walking|driving|sitting))\b",
    r"\b(the other day|last week|last night|yesterday)\b",
    r"\b(my (wife|husband|kid|son|daughter|mom|dad|friend))\b",
]


def detect_transition_phrases(
    timeline: list[TimelineEntry],
) -> list[float]:
    """
    Find speech entries that start with a transition phrase.

    Transition phrases signal that the comedian is opening a new bit
    or changing topic. Examples: "Anyway...", "So check this out...",
    "Have you ever noticed..."

    Returns list of timestamps.
    """
    break_points = []

    for entry in timeline:
        if entry.entry_type != "speech":
            continue

        text_lower = entry.text.lower().strip()
        for pattern in TRANSITION_PATTERNS:
            if re.match(pattern, text_lower):
                break_points.append(entry.start)
                break

    return break_points


# === Signal 3: Semantic Shift Detection ===

def detect_semantic_shifts(
    timeline: list[TimelineEntry],
    window_size: int = 5,
    shift_threshold: float = 0.55,
) -> list[float]:
    """
    Use sentence embeddings to detect topic changes.

    Compares sliding windows of sentences. When cosine similarity
    between consecutive windows drops below threshold, it indicates
    the comedian has shifted to a new topic.

    window_size: number of sentences per window
    shift_threshold: similarity below this = topic change
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("  Warning: sentence-transformers not available, skipping semantic shift detection")
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")

    speeches = [e for e in timeline if e.entry_type == "speech" and e.text.strip()]
    if len(speeches) < window_size * 2:
        return []

    # Build sliding windows
    windows = []
    for i in range(len(speeches) - window_size + 1):
        window_speeches = speeches[i : i + window_size]
        text = " ".join(s.text for s in window_speeches)
        windows.append({
            "text": text,
            "center_time": window_speeches[window_size // 2].start,
            "boundary_time": window_speeches[-1].end,
        })

    # Compute embeddings
    embeddings = model.encode([w["text"] for w in windows])

    # Compare consecutive windows
    break_points = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(
            embeddings[i - 1].reshape(1, -1),
            embeddings[i].reshape(1, -1),
        )[0][0]

        if sim < shift_threshold:
            break_points.append(windows[i]["center_time"])

    return break_points


# === Signal 4: Post-Big-Laugh Pause ===

def detect_post_laugh_breaks(
    timeline: list[TimelineEntry],
    min_post_laugh_pause: float = 2.0,
) -> list[float]:
    """
    Detect topic changes after big laughs.

    After a big_laugh, if the comedian pauses for > min_post_laugh_pause
    seconds before speaking again, they are likely transitioning to a new bit.

    Only considers big_laugh events (not medium or chuckle), because
    the comedian usually continues the same bit after smaller laughs.
    """
    break_points = []

    for i, entry in enumerate(timeline):
        if entry.entry_type != "laughter":
            continue
        if entry.intensity_category != "big_laugh":
            continue

        # Find next speech entry
        for j in range(i + 1, len(timeline)):
            if timeline[j].entry_type == "speech":
                pause = timeline[j].start - entry.end
                if pause > min_post_laugh_pause:
                    break_points.append(timeline[j].start)
                break

    return break_points


# === Weighted Voting: Combine All Signals ===

def find_paragraph_breaks(
    timeline: list[TimelineEntry],
    weights: dict[str, float] | None = None,
    vote_threshold: float = 0.25,
    min_paragraph_duration: float = 5.0,
    max_paragraph_duration: float = 30.0,
) -> list[float]:
    """
    Combine all 4 signals using weighted voting to find paragraph breaks.

    Each speech timestamp receives a score from the signals that detected
    a break near it. If the combined score exceeds vote_threshold,
    a paragraph break is placed there.

    Post-processing:
    - Removes breaks that are too close together (< min_paragraph_duration)
    - Force-cuts paragraphs that are too long (> max_paragraph_duration)
    """
    if weights is None:
        weights = {
            "long_pause": 0.5,            # Strongest signal
            "transition_phrase": 0.3,      # Strong but may false-positive
            "semantic_shift": 0.3,         # Strong but needs large window
            "post_big_laugh_pause": 0.25,  # Medium — not every big laugh = topic change
        }

    # Collect signals from all detectors
    signals = {
        "long_pause": detect_long_pauses(timeline),
        "transition_phrase": detect_transition_phrases(timeline),
        "semantic_shift": detect_semantic_shifts(timeline),
        "post_big_laugh_pause": detect_post_laugh_breaks(timeline),
    }

    # All speech timestamps as candidate break points
    speech_times = [e.start for e in timeline if e.entry_type == "speech"]
    if not speech_times:
        return []

    # Score each speech timestamp
    scores: dict[float, dict] = {}
    for t in speech_times:
        score = 0.0
        matched_signals = []

        for signal_name, break_points in signals.items():
            # Check if any break point is near this timestamp (within 1.5s)
            if any(abs(bp - t) < 1.5 for bp in break_points):
                score += weights[signal_name]
                matched_signals.append(signal_name)

        if score > 0:
            scores[t] = {
                "score": score,
                "signals": matched_signals,
            }

    # Select break points above threshold
    break_points = sorted([
        t for t, info in scores.items()
        if info["score"] >= vote_threshold
    ])

    # === Post-processing ===

    # 1. Remove breaks too close together
    filtered = []
    for bp in break_points:
        if not filtered or (bp - filtered[-1]) >= min_paragraph_duration:
            filtered.append(bp)
    break_points = filtered

    # 2. Force-cut paragraphs that are too long
    final_breaks = []
    last_break = speech_times[0] if speech_times else 0

    for bp in break_points:
        if bp - last_break > max_paragraph_duration:
            mid_breaks = _find_best_pause_in_range(timeline, last_break, bp)
            final_breaks.extend(mid_breaks)
        final_breaks.append(bp)
        last_break = bp

    print(f"  Segmentation: {len(final_breaks)} paragraph breaks found")
    return final_breaks


def _find_best_pause_in_range(
    timeline: list[TimelineEntry],
    start: float,
    end: float,
) -> list[float]:
    """
    Find the longest pause within [start, end] and break there.

    Used when a paragraph exceeds max_paragraph_duration.
    """
    speeches = [
        e for e in timeline
        if e.entry_type == "speech" and start <= e.start <= end
    ]

    if len(speeches) < 2:
        return []

    # Find the longest gap
    gaps = []
    for i in range(1, len(speeches)):
        gap = speeches[i].start - speeches[i - 1].end
        gaps.append((gap, speeches[i].start))

    gaps.sort(reverse=True)

    if gaps and gaps[0][0] > 1.0:
        return [gaps[0][1]]
    return []


# === Build Paragraphs ===

def build_paragraphs(
    timeline: list[TimelineEntry],
    break_points: list[float],
) -> list[dict]:
    """
    Split the timeline into paragraphs based on break points.

    Laughter tags are placed inline within paragraphs (not at the end).
    Some paragraphs may have no laughter (setup/transition).
    """
    speeches = [e for e in timeline if e.entry_type == "speech"]
    if not speeches:
        return []

    boundaries = [speeches[0].start] + sorted(break_points) + [float("inf")]
    paragraphs = []

    for seg_idx in range(len(boundaries) - 1):
        seg_start = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]

        # Collect timeline entries in this segment
        segment_entries = [
            e for e in timeline
            if e.start >= seg_start - 0.1 and e.start < seg_end - 0.1
        ]

        seg_speeches = [e for e in segment_entries if e.entry_type == "speech"]
        seg_laughs = [e for e in segment_entries if e.entry_type == "laughter"]

        if not seg_speeches:
            continue

        # Build annotated text with inline laugh tags
        annotated_text, inline_laughs = build_annotated_text(seg_speeches, seg_laughs)

        paragraphs.append({
            "paragraph_id": len(paragraphs) + 1,
            "text": " ".join(s.text for s in seg_speeches),
            "annotated_text": annotated_text,
            "start_time": seg_speeches[0].start,
            "end_time": seg_speeches[-1].end,
            "segment_reason": "topic_shift",
            "inline_laughs": inline_laughs,
            "laugh_count": len(seg_laughs),
            "has_laughs": len(seg_laughs) > 0,
        })

    with_laughs = sum(1 for p in paragraphs if p["has_laughs"])
    without_laughs = sum(1 for p in paragraphs if not p["has_laughs"])
    print(f"  Paragraphs: {len(paragraphs)} total ({with_laughs} with laughs, {without_laughs} setup/transition)")

    return paragraphs


def build_annotated_text(
    speeches: list[TimelineEntry],
    laughs: list[TimelineEntry],
) -> tuple[str, list[dict]]:
    """
    Build text with inline [big_laugh] [medium_laugh] [chuckle] tags.

    Laughter tags are placed after the last speech segment that was
    spoken before the laughter occurred.
    """
    # Merge speeches + laughs and sort by time
    events = []
    for s in speeches:
        events.append(("speech", s))
    for l in laughs:
        events.append(("laugh", l))

    events.sort(key=lambda x: (x[1].start, 0 if x[0] == "speech" else 1))

    # Build text
    parts: list[str] = []
    inline_laughs: list[dict] = []
    char_offset = 0

    for event_type, entry in events:
        if event_type == "speech":
            text = entry.text.strip()
            if text:
                if parts and not parts[-1].startswith("[") and not parts[-1].startswith("\n"):
                    parts.append(" ")
                    char_offset += 1
                parts.append(text)
                char_offset += len(text)

        elif event_type == "laugh":
            tag = f"[{entry.intensity_category}]"
            parts.append(f"\n{tag}\n")

            # Find preceding speech text for context
            preceding_text = ""
            for p in reversed(parts[:-1]):
                if not p.startswith("[") and not p.startswith("\n") and p.strip():
                    preceding_text = p.strip()
                    break

            inline_laughs.append({
                "type": entry.intensity_category,
                "position_after_text": preceding_text[-60:] if preceding_text else "",
                "char_offset": char_offset,
                "timestamp": entry.start,
                "intensity": entry.intensity,
                "duration": entry.duration,
                "latency": entry.latency,
            })

    annotated_text = "".join(parts).strip()

    # Clean up whitespace
    annotated_text = re.sub(r" +", " ", annotated_text)
    annotated_text = re.sub(r"\n{3,}", "\n\n", annotated_text)

    return annotated_text, inline_laughs
