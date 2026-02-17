"""Phase 8: Annotated Transcript Generation and Export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate_annotated_transcript(
    paragraphs: list[dict],
    metadata: dict,
    summary_stats: dict,
) -> str:
    """
    Generate a clean annotated transcript.

    - Paragraphs divided by context/topic (not by laugh)
    - [laugh] tags inline within paragraphs
    - Some paragraphs have no laughs (setup/transition)
    """
    lines = []

    # === Header ===
    lines.append(f"[COMEDIAN] {metadata.get('comedian', 'Unknown')}")

    special_line = f"[SPECIAL] {metadata.get('special_name', 'Unknown')}"
    if metadata.get("year"):
        special_line += f" ({metadata['year']})"
    lines.append(special_line)

    lines.append(f"[VIDEO_ID] {metadata.get('video_id', '')}")
    lines.append(
        f"[TOTAL_LAUGHS] {summary_stats.get('total_laughs', 0)} | "
        f"[LAUGHS_PER_MIN] {summary_stats.get('laughs_per_minute', 0)} | "
        f"[LAUGH_TIME] {summary_stats.get('laugh_time_percentage', 0)}%"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # === Body ===
    for para in paragraphs:
        lines.append(para["annotated_text"])
        lines.append("")

    # === Footer ===
    lines.append("---")
    lines.append("")
    lines.append("[STATS]")
    lines.append(f"Total paragraphs: {summary_stats.get('total_paragraphs', 0)}")
    lines.append(f"Paragraphs with laughs: {summary_stats.get('paragraphs_with_laughs', 0)}")
    lines.append(f"Paragraphs without laughs: {summary_stats.get('paragraphs_without_laughs', 0)}")
    lines.append(f"Total laughs: {summary_stats.get('total_laughs', 0)}")
    lines.append(
        f"Big laughs: {summary_stats.get('big_laughs', 0)} | "
        f"Medium laughs: {summary_stats.get('medium_laughs', 0)} | "
        f"Chuckles: {summary_stats.get('chuckles', 0)}"
    )
    lines.append(f"Avg laugh duration: {summary_stats.get('avg_laugh_duration', 0)}s")
    lines.append(f"Avg latency: {summary_stats.get('avg_latency', 0)}s")
    lines.append(f"Laughs per minute: {summary_stats.get('laughs_per_minute', 0)}")

    return "\n".join(lines)


def generate_detailed_transcript(
    paragraphs: list[dict],
    metadata: dict,
    summary_stats: dict,
) -> str:
    """Generate detailed transcript with timestamps and paragraph metadata."""
    lines = []

    # Header
    lines.append(f"[COMEDIAN] {metadata.get('comedian', 'Unknown')}")
    lines.append(f"[SPECIAL] {metadata.get('special_name', 'Unknown')}")
    lines.append(f"[VIDEO_ID] {metadata.get('video_id', '')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for para in paragraphs:
        start = _format_timestamp(para["start_time"])
        end = _format_timestamp(para["end_time"])
        laugh_info = (
            f" | laughs: {para['laugh_count']}"
            if para["has_laughs"]
            else " | (setup/transition)"
        )
        lines.append(f"[P{para['paragraph_id']}] [{start} -> {end}]{laugh_info}")
        lines.append(para["annotated_text"])
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    for key, value in summary_stats.items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def compute_summary(
    paragraphs: list[dict],
    laughter_events: list[dict],
    total_duration: float,
) -> dict:
    """Compute summary statistics including paragraph info."""
    durations = [e.get("duration", 0) for e in laughter_events]
    intensities = [e.get("intensity", 0) for e in laughter_events]
    latencies = [e.get("latency", 0) for e in laughter_events if e.get("latency", 0) > 0]

    return {
        "total_paragraphs": len(paragraphs),
        "paragraphs_with_laughs": sum(1 for p in paragraphs if p["has_laughs"]),
        "paragraphs_without_laughs": sum(1 for p in paragraphs if not p["has_laughs"]),
        "total_laughs": len(laughter_events),
        "laughs_per_minute": (
            round(len(laughter_events) / (total_duration / 60), 2)
            if total_duration
            else 0
        ),
        "total_laugh_time_sec": round(sum(durations), 1),
        "laugh_time_percentage": (
            round(sum(durations) / total_duration * 100, 1)
            if total_duration
            else 0
        ),
        "avg_laugh_duration": round(float(np.mean(durations)), 2) if durations else 0,
        "max_laugh_duration": round(max(durations), 2) if durations else 0,
        "avg_intensity": round(float(np.mean(intensities)), 3) if intensities else 0,
        "avg_latency": round(float(np.mean(latencies)), 3) if latencies else 0,
        "big_laughs": sum(
            1 for e in laughter_events if e.get("intensity_category") == "big_laugh"
        ),
        "medium_laughs": sum(
            1 for e in laughter_events if e.get("intensity_category") == "medium_laugh"
        ),
        "chuckles": sum(
            1 for e in laughter_events if e.get("intensity_category") == "chuckle"
        ),
    }


def export_dataset(
    video_id: str,
    metadata: dict,
    paragraphs: list[dict],
    laughter_events: list[dict],
    summary_stats: dict,
    output_dir: str = "./dataset",
) -> dict[str, str]:
    """
    Export 3 files:
    1. {video_id}.txt          — clean annotated transcript
    2. {video_id}_detailed.txt — annotated transcript + timestamps
    3. {video_id}.json         — full structured data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Clean transcript
    clean_text = generate_annotated_transcript(paragraphs, metadata, summary_stats)
    clean_path = f"{output_dir}/{video_id}.txt"
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    # 2. Detailed transcript
    detailed_text = generate_detailed_transcript(paragraphs, metadata, summary_stats)
    detailed_path = f"{output_dir}/{video_id}_detailed.txt"
    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write(detailed_text)

    # 3. Full JSON
    json_data = {
        "video_id": video_id,
        "video_url": f"https://youtube.com/watch?v={video_id}",
        "comedian": metadata.get("comedian", "Unknown"),
        "special_name": metadata.get("special_name", "Unknown"),
        "year": metadata.get("year"),
        "total_duration_sec": metadata.get("duration", 0),
        "transcript_type": metadata.get("transcript_type", "unknown"),
        "annotated_transcript": clean_text,
        "paragraphs": [
            {k: v for k, v in p.items() if k != "annotated_text"}
            for p in paragraphs
        ],
        "laughter_events": laughter_events,
        "summary_stats": summary_stats,
    }
    json_path = f"{output_dir}/{video_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"  Exported: {clean_path}, {detailed_path}, {json_path}")
    return {"clean": clean_path, "detailed": detailed_path, "json": json_path}
