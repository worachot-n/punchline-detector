"""Pipeline runner with parameter injection and progress callbacks for the web app."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from comedy_pipeline.phase1_acquisition import download_audio, download_metadata, get_transcript
from comedy_pipeline.phase2_preprocessing import separate_sources
from comedy_pipeline.phase3_detection import (
    detect_with_gillick,
    detect_with_yamnet,
    ensemble_detection,
    load_yamnet,
)
from comedy_pipeline.phase4_postprocessing import (
    compute_intensity,
    filter_by_duration,
    merge_close_events,
    validate_laughter_spectral,
)
from comedy_pipeline.phase5_timeline import build_unified_timeline, compute_latencies
from comedy_pipeline.phase6_qa import sample_for_verification
from comedy_pipeline.phase7_segmentation import build_paragraphs, find_paragraph_breaks
from comedy_pipeline.phase8_transcript import compute_summary, export_dataset


def run_pipeline_with_params(
    video_id: str,
    output_dir: str = "./dataset",
    skip_qa: bool = False,
    detection_mode: str = "ensemble",
    phase3_params: dict | None = None,
    phase4_params: dict | None = None,
    phase7_params: dict | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, str]:
    """
    Run the full pipeline with configurable parameters.

    detection_mode: "yamnet" | "gillick" | "ensemble"
    """
    p3 = phase3_params or {}
    p4 = phase4_params or {}
    p7 = phase7_params or {}

    def cb(phase: str, pct: int) -> None:
        if progress_callback:
            progress_callback(phase, pct)

    # Phase 1
    cb("Phase 1: Downloading audio & metadata...", 5)
    audio_path = download_audio(video_id)
    metadata_obj = download_metadata(video_id)
    metadata = asdict(metadata_obj)
    transcript_segments, transcript_type = get_transcript(video_id)
    metadata["transcript_type"] = transcript_type

    # Phase 2
    cb("Phase 2: Separating audio sources...", 15)
    sources = separate_sources(audio_path)
    audience_audio = sources.get("no_vocals", audio_path)

    # Phase 3 — detection mode branching
    cb("Phase 3: Detecting laughter...", 30)
    yamnet_model, class_names = load_yamnet()

    if detection_mode == "yamnet":
        laughter_events = detect_with_yamnet(
            audience_audio,
            yamnet_model,
            class_names,
            confidence_threshold=p3.get("yamnet_confidence_threshold", 0.3),
            frame_duration=p3.get("yamnet_frame_duration", 0.96),
            hop_duration=p3.get("yamnet_hop_duration", 0.48),
        )
    elif detection_mode == "gillick":
        laughter_events = detect_with_gillick(
            audience_audio,
            threshold=p3.get("gillick_threshold", 0.5),
            min_length=p3.get("gillick_min_length", 0.2),
        )
    else:  # ensemble
        yamnet_events = detect_with_yamnet(
            audience_audio,
            yamnet_model,
            class_names,
            confidence_threshold=p3.get("yamnet_confidence_threshold", 0.3),
            frame_duration=p3.get("yamnet_frame_duration", 0.96),
            hop_duration=p3.get("yamnet_hop_duration", 0.48),
        )
        gillick_events = detect_with_gillick(
            audience_audio,
            threshold=p3.get("gillick_threshold", 0.5),
            min_length=p3.get("gillick_min_length", 0.2),
        )
        laughter_events = ensemble_detection(
            yamnet_events,
            gillick_events,
            overlap_threshold=p3.get("ensemble_overlap_threshold", 0.3),
        )

    # Phase 4
    cb("Phase 4: Post-processing laughter events...", 50)
    laughter_events = merge_close_events(laughter_events, max_gap=p4.get("max_gap", 0.1))
    laughter_events = filter_by_duration(
        laughter_events,
        min_duration=p4.get("min_duration", 0.3),
        max_duration=p4.get("max_duration", 30.0),
    )
    laughter_events = compute_intensity(
        audience_audio,
        laughter_events,
        big_threshold=p4.get("big_threshold", 0.7),
        medium_threshold=p4.get("medium_threshold", 0.4),
    )
    laughter_events = validate_laughter_spectral(
        audience_audio,
        laughter_events,
        min_centroid=p4.get("min_centroid", 500.0),
        max_centroid=p4.get("max_centroid", 4000.0),
    )
    for i, event in enumerate(laughter_events):
        event.event_id = i + 1

    # Phase 5
    cb("Phase 5: Building unified timeline...", 65)
    timeline = build_unified_timeline(transcript_segments, laughter_events)
    timeline = compute_latencies(timeline)

    # Phase 6
    cb("Phase 6: Quality assurance...", 75)
    if not skip_qa and laughter_events:
        sample_for_verification(laughter_events)

    # Phase 7
    cb("Phase 7: Segmenting paragraphs...", 82)
    break_points = find_paragraph_breaks(
        timeline,
        vote_threshold=p7.get("vote_threshold", 0.25),
        min_paragraph_duration=p7.get("min_paragraph_duration", 5.0),
        max_paragraph_duration=p7.get("max_paragraph_duration", 30.0),
        min_pause=p7.get("min_pause", 2.0),
        shift_threshold=p7.get("shift_threshold", 0.55),
        min_post_laugh_pause=p7.get("min_post_laugh_pause", 2.0),
    )
    paragraphs = build_paragraphs(timeline, break_points)

    # Phase 8
    cb("Phase 8: Generating transcript & exporting...", 92)
    laughter_dicts = [asdict(e) for e in laughter_events]
    summary_stats = compute_summary(paragraphs, laughter_dicts, metadata.get("duration", 0))
    paths = export_dataset(
        video_id=video_id,
        metadata=metadata,
        paragraphs=paragraphs,
        laughter_events=laughter_dicts,
        summary_stats=summary_stats,
        output_dir=output_dir,
    )

    return paths
