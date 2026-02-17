"""Full pipeline orchestrator — runs Phase 1 through Phase 8."""

from __future__ import annotations

from dataclasses import asdict

from .phase1_acquisition import download_audio, download_metadata, get_transcript
from .phase2_preprocessing import separate_sources
from .phase3_detection import (
    detect_with_gillick,
    detect_with_yamnet,
    ensemble_detection,
    load_yamnet,
)
from .phase4_postprocessing import (
    compute_intensity,
    filter_by_duration,
    merge_close_events,
    validate_laughter_spectral,
)
from .phase5_timeline import build_unified_timeline, compute_latencies
from .phase6_qa import sample_for_verification
from .phase7_segmentation import build_paragraphs, find_paragraph_breaks
from .phase8_transcript import compute_summary, export_dataset


def run_full_pipeline(
    video_id: str,
    output_dir: str = "./dataset",
    skip_qa: bool = False,
) -> dict[str, str]:
    """
    Run the complete pipeline (Phase 1-8): context-based paragraph segmentation.

    Args:
        video_id: YouTube video ID
        output_dir: directory for output files
        skip_qa: if True, skip Phase 6 QA sampling

    Returns:
        dict with paths to output files: clean, detailed, json
    """
    # Phase 1: Data Acquisition
    print("[Phase 1] Downloading audio & metadata...")
    audio_path = download_audio(video_id)
    metadata_obj = download_metadata(video_id)
    metadata = asdict(metadata_obj)
    transcript_segments, transcript_type = get_transcript(video_id)
    metadata["transcript_type"] = transcript_type

    # Phase 2: Audio Preprocessing
    print("[Phase 2] Separating sources...")
    sources = separate_sources(audio_path)
    audience_audio = sources.get("no_vocals", audio_path)

    # Phase 3: Laughter Detection
    print("[Phase 3] Detecting laughter (ensemble)...")
    yamnet_model, class_names = load_yamnet()
    yamnet_events = detect_with_yamnet(audience_audio, yamnet_model, class_names)
    gillick_events = detect_with_gillick(audience_audio)
    laughter_events = ensemble_detection(yamnet_events, gillick_events)

    # Phase 4: Post-Processing
    print("[Phase 4] Post-processing laughter events...")
    laughter_events = merge_close_events(laughter_events)
    laughter_events = filter_by_duration(laughter_events)
    laughter_events = compute_intensity(audience_audio, laughter_events)
    laughter_events = validate_laughter_spectral(audience_audio, laughter_events)
    for i, event in enumerate(laughter_events):
        event.event_id = i + 1

    # Phase 5: Build Unified Timeline
    print("[Phase 5] Building unified timeline...")
    timeline = build_unified_timeline(transcript_segments, laughter_events)
    timeline = compute_latencies(timeline)

    # Phase 6: Quality Assurance
    print("[Phase 6] Quality check...")
    if not skip_qa and laughter_events:
        sample_for_verification(laughter_events)

    # Phase 7: Context-Based Paragraph Segmentation
    print("[Phase 7] Segmenting paragraphs by context/topic...")
    break_points = find_paragraph_breaks(timeline)
    paragraphs = build_paragraphs(timeline, break_points)

    # Phase 8: Generate Annotated Transcript
    print("[Phase 8] Generating annotated transcript...")
    laughter_dicts = [asdict(e) for e in laughter_events]
    summary_stats = compute_summary(
        paragraphs, laughter_dicts, metadata.get("duration", 0)
    )

    paths = export_dataset(
        video_id=video_id,
        metadata=metadata,
        paragraphs=paragraphs,
        laughter_events=laughter_dicts,
        summary_stats=summary_stats,
        output_dir=output_dir,
    )

    with_laughs = sum(1 for p in paragraphs if p["has_laughs"])
    without_laughs = sum(1 for p in paragraphs if not p["has_laughs"])
    print(f"\nDone!")
    print(f"  {len(paragraphs)} paragraphs ({with_laughs} with laughs, {without_laughs} setup/transition)")
    print(f"  {len(laughter_events)} laughter events total")
    print(f"  Output: {paths}")

    return paths
