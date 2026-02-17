"""Phase 6: Quality Assurance — sampling and manual verification support."""

from __future__ import annotations

import random

from .models import LaughterEvent


def sample_for_verification(
    events: list[LaughterEvent],
    sample_ratio: float = 0.1,
    min_samples: int = 5,
    max_samples: int = 20,
    seed: int = 42,
) -> list[LaughterEvent]:
    """
    Select a random sample of laughter events for manual verification.

    Returns a subset of events that should be manually checked to
    estimate precision and recall of the detection pipeline.
    """
    n_samples = max(min_samples, int(len(events) * sample_ratio))
    n_samples = min(n_samples, max_samples, len(events))

    rng = random.Random(seed)
    sample = rng.sample(events, n_samples)
    sample.sort(key=lambda e: e.start)

    print(f"  QA: Selected {n_samples}/{len(events)} events for verification")
    for i, evt in enumerate(sample, 1):
        print(
            f"    [{i}] {evt.start:.1f}s-{evt.end:.1f}s "
            f"({evt.intensity_category}, conf={evt.confidence:.2f})"
        )

    return sample


def compute_precision(
    verified_correct: int,
    total_verified: int,
) -> float:
    """Compute precision from manual verification results."""
    if total_verified == 0:
        return 0.0
    return verified_correct / total_verified


def generate_qa_report(
    events: list[LaughterEvent],
    sample: list[LaughterEvent],
    verified_correct: int | None = None,
) -> dict:
    """Generate a QA report summarizing detection quality."""
    report = {
        "total_events": len(events),
        "sample_size": len(sample),
        "sample_events": [
            {
                "event_id": e.event_id,
                "start": e.start,
                "end": e.end,
                "duration": e.duration,
                "intensity_category": e.intensity_category,
                "confidence": e.confidence,
                "source": e.source,
            }
            for e in sample
        ],
    }

    if verified_correct is not None:
        report["precision"] = compute_precision(verified_correct, len(sample))
        report["estimated_true_positives"] = int(
            len(events) * report["precision"]
        )

    # Confidence distribution
    confidences = [e.confidence for e in events]
    if confidences:
        report["confidence_stats"] = {
            "mean": round(sum(confidences) / len(confidences), 3),
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
        }

    # Source distribution
    sources = {}
    for e in events:
        sources[e.source] = sources.get(e.source, 0) + 1
    report["source_distribution"] = sources

    return report
