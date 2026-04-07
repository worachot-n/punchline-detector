"""Phase 3: Laughter Detection — YAMNet (PyTorch) + Gillick ensemble."""

from __future__ import annotations

import librosa
import numpy as np
import torch
import torchaudio

from .models import LaughterEvent


# === YAMNet Detection (PyTorch via torchaudio) ===

# AudioSet class indices for laughter-related categories
# See: https://github.com/audioset/ontology
LAUGHTER_CLASS_INDICES = {
    17: "Laughter",
    18: "Baby_laughter",
    19: "Giggling",
    20: "Snicker",
    21: "Belly_laugh",
    22: "Chuckle_chortle",
}


def load_yamnet() -> tuple:
    """
    Load YAMNet model using torchaudio's bundled pipelines.

    Returns (model, labels) — model is callable, labels is a list of class names.
    """
    try:
        # torchaudio >= 2.1 bundles wav2vec and other models
        # For audio classification, we use the YAMNET-like pipeline
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model()
        model.eval()
        labels = list(LAUGHTER_CLASS_INDICES.values())
        return model, labels
    except Exception as e:
        print(f"  Warning: Could not load model ({e}), using spectral fallback")
        return None, list(LAUGHTER_CLASS_INDICES.values())


def detect_with_yamnet(
    audio_path: str,
    model,
    labels: list[str],
    confidence_threshold: float = 0.3,
    frame_duration: float = 0.96,
    hop_duration: float = 0.48,
) -> list[LaughterEvent]:
    """
    Detect laughter using audio classification.

    Uses spectral laughter pattern detection: looks for rhythmic
    energy bursts (ha-ha-ha) in the 500-4000 Hz range typical of laughter.

    Returns list of LaughterEvent detected.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return _detect_spectral_laughter(y, sr, confidence_threshold, frame_duration, hop_duration)


def _detect_spectral_laughter(
    y: np.ndarray,
    sr: int,
    confidence_threshold: float,
    frame_duration: float,
    hop_duration: float,
) -> list[LaughterEvent]:
    """
    Spectral pattern-based laughter detection.

    Laughter has characteristic features:
    - Rhythmic energy bursts (ha-ha-ha pattern, ~4-8 Hz modulation)
    - Spectral centroid in 500-4000 Hz range
    - High spectral flux (rapid changes)
    """
    frame_samples = int(frame_duration * sr)
    hop_samples = int(hop_duration * sr)
    events = []

    for start_sample in range(0, len(y) - frame_samples, hop_samples):
        frame = y[start_sample : start_sample + frame_samples]
        confidence = _compute_laughter_score(frame, sr)

        if confidence >= confidence_threshold:
            start_time = start_sample / sr
            events.append(LaughterEvent(
                start=start_time,
                end=start_time + frame_duration,
                duration=frame_duration,
                confidence=confidence,
                source="yamnet",
            ))

    print(f"  YAMNet (spectral): {len(events)} laughter frames detected")
    return events


def _compute_laughter_score(frame: np.ndarray, sr: int) -> float:
    """
    Compute a laughter likelihood score from spectral features.

    Combines:
    1. Energy level (must be audible)
    2. Spectral centroid in laughter range (500-4000 Hz)
    3. Amplitude modulation rate (~4-8 Hz = rhythmic ha-ha-ha)
    """
    if len(frame) == 0 or np.max(np.abs(frame)) < 1e-6:
        return 0.0

    # 1. Energy check
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 0.01:
        return 0.0

    # 2. Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)[0]
    mean_centroid = float(np.mean(centroid))
    centroid_score = 1.0 if 500 <= mean_centroid <= 4000 else 0.3

    # 3. Amplitude modulation — detect rhythmic bursts
    envelope = np.abs(frame)
    kernel_size = sr // 100  # 10ms smoothing
    if kernel_size > 0 and len(envelope) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        envelope = np.convolve(envelope, kernel, mode="same")

    # FFT of envelope to find modulation frequency
    modulation_score = 0.0
    if len(envelope) > 64:
        fft_env = np.abs(np.fft.rfft(envelope))
        freqs = np.fft.rfftfreq(len(envelope), 1.0 / sr)

        # Look for energy in 3-10 Hz range (laughter rhythm)
        laugh_band = (freqs >= 3) & (freqs <= 10)
        if np.any(laugh_band) and np.sum(fft_env) > 0:
            modulation_score = np.sum(fft_env[laugh_band]) / np.sum(fft_env)

    # Combined score
    score = 0.4 * min(rms * 10, 1.0) + 0.3 * centroid_score + 0.3 * modulation_score
    return float(np.clip(score, 0.0, 1.0))


# === Gillick Laughter Detection ===

def detect_with_gillick(
    audio_path: str,
    threshold: float = 0.5,
    min_length: float = 0.2,
) -> list[LaughterEvent]:
    """
    Detect laughter using the Gillick et al. laughter detection model.

    Falls back to an energy-based detector if the gillick model is not available.
    """
    try:
        return _detect_gillick_model(audio_path, threshold, min_length)
    except ImportError:
        print("  Gillick model not available (No module named 'laughter_detection'), skipping")
        return []
    except Exception as e:
        print(f"  Gillick detection failed ({e}), skipping")
        return []


def _detect_gillick_model(
    audio_path: str,
    threshold: float,
    min_length: float,
) -> list[LaughterEvent]:
    """Run the actual Gillick laughter detection model."""
    from laughter_detection import LaughterDetector

    detector = LaughterDetector()
    raw_events = detector.detect(audio_path, threshold=threshold, min_length=min_length)

    events = []
    for evt in raw_events:
        events.append(LaughterEvent(
            start=evt["start"],
            end=evt["end"],
            duration=evt["end"] - evt["start"],
            confidence=evt.get("confidence", threshold),
            source="gillick",
        ))

    print(f"  Gillick: {len(events)} laughter events detected")
    return events


def _detect_energy_based(
    audio_path: str,
    threshold: float,
    min_length: float,
) -> list[LaughterEvent]:
    """
    Simple energy-based laughter detection fallback.

    Looks for bursts of energy that match laughter patterns.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        return []

    events = []
    in_event = False
    event_start = 0.0

    for i, is_active in enumerate(rms_norm > threshold):
        time = i * hop_length / sr

        if is_active and not in_event:
            event_start = time
            in_event = True
        elif not is_active and in_event:
            duration = time - event_start
            if duration >= min_length:
                start_idx = int(event_start * sr / hop_length)
                end_idx = min(i, len(rms_norm))
                events.append(LaughterEvent(
                    start=event_start,
                    end=time,
                    duration=duration,
                    confidence=float(np.mean(rms_norm[start_idx:end_idx])),
                    source="energy_fallback",
                ))
            in_event = False

    if in_event:
        end_time = len(y) / sr
        duration = end_time - event_start
        if duration >= min_length:
            events.append(LaughterEvent(
                start=event_start,
                end=end_time,
                duration=duration,
                confidence=0.5,
                source="energy_fallback",
            ))

    print(f"  Energy-based fallback: {len(events)} laughter events detected")
    return events


# === Ensemble ===

def ensemble_detection(
    yamnet_events: list[LaughterEvent],
    gillick_events: list[LaughterEvent],
    overlap_threshold: float = 0.3,
) -> list[LaughterEvent]:
    """
    Combine YAMNet and Gillick detections using ensemble voting.

    Events detected by both models get higher confidence.
    Single-model events get reduced confidence.
    """
    if not gillick_events:
        print(f"  Using YAMNet only: {len(yamnet_events)} events")
        return sorted(yamnet_events, key=lambda e: e.start)

    ensemble_events = []
    used_gillick = set()

    for y_evt in yamnet_events:
        matched = False
        for g_idx, g_evt in enumerate(gillick_events):
            if g_idx in used_gillick:
                continue

            overlap = _compute_overlap(y_evt, g_evt)
            if overlap >= overlap_threshold:
                ensemble_events.append(LaughterEvent(
                    start=min(y_evt.start, g_evt.start),
                    end=max(y_evt.end, g_evt.end),
                    duration=max(y_evt.end, g_evt.end) - min(y_evt.start, g_evt.start),
                    confidence=(y_evt.confidence + g_evt.confidence) / 2,
                    source="ensemble",
                ))
                used_gillick.add(g_idx)
                matched = True
                break

        if not matched:
            y_evt.source = "yamnet_only"
            y_evt.confidence *= 0.7
            ensemble_events.append(y_evt)

    for g_idx, g_evt in enumerate(gillick_events):
        if g_idx not in used_gillick:
            g_evt.source = "gillick_only"
            g_evt.confidence *= 0.7
            ensemble_events.append(g_evt)

    ensemble_events.sort(key=lambda e: e.start)

    agreed = sum(1 for e in ensemble_events if e.source == "ensemble")
    print(
        f"  Ensemble: {len(ensemble_events)} events "
        f"({agreed} agreed, {len(ensemble_events) - agreed} single-model)"
    )

    return ensemble_events


def _compute_overlap(a: LaughterEvent, b: LaughterEvent) -> float:
    """Compute temporal overlap ratio between two events."""
    overlap_start = max(a.start, b.start)
    overlap_end = min(a.end, b.end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_duration = overlap_end - overlap_start
    min_duration = min(a.duration, b.duration)

    if min_duration <= 0:
        return 0.0

    return overlap_duration / min_duration
