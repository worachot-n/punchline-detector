"""Phase 2: Audio Preprocessing — source separation and trimming."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def separate_sources(
    audio_path: str,
    output_dir: str | None = None,
    model: str = "htdemucs",
) -> dict[str, str]:
    """
    Separate audio into vocals (comedian) and accompaniment (audience) using Demucs.

    The 'no_vocals' track contains audience reactions (laughter, applause, etc.)
    while 'vocals' contains the comedian's speech.

    Returns dict with keys: 'vocals', 'no_vocals', 'drums', 'bass', 'other'
    """
    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = str(audio_path.parent / "separated")

    output_path = Path(output_dir)

    # Check if separation already done
    stem_name = audio_path.stem
    expected_dir = output_path / model / stem_name
    vocals_path = expected_dir / "vocals.wav"
    no_vocals_path = expected_dir / "no_vocals.wav"

    if vocals_path.exists() and no_vocals_path.exists():
        print(f"  Source separation already exists: {expected_dir}")
        return _collect_stems(expected_dir)

    # Run demucs
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",  # split into vocals + no_vocals
        "-n", model,
        "-o", str(output_path),
        str(audio_path),
    ]

    print(f"  Running source separation ({model})...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr}")

    if not expected_dir.exists():
        # Try to find the output directory
        for candidate in output_path.rglob("vocals.wav"):
            expected_dir = candidate.parent
            break

    return _collect_stems(expected_dir)


def _collect_stems(stem_dir: Path) -> dict[str, str]:
    """Collect all stem file paths from the separation output directory."""
    stems = {}
    for wav_file in stem_dir.glob("*.wav"):
        stems[wav_file.stem] = str(wav_file)
    return stems


def trim_audio(
    audio_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    output_path: str | None = None,
) -> str:
    """
    Trim audio to a specific time range.

    Useful for removing intro/outro music that could interfere with detection.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr) if end_sec else len(y)

    y_trimmed = y[start_sample:end_sample]

    if output_path is None:
        p = Path(audio_path)
        output_path = str(p.parent / f"{p.stem}_trimmed{p.suffix}")

    sf.write(output_path, y_trimmed, sr)
    print(f"  Trimmed audio: {start_sec}s → {end_sec or len(y)/sr:.1f}s → {output_path}")

    return output_path


def detect_intro_outro(
    audio_path: str,
    energy_threshold: float = 0.01,
    min_speech_duration: float = 5.0,
) -> tuple[float, float]:
    """
    Detect intro/outro boundaries based on energy levels.

    Returns (start_sec, end_sec) of the main content.
    Intro music typically has high energy without speech patterns.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Compute short-time energy in 1-second windows
    hop_length = sr  # 1 second
    n_frames = len(y) // hop_length

    energies = []
    for i in range(n_frames):
        frame = y[i * hop_length : (i + 1) * hop_length]
        rms = np.sqrt(np.mean(frame ** 2))
        energies.append(rms)

    energies = np.array(energies)

    # Find first and last second where energy is above threshold
    active = np.where(energies > energy_threshold)[0]

    if len(active) == 0:
        return 0.0, len(y) / sr

    start_sec = float(active[0])
    end_sec = float(active[-1] + 1)

    return start_sec, end_sec
