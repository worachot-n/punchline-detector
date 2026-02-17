"""Phase 1: Data Acquisition — download audio, metadata, and transcript."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from .models import TranscriptSegment, VideoMetadata


def download_audio(
    video_id: str,
    output_dir: str = "./downloads",
    sample_rate: int = 16000,
) -> str:
    """
    Download audio from YouTube as WAV 16kHz mono.

    Uses yt-dlp to download and ffmpeg to convert.
    Returns path to the downloaded WAV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_path = output_path / f"{video_id}.wav"
    if wav_path.exists():
        print(f"  Audio already exists: {wav_path}")
        return str(wav_path)

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Download best audio and convert to WAV 16kHz mono
    cmd = [
        "yt-dlp",
        "-x",                          # extract audio only
        "--audio-format", "wav",        # output as WAV
        "--postprocessor-args",
        f"ffmpeg:-ar {sample_rate} -ac 1",  # 16kHz mono
        "-o", str(wav_path),
        url,
    ]

    print(f"  Downloading audio for {video_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # yt-dlp may append extension — find the actual file
    if not wav_path.exists():
        # Try common alternative names
        for candidate in output_path.glob(f"{video_id}*"):
            if candidate.suffix in (".wav", ".webm", ".m4a", ".mp3"):
                # Convert to WAV if not already
                if candidate.suffix != ".wav":
                    _convert_to_wav(str(candidate), str(wav_path), sample_rate)
                    candidate.unlink()
                else:
                    candidate.rename(wav_path)
                break

    if not wav_path.exists():
        raise FileNotFoundError(f"Failed to download audio for {video_id}")

    print(f"  Audio saved: {wav_path}")
    return str(wav_path)


def _convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> None:
    """Convert any audio file to WAV 16kHz mono using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")


def download_metadata(video_id: str) -> VideoMetadata:
    """
    Fetch video metadata using yt-dlp --dump-json.

    Returns VideoMetadata with comedian name, special name, year, duration.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ["yt-dlp", "--dump-json", "--no-download", url]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Could not fetch metadata: {result.stderr}")
        return VideoMetadata(video_id=video_id)

    data = json.loads(result.stdout)

    # Try to parse comedian and special from title/channel
    title = data.get("title", "")
    channel = data.get("channel", data.get("uploader", ""))
    upload_date = data.get("upload_date", "")
    year = int(upload_date[:4]) if upload_date and len(upload_date) >= 4 else None

    return VideoMetadata(
        video_id=video_id,
        comedian=channel,
        special_name=title,
        year=year,
        duration=float(data.get("duration", 0)),
    )


def get_transcript(
    video_id: str,
    preferred_languages: list[str] | None = None,
) -> tuple[list[TranscriptSegment], str]:
    """
    Fetch transcript for a YouTube video.

    Prefers manually created transcripts over auto-generated.
    Returns (segments, transcript_type) where type is 'manual' or 'auto'.
    """
    if preferred_languages is None:
        preferred_languages = ["en"]

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
    except Exception as e:
        print(f"  Warning: Could not fetch transcript list: {e}")
        return [], "none"

    # Try manual first, then auto-generated
    transcript_type = "manual"
    try:
        transcript = transcript_list.find_manually_created_transcript(preferred_languages)
    except Exception:
        transcript_type = "auto"
        try:
            transcript = transcript_list.find_generated_transcript(preferred_languages)
        except Exception as e:
            print(f"  Warning: No transcript available: {e}")
            return [], "none"

    fetched = transcript.fetch()
    segments = [
        TranscriptSegment(
            text=entry.text,
            start=float(entry.start),
            end=float(entry.start + entry.duration),
            duration=float(entry.duration),
        )
        for entry in fetched
    ]

    print(f"  Transcript: {len(segments)} segments ({transcript_type})")
    return segments, transcript_type
