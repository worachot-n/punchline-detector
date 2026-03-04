# Comedy Audio Transcript Pipeline

> Stand-up Comedy Laughter Detection & Annotated Transcript Generator — v3

Automatically detects audience laughter in stand-up comedy videos and builds
**context-aware annotated transcripts** where paragraphs follow the comedian's
natural topic/bit structure — not laughter positions.

---

## Features

- **8-phase automated pipeline** — download, separate, detect, annotate, export
- **Ensemble laughter detection** — spectral pattern analysis + energy-based fallback
- **Context-based paragraph segmentation** — splits by comedy bit, not by laugh event
- **3 laugh intensity levels** — `[big_laugh]` / `[medium_laugh]` / `[chuckle]`
- **3 output formats** — clean `.txt`, timestamped `.txt`, full `.json`
- **Batch processing** — run multiple YouTube videos from a file

---

## Requirements

### System Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| **ffmpeg** | Audio conversion (WAV 16kHz) | `scoop install ffmpeg` |
| **deno** | YouTube JS extraction (yt-dlp) | `scoop install deno` |
| **Python 3.11–3.12** | Runtime (3.13+ breaks torch 2.4) | `scoop install python312` |
| **uv** | Python package manager | `scoop install uv` |

> On Linux/macOS: replace `scoop install` with `brew install` or your distro's package manager.

---

## Installation

```bash
git clone <repo-url>
cd Comedy_Audio_Transcript
uv sync
```

This creates `.venv/` and installs all Python dependencies automatically.

---

## Usage

### Single Video

```bash
uv run comedy-pipeline run <VIDEO_ID>
```

```bash
# With options
uv run comedy-pipeline run QhMO5SSmiaA --output-dir ./dataset --skip-qa
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir, -o` | `./dataset` | Directory for output files |
| `--skip-qa` | off | Skip Phase 6 random sampling |

### Batch Processing

Create a text file with one YouTube video ID per line:

```text
# comedy_batch.txt
QhMO5SSmiaA
dQw4w9WgXcQ
# Lines starting with # are skipped
abc123xyz
```

```bash
uv run comedy-pipeline batch comedy_batch.txt --output-dir ./dataset
```

---

## Output

Three files are generated per video under `{output-dir}/`:

| File | Description |
|------|-------------|
| `{video_id}.txt` | Clean annotated transcript — human-readable |
| `{video_id}_detailed.txt` | Same + timestamps and paragraph metadata |
| `{video_id}.json` | Full structured data for downstream processing |

### Example: Clean Transcript

```
[COMEDIAN] Trevor Noah
[SPECIAL] "How The British Took Over India" (2021)
[VIDEO_ID] QhMO5SSmiaA
[TOTAL_LAUGHS] 6 | [LAUGHS_PER_MIN] 1.03 | [LAUGH_TIME] 5.9%

---

- When you think about colonization, it is the strangest thing you can think
about. 'Cause conquering is one thing. You go to another country, you take
what's theirs... But colonization is strange because you go there, and you
don't just take over. You then force the people to become you.
[Paragraph continues — no laugh yet, comedian is building the setup]

And I'm glad that I can tell you that India is exactly where it was yesterday.
[big_laugh]
No, no, no, I feel you're not understanding what I'm saying. Who is the queen?
The Queen of England! She who was ordained by God. Which god?
[big_laugh]

How dare you speak to me like that! We are Great Britain.
Well, in that case, welcome to Great India.
[big_laugh]
Damn you, we are going to run this country whether you like it or not!
You're not taking...
[big_laugh]
She is all yours, take, take...
[big_laugh]

---

[STATS]
Total paragraphs: 3
Paragraphs with laughs: 2
Paragraphs without laughs: 1 (setup/transition)
Total laughs: 6
Big laughs: 6 | Medium laughs: 0 | Chuckles: 0
Avg laugh duration: 3.44s
Laughs per minute: 1.03
```

### Example: Detailed Transcript

```
[P1] [00:00 -> 01:27] (setup/transition)
- When you think about colonization...

[P2] [01:30 -> 03:22] | laughs: 2
And I'm glad that I can tell you...
[big_laugh]
...
```

### JSON Schema

```json
{
  "video_id": "QhMO5SSmiaA",
  "comedian": "Trevor Noah",
  "special_name": "...",
  "year": 2021,
  "total_duration_sec": 351.0,
  "transcript_type": "manual",
  "paragraphs": [
    {
      "paragraph_id": 1,
      "text": "...",
      "start_time": 0.11,
      "end_time": 87.48,
      "has_laughs": false,
      "laugh_count": 0,
      "inline_laughs": []
    }
  ],
  "laughter_events": [...],
  "summary_stats": {...}
}
```

---

## How It Works

```
Video URL
    │
    ▼
Phase 1 ── Data Acquisition
              yt-dlp  →  WAV 16kHz mono
              youtube-transcript-api  →  transcript segments
    │
    ▼
Phase 2 ── Audio Preprocessing
              Demucs (htdemucs)  →  vocals / no_vocals stems
              audience track isolated for detection
    │
    ▼
Phase 3 ── Laughter Detection (Ensemble)
              Spectral detector  →  rhythmic energy bursts, 500–4000 Hz centroid
              Energy-based fallback  →  RMS threshold scan
              Ensemble merge  →  agreed events get higher confidence
    │
    ▼
Phase 4 ── Post-Processing
              merge_close_events  →  gap ≤ 0.5s merged into one event
              filter_by_duration  →  0.3s – 30s kept
              compute_intensity   →  RMS → big / medium / chuckle
              validate_spectral   →  reject non-laughter noise
    │
    ▼
Phase 5 ── Unified Timeline
              speech + laughter placed on single sorted timeline
              latency computed per laugh (time after preceding speech)
    │
    ▼
Phase 6 ── Quality Assurance
              10% random sample printed for manual spot-check
    │
    ▼
Phase 7 ── Context-Based Paragraph Segmentation  ◄── key innovation
              4 signals → weighted voting → break points
              build_paragraphs: laughs inline, not at end
    │
    ▼
Phase 8 ── Annotated Transcript Export
              .txt  /  _detailed.txt  /  .json
```

---

## Context-Based Segmentation (v3)

The core innovation of v3 is **segmenting by comedy bit**, not by laughter.

### Why it matters

```
❌ Laughter-based (old):          ✅ Context-based (v3):

"India is where it was."          "I'm glad that I can tell you that
[big_laugh]                        India is exactly where it was yesterday.
                                   [big_laugh]
"No no, I feel you're not          No, no, no — I feel you're not
understanding."                    understanding what I'm saying.
[big_laugh]                        Who is the queen? The Queen of England!
                                   She who was ordained by God. Which god?
"Which god?"                       [big_laugh]"
[big_laugh]
                                   → one paragraph = one bit
→ context shredded                 → laughs inline at natural positions
→ every para ends with laugh       → some paras have no laugh (setup)
```

### 4-Signal Weighted Voting

| Signal | Weight | How it works |
|--------|--------|--------------|
| **Long pause** | 0.50 | Silence > 3s (excluding laugh pauses) = clear topic break |
| **Transition phrase** | 0.30 | "Anyway...", "Have you ever noticed...", "The other day..." |
| **Semantic shift** | 0.30 | Cosine similarity of sentence embeddings drops below 0.55 |
| **Post-big-laugh pause** | 0.25 | Comedian pauses > 2s after a big laugh = new bit |

A timestamp accumulates votes from all signals that fire near it.
If the combined score reaches `0.4`, a paragraph break is placed there.

---

## Project Structure

```
Comedy_Audio_Transcript/
├── pyproject.toml
├── src/
│   └── comedy_pipeline/
│       ├── models.py               # TimelineEntry, LaughterEvent, Paragraph
│       ├── phase1_acquisition.py   # yt-dlp + transcript API
│       ├── phase2_preprocessing.py # Demucs source separation
│       ├── phase3_detection.py     # Spectral + energy ensemble detection
│       ├── phase4_postprocessing.py# Merge, filter, intensity, spectral QA
│       ├── phase5_timeline.py      # Unified timeline + latency
│       ├── phase6_qa.py            # Random sampling for manual QA
│       ├── phase7_segmentation.py  # 4-signal paragraph segmentation
│       ├── phase8_transcript.py    # .txt / .json export
│       ├── pipeline.py             # run_full_pipeline() orchestrator
│       └── cli.py                  # Click CLI entry point
├── tests/
│   └── test_pipeline.py            # 22 unit tests
└── dataset/                        # Output files (git-ignored)
```

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run a single video (dry run check)
uv run comedy-pipeline run QhMO5SSmiaA --skip-qa
```

---

## Dependency Notes

| Package | Constraint | Reason |
|---------|-----------|--------|
| `torch` | `==2.4.0` | Latest Windows-compatible version on PyPI without torchcodec dependency |
| `torchaudio` | `==2.4.0` | Matches torch; v2.9+ requires torchcodec (broken DLL on Windows) |
| `numpy` | `<2.0.0` | torch 2.4.0 compiled against NumPy 1.x — NumPy 2.x causes `_ARRAY_API not found` crash |
| `demucs` | `>=4.0.0` | htdemucs model required for two-stem vocal separation |
| Python | `<3.13` | torch 2.4.0 has no wheels for CPython 3.13+ |
