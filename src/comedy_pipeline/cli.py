"""CLI entry point for the comedy laughter detection pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .pipeline import run_full_pipeline


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Stand-up Comedy Laughter Detection Pipeline (v3).

    Detects laughter in stand-up comedy audio and generates annotated
    transcripts with context-based paragraph segmentation.
    """
    pass


@main.command()
@click.argument("video_id")
@click.option(
    "--output-dir", "-o",
    default="./dataset",
    help="Output directory for generated files.",
)
@click.option(
    "--skip-qa",
    is_flag=True,
    default=False,
    help="Skip Phase 6 QA sampling.",
)
def run(video_id: str, output_dir: str, skip_qa: bool):
    """Run the full pipeline for a single YouTube video.

    VIDEO_ID is the YouTube video ID (e.g., 'abc123' from youtube.com/watch?v=abc123).
    """
    try:
        paths = run_full_pipeline(
            video_id=video_id,
            output_dir=output_dir,
            skip_qa=skip_qa,
        )
        click.echo(f"\nOutput files:")
        for name, path in paths.items():
            click.echo(f"  {name}: {path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("video_ids_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    default="./dataset",
    help="Output directory for generated files.",
)
@click.option(
    "--skip-qa",
    is_flag=True,
    default=False,
    help="Skip Phase 6 QA sampling.",
)
def batch(video_ids_file: str, output_dir: str, skip_qa: bool):
    """Run the pipeline for multiple videos from a file.

    VIDEO_IDS_FILE is a text file with one YouTube video ID per line.
    Lines starting with # are treated as comments.
    """
    video_ids = []
    with open(video_ids_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                video_ids.append(line)

    click.echo(f"Processing {len(video_ids)} videos...")
    results = {}
    errors = {}

    for i, video_id in enumerate(video_ids, 1):
        click.echo(f"\n{'='*60}")
        click.echo(f"[{i}/{len(video_ids)}] Processing: {video_id}")
        click.echo(f"{'='*60}")

        try:
            paths = run_full_pipeline(
                video_id=video_id,
                output_dir=output_dir,
                skip_qa=skip_qa,
            )
            results[video_id] = paths
        except Exception as e:
            click.echo(f"Error processing {video_id}: {e}", err=True)
            errors[video_id] = str(e)

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo(f"Batch complete: {len(results)} success, {len(errors)} failed")
    if errors:
        click.echo("Failed videos:")
        for vid, err in errors.items():
            click.echo(f"  {vid}: {err}")


if __name__ == "__main__":
    main()
