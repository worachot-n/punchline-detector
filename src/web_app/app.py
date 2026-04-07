"""Flask web application for Punchline Detector."""

from __future__ import annotations

import json
import re
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from markupsafe import Markup

from . import job_manager


DEFAULTS = {
    # Phase 3
    "p3_yamnet_confidence_threshold": 0.3,
    "p3_yamnet_frame_duration": 0.96,
    "p3_yamnet_hop_duration": 0.48,
    "p3_gillick_threshold": 0.5,
    "p3_gillick_min_length": 0.2,
    "p3_ensemble_overlap_threshold": 0.3,
    # Phase 4
    "p4_max_gap": 0.1,
    "p4_min_duration": 0.3,
    "p4_max_duration": 30.0,
    "p4_big_threshold": 0.7,
    "p4_medium_threshold": 0.4,
    "p4_min_centroid": 500.0,
    "p4_max_centroid": 4000.0,
    # Phase 7
    "p7_min_pause": 2.0,
    "p7_shift_threshold": 0.55,
    "p7_min_post_laugh_pause": 2.0,
    "p7_vote_threshold": 0.25,
    "p7_min_paragraph_duration": 5.0,
    "p7_max_paragraph_duration": 30.0,
}


def _f(form, name: str, cast=float):
    try:
        return cast(form.get(name, DEFAULTS.get(name)))
    except (ValueError, TypeError):
        return DEFAULTS.get(name)


def parse_params(form) -> dict:
    phase3 = {
        "yamnet_confidence_threshold": _f(form, "p3_yamnet_confidence_threshold"),
        "yamnet_frame_duration":       _f(form, "p3_yamnet_frame_duration"),
        "yamnet_hop_duration":         _f(form, "p3_yamnet_hop_duration"),
        "gillick_threshold":           _f(form, "p3_gillick_threshold"),
        "gillick_min_length":          _f(form, "p3_gillick_min_length"),
        "ensemble_overlap_threshold":  _f(form, "p3_ensemble_overlap_threshold"),
    }
    phase4 = {
        "max_gap":          _f(form, "p4_max_gap"),
        "min_duration":     _f(form, "p4_min_duration"),
        "max_duration":     _f(form, "p4_max_duration"),
        "big_threshold":    _f(form, "p4_big_threshold"),
        "medium_threshold": _f(form, "p4_medium_threshold"),
        "min_centroid":     _f(form, "p4_min_centroid"),
        "max_centroid":     _f(form, "p4_max_centroid"),
    }
    phase7 = {
        "min_pause":              _f(form, "p7_min_pause"),
        "shift_threshold":        _f(form, "p7_shift_threshold"),
        "min_post_laugh_pause":   _f(form, "p7_min_post_laugh_pause"),
        "vote_threshold":         _f(form, "p7_vote_threshold"),
        "min_paragraph_duration": _f(form, "p7_min_paragraph_duration"),
        "max_paragraph_duration": _f(form, "p7_max_paragraph_duration"),
    }
    return {
        "phase3": phase3,
        "phase4": phase4,
        "phase7": phase7,
        "detection_mode": form.get("detection_mode", "ensemble"),
        "skip_qa": "skip_qa" in form,
        "output_dir": "./dataset",
    }


def _reconstruct_annotated(para: dict) -> str:
    """Rebuild annotated text from text + inline_laughs char offsets."""
    text = para.get("text", "")
    inline_laughs = para.get("inline_laughs", [])
    if not inline_laughs:
        return text
    chars = list(text)
    for laugh in sorted(inline_laughs, key=lambda x: x.get("char_offset", 0), reverse=True):
        tag = f"[{laugh.get('type', 'chuckle')}]"
        offset = min(laugh.get("char_offset", len(chars)), len(chars))
        chars.insert(offset, tag)
    return "".join(chars)


def _load_result_data(video_id: str, output_dir: str) -> dict | None:
    """Load and enrich result JSON for a given video_id."""
    json_path = Path(output_dir) / f"{video_id}.json"
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    for para in data.get("paragraphs", []):
        if "annotated_text" not in para:
            para["annotated_text"] = _reconstruct_annotated(para)
    return data


def create_app(output_dir: str = "./dataset", downloads_dir: str = "./downloads") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["OUTPUT_DIR"] = output_dir
    app.config["DOWNLOADS_DIR"] = downloads_dir
    app.secret_key = "punchline-detector-dev"

    @app.template_filter("render_annotated")
    def render_annotated(text) -> str:
        if not isinstance(text, str):
            return Markup("")
        text = re.sub(r"\[big_laugh\]",
                      '<span class="badge bg-danger ms-1">big laugh</span>', text)
        text = re.sub(r"\[medium_laugh\]",
                      '<span class="badge bg-warning text-dark ms-1">medium laugh</span>', text)
        text = re.sub(r"\[chuckle\]",
                      '<span class="badge bg-info text-dark ms-1">chuckle</span>', text)
        text = text.replace("\n", "<br>")
        return Markup(text)

    @app.template_filter("fmt_time")
    def fmt_time(seconds: float) -> str:
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # ── Pages ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html", defaults=DEFAULTS)

    @app.route("/run", methods=["POST"])
    def run():
        video_id = request.form.get("video_id", "").strip()
        if not re.match(r"^[A-Za-z0-9_-]{1,20}$", video_id):
            return render_template("index.html", defaults=DEFAULTS,
                                   error="Invalid YouTube Video ID.")
        params = parse_params(request.form)
        job_id = job_manager.submit_job(video_id, params)
        return redirect(url_for("job_status_page", job_id=job_id))

    @app.route("/job/<job_id>")
    def job_status_page(job_id: str):
        entry = job_manager.get_job(job_id)
        if not entry:
            return "Job not found", 404
        if entry.status == "done":
            return redirect(url_for("results", job_id=job_id))
        return render_template("job.html", job=entry)

    @app.route("/api/job/<job_id>/status")
    def job_status_api(job_id: str):
        entry = job_manager.get_job(job_id)
        if not entry:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "status": entry.status,
            "phase": entry.phase,
            "progress_pct": entry.progress_pct,
            "error": entry.error,
        })

    @app.route("/results/<job_id>")
    def results(job_id: str):
        entry = job_manager.get_job(job_id)
        if not entry:
            return "Job not found", 404
        if entry.status != "done":
            return redirect(url_for("job_status_page", job_id=job_id))

        data = _load_result_data(entry.video_id, app.config["OUTPUT_DIR"])
        if data is None:
            return "Result file not found", 404

        has_audio = Path(app.config["DOWNLOADS_DIR"]).joinpath(
            f"{entry.video_id}.wav").exists()

        return render_template(
            "results.html",
            job=entry,
            data=data,
            video_id=entry.video_id,
            params=entry.params,
            has_audio=has_audio,
            audio_url=url_for("serve_audio", video_id=entry.video_id) if has_audio else None,
        )

    @app.route("/view/<video_id>")
    def view(video_id: str):
        """View results for any previously analysed video (no job required)."""
        if not re.match(r"^[A-Za-z0-9_-]{1,20}$", video_id):
            return "Invalid video ID", 400

        data = _load_result_data(video_id, app.config["OUTPUT_DIR"])
        if data is None:
            return "No analysis found for this video.", 404

        has_audio = Path(app.config["DOWNLOADS_DIR"]).joinpath(f"{video_id}.wav").exists()

        return render_template(
            "results.html",
            job=None,
            data=data,
            video_id=video_id,
            params=None,
            has_audio=has_audio,
            audio_url=url_for("serve_audio", video_id=video_id) if has_audio else None,
        )

    @app.route("/results/<job_id>/download/<filetype>")
    def download(job_id: str, filetype: str):
        entry = job_manager.get_job(job_id)
        if not entry or not entry.result_paths:
            return "Not found", 404
        allowed = {"json", "clean", "detailed"}
        if filetype not in allowed:
            return "Invalid file type", 400
        path = entry.result_paths.get(filetype)
        if not path or not Path(path).exists():
            return "File not found", 404
        return send_file(Path(path).resolve(), as_attachment=True)

    @app.route("/download/<video_id>/<filetype>")
    def download_by_video(video_id: str, filetype: str):
        """Download output files by video_id (for history page)."""
        if not re.match(r"^[A-Za-z0-9_-]{1,20}$", video_id):
            return "Invalid video ID", 400
        allowed = {"json": f"{video_id}.json",
                   "clean": f"{video_id}.txt",
                   "detailed": f"{video_id}_detailed.txt"}
        if filetype not in allowed:
            return "Invalid file type", 400
        path = Path(app.config["OUTPUT_DIR"]) / allowed[filetype]
        if not path.exists():
            return "File not found", 404
        return send_file(path.resolve(), as_attachment=True)

    @app.route("/audio/<video_id>")
    def serve_audio(video_id: str):
        """Serve the local WAV file for a video."""
        if not re.match(r"^[A-Za-z0-9_-]{1,20}$", video_id):
            return "Invalid video ID", 400
        wav = Path(app.config["DOWNLOADS_DIR"]) / f"{video_id}.wav"
        if not wav.exists():
            return "Audio not found", 404
        return send_file(wav.resolve(), mimetype="audio/wav")

    @app.route("/history")
    def history():
        output_dir = Path(app.config["OUTPUT_DIR"])
        downloads_dir = Path(app.config["DOWNLOADS_DIR"])
        results_list = []
        if output_dir.exists():
            for json_file in sorted(output_dir.glob("*.json"),
                                    key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        d = json.load(f)
                    vid = d.get("video_id", json_file.stem)
                    results_list.append({
                        "video_id": vid,
                        "comedian": d.get("comedian", "Unknown"),
                        "special_name": d.get("special_name", ""),
                        "year": d.get("year", ""),
                        "total_laughs": d.get("summary_stats", {}).get("total_laughs", 0),
                        "laughs_per_minute": d.get("summary_stats", {}).get("laughs_per_minute", 0),
                        "total_duration_sec": d.get("total_duration_sec", 0),
                        "has_audio": (downloads_dir / f"{vid}.wav").exists(),
                    })
                except Exception:
                    continue
        return render_template("history.html", results=results_list)

    return app


def main():
    app = create_app()
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
