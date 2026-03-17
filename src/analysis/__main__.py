"""Entry point: python -m analysis [result.json] [audio.wav]"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from .models import PipelineResult
from .viewer import MainWindow, _find_audio


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Comedy Pipeline Viewer")

    args = sys.argv[1:]
    json_path: Path | None = Path(args[0]) if args else None
    audio_path: Path | None = Path(args[1]) if len(args) > 1 else None

    if json_path is None:
        path_str, _ = QFileDialog.getOpenFileName(
            None, "Open Pipeline Result", "", "JSON Files (*.json)"
        )
        if not path_str:
            sys.exit(0)
        json_path = Path(path_str)

    try:
        result = PipelineResult.from_json(json_path)
    except Exception as exc:
        QMessageBox.critical(None, "Load error", f"Could not load {json_path.name}:\n{exc}")
        sys.exit(1)

    if audio_path is None:
        audio_path = _find_audio(json_path)

    if audio_path is None:
        reply = QMessageBox.question(
            None,
            "Audio not found",
            f"No audio file found near {json_path.name}.\nBrowse for audio file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            path_str, _ = QFileDialog.getOpenFileName(
                None,
                "Open Audio File",
                str(json_path.parent),
                "Audio Files (*.wav *.mp3 *.m4a *.ogg)",
            )
            audio_path = Path(path_str) if path_str else None

    window = MainWindow(result, audio_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
