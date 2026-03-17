"""Qt UI for the comedy pipeline analysis viewer."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .models import LAUGH_COLORS, LaughEvent, ParagraphViewModel, PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _fmt_ms(ms: int) -> str:
    return _fmt_time(ms / 1000.0)


def _is_dark() -> bool:
    """Return True when the current system/app palette is dark-themed."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        return False
    return app.palette().window().color().lightness() < 128


def _row_colors() -> dict[str, QColor]:
    """Table row tint colors that work on both light and dark palettes."""
    if _is_dark():
        return {
            "big_laugh":    QColor("#4a2020"),
            "medium_laugh": QColor("#4a3010"),
            "chuckle":      QColor("#3a3a10"),
        }
    return {
        "big_laugh":    QColor("#FFE8E8"),
        "medium_laugh": QColor("#FFF0E0"),
        "chuckle":      QColor("#FFFDE8"),
    }


# ---------------------------------------------------------------------------
# SummaryBar
# ---------------------------------------------------------------------------

class SummaryBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(16)
        self._labels: list[QLabel] = []

    def load(self, result: PipelineResult) -> None:
        for lbl in self._labels:
            lbl.deleteLater()
        self._labels.clear()

        stats = result.summary_stats
        items = [
            f"<b>{result.comedian}</b>",
            result.special_name or "",
            str(result.year) if result.year else "",
            f"Laughs: <b>{stats.get('total_laughs', 0)}</b>",
            f"{stats.get('laughs_per_minute', 0)}/min",
            f"Laugh time: {stats.get('laugh_time_percentage', 0)}%",
            f"Big: {stats.get('big_laughs', 0)} | Med: {stats.get('medium_laughs', 0)} | Chuckle: {stats.get('chuckles', 0)}",
        ]
        for text in items:
            if not text:
                continue
            lbl = QLabel(text)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            self._layout.addWidget(lbl)
            self._labels.append(lbl)
        self._layout.addStretch()


# ---------------------------------------------------------------------------
# ParagraphBlock
# ---------------------------------------------------------------------------

class ParagraphBlock(QFrame):
    seek_requested = pyqtSignal(float)

    def __init__(self, para: ParagraphViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._para = para
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._apply_style(active=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header row
        header_row = QHBoxLayout()
        laugh_info = f"  [{para.laugh_count} laugh{'s' if para.laugh_count != 1 else ''}]" if para.has_laughs else "  [setup]"
        header = QLabel(
            f"<b>P{para.paragraph_id}</b>  "
            f"{_fmt_time(para.start_time)} → {_fmt_time(para.end_time)}"
            f"{laugh_info}"
        )
        header.setTextFormat(Qt.TextFormat.RichText)
        seek_btn = QPushButton("▶")
        seek_btn.setFixedWidth(28)
        seek_btn.setToolTip("Seek audio to this paragraph")
        seek_btn.clicked.connect(lambda: self.seek_requested.emit(para.start_time))
        header_row.addWidget(header)
        header_row.addStretch()
        header_row.addWidget(seek_btn)
        layout.addLayout(header_row)

        # Transcript text
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setReadOnly(True)
        browser.document().setDefaultStyleSheet(
            "body { font-family: 'Segoe UI', sans-serif; font-size: 13px; }"
        )
        browser.setHtml(para.html_text)
        browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Fit to content height
        browser.document().adjustSize()
        h = int(browser.document().size().height()) + 12
        browser.setFixedHeight(max(h, 40))
        layout.addWidget(browser)

    def set_active(self, active: bool) -> None:
        self._apply_style(active)

    def _apply_style(self, active: bool) -> None:
        if active:
            bg = "#2d2018" if _is_dark() else "#fff8f0"
            self.setStyleSheet(
                f"QFrame {{ border-left: 4px solid #FF6B6B; background: {bg}; border-radius: 4px; }}"
            )
        else:
            self.setStyleSheet(
                "QFrame { border-left: 4px solid transparent; background: transparent; border-radius: 4px; }"
            )


# ---------------------------------------------------------------------------
# TranscriptPanel
# ---------------------------------------------------------------------------

class TranscriptPanel(QWidget):
    paragraph_seek_requested = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._inner = QVBoxLayout(self._container)
        self._inner.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._inner.setSpacing(8)
        self._inner.setContentsMargins(8, 8, 8, 8)
        self._scroll.setWidget(self._container)

        layout.addWidget(self._scroll)

        self._blocks: dict[int, ParagraphBlock] = {}
        self._active_id: int | None = None

    def load(self, paragraphs: list[ParagraphViewModel]) -> None:
        for block in self._blocks.values():
            block.deleteLater()
        self._blocks.clear()
        self._active_id = None

        for para in paragraphs:
            block = ParagraphBlock(para)
            block.seek_requested.connect(self.paragraph_seek_requested)
            self._inner.addWidget(block)
            self._blocks[para.paragraph_id] = block

    def highlight_paragraph(self, paragraph_id: int) -> None:
        if self._active_id == paragraph_id:
            return
        if self._active_id is not None and self._active_id in self._blocks:
            self._blocks[self._active_id].set_active(False)
        self._active_id = paragraph_id
        if paragraph_id in self._blocks:
            block = self._blocks[paragraph_id]
            block.set_active(True)
            self._scroll.ensureWidgetVisible(block)

    def clear(self) -> None:
        for block in self._blocks.values():
            block.deleteLater()
        self._blocks.clear()
        self._active_id = None


# ---------------------------------------------------------------------------
# LaughEventsPanel
# ---------------------------------------------------------------------------

_COLUMNS = ["#", "Time", "Type", "Duration", "Intensity", "Conf", "Source"]


class LaughEventsPanel(QWidget):
    event_seek_requested = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._stats_label = QLabel("No events loaded")
        layout.addWidget(self._stats_label)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 40)
        self._table.setColumnWidth(1, 70)
        self._table.setColumnWidth(2, 110)
        self._table.setColumnWidth(3, 65)
        self._table.setColumnWidth(4, 90)
        self._table.setColumnWidth(5, 55)
        self._table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table)

        self._event_starts: dict[int, float] = {}  # row -> start_sec

    def load(self, events: list[LaughEvent], stats: dict) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._event_starts.clear()

        total = stats.get("total_laughs", len(events))
        big = stats.get("big_laughs", 0)
        med = stats.get("medium_laughs", 0)
        chk = stats.get("chuckles", 0)
        avg_dur = stats.get("avg_laugh_duration", 0)
        self._stats_label.setText(
            f"{total} total  |  {big} big / {med} medium / {chk} chuckle  |  avg {avg_dur}s"
        )

        for row, event in enumerate(events):
            self._table.insertRow(row)
            self._event_starts[row] = event.start
            row_color = _row_colors().get(event.intensity_category, QColor())

            def _item(text: str) -> QTableWidgetItem:
                item = QTableWidgetItem(text)
                item.setBackground(row_color)
                return item

            self._table.setItem(row, 0, _item(str(event.event_id)))
            self._table.setItem(row, 1, _item(_fmt_time(event.start)))

            # Type — colored badge label
            badge = QLabel(f"  {event.intensity_category}  ")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color = LAUGH_COLORS.get(event.intensity_category, "#cccccc")
            badge.setStyleSheet(
                f"background:{color}; color:#111; border-radius:3px; font-size:11px; font-weight:600;"
            )
            self._table.setCellWidget(row, 2, badge)

            self._table.setItem(row, 3, _item(f"{event.duration:.1f}s"))

            # Intensity — progress bar
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(event.intensity * 100))
            bar.setTextVisible(True)
            bar.setFormat(f"{event.intensity:.2f}")
            self._table.setCellWidget(row, 4, bar)

            self._table.setItem(row, 5, _item(f"{event.confidence:.3f}"))
            self._table.setItem(row, 6, _item(event.source))

        self._table.setSortingEnabled(True)

    def highlight_event(self, event_id: int) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.text() == str(event_id):
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                break

    def _on_double_click(self, row: int, _col: int) -> None:
        start = self._event_starts.get(row)
        if start is not None:
            self.event_seek_requested.emit(start)

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._event_starts.clear()
        self._stats_label.setText("No events loaded")


# ---------------------------------------------------------------------------
# AudioControlBar
# ---------------------------------------------------------------------------

class AudioControlBar(QWidget):
    seek_requested = pyqtSignal(int)  # milliseconds
    volume_changed = pyqtSignal(float)  # 0.0-1.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.setEnabled(False)
        layout.addWidget(self._play_btn)

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setEnabled(False)
        layout.addWidget(self._seek_slider, stretch=1)

        self._time_label = QLabel("--:-- / --:--")
        self._time_label.setMinimumWidth(100)
        layout.addWidget(self._time_label)

        layout.addWidget(QLabel("Vol"))
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(80)
        layout.addWidget(self._vol_slider)

        self._duration_ms = 0
        self._user_seeking = False

        self._seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)
        self._vol_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(v / 100.0)
        )

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = duration_ms
        self._seek_slider.setRange(0, max(duration_ms, 1))
        self._seek_slider.setEnabled(duration_ms > 0)
        self._play_btn.setEnabled(duration_ms > 0)

    def set_position(self, position_ms: int) -> None:
        if self._user_seeking:
            return
        self._seek_slider.setValue(position_ms)
        self._time_label.setText(
            f"{_fmt_ms(position_ms)} / {_fmt_ms(self._duration_ms)}"
        )

    def set_playing(self, playing: bool) -> None:
        self._play_btn.setText("⏸" if playing else "▶")

    @property
    def play_button(self) -> QPushButton:
        return self._play_btn

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._user_seeking = False
        self.seek_requested.emit(self._seek_slider.value())


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(
        self,
        result: PipelineResult | None,
        audio_path: Path | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Comedy Pipeline Viewer")
        self.resize(1280, 800)

        self._result: PipelineResult | None = None
        self._active_para_id: int | None = None

        # --- Media player ---
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.8)

        # --- Build UI ---
        self._build_ui()
        self._build_menu()
        self._connect_signals()

        # --- Sync timer ---
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(250)
        self._sync_timer.timeout.connect(self._on_sync_tick)

        if result is not None:
            self._load_result(result, audio_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Summary bar
        self._summary_bar = SummaryBar()
        self._summary_bar.setFixedHeight(36)
        if _is_dark():
            self._summary_bar.setStyleSheet("background:#2b2b2b; border-bottom:1px solid #444;")
        else:
            self._summary_bar.setStyleSheet("background:#f0f0f0; border-bottom:1px solid #ddd;")
        root.addWidget(self._summary_bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._transcript_panel = TranscriptPanel()
        self._laughs_panel = LaughEventsPanel()
        splitter.addWidget(self._transcript_panel)
        splitter.addWidget(self._laughs_panel)
        splitter.setSizes([760, 520])
        root.addWidget(splitter, stretch=1)

        # Audio bar (dock at bottom)
        self._audio_bar = AudioControlBar()
        dock = QDockWidget("Audio", self)
        dock.setWidget(self._audio_bar)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QWidget())  # hide title bar
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        open_json = QAction("Open JSON result…", self)
        open_json.setShortcut("Ctrl+O")
        open_json.triggered.connect(self._on_open_file)
        file_menu.addAction(open_json)

        open_audio = QAction("Open audio file…", self)
        open_audio.setShortcut("Ctrl+Shift+O")
        open_audio.triggered.connect(self._on_open_audio)
        file_menu.addAction(open_audio)

    def _connect_signals(self) -> None:
        self._audio_bar.play_button.clicked.connect(self._on_play_pause)
        self._audio_bar.seek_requested.connect(self._player.setPosition)
        self._audio_bar.volume_changed.connect(self._audio_output.setVolume)

        self._player.durationChanged.connect(self._audio_bar.set_duration)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._transcript_panel.paragraph_seek_requested.connect(self._on_seek_to_sec)
        self._laughs_panel.event_seek_requested.connect(self._on_seek_to_sec)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Pipeline Result", "", "JSON Files (*.json)"
        )
        if not path_str:
            return
        from .models import PipelineResult as PR
        result = PR.from_json(Path(path_str))
        audio = _find_audio(Path(path_str))
        self._load_result(result, audio)

    def _on_open_audio(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.m4a *.ogg)"
        )
        if path_str:
            self._load_audio(Path(path_str))

    def _on_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_playback_state_changed(
        self, state: QMediaPlayer.PlaybackState
    ) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._audio_bar.set_playing(playing)
        if playing:
            self._sync_timer.start()
        else:
            self._sync_timer.stop()

    def _on_sync_tick(self) -> None:
        pos_ms = self._player.position()
        self._audio_bar.set_position(pos_ms)

        if self._result is None:
            return
        pos_sec = pos_ms / 1000.0
        para = self._result.paragraph_at(pos_sec)
        if para:
            self._transcript_panel.highlight_paragraph(para.paragraph_id)

        nearby = self._result.laughs_near(pos_sec, window=1.0)
        if nearby:
            self._laughs_panel.highlight_event(nearby[0].event_id)

    def _on_seek_to_sec(self, seconds: float) -> None:
        self._player.setPosition(int(seconds * 1000))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_result(self, result: PipelineResult, audio_path: Path | None) -> None:
        self._result = result
        self._active_para_id = None
        self.setWindowTitle(f"Comedy Pipeline Viewer — {result.comedian} · {result.video_id}")

        self._summary_bar.load(result)
        self._transcript_panel.load(result.paragraphs)
        self._laughs_panel.load(result.laughter_events, result.summary_stats)

        if audio_path is not None:
            self._load_audio(audio_path)

    def _load_audio(self, path: Path) -> None:
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._audio_bar.set_position(0)
        self.statusBar().showMessage(f"Audio: {path.name}", 5000)


# ---------------------------------------------------------------------------
# Audio auto-detection helper (also used by __main__)
# ---------------------------------------------------------------------------

def _find_audio(json_path: Path) -> Path | None:
    video_id = json_path.stem
    parent = json_path.parent
    candidates = [
        parent / f"{video_id}.wav",
        parent / f"{video_id}_vocals.wav",
        parent / f"{video_id}_no_vocals.wav",
        parent / f"{video_id}.mp3",
        parent / f"{video_id}.m4a",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
