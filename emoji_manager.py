from __future__ import annotations

import os
import re
import sys
import json
import hashlib
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import platform

import requests
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
    Slot,
    QSize,
    QByteArray,
    QBuffer,
    QIODevice,
    QStandardPaths
)
from PySide6.QtGui import QPixmap, QMovie, QPainter

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# -----------------------------
# Config / Validation
# -----------------------------
ALLOWED_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ALLOWED_NAME_RULE = "使用可能: 小文字 a-z、数字 0-9、_ と -。パターン: ^[a-z0-9][a-z0-9_-]*$"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}

DEFAULT_SOFT_WARN_KIB = 256
DEFAULT_HARD_MAX_KIB = 512
DEFAULT_MAX_SUFFIX = 200


class ItemStatus(Enum):
    LOCAL_ONLY = auto()
    SERVER_ONLY = auto()
    BOTH_SAME_NAME = auto()   # same name exists both sides (may or may not be same image)
    INVALID_NAME = auto()
    TOO_LARGE = auto()


def kib_of_file(path: Path) -> int:
    return (path.stat().st_size + 1023) // 1024


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate_name(raw: str) -> Tuple[bool, str, str]:
    """Return (ok, normalized_name, reason). No sanitization; only lowercasing."""
    name = (raw or "").strip()
    if not name:
        return False, "", "EMPTY_NAME"

    lower = name.lower()
    if len(lower) > 64:
        return False, lower, "TOO_LONG (>64)"

    if not ALLOWED_NAME_PATTERN.match(lower):
        return False, lower, "INVALID_CHARS"

    return True, lower, ""


# -----------------------------
# Data Model
# -----------------------------
@dataclass
class EmojiItem:
    checked: bool
    # display/identity
    base_name: str          # derived from file or server
    desired_name: str       # editable target name for upload (or rename/recreate)
    # local
    local_path: Optional[Path] = None
    local_kib: Optional[int] = None
    local_thumb: Optional[QPixmap] = None
    # server
    server_id: Optional[str] = None
    server_has: bool = False
    # computed
    status: ItemStatus = ItemStatus.LOCAL_ONLY
    detail: str = ""
    server_thumb: Optional[QPixmap] = None
    thumb_fetching: bool = False



# -----------------------------
# Mattermost API Client
# -----------------------------
class MattermostClient:
    def __init__(self, server_url: str, token: str, timeout_sec: int = 30) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })
        self._movie: QMovie | None = None

    def list_emojis(self) -> List[Dict[str, Any]]:
        """GET /api/v4/emoji?page=&per_page=200 until empty."""
        page = 0
        per_page = 200
        out: List[Dict[str, Any]] = []

        while True:
            url = f"{self.server_url}/api/v4/emoji"
            r = self.session.get(url, params={"page": page, "per_page": per_page}, timeout=self.timeout_sec)
            r.raise_for_status()
            items = r.json()
            if not items:
                break
            out.extend(items)
            page += 1

        return out
    
    def list_my_emojis(client: MattermostClient) -> list[dict]:
        me = client.get_me()
        my_id = me["id"]

        all_emojis = client.list_emojis()
        return [e for e in all_emojis if e.get("creator_id") == my_id]


    def get_emoji_image(self, emoji_id: str) -> bytes:
        """GET /api/v4/emoji/{emoji_id}/image"""
        url = f"{self.server_url}/api/v4/emoji/{emoji_id}/image"
        r = self.session.get(url, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.content

    def get_me(self) -> dict:
        url = f"{self.server_url}/api/v4/users/me"
        r = self.session.get(url, timeout=self.timeout_sec)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}\nURL: {r.url}\nBody:\n{r.text}")
        return r.json()

    def create_emoji(self, name: str, image_path: Path, creator_id: str) -> dict:
        url = f"{self.server_url}/api/v4/emoji"

        with open(image_path, "rb") as f:
            files = {
                "image": (image_path.name, f, "application/octet-stream"),
            }
            data = {
                "emoji": json.dumps({
                    "name": name,
                    "creator_id": creator_id,
                })
            }
            r = self.session.post(url, data=data, files=files, timeout=self.timeout_sec)

        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}\nURL: {r.url}\nBody:\n{r.text}")

        return r.json()


    def delete_emoji(self, emoji_id: str) -> None:
        """DELETE /api/v4/emoji/{emoji_id}"""
        url = f"{self.server_url}/api/v4/emoji/{emoji_id}"
        r = self.session.delete(url, timeout=self.timeout_sec)
        r.raise_for_status()


# -----------------------------
# Worker Infrastructure
# -----------------------------
class WorkerSignals(QObject):
    finished = Signal(object)      # result
    error = Signal(str)            # error message
    progress = Signal(str)         # status string


class ApiWorker(QRunnable):
    """Run an API task in background, return result or error."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

class BatchUploadWorker(QRunnable):
    """Upload multiple emojis sequentially in background (continue on error)."""
    def __init__(self, client: MattermostClient, tasks: list[tuple[int, str, Path]]):
        """
        tasks: list of (row_index, emoji_name, image_path)
        """
        super().__init__()
        self.client = client
        self.tasks = tasks
        self.signals = WorkerSignals()
        self.me = self.client.get_me()

    @Slot()
    def run(self) -> None:
        results = []
        total = len(self.tasks)

        for i, (row, name, path) in enumerate(self.tasks, start=1):
            try:
                self.signals.progress.emit(f"登録中 {i}/{total}: :{name}:")
                
                creator_id = self.me["id"]
                created = self.client.create_emoji(name, path, creator_id)
                results.append((row, True, created, ""))
            except Exception as e:
                # 失敗しても続行
                results.append((row, False, None, str(e)))

        self.signals.finished.emit(results)

class BatchDeleteWorker(QRunnable):
    """Delete multiple emojis sequentially in background (continue on error)."""
    def __init__(self, client: MattermostClient, tasks: list[tuple[int, str, str]]):
        """
        tasks: list of (row_index, emoji_name, emoji_id)
        """
        super().__init__()
        self.client = client
        self.tasks = tasks
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        results = []
        total = len(self.tasks)

        for i, (row, name, emoji_id) in enumerate(self.tasks, start=1):
            try:
                self.signals.progress.emit(f"削除中 {i}/{total}: :{name}:")
                self.client.delete_emoji(emoji_id)
                results.append((row, True, None, ""))
            except Exception as e:
                results.append((row, False, None, str(e)))

        self.signals.finished.emit(results)

class FetchEmojiImageWorker(QRunnable):
    """Fetch emoji image bytes by emoji_id in background."""
    def __init__(self, client: MattermostClient, emoji_id: str):
        super().__init__()
        self.client = client
        self.emoji_id = emoji_id
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.progress.emit(f"Fetching server image: {self.emoji_id}")
            b = self.client.get_emoji_image(self.emoji_id)  # bytes
            self.signals.finished.emit((self.emoji_id, b))
        except Exception as e:
            self.signals.error.emit(str(e))

class FetchEmojiThumbWorker(QRunnable):
    def __init__(self, client: MattermostClient, row: int, emoji_id: str):
        super().__init__()
        self.client = client
        self.row = row
        self.emoji_id = emoji_id
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            b = self.client.get_emoji_image(self.emoji_id)
            self.signals.finished.emit((self.row, self.emoji_id, b))
        except Exception as e:
            self.signals.error.emit(str(e))


# -----------------------------
# Table Model
# -----------------------------
class EmojiTableModel(QAbstractTableModel):
    COL_CHECK = 0
    COL_THUMB = 1
    COL_BASE = 2
    COL_DESIRED = 3
    COL_STATUS = 4
    COL_DETAIL = 5
    COL_LOCAL_PATH = 6
    COL_SERVER_ID = 7

    HEADERS = [
        "✅",
        "アイコン",
        "元の名前",
        "登録名（編集可）",
        "状態",
        "詳細",
        "ローカルパス",
        "サーバID",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.items: List[EmojiItem] = []

    def set_items(self, items: List[EmojiItem]) -> None:
        self.beginResetModel()
        self.items = items
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.items)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags

        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if index.column() == self.COL_CHECK:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == self.COL_DESIRED:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags


    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self.items[index.row()]
        col = index.column()

        if col == self.COL_CHECK:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if item.checked else Qt.CheckState.Unchecked
            return None

        # if col == self.COL_THUMB:
        #     if role == Qt.DecorationRole and item.local_thumb is not None:
        #         return item.local_thumb
        #     return None
        if col == self.COL_THUMB:
            if role == Qt.ItemDataRole.DecorationRole:
                if item.local_thumb is not None:
                    return item.local_thumb
                if item.server_thumb is not None:
                    return item.server_thumb
            return None


        if role == Qt.DisplayRole:
            if col == self.COL_BASE:
                return item.base_name
            if col == self.COL_DESIRED:
                return item.desired_name
            if col == self.COL_STATUS:
                return item.status.name
            if col == self.COL_DETAIL:
                return item.detail
            if col == self.COL_LOCAL_PATH:
                return str(item.local_path) if item.local_path else ""
            if col == self.COL_SERVER_ID:
                return item.server_id or ""
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if not index.isValid():
            return False
        item = self.items[index.row()]
        col = index.column()

        
        if col == self.COL_CHECK and role == Qt.ItemDataRole.CheckStateRole:
            item.checked = (value == Qt.CheckState.Checked or int(value) == int(Qt.CheckState.Checked.value))
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True

        if col == self.COL_DESIRED and role in (Qt.EditRole, Qt.DisplayRole):
            item.desired_name = str(value).strip()
            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            return True

        return False


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mattermost カスタム絵文字マネージャ")
        self.resize(1200, 700)

        self.thread_pool = QThreadPool.globalInstance()

        # cached server emojis: name(lower) -> (id, raw json)
        self.server_cache: Dict[str, Dict[str, Any]] = {}
        self.session_user_id: str | None = None
        # server image cache: emoji_id -> bytes
        self.server_image_cache: dict[str, bytes] = {}

        # selection guard (race prevention)
        self._current_selected_server_id: str | None = None

        # for in-memory GIF playback (must keep alive)
        self._movie: QMovie | None = None
        self._movie_buffer: QBuffer | None = None


        # UI
        self.server_edit = QLineEdit()
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)

        self.remember_token_chk = QCheckBox("Remember token")
        self.remember_token_chk.setChecked(False)


        self.soft_warn_edit = QLineEdit(str(DEFAULT_SOFT_WARN_KIB))
        self.hard_max_edit = QLineEdit(str(DEFAULT_HARD_MAX_KIB))
        self.max_suffix_edit = QLineEdit(str(DEFAULT_MAX_SUFFIX))

        self.btn_connect = QPushButton("接続・サーバキャッシュ更新")
        self.btn_pick_folder = QPushButton("フォルダ読み込み…")
        self.btn_load_server_mine = QPushButton("サーバ絵文字読込")

        self.btn_validate = QPushButton("検証・状態更新")
        self.btn_go_upload = QPushButton("登録")
        self.btn_delete = QPushButton("削除")
        # self.btn_rename = QPushButton("リネーム（再作成）（未実装）")

        self.status_label = QLabel("準備完了")

        # Table + preview
        self.model = EmojiTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(QSize(32, 32))
        self.table.setColumnWidth(EmojiTableModel.COL_CHECK, 50)
        self.table.setColumnWidth(EmojiTableModel.COL_THUMB, 60)
        self.table.setColumnWidth(EmojiTableModel.COL_BASE, 180)
        self.table.setColumnWidth(EmojiTableModel.COL_DESIRED, 220)
        self.table.setColumnWidth(EmojiTableModel.COL_STATUS, 160)
        self.table.setColumnWidth(EmojiTableModel.COL_DETAIL, 260)

        self.preview_label = QLabel("プレビュー")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(260)

        # Layout
        top = QWidget()
        top_layout = QFormLayout(top)
        top_layout.addRow("Server URL", self.server_edit)
        top_layout.addRow("Token", self.token_edit)
        top_layout.addRow("", self.remember_token_chk)


        limits_row = QHBoxLayout()
        limits_row.addWidget(QLabel("SoftWarn KiB"))
        limits_row.addWidget(self.soft_warn_edit)
        limits_row.addWidget(QLabel("HardMax KiB"))
        limits_row.addWidget(self.hard_max_edit)
        limits_row.addWidget(QLabel("MaxSuffix"))
        limits_row.addWidget(self.max_suffix_edit)
        limits_row.addStretch(1)

        limits_wrap = QWidget()
        limits_wrap.setLayout(limits_row)
        top_layout.addRow("Limits", limits_wrap)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_pick_folder)
        btn_row.addWidget(self.btn_load_server_mine)
        btn_row.addWidget(self.btn_validate)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_go_upload)
        btn_row.addWidget(self.btn_delete)
        # btn_row.addWidget(self.btn_rename)

        btn_wrap = QWidget()
        btn_wrap.setLayout(btn_row)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(top)
        left_layout.addWidget(btn_wrap)
        left_layout.addWidget(self.table)
        left_layout.addWidget(self.status_label)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

        # Signals
        self.btn_connect.clicked.connect(self.on_connect_refresh)
        self.btn_pick_folder.clicked.connect(self.on_load_folder)
        self.btn_load_server_mine.clicked.connect(self.on_load_server_mine)
        self.btn_validate.clicked.connect(self.on_validate)
        self.btn_go_upload.clicked.connect(self.on_upload)
        self.btn_delete.clicked.connect(self.on_delete)
        # self.btn_rename.clicked.connect(self.on_rename_stub)

        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # seed empty
        self.model.set_items([])

        self.table.verticalScrollBar().valueChanged.connect(lambda _v: self._prefetch_visible_thumbs())


        self._load_config()

    def closeEvent(self, event):
        self._save_config()
        super().closeEvent(event)


    # -------------------------
    # Helpers
    # -------------------------
    def _get_limits(self) -> Tuple[int, int, int]:
        try:
            soft_warn = int(self.soft_warn_edit.text().strip())
            hard_max = int(self.hard_max_edit.text().strip())
            max_suffix = int(self.max_suffix_edit.text().strip())
            return soft_warn, hard_max, max_suffix
        except ValueError:
            raise ValueError("上限値は整数で入力してください。")

    def _make_client(self) -> MattermostClient:
        server = self.server_edit.text().strip()
        token = self.token_edit.text().strip()
        if not server or not token:
            raise ValueError("Server URL と Token は必須です。")
        return MattermostClient(server, token)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "エラー", message)

    def _is_gif_bytes(self, b: bytes) -> bool:
        return len(b) >= 6 and (b[:6] == b"GIF87a" or b[:6] == b"GIF89a")

    def _show_image_bytes_in_preview(self, b: bytes) -> None:
        # Stop old movie
        if self._movie is not None:
            self._movie.stop()
        self._movie = None
        self._movie_buffer = None

        # GIF: animate from memory via QMovie + QBuffer
        if self._is_gif_bytes(b):
            self.preview_label.setText("")
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setMovie(None)

            ba = QByteArray(b)
            buf = QBuffer()
            buf.setData(ba)
            buf.open(QIODevice.OpenModeFlag.ReadOnly)

            movie = QMovie(buf)  # QMovie keeps a pointer; buffer must stay alive
            self._movie = movie
            self._movie_buffer = buf

            def render_frame():
                if self._movie is None:
                    return
                px = self._movie.currentPixmap()
                if px.isNull():
                    return
                w = max(64, self.preview_label.width() - 16)
                h = max(64, self.preview_label.height() - 16)
                px = px.scaled(
                    w, h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(px)

            self._movie.frameChanged.connect(lambda _i: render_frame())
            self._movie.start()
            render_frame()
            return

        # Non-GIF: show as pixmap
        px = QPixmap()
        px.loadFromData(b)
        if px.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("サーバ画像のデコードに失敗しました")
            return

        w = max(64, self.preview_label.width() - 16)
        h = max(64, self.preview_label.height() - 16)
        px = px.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setMovie(None)
        self.preview_label.setPixmap(px)
        self.preview_label.setText("")

    def _make_icon_pixmap_from_bytes(self, b: bytes, logical_size: int = 32) -> QPixmap | None:
        src = QPixmap()
        if not src.loadFromData(b):
            return None

        # HiDPI: 表示上の 32px を、実ピクセル（DPR倍）で作る
        dpr = self.table.devicePixelRatioF()  # 1.0, 1.25, 1.5, 2.0...
        target_px = int(round(logical_size * dpr))
        target = QSize(target_px, target_px)

        # 透明キャンバス（最終表示サイズ）
        canvas = QPixmap(target)
        canvas.fill(Qt.GlobalColor.transparent)

        # 画像を縦横比維持で縮小
        # ※アップスケールしたいなら min() を外してください
        scaled = src
        if src.width() > target_px or src.height() > target_px:
            scaled = src.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # 中央合成（余白は透明）
        x = (target_px - scaled.width()) // 2
        y = (target_px - scaled.height()) // 2
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(x, y, scaled)
        painter.end()

        # 論理ピクセルに対するDPRを設定（ぼけ防止）
        canvas.setDevicePixelRatio(dpr)
        return canvas
    

    def _make_icon_pixmap_from_path(self, image_path: Path, logical_size: int = 32) -> QPixmap | None:
        src = QPixmap(str(image_path))
        if src.isNull():
            return None

        dpr = self.table.devicePixelRatioF()
        target_px = int(round(logical_size * dpr))
        target = QSize(target_px, target_px)

        canvas = QPixmap(target)
        canvas.fill(Qt.GlobalColor.transparent)

        scaled = src
        if src.width() > target_px or src.height() > target_px:
            scaled = src.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        x = (target_px - scaled.width()) // 2
        y = (target_px - scaled.height()) // 2
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(x, y, scaled)
        painter.end()

        canvas.setDevicePixelRatio(dpr)
        return canvas


    def _prefetch_visible_thumbs(self) -> None:
        # テーブルが空なら何もしない
        if self.model.rowCount() == 0:
            return

        # 可視範囲の上端・下端の行を推定
        top_index = self.table.indexAt(self.table.viewport().rect().topLeft())
        bottom_index = self.table.indexAt(self.table.viewport().rect().bottomLeft())

        if not top_index.isValid():
            return

        top_row = top_index.row()
        bottom_row = bottom_index.row() if bottom_index.isValid() else min(self.model.rowCount() - 1, top_row + 30)

        # 取得要求
        for row in range(top_row, bottom_row + 1):
            self._ensure_thumb_for_row(row)

    def _ensure_thumb_for_row(self, row: int) -> None:
        it = self.model.items[row]

        # ローカルサムネがあるなら不要
        if it.local_thumb is not None:
            return

        # サーバIDが無いなら不要
        if not it.server_id:
            return

        # 既に持っているなら不要
        if it.server_thumb is not None:
            return

        emoji_id = it.server_id

        # bytesキャッシュがあれば、それから作る（通信なし）
        if emoji_id in self.server_image_cache:
            px = self._make_icon_pixmap_from_bytes(self.server_image_cache[emoji_id], size=32)
            if px is not None:
                it.server_thumb = px
                idx = self.model.index(row, EmojiTableModel.COL_THUMB)
                self.model.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])
            return

        # 取得中なら不要
        if it.thumb_fetching:
            return

        # 取得開始
        try:
            client = self._make_client()
        except Exception:
            return

        it.thumb_fetching = True
        worker = FetchEmojiThumbWorker(client, row, emoji_id)
        worker.signals.finished.connect(self._on_thumb_fetched)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _config_path(self) -> Path:
        # QStandardPaths はOSごとの「設定保存先」を返す
        base = Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation))
        # 例: C:\Users\<you>\AppData\Roaming\<AppName> になりがちなので、フォルダ名を固定
        base = base.parent / "MmEmojiManager"
        base.mkdir(parents=True, exist_ok=True)
        return base / "config.json"

    def _load_config(self) -> None:
        p = self._config_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            server = (data.get("server_url") or "").strip()
            token = (data.get("token") or "").strip()
            remember = bool(data.get("remember_token", False))

            if server:
                self.server_edit.setText(server)

            self.remember_token_chk.setChecked(remember)
            if remember and token:
                self.token_edit.setText(token)
        except Exception as e:
            # 設定が壊れていてもアプリが起動できるようにする
            self._set_status(f"Config load failed: {e}")

    def _save_config(self) -> None:
        p = self._config_path()
        try:
            remember = self.remember_token_chk.isChecked()
            payload = {
                "server_url": self.server_edit.text().strip(),
                "remember_token": remember,
                # トークンは remember がONのときだけ保存
                "token": self.token_edit.text().strip() if remember else "",
            }
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self._set_status(f"Config save failed: {e}")


    # -------------------------
    # Actions
    # -------------------------
    @Slot()
    def on_connect_refresh(self) -> None:
        """Fetch server emoji list and cache name->id."""
        try:
            client = self._make_client()
        except Exception as e:
            self._error(str(e))
            return

        self._set_status("サーバキャッシュ更新中...")
        worker = ApiWorker(client.list_emojis)
        worker.signals.finished.connect(self._on_server_cache_loaded)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    @Slot(object)
    def _on_server_cache_loaded(self, result: object) -> None:
        items = result  # list of dict
        assert isinstance(items, list)
        cache: Dict[str, Dict[str, Any]] = {}
        for e in items:
            name = str(e.get("name", "")).strip().lower()
            if name:
                cache[name] = e
        self.server_cache = cache
        self._set_status(f"Server cache loaded: {len(self.server_cache)} emojis.")
        # recompute statuses if local loaded
        self.on_validate()

        self._save_config()


    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self._set_status("Error")
        self._error(msg)

    def _normalize_key(self, name: str) -> str:
        return name.strip().lower()

    @Slot()
    def on_load_folder(self) -> None:
        # ... folder選択までは既存のまま ...
        folder = QFileDialog.getExistingDirectory(self, "Select Emoji Folder")
        if not folder:
            return

        self._set_status(f"フォルダ読み込み中: {folder}")
        folder_path = Path(folder)

        # 既存行の辞書（desired_name基準。必要なら base_name でも良い）
        by_name: dict[str, EmojiItem] = {}
        for it in self.model.items:
            k = self._normalize_key(it.desired_name or it.base_name)
            if k:
                by_name[k] = it

        # 今回フォルダに存在したキー（後で「消えたローカル」を外すなら使う）
        seen_local: set[str] = set()

        local_files: list[Path] = []
        for p in folder_path.rglob("*"):  # 再帰したくないなら iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                local_files.append(p)

        for p in local_files:
            base = p.stem
            k = self._normalize_key(base)
            if not k:
                continue

            seen_local.add(k)

            # ローカルサムネを生成（後述の高品質関数を使う）
            thumb = self._make_icon_pixmap_from_path(p, logical_size=32)
            kib = (p.stat().st_size + 1023) // 1024

            if k in by_name:
                it = by_name[k]
                it.local_path = p
                it.local_thumb = thumb
                it.local_kib = int(kib)
                # desired_name はユーザーが変えている可能性があるので基本触らない
            else:
                it = EmojiItem(
                    checked=False,
                    base_name=base,
                    desired_name=base,
                    local_path=p,
                    local_kib=int(kib),
                    local_thumb=thumb,
                    server_id=None,
                    server_has=False,
                    status=ItemStatus.LOCAL_ONLY,
                    detail="",
                    server_thumb=None,
                    thumb_fetching=False,
                )
                self.model.items.append(it)
                by_name[k] = it

        # 任意：フォルダから消えたローカルを外したい場合（置換に近い挙動）
        for it in self.model.items:
            k = self._normalize_key(it.base_name)
            if it.local_path and k not in seen_local:
                it.local_path = None
                it.local_thumb = None
                it.local_kib = None

        self.on_validate()
        self.model.layoutChanged.emit()
        self._prefetch_visible_thumbs()  # サーバサムネもあるなら


    # @Slot()
    # def on_load_folder(self) -> None:
    #     folder = QFileDialog.getExistingDirectory(self, "Select Emoji Folder")
    #     if not folder:
    #         return

    #     self._set_status(f"フォルダ読み込み中: {folder}")
    #     folder_path = Path(folder)

    #     # Build local items
    #     local_files = []
    #     for p in folder_path.iterdir():
    #         if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
    #             local_files.append(p)

    #     items: List[EmojiItem] = []
    #     for p in sorted(local_files):
    #         base = p.stem  # filename without extension
    #         ok, norm, reason = validate_name(base)
    #         size_kib = kib_of_file(p)

    #         thumb = QPixmap(str(p))
    #         if not thumb.isNull():
    #             thumb = thumb.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    #         it = EmojiItem(
    #             checked=False,
    #             base_name=base,
    #             desired_name=norm if ok else base,
    #             local_path=p,
    #             local_kib=size_kib,
    #             local_thumb=thumb,
    #         )
    #         # status computed later
    #         items.append(it)

    #     self.model.set_items(items)
    #     self._set_status(f"Loaded {len(items)} local files. Click Validate.")

    @Slot()
    def on_validate(self) -> None:
        """Recompute validation, size, collision, and server existence statuses."""
        try:
            soft_warn, hard_max, _max_suffix = self._get_limits()
        except Exception as e:
            self._error(str(e))
            return

        # Compute status for each row
        for it in self.model.items:
            it.detail = ""
            # derive name from desired_name; validate it (no sanitize)
            ok, norm, reason = validate_name(it.desired_name)
            if not ok:
                it.status = ItemStatus.INVALID_NAME
                it.detail = f"{reason}. {ALLOWED_NAME_RULE}"
                continue
            it.desired_name = norm

            # size checks (only if local)
            if it.local_path is not None:
                it.local_kib = kib_of_file(it.local_path)
                if it.local_kib > hard_max:
                    it.status = ItemStatus.TOO_LARGE
                    it.detail = f"TOO_LARGE: {it.local_kib}KiB > HardMax {hard_max}KiB (will not upload)"
                    continue
                elif it.local_kib > soft_warn:
                    it.detail = f"WARN: {it.local_kib}KiB > SoftWarn {soft_warn}KiB"

            # server presence
            s = self.server_cache.get(it.desired_name.lower())
            if s:
                it.server_has = True
                it.server_id = str(s.get("id", "")) or None
            else:
                it.server_has = False
                it.server_id = None

            # decide overall status
            if it.local_path is not None and it.server_has:
                it.status = ItemStatus.BOTH_SAME_NAME
            elif it.local_path is not None and not it.server_has:
                it.status = ItemStatus.LOCAL_ONLY
            elif it.local_path is None and it.server_has:
                it.status = ItemStatus.SERVER_ONLY
            else:
                it.status = ItemStatus.LOCAL_ONLY  # fallback

        # refresh view
        self.model.layoutChanged.emit()
        self._set_status("Validate complete.")

    @Slot()
    def on_load_server_mine(self) -> None:
        # 常に最新化（安全）
        self._set_status("サーバキャッシュ更新中...")
        if not self._refresh_server_cache_sync():
            return

        if not self.session_user_id:
            self._error("現在のユーザーIDの取得に失敗しました。")
            return

        my_id = self.session_user_id

        # 既存モデルを name->item で引けるように（desired_name基準）
        by_name: dict[str, EmojiItem] = {}
        for it in self.model.items:
            ok, norm, _reason = validate_name(it.desired_name)
            key = (norm if ok else it.desired_name).strip().lower()
            if key:
                by_name[key] = it

        added = 0
        updated = 0

        # server_cache から creator_id で絞り込み
        for name, e in self.server_cache.items():
            if e.get("creator_id") != my_id:
                continue

            server_id = str(e.get("id", "")).strip() or None
            if not server_id:
                continue

            if name in by_name:
                # 既にローカル行がある → サーバ情報を付与
                it = by_name[name]
                it.server_has = True
                it.server_id = server_id
                updated += 1
            else:
                # サーバのみの行を追加
                it = EmojiItem(
                    checked=False,
                    base_name=name,
                    desired_name=name,
                    local_path=None,
                    local_kib=None,
                    local_thumb=None,
                    server_id=server_id,
                    server_has=True,
                    status=ItemStatus.SERVER_ONLY,
                    detail="",
                )
                self.model.items.append(it)
                added += 1

        self.on_validate()
        self.model.layoutChanged.emit()
        self._set_status(f"サーバ絵文字を読み込みました。追加={added} 更新={updated}")

        self._prefetch_visible_thumbs()

    @Slot()
    def on_upload(self) -> None:
        """Upload checked LOCAL_ONLY items. Abort if any checked item is not uploadable."""
        # --- ALWAYS refresh server cache before upload ---
        self._set_status("サーバキャッシュ更新中...")
        if not self._refresh_server_cache_sync():
            return

        # 最新キャッシュで再判定
        self.on_validate()

        try:
            client = self._make_client()
        except Exception as e:
            self._error(str(e))
            return

        checked_rows = [(idx, it) for idx, it in enumerate(self.model.items) if it.checked]
        if not checked_rows:
            QMessageBox.information(self, "登録", "チェックされた行がありません。")
            return

        # 必ず最新の状態で判定したいので一度 validate をかける
        self.on_validate()

        # ブロッキング条件（方針固定）
        blockers = []
        for idx, it in checked_rows:
            if it.status in (ItemStatus.BOTH_SAME_NAME, ItemStatus.INVALID_NAME, ItemStatus.TOO_LARGE):
                blockers.append((idx, it))

        if blockers:
            lines = []
            for idx, it in blockers:
                lines.append(f"- 行 {idx+1}: :{it.desired_name}: 状態={it.status.name}")
            QMessageBox.critical(
                self,
                "登録中断",
                "チェックされた項目の中に、登録できないものがあります。\n\n"
                + "\n".join(lines)
                + "\n\n問題を解消してから再実行してください（例: 未使用の名前に変更／命名規則に合わせる／ファイルサイズを削減）。",
            )
            return

        # アップロード対象は LOCAL_ONLY かつ local_path があるものだけ
        tasks: list[tuple[int, str, Path]] = []
        for idx, it in checked_rows:
            if it.status != ItemStatus.LOCAL_ONLY:
                # ここに来るのは主に SERVER_ONLY 等。安全側で無視。
                continue
            if not it.local_path or not it.local_path.exists():
                QMessageBox.critical(self, "登録中断", f"行 {idx+1}.")
                return
            tasks.append((idx, it.desired_name, it.local_path))

        if not tasks:
            QMessageBox.information(self, "登録", "チェックされた行の中に登録可能（LOCAL_ONLY）な項目がありません。")
            return

        # 実行確認（事故防止）
        if QMessageBox.question(
            self,
            "登録確認",
            f"Upload {len(tasks)} emojis to server?\n\n"
            "注意：同名がサーバに存在する場合は中断します（自動連番なし・上書きなし）。",
        ) != QMessageBox.Yes:
            return

        self._set_status("登録中...")
        worker = BatchUploadWorker(client, tasks)
        worker.signals.progress.connect(self._set_status)
        worker.signals.finished.connect(self._on_upload_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)


    @Slot()
    def on_delete(self) -> None:
        """Delete checked items that exist on server (SERVER_ONLY or BOTH)."""
        try:
            client = self._make_client()
        except Exception as e:
            self._error(str(e))
            return

        checked_rows = [(idx, it) for idx, it in enumerate(self.model.items) if it.checked]
        if not checked_rows:
            QMessageBox.information(self, "削除", "チェックされた行がありません。")
            return

        # Always refresh cache before destructive ops
        self._set_status("サーバキャッシュ更新中...")
        if not self._refresh_server_cache_sync():
            return
        self.on_validate()

        # Delete対象：server_id がある行のみ
        tasks: list[tuple[int, str, str]] = []
        invalid = []
        for idx, it in checked_rows:
            name = (it.desired_name or it.base_name).strip()
            if not it.server_id:
                invalid.append((idx, name, it.status.name))
                continue
            tasks.append((idx, name, it.server_id))

        if not tasks:
            QMessageBox.information(
                self,
                "削除",
                "チェックされた行の中に削除可能な項目がありません。\n"
                "Server ID を持つ項目のみ削除できます。",
            )
            return

        # まとめて確認（名前一覧を見せる）
        preview = "\n".join([f"- :{name}:" for (_row, name, _id) in tasks[:30]])
        if len(tasks) > 30:
            preview += f"\n... and {len(tasks) - 30} more"

        extra = ""
        if invalid:
            extra_lines = "\n".join([f"- row {r+1}: :{n}: status={s} (no server_id)" for (r, n, s) in invalid[:10]])
            if len(invalid) > 10:
                extra_lines += f"\n... and {len(invalid) - 10} more"
            extra = "\n\n次の行は無視されます（Server ID がありません）:\n" + extra_lines

        if QMessageBox.question(
            self,
            "削除確認",
            f"{len(tasks)} 件の絵文字をサーバから削除しますか？\n\n{preview}{extra}\n\nこの操作は元に戻せません。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        self._set_status("削除中...")
        worker = BatchDeleteWorker(client, tasks)
        worker.signals.progress.connect(self._set_status)
        worker.signals.finished.connect(self._on_delete_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    @Slot(object)
    def _on_server_image_fetched(self, result: object) -> None:
        emoji_id, b = result
        assert isinstance(emoji_id, str)
        assert isinstance(b, (bytes, bytearray))

        self.server_image_cache[emoji_id] = bytes(b)

        # still selected?
        if self._current_selected_server_id != emoji_id:
            return

        self._show_image_bytes_in_preview(self.server_image_cache[emoji_id])
        self._set_status("サーバ画像を読み込みました。")

    @Slot(object)
    def _on_thumb_fetched(self, result: object) -> None:
        row, emoji_id, b = result
        b = bytes(b)
        self.server_image_cache[emoji_id] = b

        it = self.model.items[row]
        it.thumb_fetching = False

        px = self._make_icon_pixmap_from_bytes(b, logical_size=32)
        if px is not None:
            it.server_thumb = px
            idx = self.model.index(row, EmojiTableModel.COL_THUMB)
            self.model.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])

    def _show_local_preview(self, path: Path) -> None:
        # 以前の movie を止める
        if self._movie is not None:
            self._movie.stop()
        self._movie = None
        self._movie_buffer = None  # ローカルでは未使用
        self.preview_label.setMovie(None)

        if path.suffix.lower() == ".gif":
            # GIFはQMovieで再生しつつ、縦横比維持で毎フレーム描画
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("")

            self._movie = QMovie(str(path))

            def render_frame():
                if self._movie is None:
                    return
                px = self._movie.currentPixmap()
                if px.isNull():
                    return
                w = max(64, self.preview_label.width() - 16)
                h = max(64, self.preview_label.height() - 16)
                px = px.scaled(
                    w, h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(px)

            self._movie.frameChanged.connect(lambda _i: render_frame())
            self._movie.start()
            render_frame()
            return

        # 静止画
        px = QPixmap(str(path))
        if px.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("画像を読み込めませんでした")
            return

        w = max(64, self.preview_label.width() - 16)
        h = max(64, self.preview_label.height() - 16)
        px = px.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(px)
        self.preview_label.setText("")


    # -------------------------
    # Stubs for next steps
    # -------------------------
    @Slot()
    def on_rename_stub(self) -> None:
        """Placeholder: rename is generally 'recreate new name + delete old'."""
        QMessageBox.information(self, "リネーム（未実装）", "リネーム（再作成）の処理は未実装です。")

    @Slot()
    def on_selection_changed(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self._current_selected_server_id = None
            if self._movie is not None:
                self._movie.stop()
            self._movie = None
            self._movie_buffer = None
            self.preview_label.setMovie(None)
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("プレビュー")
            return

        row = sel[0].row()
        it = self.model.items[row]

        # まずローカル優先
        if it.local_path and it.local_path.exists():
            self._current_selected_server_id = None
            self._show_local_preview(it.local_path)
            return

        # ローカルが無い → サーバ画像プレビュー
        if not it.server_id:
            self._current_selected_server_id = None
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("ローカルファイルがなく、サーバIDもありません。")
            return

        emoji_id = it.server_id
        self._current_selected_server_id = emoji_id

        # キャッシュがあれば即表示
        if emoji_id in self.server_image_cache:
            self._show_image_bytes_in_preview(self.server_image_cache[emoji_id])
            return

        # 無ければ取得
        try:
            client = self._make_client()
        except Exception as e:
            self._error(str(e))
            return

        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("サーバ画像を取得中...")

        worker = FetchEmojiImageWorker(client, emoji_id)
        worker.signals.progress.connect(self._set_status)
        worker.signals.finished.connect(self._on_server_image_fetched)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)


    @Slot(object)
    def _on_upload_finished(self, result: object) -> None:
        """
        result: list of (row_index, ok, created_json_or_None, message)
        """
        assert isinstance(result, list)

        success = []
        failed = []

        for (row, ok, created, msg) in result:
            it = self.model.items[row]
            it.checked = False

            if ok and created:
                name = str(created.get("name", "")).strip().lower()
                if name:
                    self.server_cache[name] = created
                it.server_has = True
                it.server_id = str(created.get("id", "")) or None
                success.append(it.desired_name)
            else:
                failed.append((it.desired_name, msg))

        self.on_validate()
        self.model.layoutChanged.emit()

        # サマリ表示
        lines = []
        if success:
            lines.append(f"✓ 成功: {len(success)}")
            for n in success:
                lines.append(f"  - :{n}:")
        if failed:
            lines.append("")
            lines.append(f"✗ 失敗: {len(failed)}")
            for n, msg in failed:
                lines.append(f"  - :{n}: {msg}")

        QMessageBox.information(
            self,
            "登録結果",
            "\n".join(lines) if lines else "結果なし",
        )

        self._set_status(f"登録完了 成功={len(success)} 失敗={len(failed)}")

    @Slot(object)
    def _on_delete_finished(self, result: object) -> None:
        """
        result: list of (row_index, ok, _, message)
        """
        assert isinstance(result, list)

        success = []
        failed = []

        for (row, ok, _created, msg) in result:
            it = self.model.items[row]
            it.checked = False

            if ok:
                # server_cache から除去
                key = (it.desired_name or it.base_name).strip().lower()
                if key in self.server_cache:
                    del self.server_cache[key]

                # モデル側のサーバ情報もクリア（ローカルがあればLOCAL_ONLYへ、なければ空状態）
                it.server_has = False
                it.server_id = None
                success.append(key)
            else:
                name = (it.desired_name or it.base_name).strip().lower()
                failed.append((name, msg))

        self.on_validate()
        self.model.layoutChanged.emit()

        lines = []
        if success:
            lines.append(f"✓ 削除成功: {len(success)}")
            for n in success:
                lines.append(f"  - :{n}:")
        if failed:
            lines.append("")
            lines.append(f"✗ 失敗: {len(failed)}")
            for n, msg in failed:
                lines.append(f"  - :{n}: {msg}")

        QMessageBox.information(self, "削除結果", "\n".join(lines) if lines else "結果なし")
        self._set_status(f"削除完了 成功={len(success)} 失敗={len(failed)}")

        
    def _refresh_server_cache_sync(self) -> bool:
        """Refresh server_cache and session_user_id synchronously. Return True if success."""
        try:
            client = self._make_client()

            me = client.get_me()
            self.session_user_id = me["id"]

            items = client.list_emojis()
            cache: dict[str, dict] = {}
            for e in items:
                name = str(e.get("name", "")).strip().lower()
                if name:
                    cache[name] = e
            self.server_cache = cache
            return True
        except Exception as e:
            self._error(f"サーバキャッシュの更新に失敗しました:\n{e}")
            return False



def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
