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
    QSize
)
from PySide6.QtGui import QPixmap, QMovie

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
ALLOWED_NAME_RULE = "Allowed: lowercase a-z, digits 0-9, '_' and '-'. Pattern: ^[a-z0-9][a-z0-9_-]*$"

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

    def get_emoji_image(self, emoji_id: str) -> bytes:
        """GET /api/v4/emoji/{emoji_id}/image"""
        url = f"{self.server_url}/api/v4/emoji/{emoji_id}/image"
        r = self.session.get(url, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.content

    def create_emoji(self, name: str, image_path: Path) -> Dict[str, Any]:
        """POST /api/v4/emoji multipart: emoji(json) + image(file)"""
        url = f"{self.server_url}/api/v4/emoji"
        payload = {"name": name}
        files = {
            "emoji": (None, json.dumps(payload), "application/json"),
            "image": (image_path.name, image_path.read_bytes(), "application/octet-stream"),
        }
        r = self.session.post(url, files=files, timeout=self.timeout_sec)
        r.raise_for_status()
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

    @Slot()
    def run(self) -> None:
        results = []
        total = len(self.tasks)

        for i, (row, name, path) in enumerate(self.tasks, start=1):
            try:
                self.signals.progress.emit(f"Uploading {i}/{total}: :{name}:")
                created = self.client.create_emoji(name, path)
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
                self.signals.progress.emit(f"Deleting {i}/{total}: :{name}:")
                self.client.delete_emoji(emoji_id)
                results.append((row, True, None, ""))
            except Exception as e:
                results.append((row, False, None, str(e)))

        self.signals.finished.emit(results)


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
        "Use",
        "Icon",
        "Base Name",
        "Target Name (editable)",
        "Status",
        "Detail",
        "Local Path",
        "Server ID",
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

        if col == self.COL_THUMB:
            if role == Qt.DecorationRole and item.local_thumb is not None:
                return item.local_thumb
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
        self.setWindowTitle("Mattermost Custom Emoji Manager (Skeleton)")
        self.resize(1200, 700)

        self.thread_pool = QThreadPool.globalInstance()

        # cached server emojis: name(lower) -> (id, raw json)
        self.server_cache: Dict[str, Dict[str, Any]] = {}

        # UI
        self.server_edit = QLineEdit()
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)

        self.soft_warn_edit = QLineEdit(str(DEFAULT_SOFT_WARN_KIB))
        self.hard_max_edit = QLineEdit(str(DEFAULT_HARD_MAX_KIB))
        self.max_suffix_edit = QLineEdit(str(DEFAULT_MAX_SUFFIX))

        self.btn_connect = QPushButton("Connect & Refresh Server Cache")
        self.btn_pick_folder = QPushButton("Load Folder…")
        self.btn_validate = QPushButton("Validate / Recompute Status")
        self.btn_go_upload = QPushButton("GO: Upload Checked")
        self.btn_delete = QPushButton("Delete Checked (stub)")
        self.btn_rename = QPushButton("Rename Checked (recreate stub)")

        self.status_label = QLabel("Ready")

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

        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(260)

        # Layout
        top = QWidget()
        top_layout = QFormLayout(top)
        top_layout.addRow("Server URL", self.server_edit)
        top_layout.addRow("Token", self.token_edit)

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
        btn_row.addWidget(self.btn_validate)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_go_upload)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_rename)

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
        self.btn_validate.clicked.connect(self.on_validate)
        self.btn_go_upload.clicked.connect(self.on_upload)
        self.btn_delete.clicked.connect(self.on_delete)
        self.btn_rename.clicked.connect(self.on_rename_stub)

        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # seed empty
        self.model.set_items([])

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
            raise ValueError("Limits must be integers.")

    def _make_client(self) -> MattermostClient:
        server = self.server_edit.text().strip()
        token = self.token_edit.text().strip()
        if not server or not token:
            raise ValueError("Server URL and Token are required.")
        return MattermostClient(server, token)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

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

        self._set_status("Refreshing server cache...")
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

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self._set_status("Error")
        self._error(msg)

    @Slot()
    def on_load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Emoji Folder")
        if not folder:
            return

        self._set_status(f"Loading folder: {folder}")
        folder_path = Path(folder)

        # Build local items
        local_files = []
        for p in folder_path.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                local_files.append(p)

        items: List[EmojiItem] = []
        for p in sorted(local_files):
            base = p.stem  # filename without extension
            ok, norm, reason = validate_name(base)
            size_kib = kib_of_file(p)

            thumb = QPixmap(str(p))
            if not thumb.isNull():
                thumb = thumb.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            it = EmojiItem(
                checked=False,
                base_name=base,
                desired_name=norm if ok else base,
                local_path=p,
                local_kib=size_kib,
                local_thumb=thumb,
            )
            # status computed later
            items.append(it)

        self.model.set_items(items)
        self._set_status(f"Loaded {len(items)} local files. Click Validate.")

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

    # -------------------------
    # Stubs for next steps
    # -------------------------
    @Slot()
    def on_upload(self) -> None:
        """Upload checked LOCAL_ONLY items. Abort if any checked item is not uploadable."""
        # --- ALWAYS refresh server cache before upload ---
        self._set_status("Refreshing server cache...")
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
            QMessageBox.information(self, "Upload", "No rows checked.")
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
                lines.append(f"- row {idx+1}: :{it.desired_name}: status={it.status.name}")
            QMessageBox.critical(
                self,
                "Upload aborted",
                "Some checked items cannot be uploaded.\n\n"
                + "\n".join(lines)
                + "\n\nResolve the issues (rename to a non-existing name, fix name rule, or reduce file size) and try again.",
            )
            return

        # アップロード対象は LOCAL_ONLY かつ local_path があるものだけ
        tasks: list[tuple[int, str, Path]] = []
        for idx, it in checked_rows:
            if it.status != ItemStatus.LOCAL_ONLY:
                # ここに来るのは主に SERVER_ONLY 等。安全側で無視。
                continue
            if not it.local_path or not it.local_path.exists():
                QMessageBox.critical(self, "Upload aborted", f"Missing local file for row {idx+1}.")
                return
            tasks.append((idx, it.desired_name, it.local_path))

        if not tasks:
            QMessageBox.information(self, "Upload", "No uploadable items (LOCAL_ONLY) among checked rows.")
            return

        # 実行確認（事故防止）
        if QMessageBox.question(
            self,
            "Confirm Upload",
            f"Upload {len(tasks)} emojis to server?\n\n"
            "Note: If a name already exists on server, this tool will abort (no auto-suffix, no overwrite).",
        ) != QMessageBox.Yes:
            return

        self._set_status("Uploading...")
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
            QMessageBox.information(self, "Delete", "No rows checked.")
            return

        # Always refresh cache before destructive ops
        self._set_status("Refreshing server cache...")
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
                "Delete",
                "No deletable items among checked rows.\n"
                "Only items with Server ID can be deleted.",
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
            extra = "\n\nThese checked rows will be ignored (no Server ID):\n" + extra_lines

        if QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {len(tasks)} emojis from server?\n\n{preview}{extra}\n\nThis operation cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        self._set_status("Deleting...")
        worker = BatchDeleteWorker(client, tasks)
        worker.signals.progress.connect(self._set_status)
        worker.signals.finished.connect(self._on_delete_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    @Slot()
    def on_rename_stub(self) -> None:
        """Placeholder: rename is generally 'recreate new name + delete old'."""
        QMessageBox.information(self, "Rename (stub)", "Rename (recreate) logic is not implemented yet.")

    @Slot()
    def on_selection_changed(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self._movie = None
            self.preview_label.setMovie(None)
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Preview")
            return

        row = sel[0].row()
        it = self.model.items[row]

        if not it.local_path or not it.local_path.exists():
            self._movie = None
            self.preview_label.setMovie(None)
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("No local preview")
            return

        # GIFはアニメで再生
        if it.local_path.suffix.lower() == ".gif":
            # 以前の movie を止める
            if self._movie is not None:
                self._movie.stop()

            self._movie = QMovie(str(it.local_path))
            self.preview_label.setMovie(None)      # QLabelのmovie機能は使わない
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("")

            def render_frame():
                if self._movie is None:
                    return
                px = self._movie.currentPixmap()
                if px.isNull():
                    return
                max_size = getattr(self, "preview_max", 256)
                px = px.scaled(
                    max_size, max_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(px)

            # フレーム更新のたびに描画（縦横比を維持）
            self._movie.frameChanged.connect(lambda _i: render_frame())
            self._movie.start()
            render_frame()
            return


        # それ以外は静止画
        self._movie = None
        self.preview_label.setMovie(None)

        px = QPixmap(str(it.local_path))
        if not px.isNull():
            px = px.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_label.setPixmap(px)
            self.preview_label.setText("")
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Failed to load image")

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
            lines.append(f"✓ Success: {len(success)}")
            for n in success:
                lines.append(f"  - :{n}:")
        if failed:
            lines.append("")
            lines.append(f"✗ Failed: {len(failed)}")
            for n, msg in failed:
                lines.append(f"  - :{n}: {msg}")

        QMessageBox.information(
            self,
            "Upload result",
            "\n".join(lines) if lines else "No result.",
        )

        self._set_status(f"Upload finished. success={len(success)} failed={len(failed)}")

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
            lines.append(f"✓ Deleted: {len(success)}")
            for n in success:
                lines.append(f"  - :{n}:")
        if failed:
            lines.append("")
            lines.append(f"✗ Failed: {len(failed)}")
            for n, msg in failed:
                lines.append(f"  - :{n}: {msg}")

        QMessageBox.information(self, "Delete result", "\n".join(lines) if lines else "No result.")
        self._set_status(f"Delete finished. deleted={len(success)} failed={len(failed)}")



    def _refresh_server_cache_sync(self) -> bool:
        """Refresh server_cache synchronously. Return True if success."""
        try:
            client = self._make_client()
            items = client.list_emojis()
            cache: dict[str, dict] = {}
            for e in items:
                name = str(e.get("name", "")).strip().lower()
                if name:
                    cache[name] = e
            self.server_cache = cache
            return True
        except Exception as e:
            self._error(f"Failed to refresh server cache:\n{e}")
            return False


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
