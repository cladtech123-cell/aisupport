"""Persistent local storage for chat history, deduplication, and runtime state."""

from __future__ import annotations

from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Union


def ensure_private_dir(path: Union[str, Path]) -> Path:
    """Create directory if needed and set mode 0700."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target, 0o700)
    except OSError:
        pass
    return target


class JsonFile:
    """Atomic JSON file reader/writer with 0600 permissions and safe fallback."""

    def __init__(self, path: Union[str, Path], default: Any) -> None:
        self.path = Path(path)
        self._default = default

    def read(self) -> Any:
        if not self.path.exists():
            return copy.deepcopy(self._default)
        try:
            raw = self.path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return copy.deepcopy(self._default)

    def write(self, data: Any) -> None:
        ensure_private_dir(self.path.parent)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            os.replace(str(tmp_path), str(self.path))
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


class HistoryStore:
    """Per-(connection_id, chat_id) bounded message history store."""

    def __init__(self, path: Union[str, Path], limit: int = 30) -> None:
        self.file = JsonFile(path, {})
        self.limit = max(1, int(limit))
        raw = self.file.read()
        self._data: Dict[str, List[Dict[str, Any]]] = (
            raw if isinstance(raw, dict) else {}
        )

    @staticmethod
    def _key(connection_id: str, chat_id: int) -> str:
        return f"{connection_id}:{chat_id}"

    def get(self, connection_id: str, chat_id: int) -> List[Dict[str, Any]]:
        key = self._key(connection_id, chat_id)
        items = self._data.get(key, [])
        if not isinstance(items, list):
            return []
        return [dict(item) for item in items[-self.limit :]]

    def add(
        self,
        connection_id: str,
        chat_id: int,
        *,
        role: str,
        name: str,
        text: str,
        lang: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        key = self._key(connection_id, chat_id)
        items = self._data.get(key)
        if not isinstance(items, list):
            items = []
        items.append(
            {
                "role": role,
                "name": name,
                "text": text,
                "lang": lang,
                "ts": float(ts if ts is not None else time.time()),
            }
        )
        if len(items) > self.limit:
            items = items[-self.limit :]
        self._data[key] = items
        self.file.write(self._data)

    def last_language(self, connection_id: str, chat_id: int) -> Optional[str]:
        items = self.get(connection_id, chat_id)
        for item in reversed(items):
            lang = item.get("lang")
            if lang in ("uz", "ru", "en"):
                return str(lang)
        return None

    def chat_count(self) -> int:
        return sum(1 for items in self._data.values() if items)


class SeenMessages:
    """Bounded persistent deduplication set backed by an OrderedDict."""

    def __init__(self, path: Union[str, Path], capacity: int = 5000) -> None:
        self.file = JsonFile(path, [])
        self.capacity = max(1, int(capacity))
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        raw = self.file.read()
        if isinstance(raw, list):
            for key in raw[-self.capacity :]:
                if isinstance(key, str):
                    self._seen[key] = None

    def first_time(self, key: str) -> bool:
        if key in self._seen:
            self._seen.move_to_end(key)
            return False
        self._seen[key] = None
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        self.file.write(list(self._seen.keys()))
        return True


class RuntimeState:
    """Persistent runtime state for pause flag and rate-limit cooldown."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.file = JsonFile(path, {"paused": False, "limit_until": 0.0})
        raw = self.file.read()
        if not isinstance(raw, dict):
            raw = {}
        self.paused: bool = bool(raw.get("paused", False))
        try:
            self.limit_until: float = float(raw.get("limit_until", 0.0) or 0.0)
        except (TypeError, ValueError):
            self.limit_until = 0.0

    def save(self) -> None:
        self.file.write(
            {
                "paused": self.paused,
                "limit_until": self.limit_until,
            }
        )

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        self.save()

    def set_limit_until(self, timestamp: float) -> None:
        self.limit_until = float(timestamp)
        self.save()
