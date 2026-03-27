"""Utility helpers for file IO and text handling."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save dictionary as pretty JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def clean_text(value: str) -> str:
    """Normalize text for vectorization."""
    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text

