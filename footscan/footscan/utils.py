"""Utility helpers for FootScan."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_iso() -> str:
    """Return local timestamp in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def load_image_any(path: Path) -> np.ndarray:
    """Load an image from disk supporting PNG/JPG/TIF and unicode paths."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de imagen: {path}")

    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save image to disk handling unicode paths on Windows."""
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"No se pudo codificar imagen para guardar en {path}")
    encoded.tofile(str(path))


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses/numpy objects to JSON-compatible structures."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON with UTF-8 and pretty formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def pixels_to_mm(value_px: float, dpi: Optional[float]) -> Optional[float]:
    """Convert pixels to millimeters if DPI is provided."""
    if dpi is None or dpi <= 0:
        return None
    return float(value_px) * 25.4 / float(dpi)


def area_px_to_mm2(area_px: float, dpi: Optional[float]) -> Optional[float]:
    """Convert area in px^2 to mm^2 if DPI is provided."""
    if dpi is None or dpi <= 0:
        return None
    scale = 25.4 / float(dpi)
    return float(area_px) * scale * scale


def clamp_bbox(x: int, y: int, w: int, h: int, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Clamp bounding box to image boundaries."""
    max_h, max_w = shape
    x = max(0, min(x, max_w - 1))
    y = max(0, min(y, max_h - 1))
    w = max(1, min(w, max_w - x))
    h = max(1, min(h, max_h - y))
    return x, y, w, h
