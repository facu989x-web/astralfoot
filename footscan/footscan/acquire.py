"""Image acquisition for FootScan (scanner + file fallback)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .utils import ensure_dir, timestamp_iso


def _try_wia_scan(output_path: Path, dpi: int) -> bool:
    """Stub for WIA scanner acquisition.

    Returns True on success and False when no scanner backend is available.
    """
    try:
        import win32com.client  # type: ignore  # Optional dependency
    except Exception:
        return False

    # Placeholder: WIA implementation skeleton. This intentionally returns False
    # so the app remains functional without scanner-specific dependencies.
    _ = (win32com.client, dpi, output_path)
    return False


def _try_twain_scan(output_path: Path, dpi: int) -> bool:
    """Stub for TWAIN scanner acquisition.

    Returns True on success and False when no scanner backend is available.
    """
    try:
        import twain  # type: ignore  # Optional dependency
    except Exception:
        return False

    # Placeholder: TWAIN implementation skeleton. This intentionally returns False
    # so the app remains functional without scanner-specific dependencies.
    _ = (twain, dpi, output_path)
    return False


def acquire_from_scanner(output_dir: Path, dpi: int = 300) -> Path:
    """Attempt scanner acquisition via WIA/TWAIN and return saved raw image path.

    Raises:
        RuntimeError: If no scanner backend succeeded.
    """
    ensure_dir(output_dir)
    out_path = output_dir / f"scan_raw_{timestamp_iso().replace(':', '-')}.png"

    if _try_wia_scan(out_path, dpi):
        return out_path
    if _try_twain_scan(out_path, dpi):
        return out_path

    raise RuntimeError(
        "No se pudo adquirir desde escáner (WIA/TWAIN no disponible o no implementado). "
        "Usá '--input <ruta_imagen>' para fallback por archivo."
    )


def acquire_from_file(input_path: Path, output_dir: Path) -> Path:
    """Copy input image into output directory as raw source for traceability."""
    if not input_path.exists():
        raise FileNotFoundError(f"No existe archivo de entrada: {input_path}")
    ensure_dir(output_dir)
    dst = output_dir / f"raw_from_file_{timestamp_iso().replace(':', '-')}{input_path.suffix.lower()}"
    shutil.copy2(input_path, dst)
    return dst
