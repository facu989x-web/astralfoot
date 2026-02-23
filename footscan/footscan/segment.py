"""Segmentation of plantar footprint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Container for segmentation artifacts."""

    mask: np.ndarray
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    debug_images: Dict[str, np.ndarray]


def _pick_reasonable_binary(candidates: Dict[str, np.ndarray]) -> np.ndarray:
    """Choose threshold result based on expected foreground ratio."""
    best = None
    best_score = -1.0
    for _, binary in candidates.items():
        ratio = float(np.count_nonzero(binary)) / binary.size
        score = 1.0 - min(abs(ratio - 0.18), 0.18) / 0.18
        if 0.01 <= ratio <= 0.75 and score > best_score:
            best = binary
            best_score = score
    if best is not None:
        return best
    return next(iter(candidates.values()))


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary)

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == largest_idx] = 255
    return out


def segment_footprint(preprocessed_gray: np.ndarray) -> SegmentationResult:
    """Segment the footprint mask from a corrected grayscale image."""
    _, otsu_inv = cv2.threshold(preprocessed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(preprocessed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive_inv = cv2.adaptiveThreshold(
        preprocessed_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        5,
    )

    binary = _pick_reasonable_binary({"otsu_inv": otsu_inv, "otsu": otsu, "adaptive_inv": adaptive_inv})

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    largest = _largest_component(closed)

    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se pudo segmentar la huella. Probá con otra imagen o mejor contraste.")

    contour = max(contours, key=cv2.contourArea)

    # Fill principal contour to get a solid plantar contact region.
    # Some scans keep strong borders but weak interior texture; filling avoids
    # hollow masks that later produce contour-only heatmaps.
    filled_mask = np.zeros_like(largest)
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)

    x, y, w, h = cv2.boundingRect(contour)

    return SegmentationResult(
        mask=filled_mask,
        contour=contour,
        bbox=(x, y, w, h),
        debug_images={
            "otsu_inv": otsu_inv,
            "otsu": otsu,
            "adaptive_inv": adaptive_inv,
            "opened": opened,
            "closed": closed,
            "filled": filled_mask,
        },
    )
