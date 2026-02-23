"""Preprocessing: grayscale conversion and illumination correction."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _odd_kernel_size(size: int) -> int:
    """Return the nearest odd kernel size >= 3."""
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return size


def preprocess_image(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Preprocess image with flat-field correction and denoising.

    Returns a dictionary of intermediate images for debug/export.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    k = _odd_kernel_size(min(gray.shape[:2]) // 8)
    background = cv2.GaussianBlur(gray, (k, k), 0)

    gray_f = gray.astype(np.float32)
    bg_f = background.astype(np.float32) + 1.0
    corrected = gray_f / bg_f
    corrected *= np.mean(bg_f)

    corrected_norm = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    corrected_blur = cv2.GaussianBlur(corrected_norm, (5, 5), 0)

    return {
        "gray": gray,
        "background": background,
        "corrected": corrected_norm,
        "denoised": corrected_blur,
    }
