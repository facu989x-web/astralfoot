"""Metric extraction for footprint analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import area_px_to_mm2, pixels_to_mm


@dataclass
class FootMetrics:
    """Computed footprint metrics."""

    foot_side: str
    length_px: float
    forefoot_width_px: float
    heel_width_px: float
    midfoot_min_width_px: float
    arch_index_chippaux_smirak: float
    contact_area_px2: float
    centroid_xy_px: Tuple[float, float]
    principal_axis_angle_deg: float
    forefoot_line_yx: Tuple[Tuple[float, float], Tuple[float, float]]
    heel_line_yx: Tuple[Tuple[float, float], Tuple[float, float]]
    midfoot_line_yx: Tuple[Tuple[float, float], Tuple[float, float]]
    length_endpoints_yx: Tuple[Tuple[float, float], Tuple[float, float]]
    length_mm: Optional[float] = None
    forefoot_width_mm: Optional[float] = None
    heel_width_mm: Optional[float] = None
    midfoot_min_width_mm: Optional[float] = None
    contact_area_mm2: Optional[float] = None


def _pca_axis(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid and major principal axis unit vector for points in xy."""
    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    major = major / (np.linalg.norm(major) + 1e-8)
    return mean, major


def _line_width(points_lw: np.ndarray, mask_l: np.ndarray) -> Tuple[float, float, float, Tuple[float, float]]:
    """Compute width profile over normalized longitudinal axis.

    Returns (max_width, min_width, chosen_l_for_max, (wmin_l, wmax_l)).
    """
    bins = np.linspace(0.0, 1.0, 120)
    widths = np.full(bins.shape[0] - 1, np.nan, dtype=np.float32)
    centers = (bins[:-1] + bins[1:]) * 0.5

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (mask_l >= lo) & (mask_l < hi)
        if np.any(m):
            w_vals = points_lw[m, 1]
            widths[i] = float(np.max(w_vals) - np.min(w_vals))

    valid = ~np.isnan(widths)
    if not np.any(valid):
        return 0.0, 0.0, 0.0, (0.0, 0.0)

    max_idx = int(np.nanargmax(widths))
    min_idx = int(np.nanargmin(np.where(valid, widths, np.nanmax(widths))))
    return float(widths[max_idx]), float(widths[min_idx]), float(centers[max_idx]), (float(centers[min_idx]), float(centers[max_idx]))


def _line_endpoints_from_l(
    points_lw: np.ndarray,
    points_xy: np.ndarray,
    target_l: float,
    tol: float = 0.01,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get two edge points in yx for a longitudinal location."""
    m = np.abs(points_lw[:, 0] - target_l) <= tol
    if not np.any(m):
        m = np.abs(points_lw[:, 0] - target_l) <= (tol * 2.0)
    if not np.any(m):
        idx = np.argsort(np.abs(points_lw[:, 0] - target_l))[:25]
        pts = points_xy[idx]
    else:
        pts = points_xy[m]

    order = np.argsort(pts[:, 1])
    p1 = pts[order[0]]
    p2 = pts[order[-1]]
    return (float(p1[1]), float(p1[0])), (float(p2[1]), float(p2[0]))


def compute_metrics(
    mask: np.ndarray,
    corrected_gray: np.ndarray,
    foot_hint: str = "auto",
    dpi: Optional[float] = None,
) -> Tuple[FootMetrics, np.ndarray]:
    """Compute geometric and contact-intensity metrics from segmented footprint."""
    ys, xs = np.where(mask > 0)
    if xs.size < 50:
        raise ValueError("Máscara insuficiente para calcular métricas.")

    points_xy = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centroid_xy, axis_xy = _pca_axis(points_xy)
    perp_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float32)

    rel = points_xy - centroid_xy
    l = rel @ axis_xy
    w = rel @ perp_xy
    l_min, l_max = float(np.min(l)), float(np.max(l))
    length_px = float(l_max - l_min)

    l_norm = (l - l_min) / (length_px + 1e-8)

    end_a = w[l_norm < 0.2]
    end_b = w[l_norm > 0.8]
    width_a = float(np.max(end_a) - np.min(end_a)) if end_a.size else 0.0
    width_b = float(np.max(end_b) - np.min(end_b)) if end_b.size else 0.0

    if width_a > width_b:
        l_norm = 1.0 - l_norm
        l = -l
        axis_xy = -axis_xy

    fore_m = l_norm >= (2.0 / 3.0)
    heel_m = l_norm <= (1.0 / 3.0)
    mid_m = (l_norm > (1.0 / 3.0)) & (l_norm < (2.0 / 3.0))

    forefoot_width_px = float(np.max(w[fore_m]) - np.min(w[fore_m])) if np.any(fore_m) else 0.0
    heel_width_px = float(np.max(w[heel_m]) - np.min(w[heel_m])) if np.any(heel_m) else 0.0
    midfoot_min_width_px = float(np.max(w[mid_m]) - np.min(w[mid_m])) if np.any(mid_m) else 0.0

    points_lw = np.column_stack([l_norm.astype(np.float32), w.astype(np.float32)])
    fore_max, mid_min, fore_l, (mid_l, _) = _line_width(points_lw[fore_m | mid_m | heel_m], l_norm)

    if fore_max > 0:
        arch_index = (mid_min / fore_max) * 100.0
    else:
        arch_index = 0.0

    area_px = float(np.count_nonzero(mask))
    moments = cv2.moments(mask)
    if abs(moments["m00"]) < 1e-8:
        centroid = (float(centroid_xy[0]), float(centroid_xy[1]))
    else:
        centroid = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))

    theta = float(np.degrees(np.arctan2(axis_xy[1], axis_xy[0])))

    pt_heel = centroid_xy + axis_xy * np.min(l)
    pt_toe = centroid_xy + axis_xy * np.max(l)

    fore_line = _line_endpoints_from_l(points_lw, points_xy, fore_l if fore_l > 0 else 0.83)
    heel_line = _line_endpoints_from_l(points_lw, points_xy, 0.17)
    mid_line = _line_endpoints_from_l(points_lw, points_xy, mid_l if mid_l > 0 else 0.5)

    side = foot_hint
    if foot_hint == "auto":
        side = "left" if centroid[0] > (mask.shape[1] / 2.0) else "right"

    corrected = corrected_gray.astype(np.float32)
    normalized = cv2.normalize(corrected, None, 0, 1.0, cv2.NORM_MINMAX)
    contact_rel = (1.0 - normalized) * (mask > 0).astype(np.float32)

    metrics = FootMetrics(
        foot_side=side,
        length_px=length_px,
        forefoot_width_px=forefoot_width_px,
        heel_width_px=heel_width_px,
        midfoot_min_width_px=midfoot_min_width_px,
        arch_index_chippaux_smirak=float(arch_index),
        contact_area_px2=area_px,
        centroid_xy_px=centroid,
        principal_axis_angle_deg=theta,
        forefoot_line_yx=fore_line,
        heel_line_yx=heel_line,
        midfoot_line_yx=mid_line,
        length_endpoints_yx=((float(pt_heel[1]), float(pt_heel[0])), (float(pt_toe[1]), float(pt_toe[0]))),
        length_mm=pixels_to_mm(length_px, dpi),
        forefoot_width_mm=pixels_to_mm(forefoot_width_px, dpi),
        heel_width_mm=pixels_to_mm(heel_width_px, dpi),
        midfoot_min_width_mm=pixels_to_mm(midfoot_min_width_px, dpi),
        contact_area_mm2=area_px_to_mm2(area_px, dpi),
    )
    return metrics, contact_rel
