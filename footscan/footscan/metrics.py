"""Metric extraction for footprint analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    quality_status: str = "ok"
    quality_warnings: Tuple[str, ...] = ()


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only largest connected component in binary uint8 image."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == idx] = 255
    return out


def _build_metrics_mask(base_mask: np.ndarray, corrected_gray: np.ndarray) -> np.ndarray:
    """Build stricter mask for geometry metrics to avoid overfilled silhouettes."""
    inside = corrected_gray[base_mask > 0]
    if inside.size < 100:
        return base_mask

    p35 = float(np.percentile(inside, 35))
    p65 = float(np.percentile(inside, 65))

    dark_cand = np.zeros_like(base_mask)
    bright_cand = np.zeros_like(base_mask)
    dark_cand[(base_mask > 0) & (corrected_gray <= p65)] = 255
    bright_cand[(base_mask > 0) & (corrected_gray >= p35)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidates = []
    for cand in (dark_cand, bright_cand):
        c = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel)
        c = _largest_component(c)
        candidates.append(c)

    base_area = float(np.count_nonzero(base_mask))
    best = base_mask
    best_score = -1.0
    for c in candidates:
        area = float(np.count_nonzero(c))
        if area <= 0:
            continue
        area_ratio = area / max(base_area, 1.0)
        # prefer masks that keep most plantar region but avoid full overfill.
        score = 1.0 - min(abs(area_ratio - 0.7), 0.7) / 0.7
        if 0.35 <= area_ratio <= 0.98 and score > best_score:
            best = c
            best_score = score
    return best


def _build_heatmap_mask(base_mask: np.ndarray) -> np.ndarray:
    """Build stable mask for heatmap rendering (favor continuity over strictness)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
    m = _largest_component(m)
    return m


def _quality_checks(
    length_px: float,
    forefoot_width_px: float,
    heel_width_px: float,
    midfoot_min_width_px: float,
    arch_index: float,
    area_px: float,
    ys: np.ndarray,
    xs: np.ndarray,
) -> Tuple[str, Tuple[str, ...]]:
    warnings: List[str] = []

    if length_px <= 0 or forefoot_width_px <= 0 or heel_width_px <= 0:
        warnings.append("Medidas geométricas inválidas (<= 0).")

    tol = max(8.0, 0.015 * max(length_px, 1.0))
    if abs(forefoot_width_px - heel_width_px) < tol and abs(midfoot_min_width_px - forefoot_width_px) < tol:
        warnings.append("Anchos casi idénticos en antepié/talón/mediopié; posible sobre-segmentación.")

    if arch_index >= 95.0 or arch_index <= 5.0:
        warnings.append("Índice de arco extremo; revisar máscara y calidad de captura.")

    if midfoot_min_width_px > (forefoot_width_px * 0.98):
        warnings.append("Mediopié igual o más ancho que antepié; posible contaminación lateral en máscara.")

    if xs.size > 10 and ys.size > 10:
        bbox_area = float((np.max(xs) - np.min(xs) + 1) * (np.max(ys) - np.min(ys) + 1))
        fill_ratio = area_px / max(bbox_area, 1.0)
        if fill_ratio > 0.88:
            warnings.append("Huella muy compacta respecto al bounding box; posible relleno excesivo.")
        if fill_ratio < 0.18:
            warnings.append("Huella muy dispersa/fragmentada; posible sub-segmentación.")

    status = "ok" if not warnings else "warn"
    return status, tuple(warnings)


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


def _section_width_at_l(
    l_norm: np.ndarray,
    w: np.ndarray,
    l_start: float,
    l_end: float,
    mode: str,
) -> Tuple[float, float]:
    """Compute width extrema inside a longitudinal section.

    Returns (width_px, representative_l).
    """
    bins = np.linspace(l_start, l_end, 40)
    widths = []
    centers = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (l_norm >= lo) & (l_norm < hi)
        if np.any(m):
            q95 = float(np.percentile(w[m], 95))
            q05 = float(np.percentile(w[m], 5))
            widths.append(q95 - q05)
            centers.append(float((lo + hi) * 0.5))

    if not widths:
        return 0.0, float((l_start + l_end) * 0.5)

    arr = np.array(widths, dtype=np.float32)
    if mode == "max":
        idx = int(np.argmax(arr))
    else:
        idx = int(np.argmin(arr))
    return float(arr[idx]), centers[idx]


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
    original_gray: np.ndarray,
    foot_hint: str = "auto",
    dpi: Optional[float] = None,
) -> Tuple[FootMetrics, np.ndarray]:
    """Compute geometric and contact-intensity metrics from segmented footprint."""
    metric_mask = _build_metrics_mask(mask, corrected_gray)
    ys, xs = np.where(metric_mask > 0)
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

    forefoot_width_px = float(np.percentile(w[fore_m], 95) - np.percentile(w[fore_m], 5)) if np.any(fore_m) else 0.0
    heel_width_px = float(np.percentile(w[heel_m], 95) - np.percentile(w[heel_m], 5)) if np.any(heel_m) else 0.0
    midfoot_min_width_px = float(np.percentile(w[mid_m], 95) - np.percentile(w[mid_m], 5)) if np.any(mid_m) else 0.0

    points_lw = np.column_stack([l_norm.astype(np.float32), w.astype(np.float32)])
    fore_max, fore_l = _section_width_at_l(l_norm, w, 2.0 / 3.0, 0.98, mode="max")
    mid_min, mid_l = _section_width_at_l(l_norm, w, 0.34, 0.66, mode="min")

    if fore_max > 0:
        arch_index = (mid_min / fore_max) * 100.0
    else:
        arch_index = 0.0

    area_px = float(np.count_nonzero(metric_mask))
    moments = cv2.moments(metric_mask)
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
        side = "left" if centroid[0] > (metric_mask.shape[1] / 2.0) else "right"

    # Contact intensity map (relative): computed on stable heatmap mask, independent
    # from stricter metrics mask to avoid fragmented/spotty rendering.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(original_gray)
    enhanced_f = enhanced.astype(np.float32)

    heatmap_mask = _build_heatmap_mask(mask)

    inside_gray = enhanced_f[heatmap_mask > 0]
    if inside_gray.size > 0:
        lo = float(np.percentile(inside_gray, 2))
        hi = float(np.percentile(inside_gray, 98))
        if hi > lo:
            intensity_rel = np.clip((enhanced_f - lo) / (hi - lo), 0.0, 1.0)
        else:
            intensity_rel = np.zeros_like(enhanced_f, dtype=np.float32)
    else:
        intensity_rel = np.zeros_like(enhanced_f, dtype=np.float32)

    dist = cv2.distanceTransform((heatmap_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if float(np.max(dist)) > 0:
        dist_rel = dist / float(np.max(dist))
    else:
        dist_rel = np.zeros_like(dist, dtype=np.float32)

    # Blend brightness-driven contact with center weighting to avoid black foot interiors.
    contact_rel = (0.9 * intensity_rel) + (0.1 * dist_rel.astype(np.float32))
    contact_rel = cv2.GaussianBlur(contact_rel, (0, 0), sigmaX=1.3, sigmaY=1.3)
    contact_rel *= (heatmap_mask > 0).astype(np.float32)

    inside = contact_rel[heatmap_mask > 0]
    if inside.size > 0:
        lo = float(np.percentile(inside, 1))
        hi = float(np.percentile(inside, 99))
        if hi > lo:
            contact_rel = np.clip((contact_rel - lo) / (hi - lo), 0.0, 1.0)

        # Keep interior visible in heatmap while preserving dynamic contrast.
        contact_rel[heatmap_mask > 0] = 0.08 + (0.92 * contact_rel[heatmap_mask > 0])

    quality_status, quality_warnings = _quality_checks(
        length_px,
        forefoot_width_px,
        heel_width_px,
        midfoot_min_width_px,
        float(arch_index),
        area_px,
        ys,
        xs,
    )

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
        quality_status=quality_status,
        quality_warnings=quality_warnings,
    )
    return metrics, contact_rel
