"""FootScan CLI - MVP for plantar footprint analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from footscan.acquire import acquire_from_file, acquire_from_scanner
from footscan.metrics import compute_metrics
from footscan.preprocess import preprocess_image
from footscan.report import create_report_pdf
from footscan.segment import segment_footprint
from footscan.utils import ensure_dir, load_image_any, save_image, save_json, timestamp_iso


def _draw_overlay(image_bgr, contour, metrics) -> Any:
    overlay = image_bgr.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    heel_pt, toe_pt = metrics.length_endpoints_yx
    cv2.line(overlay, (int(heel_pt[1]), int(heel_pt[0])), (int(toe_pt[1]), int(toe_pt[0])), (255, 200, 0), 2)

    for line, color in [
        (metrics.forefoot_line_yx, (255, 0, 255)),
        (metrics.heel_line_yx, (0, 140, 255)),
        (metrics.midfoot_line_yx, (0, 255, 255)),
    ]:
        p1, p2 = line
        cv2.line(overlay, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), color, 2)

    cx, cy = metrics.centroid_xy_px
    cv2.circle(overlay, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    labels = [
        f"Length: {metrics.length_px:.1f}px",
        f"Forefoot width: {metrics.forefoot_width_px:.1f}px",
        f"Heel width: {metrics.heel_width_px:.1f}px",
        f"Arch idx (C-S): {metrics.arch_index_chippaux_smirak:.1f}%",
    ]
    y = 30
    for text in labels:
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        y += 24

    return overlay


def _draw_bbox(image_bgr, crop_meta: Dict[str, Any]) -> Any:
    """Draw ROI bounding rectangle over full image for debug."""
    out = image_bgr.copy()
    x, y, w, h = int(crop_meta["x"]), int(crop_meta["y"]), int(crop_meta["w"]), int(crop_meta["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.putText(out, "ROI crop", (x + 8, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_mask_overlay(image_bgr, mask) -> Any:
    """Overlay mask in semi-transparent color for debug inspection."""
    out = image_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[:, :] = (0, 220, 255)
    m = mask > 0
    out[m] = cv2.addWeighted(out[m], 0.45, color_layer[m], 0.55, 0)
    return out


def _maybe_resize_for_processing(
    image_bgr,
    dpi: Optional[int],
    max_pixels: int = 40_000_000,
) -> Tuple[Any, Optional[float], Optional[Dict[str, Any]]]:
    """Resize very large inputs to keep processing stable.

    Returns resized image, effective dpi and resize metadata (or None if unchanged).
    """
    h, w = image_bgr.shape[:2]
    total = h * w
    if total <= max_pixels:
        return image_bgr, float(dpi) if dpi else dpi, None

    scale = (max_pixels / float(total)) ** 0.5
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    effective_dpi = (float(dpi) * scale) if dpi else dpi

    meta = {
        "original_shape": [h, w],
        "processed_shape": [new_h, new_w],
        "scale": scale,
        "max_pixels": max_pixels,
    }
    return resized, effective_dpi, meta


def _crop_to_bbox(image_bgr, bbox: Tuple[int, int, int, int], margin_ratio: float = 0.12, min_margin_px: int = 24):
    """Crop image around bbox with a margin to suppress irrelevant background."""
    x, y, w, h = bbox
    ih, iw = image_bgr.shape[:2]

    margin = int(max(min_margin_px, round(max(w, h) * margin_ratio)))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(iw, x + w + margin)
    y1 = min(ih, y + h + margin)

    cropped = image_bgr[y0:y1, x0:x1].copy()
    crop_meta = {
        "x": int(x0),
        "y": int(y0),
        "w": int(x1 - x0),
        "h": int(y1 - y0),
        "margin_px": int(margin),
    }
    return cropped, crop_meta




def _derive_findings(mask: np.ndarray, contact_rel: np.ndarray, metrics) -> Dict[str, Any]:
    """Build simple anatomical-zone findings for technical triage."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return {
            "zones": {},
            "flags": ["Máscara vacía; repetir captura."],
            "action": "repeat_scan",
            "review_score": 100,
            "severity": "high",
        }

    heel_pt, toe_pt = metrics.length_endpoints_yx
    heel_xy = np.array([float(heel_pt[1]), float(heel_pt[0])], dtype=np.float32)
    toe_xy = np.array([float(toe_pt[1]), float(toe_pt[0])], dtype=np.float32)
    axis = toe_xy - heel_xy
    axis_norm2 = float(np.dot(axis, axis))
    if axis_norm2 <= 1e-6:
        t = (ys.astype(np.float32) - ys.min()) / max(float(ys.max() - ys.min()), 1.0)
    else:
        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        t = np.dot(pts - heel_xy[None, :], axis) / axis_norm2
    t = np.clip(t, 0.0, 1.0)

    cvals = contact_rel[ys, xs].astype(np.float32)

    zone_defs = [
        ("heel", 0.00, 0.33),
        ("midfoot", 0.33, 0.67),
        ("forefoot", 0.67, 0.90),
        ("toes", 0.90, 1.01),
    ]
    zones: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in zone_defs:
        z = (t >= lo) & (t < hi)
        if not np.any(z):
            zones[name] = {"mean_contact_rel": 0.0, "high_contact_ratio": 0.0, "pixel_share": 0.0}
            continue
        zvals = cvals[z]
        zones[name] = {
            "mean_contact_rel": float(np.mean(zvals)),
            "high_contact_ratio": float(np.mean(zvals >= 0.70)),
            "pixel_share": float(np.mean(z)),
        }

    flags: List[str] = []
    score = 0
    mid = zones["midfoot"]["mean_contact_rel"]
    fore = zones["forefoot"]["mean_contact_rel"]
    heel = zones["heel"]["mean_contact_rel"]
    toes_high = zones["toes"]["high_contact_ratio"]
    toes_share = zones["toes"]["pixel_share"]

    mid_fore_ratio = (mid / fore) if fore > 1e-6 else 0.0
    heel_fore_gap = abs(heel - fore)

    if fore > 0 and mid_fore_ratio < 0.68:
        flags.append("Mediopié relativamente bajo frente a antepié (revisar arco y/o posible sobre-recorte).")
        score += 24
    if zones["toes"]["high_contact_ratio"] < 0.05:
        flags.append("Contacto alto en dedos bajo; revisar postura/captura.")
        score += 15
    if heel_fore_gap > 0.22:
        flags.append("Desbalance marcado entre talón y antepié en contacto relativo.")
        score += 20
    if toes_high > 0.995 and toes_share > 0.08 and fore < 0.80:
        flags.append("Dedos con saturación alta de contacto relativo; verificar contraste o normalización.")
        score += 12

    # Guard rail to avoid over-flagging technically clean captures with plausible arch ranges.
    if metrics.quality_status == "ok" and 12.0 <= float(metrics.arch_index_chippaux_smirak) <= 45.0:
        if mid_fore_ratio >= 0.68 and heel_fore_gap <= 0.22:
            flags = [f for f in flags if "Mediopié relativamente bajo" not in f and "Desbalance marcado" not in f]
            score = min(score, 20)

    if metrics.quality_status != "ok":
        score += 35

    score = int(max(0, min(100, score)))
    severity = "low" if score < 30 else ("medium" if score < 60 else "high")

    action = "ok"
    if flags or score >= 30:
        action = "review"
    if metrics.quality_status != "ok":
        action = "repeat_scan" if any("contaminación" in str(w).lower() for w in metrics.quality_warnings) else "review"

    return {
        "zones": zones,
        "flags": flags,
        "action": action,
        "review_score": score,
        "severity": severity,
    }


def _analyze_one(
    input_path: Path,
    output_dir: Path,
    dpi: Optional[int],
    foot: str,
    debug: bool,
    profile_path: Optional[Path] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Path]:
    ensure_dir(output_dir)

    def _progress(msg: str) -> None:
        if progress_fn is not None:
            progress_fn(msg)

    _progress("1/8 Cargando imagen")
    image_raw = load_image_any(input_path)
    image, effective_dpi, resize_meta = _maybe_resize_for_processing(image_raw, dpi)

    _progress("2/8 Preprocesado y segmentación inicial")
    prep = preprocess_image(image)
    seg = segment_footprint(prep["denoised"])

    _progress("3/8 Recorte ROI y segunda segmentación")
    image_roi, crop_meta = _crop_to_bbox(image, seg.bbox)
    prep = preprocess_image(image_roi)
    seg = segment_footprint(prep["denoised"])

    _progress("4/8 Cálculo de métricas y mapa de contacto")
    metrics, contact_rel, metrics_debug = compute_metrics(
        seg.mask,
        prep["corrected"],
        prep["gray"],
        foot_hint=foot,
        dpi=effective_dpi,
    )

    calibration_meta: Optional[Dict[str, Any]] = None
    if profile_path is not None:
        _progress("5/8 Aplicando perfil de calibración")
        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        mm_per_px = profile.get("mm_per_px")
        if mm_per_px is not None:
            mm_per_px = float(mm_per_px)
            profile_dpi = profile.get("source_dpi")
            dpi_scale_applied = 1.0
            if profile_dpi is not None and effective_dpi and float(profile_dpi) > 0:
                dpi_scale_applied = float(profile_dpi) / float(effective_dpi)
                mm_per_px *= dpi_scale_applied

            metrics.length_mm = metrics.length_px * mm_per_px
            metrics.forefoot_width_mm = metrics.forefoot_width_px * mm_per_px
            metrics.heel_width_mm = metrics.heel_width_px * mm_per_px
            metrics.midfoot_min_width_mm = metrics.midfoot_min_width_px * mm_per_px
            metrics.contact_area_mm2 = metrics.contact_area_px2 * (mm_per_px * mm_per_px)

            calibration_meta = {
                "profile_path": str(profile_path),
                "profile_name": profile.get("name", profile_path.stem),
                "mm_per_px": mm_per_px,
                "equivalent_dpi": (25.4 / mm_per_px) if mm_per_px > 0 else None,
                "profile_source_dpi": profile_dpi,
                "analysis_effective_dpi": effective_dpi,
                "dpi_scale_applied": dpi_scale_applied,
            }

    heatmap_u8 = (contact_rel * 255.0).clip(0, 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap[seg.mask == 0] = (0, 0, 0)

    overlay = _draw_overlay(image_roi, seg.contour, metrics)
    findings = _derive_findings(seg.mask, contact_rel, metrics)

    stem = input_path.stem
    overlay_path = output_dir / f"{stem}_overlay.png"
    mask_path = output_dir / f"{stem}_mask.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    json_path = output_dir / f"{stem}_results.json"
    pdf_path = output_dir / f"{stem}_report.pdf"

    save_image(overlay_path, overlay)
    save_image(mask_path, seg.mask)
    save_image(heatmap_path, heatmap)

    _progress("6/8 Exportando JSON/PDF")

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp_iso(),
            "input_file": str(input_path),
            "dpi": dpi,
            "effective_dpi": effective_dpi,
            "version": "0.1.0",
        },
        "findings": findings,
        "metrics": {
            "foot_side": metrics.foot_side,
            "length_px": metrics.length_px,
            "length_mm": metrics.length_mm,
            "forefoot_width_px": metrics.forefoot_width_px,
            "forefoot_width_mm": metrics.forefoot_width_mm,
            "heel_width_px": metrics.heel_width_px,
            "heel_width_mm": metrics.heel_width_mm,
            "midfoot_min_width_px": metrics.midfoot_min_width_px,
            "midfoot_min_width_mm": metrics.midfoot_min_width_mm,
            "arch_index_chippaux_smirak": metrics.arch_index_chippaux_smirak,
            "contact_area_px2": metrics.contact_area_px2,
            "contact_area_mm2": metrics.contact_area_mm2,
            "centroid_xy_px": metrics.centroid_xy_px,
            "principal_axis_angle_deg": metrics.principal_axis_angle_deg,
            "length_endpoints_yx": metrics.length_endpoints_yx,
            "forefoot_line_yx": metrics.forefoot_line_yx,
            "heel_line_yx": metrics.heel_line_yx,
            "midfoot_line_yx": metrics.midfoot_line_yx,
            "quality_status": metrics.quality_status,
            "quality_warnings": list(metrics.quality_warnings),
        },
    }
    if resize_meta is not None:
        results["metadata"]["resize"] = resize_meta
    results["metadata"]["roi_crop"] = crop_meta
    if calibration_meta is not None:
        results["metadata"]["calibration"] = calibration_meta
    results["metadata"]["adaptive_cleanup"] = {
        "garbage_ratio": metrics_debug.get("garbage_ratio", 0.0),
        "trim_ratio": metrics_debug.get("trim_ratio", 0.0),
        "trim_aggressiveness": metrics_debug.get("trim_aggressiveness", 0.0),
        "trim_recovery_applied": metrics_debug.get("trim_recovery_applied", False),
        "trim_recovery_level": metrics_debug.get("trim_recovery_level", 0),
        "mid_span_after_trim_px": metrics_debug.get("mid_span_after_trim_px", 0.0),
        "mid_core_span_before_trim_px": metrics_debug.get("mid_core_span_before_trim_px", 0.0),
        "mid_plausible_floor_px": metrics_debug.get("mid_plausible_floor_px", 0.0),
    }

    save_json(json_path, results)
    create_report_pdf(pdf_path, input_path, overlay_path, heatmap_path, results)

    if debug:
        _progress("7/8 Guardando debug")
        save_image(output_dir / f"{stem}_debug_full_with_roi_box.png", _draw_bbox(image, crop_meta))
        save_image(output_dir / f"{stem}_debug_roi_crop.png", image_roi)
        save_image(output_dir / f"{stem}_debug_roi_mask_overlay.png", _draw_mask_overlay(image_roi, seg.mask))
        clean_model_mask = metrics_debug.get("clean_model_mask")
        if isinstance(clean_model_mask, np.ndarray) and clean_model_mask.size > 0:
            save_image(output_dir / f"{stem}_debug_clean_model_mask.png", clean_model_mask)
            save_image(
                output_dir / f"{stem}_debug_clean_model_overlay.png",
                _draw_mask_overlay(image_roi, clean_model_mask),
            )
        for key, img in prep.items():
            save_image(output_dir / f"{stem}_debug_pre_{key}.png", img)
        for key, img in seg.debug_images.items():
            save_image(output_dir / f"{stem}_debug_seg_{key}.png", img)

    _progress("8/8 Completado")

    return {
        "overlay": overlay_path,
        "mask": mask_path,
        "heatmap": heatmap_path,
        "json": json_path,
        "pdf": pdf_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="footscan", description="MVP para análisis de huella plantar desde escáner plano.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_p = sub.add_parser("scan", help="Intenta adquirir imagen del escáner y guarda raw.")
    scan_p.add_argument("--input", type=str, default=None, help="Fallback de archivo si el scanner no está disponible.")
    scan_p.add_argument("--output_dir", type=str, default="outputs")
    scan_p.add_argument("--dpi", type=int, default=300)

    analyze_p = sub.add_parser("analyze", help="Procesa una imagen y genera PNG/JSON/PDF.")
    analyze_p.add_argument("--input", type=str, required=True)
    analyze_p.add_argument("--output_dir", type=str, default="outputs")
    analyze_p.add_argument("--dpi", type=int, default=300)
    analyze_p.add_argument("--foot", type=str, default="auto", choices=["left", "right", "auto"])
    analyze_p.add_argument("--profile", type=str, default=None, help="Perfil de calibración JSON (mm_per_px).")
    analyze_p.add_argument("--debug", action="store_true")

    batch_p = sub.add_parser("batch", help="Procesa todas las imágenes de un folder.")
    batch_p.add_argument("--input", type=str, required=True, help="Carpeta de imágenes.")
    batch_p.add_argument("--output_dir", type=str, default="outputs")
    batch_p.add_argument("--dpi", type=int, default=300)
    batch_p.add_argument("--foot", type=str, default="auto", choices=["left", "right", "auto"])
    batch_p.add_argument("--profile", type=str, default=None, help="Perfil de calibración JSON (mm_per_px).")
    batch_p.add_argument("--debug", action="store_true")

    cal_p = sub.add_parser("calibrate", help="Crea perfil de calibración mm/px usando largo real del pie.")
    cal_p.add_argument("--input", type=str, required=True)
    cal_p.add_argument("--ref_mm", type=float, required=True, help="Largo real del pie en mm (ej: 225).")
    cal_p.add_argument("--output_profile", type=str, default="outputs/scanner_profile.json")
    cal_p.add_argument("--dpi", type=int, default=300)
    cal_p.add_argument("--name", type=str, default="default_scanner")

    cal_m_p = sub.add_parser("calibrate-manual", help="Crea perfil de calibración usando y_heel/y_toe medidos manualmente.")
    cal_m_p.add_argument("--y_heel", type=float, required=True, help="Coordenada Y del talón en px (imagen completa).")
    cal_m_p.add_argument("--y_toe", type=float, required=True, help="Coordenada Y del dedo más largo en px (imagen completa).")
    cal_m_p.add_argument("--ref_mm", type=float, required=True, help="Largo real del pie en mm (ej: 225).")
    cal_m_p.add_argument("--output_profile", type=str, default="outputs/scanner_profile_manual.json")
    cal_m_p.add_argument("--name", type=str, default="manual_scanner")
    cal_m_p.add_argument("--input", type=str, default=None, help="Ruta de imagen usada como referencia manual (opcional).")
    cal_m_p.add_argument("--dpi", type=int, default=300, help="DPI nominal de la imagen usada para calibración manual.")

    return parser


def cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        input_path = Path(args.input)
        image_raw = load_image_any(input_path)
        image, _, resize_meta = _maybe_resize_for_processing(image_raw, args.dpi)

        prep = preprocess_image(image)
        seg = segment_footprint(prep["denoised"])
        image_roi, crop_meta = _crop_to_bbox(image, seg.bbox)

        prep = preprocess_image(image_roi)
        seg = segment_footprint(prep["denoised"])
        metrics, _, _ = compute_metrics(seg.mask, prep["corrected"], prep["gray"], foot_hint="auto", dpi=None)

        if args.ref_mm <= 0 or metrics.length_px <= 0:
            raise ValueError("No se pudo calcular calibración: largo de referencia o largo en px inválido.")

        mm_per_px = float(args.ref_mm) / float(metrics.length_px)
        profile = {
            "name": args.name,
            "timestamp": timestamp_iso(),
            "input_file": str(input_path),
            "source_dpi": float(args.dpi),
            "reference_length_mm": float(args.ref_mm),
            "measured_length_px": float(metrics.length_px),
            "mm_per_px": mm_per_px,
            "equivalent_dpi": (25.4 / mm_per_px) if mm_per_px > 0 else None,
            "resize": resize_meta,
            "roi_crop": crop_meta,
        }

        output_profile = Path(args.output_profile)
        ensure_dir(output_profile.parent)
        save_json(output_profile, profile)
        print(f"Perfil guardado: {output_profile}")
        print(f"  mm_per_px: {mm_per_px:.6f}")
        print(f"  equivalent_dpi: {profile['equivalent_dpi']:.2f}")
        return 0
    except Exception as e:
        print(f"Error calibrate: {e}")
        return 1


def cmd_calibrate_manual(args: argparse.Namespace) -> int:
    """Build calibration profile from manually observed heel/toe pixel coordinates."""
    try:
        if args.ref_mm <= 0:
            raise ValueError("--ref_mm debe ser > 0.")

        length_px = abs(float(args.y_toe) - float(args.y_heel))
        if length_px <= 1:
            raise ValueError("La distancia en píxeles entre talón y dedo es inválida.")

        mm_per_px = float(args.ref_mm) / length_px
        profile = {
            "name": args.name,
            "timestamp": timestamp_iso(),
            "mode": "manual_y_points",
            "input_file": str(args.input) if args.input else None,
            "source_dpi": float(args.dpi) if args.dpi else None,
            "reference_length_mm": float(args.ref_mm),
            "manual_y_heel_px": float(args.y_heel),
            "manual_y_toe_px": float(args.y_toe),
            "measured_length_px": float(length_px),
            "mm_per_px": mm_per_px,
            "equivalent_dpi": (25.4 / mm_per_px) if mm_per_px > 0 else None,
        }

        output_profile = Path(args.output_profile)
        ensure_dir(output_profile.parent)
        save_json(output_profile, profile)
        print(f"Perfil manual guardado: {output_profile}")
        print(f"  length_px: {length_px:.2f}")
        print(f"  mm_per_px: {mm_per_px:.6f}")
        print(f"  equivalent_dpi: {profile['equivalent_dpi']:.2f}")
        return 0
    except Exception as e:
        print(f"Error calibrate-manual: {e}")
        return 1


def cmd_scan(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(Path(args.output_dir))
    try:
        raw_path = acquire_from_scanner(out_dir, dpi=args.dpi)
        print(f"OK scanner -> {raw_path}")
        return 0
    except Exception as e:
        if args.input:
            raw_path = acquire_from_file(Path(args.input), out_dir)
            print(f"Scanner no disponible ({e}). Fallback archivo -> {raw_path}")
            return 0
        print(f"Error scan: {e}")
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        def _progress(msg: str) -> None:
            print(f"[analyze] {msg}")

        outputs = _analyze_one(
            Path(args.input),
            Path(args.output_dir),
            args.dpi,
            args.foot,
            args.debug,
            Path(args.profile) if args.profile else None,
            _progress,
        )
        print("Análisis completado:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        return 0
    except Exception as e:
        print(f"Error analyze: {e}")
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.input)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"La ruta --input debe ser una carpeta válida: {in_dir}")
        return 1

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    files = []
    for p in patterns:
        files.extend(sorted(in_dir.glob(p)))

    if not files:
        print(f"No se encontraron imágenes en {in_dir}")
        return 1

    success = 0
    quality_ok = 0
    warn = 0
    trim_warn = 0
    trim_ratios: List[float] = []
    flagged: List[Dict[str, Any]] = []
    analyzed: List[Dict[str, Any]] = []
    out_dir = Path(args.output_dir)
    for idx, image_path in enumerate(files, start=1):
        try:
            def _progress(msg: str, image_name=image_path.name, i=idx, n=len(files)) -> None:
                print(f"[batch {i}/{n} {image_name}] {msg}")

            outputs = _analyze_one(
                image_path,
                out_dir,
                args.dpi,
                args.foot,
                args.debug,
                Path(args.profile) if args.profile else None,
                _progress,
            )

            result_json = outputs.get("json")
            if result_json is not None and Path(result_json).exists():
                with Path(result_json).open("r", encoding="utf-8") as f:
                    data = json.load(f)
                metrics = data.get("metrics", {})
                meta = data.get("metadata", {}).get("adaptive_cleanup", {})
                status = str(metrics.get("quality_status", "ok"))
                if status == "warn":
                    warn += 1
                else:
                    quality_ok += 1
                trim_ratio = float(meta.get("trim_ratio", 0.0))
                trim_ratios.append(trim_ratio)
                quality_warnings = metrics.get("quality_warnings", [])
                has_trim_warning = any("Recorte lateral adaptativo en mediopié" in str(w) for w in quality_warnings)
                if has_trim_warning:
                    trim_warn += 1

                row = {
                    "file": str(image_path),
                    "quality_status": status,
                    "trim_ratio": trim_ratio,
                    "midfoot_mm": metrics.get("midfoot_min_width_mm"),
                    "forefoot_mm": metrics.get("forefoot_width_mm"),
                    "quality_warnings": quality_warnings,
                }
                analyzed.append(row)

                if status == "warn":
                    flagged.append(row)

            success += 1
            print(f"OK: {image_path}")
        except Exception as e:
            print(f"Fallo: {image_path} -> {e}")

    summary = {
        "timestamp": timestamp_iso(),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "total": len(files),
        "success": success,
        "ok": quality_ok,
        "warn": warn,
        "warn_trim": trim_warn,
        "fail": len(files) - success,
        "trim_ratio_avg": (sum(trim_ratios) / len(trim_ratios)) if trim_ratios else 0.0,
        "top_trim_files": sorted(analyzed, key=lambda x: float(x.get("trim_ratio", 0.0)), reverse=True)[:5],
        "top_warn_files": sorted(flagged, key=lambda x: float(x.get("trim_ratio", 0.0)), reverse=True)[:5],
    }
    ensure_dir(out_dir)
    summary_path = out_dir / "batch_summary.json"
    save_json(summary_path, summary)

    print(f"Batch finalizado. Éxitos: {success}/{len(files)} | OK calidad: {quality_ok} | Warn: {warn} | Warn(trim): {trim_warn}")
    print(f"Resumen batch: {summary_path}")
    return 0 if success > 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "batch":
        return cmd_batch(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "calibrate-manual":
        return cmd_calibrate_manual(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
