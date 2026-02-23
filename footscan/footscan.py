"""FootScan CLI - MVP for plantar footprint analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

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


def _analyze_one(
    input_path: Path,
    output_dir: Path,
    dpi: Optional[int],
    foot: str,
    debug: bool,
) -> Dict[str, Path]:
    ensure_dir(output_dir)

    image_raw = load_image_any(input_path)
    image, effective_dpi, resize_meta = _maybe_resize_for_processing(image_raw, dpi)

    prep = preprocess_image(image)
    seg = segment_footprint(prep["denoised"])
    metrics, contact_rel = compute_metrics(
        seg.mask,
        prep["corrected"],
        prep["gray"],
        foot_hint=foot,
        dpi=effective_dpi,
    )

    heatmap_u8 = (contact_rel * 255.0).clip(0, 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap[seg.mask == 0] = (0, 0, 0)

    overlay = _draw_overlay(image, seg.contour, metrics)

    stem = input_path.stem
    overlay_path = output_dir / f"{stem}_overlay.png"
    mask_path = output_dir / f"{stem}_mask.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    json_path = output_dir / f"{stem}_results.json"
    pdf_path = output_dir / f"{stem}_report.pdf"

    save_image(overlay_path, overlay)
    save_image(mask_path, seg.mask)
    save_image(heatmap_path, heatmap)

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp_iso(),
            "input_file": str(input_path),
            "dpi": dpi,
            "effective_dpi": effective_dpi,
            "version": "0.1.0",
        },
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
        },
    }
    if resize_meta is not None:
        results["metadata"]["resize"] = resize_meta

    save_json(json_path, results)
    create_report_pdf(pdf_path, input_path, overlay_path, heatmap_path, results)

    if debug:
        for key, img in prep.items():
            save_image(output_dir / f"{stem}_debug_pre_{key}.png", img)
        for key, img in seg.debug_images.items():
            save_image(output_dir / f"{stem}_debug_seg_{key}.png", img)

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
    analyze_p.add_argument("--debug", action="store_true")

    batch_p = sub.add_parser("batch", help="Procesa todas las imágenes de un folder.")
    batch_p.add_argument("--input", type=str, required=True, help="Carpeta de imágenes.")
    batch_p.add_argument("--output_dir", type=str, default="outputs")
    batch_p.add_argument("--dpi", type=int, default=300)
    batch_p.add_argument("--foot", type=str, default="auto", choices=["left", "right", "auto"])
    batch_p.add_argument("--debug", action="store_true")

    return parser


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
        outputs = _analyze_one(Path(args.input), Path(args.output_dir), args.dpi, args.foot, args.debug)
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

    ok = 0
    for image_path in files:
        try:
            _analyze_one(image_path, Path(args.output_dir), args.dpi, args.foot, args.debug)
            ok += 1
            print(f"OK: {image_path}")
        except Exception as e:
            print(f"Fallo: {image_path} -> {e}")

    print(f"Batch finalizado. Éxitos: {ok}/{len(files)}")
    return 0 if ok > 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "batch":
        return cmd_batch(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
