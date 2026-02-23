"""PDF reporting for FootScan outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _metric_rows(results: Dict) -> List[Tuple[str, str]]:
    m = results["metrics"]
    rows = [
        ("Calidad automática", m.get("quality_status", "n/a")),
        ("Largo total", f"{m['length_px']:.2f} px" + (f" | {m['length_mm']:.2f} mm" if m.get("length_mm") else "")),
        (
            "Ancho antepié",
            f"{m['forefoot_width_px']:.2f} px"
            + (f" | {m['forefoot_width_mm']:.2f} mm" if m.get("forefoot_width_mm") else ""),
        ),
        ("Ancho talón", f"{m['heel_width_px']:.2f} px" + (f" | {m['heel_width_mm']:.2f} mm" if m.get("heel_width_mm") else "")),
        (
            "Ancho mínimo mediopié",
            f"{m['midfoot_min_width_px']:.2f} px"
            + (f" | {m['midfoot_min_width_mm']:.2f} mm" if m.get("midfoot_min_width_mm") else ""),
        ),
        (
            "Área de contacto",
            f"{m['contact_area_px2']:.2f} px²"
            + (f" | {m['contact_area_mm2']:.2f} mm²" if m.get("contact_area_mm2") else ""),
        ),
        ("Índice de arco (Chippaux–Smirak)", f"{m['arch_index_chippaux_smirak']:.2f} %"),
        (
            "Centroide",
            f"({m['centroid_xy_px'][0]:.1f}, {m['centroid_xy_px'][1]:.1f}) px",
        ),
        ("Ángulo eje principal", f"{m['principal_axis_angle_deg']:.2f}°"),
    ]
    warns = m.get("quality_warnings") or []
    if warns:
        rows.append(("Avisos calidad", " | ".join(warns)))
    return rows


def create_report_pdf(
    output_pdf_path: Path,
    original_path: Path,
    overlay_path: Path,
    heatmap_path: Path,
    results: Dict,
) -> None:
    """Generate one-page PDF report with images and metrics table."""
    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4, leftMargin=1.2 * cm, rightMargin=1.2 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>FootScan - Reporte de Huella Plantar (MVP)</b>", styles["Title"]))
    story.append(Spacer(1, 0.25 * cm))

    meta = results["metadata"]
    story.append(
        Paragraph(
            f"Fecha: {meta['timestamp']} | Archivo: {meta['input_file']} | DPI: {meta.get('dpi', 'N/A')} | Pie: {results['metrics']['foot_side']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    img_w = 5.9 * cm
    img_h = 5.9 * cm
    image_table = Table(
        [
            [Image(str(original_path), width=img_w, height=img_h), Image(str(overlay_path), width=img_w, height=img_h), Image(str(heatmap_path), width=img_w, height=img_h)],
            [Paragraph("Original", styles["Normal"]), Paragraph("Overlay", styles["Normal"]), Paragraph("Heatmap contacto relativo", styles["Normal"])],
        ],
        colWidths=[6.1 * cm, 6.1 * cm, 6.1 * cm],
    )
    image_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(image_table)
    story.append(Spacer(1, 0.4 * cm))

    metric_rows = [("Métrica", "Valor")] + _metric_rows(results)
    metrics_table = Table(metric_rows, colWidths=[7.4 * cm, 11.1 * cm])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9edf7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    story.append(metrics_table)

    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Nota: este reporte describe contacto/huella en imagen escaneada. No reemplaza medición de presión plantar clínica certificada.",
            styles["Italic"],
        )
    )

    doc.build(story)
