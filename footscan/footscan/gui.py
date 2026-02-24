"""Simple FootScan GUI for image import/scan, heatmap view and manual markings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from footscan.acquire import acquire_from_scanner
from footscan.metrics import compute_metrics
from footscan.preprocess import preprocess_image
from footscan.segment import segment_footprint
from footscan.utils import ensure_dir, load_image_any, save_image, save_json, timestamp_iso


class FootScanGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FootScan GUI (MVP)")
        self.root.geometry("1200x780")

        self.current_path: Optional[Path] = None
        self.image_bgr: Optional[np.ndarray] = None
        self.display_bgr: Optional[np.ndarray] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.points_image: List[Tuple[float, float]] = []
        self.measure_segments: List[Dict[str, Any]] = []
        self._pending_measure_start: Optional[Tuple[float, float]] = None
        self.metrics: Optional[Dict[str, Any]] = None

        self.mode_var = tk.StringVar(value="mark")
        self.foot_var = tk.StringVar(value="right")
        self.dpi_var = tk.StringVar(value="300")

        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(controls, text="Importar imagen", command=self.on_import).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Escanear (stub)", command=self.on_scan).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Generar heatmap", command=self.on_analyze).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Importar medidas", command=self.on_import_measures).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Limpiar", command=self.on_clear_points).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Exportar", command=self.on_export).pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Pie:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Combobox(controls, textvariable=self.foot_var, values=["left", "right", "auto"], width=8, state="readonly").pack(side=tk.LEFT)

        ttk.Label(controls, text="DPI:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(controls, textvariable=self.dpi_var, width=8).pack(side=tk.LEFT)

        ttk.Label(controls, text="Modo:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Radiobutton(controls, text="Marcar", value="mark", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(controls, text="Medir", value="measure", variable=self.mode_var).pack(side=tk.LEFT)

        main = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=4)
        main.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click_add)
        self.canvas.bind("<Button-3>", self.on_click_remove)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

        ttk.Label(right, text="Puntos / Medidas:").pack(anchor="w")
        self.points_text = tk.Text(right, height=16, width=38)
        self.points_text.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(right, text="Comentarios / realces:").pack(anchor="w")
        self.comments_text = tk.Text(right, height=14, width=38)
        self.comments_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(right, textvariable=self.status_var, foreground="#444").pack(anchor="w")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def on_import(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")])
        if not path:
            return
        try:
            image = load_image_any(Path(path))
            self.current_path = Path(path)
            self.image_bgr = image
            self.display_bgr = image.copy()
            self.points_image.clear()
            self.measure_segments.clear()
            self._pending_measure_start = None
            self.metrics = None
            self._set_status(f"Imagen cargada: {self.current_path.name}")
            self._refresh_points_box()
            self._redraw()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo abrir imagen: {exc}")

    def on_scan(self) -> None:
        try:
            out_dir = ensure_dir(Path("outputs"))
            dpi = int(self.dpi_var.get())
            raw_path = acquire_from_scanner(out_dir, dpi=dpi)
            image = load_image_any(raw_path)
            self.current_path = raw_path
            self.image_bgr = image
            self.display_bgr = image.copy()
            self.points_image.clear()
            self.measure_segments.clear()
            self._pending_measure_start = None
            self.metrics = None
            self._set_status(f"Escaneo cargado: {raw_path.name}")
            self._refresh_points_box()
            self._redraw()
        except Exception as exc:
            messagebox.showerror("Scan", f"No se pudo escanear. Usá Importar imagen.\n\nDetalle: {exc}")

    def on_analyze(self) -> None:
        if self.image_bgr is None:
            messagebox.showinfo("Info", "Primero cargá una imagen.")
            return
        try:
            prep = preprocess_image(self.image_bgr)
            seg = segment_footprint(prep["denoised"])
            dpi = float(self.dpi_var.get()) if self.dpi_var.get().strip() else None
            metrics, contact_rel, _ = compute_metrics(seg.mask, prep["corrected"], prep["gray"], foot_hint=self.foot_var.get(), dpi=dpi)
            heat_u8 = (contact_rel * 255.0).clip(0, 255).astype("uint8")
            heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
            heat_bgr[seg.mask == 0] = (0, 0, 0)
            self.display_bgr = heat_bgr
            self.metrics = {
                "length_mm": metrics.length_mm,
                "forefoot_mm": metrics.forefoot_width_mm,
                "heel_mm": metrics.heel_width_mm,
                "midfoot_mm": metrics.midfoot_min_width_mm,
                "arch_index": metrics.arch_index_chippaux_smirak,
                "length_px": metrics.length_px,
            }
            self._set_status("Heatmap generado. Modo Marcar o Medir.")
            self._redraw()
        except Exception as exc:
            messagebox.showerror("Analyze", f"Error al generar heatmap: {exc}")

    def on_clear_points(self) -> None:
        self.points_image.clear()
        self.measure_segments.clear()
        self._pending_measure_start = None
        self._refresh_points_box()
        self._redraw()

    def on_click_add(self, event: tk.Event) -> None:
        if self.display_bgr is None:
            return
        ix, iy = self._canvas_to_image(float(event.x), float(event.y))
        if ix is None:
            return

        if self.mode_var.get() == "measure":
            if self._pending_measure_start is None:
                self._pending_measure_start = (ix, iy)
                self._set_status("Medición: elegí punto final.")
            else:
                x1, y1 = self._pending_measure_start
                x2, y2 = ix, iy
                dist_px = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                dpi = float(self.dpi_var.get()) if self.dpi_var.get().strip() else 0.0
                dist_mm = (dist_px * 25.4 / dpi) if dpi > 0 else None
                self.measure_segments.append(
                    {
                        "p1": {"x": x1, "y": y1},
                        "p2": {"x": x2, "y": y2},
                        "distance_px": dist_px,
                        "distance_mm": dist_mm,
                    }
                )
                self._pending_measure_start = None
                self._set_status(f"Medida: {dist_px:.1f}px" + (f" | {dist_mm:.2f}mm" if dist_mm is not None else ""))
            self._refresh_points_box()
            self._redraw()
            return

        self.points_image.append((ix, iy))
        self._refresh_points_box()
        self._redraw()

    def on_click_remove(self, _event: tk.Event) -> None:
        if self.mode_var.get() == "measure":
            if self._pending_measure_start is not None:
                self._pending_measure_start = None
            elif self.measure_segments:
                self.measure_segments.pop()
            self._refresh_points_box()
            self._redraw()
            return

        if self.points_image:
            self.points_image.pop()
            self._refresh_points_box()
            self._redraw()

    def _canvas_to_image(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        if self.display_bgr is None:
            return None, None
        ix = (x - self.off_x) / max(self.scale, 1e-6)
        iy = (y - self.off_y) / max(self.scale, 1e-6)
        h, w = self.display_bgr.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None, None
        return float(ix), float(iy)

    def _refresh_points_box(self) -> None:
        self.points_text.delete("1.0", tk.END)
        for i, (x, y) in enumerate(self.points_image, start=1):
            self.points_text.insert(tk.END, f"P{i:02d}: ({x:.1f}, {y:.1f})\n")
        if self.measure_segments:
            self.points_text.insert(tk.END, "\n--- Medidas ---\n")
        for i, seg in enumerate(self.measure_segments, start=1):
            mm = seg.get("distance_mm")
            mm_txt = f" | {mm:.2f} mm" if isinstance(mm, (int, float)) and mm is not None else ""
            self.points_text.insert(tk.END, f"M{i:02d}: {seg.get('distance_px', 0.0):.1f} px{mm_txt}\n")

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.display_bgr is None:
            return

        h, w = self.display_bgr.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        self.off_x = (cw - nw) // 2
        self.off_y = (ch - nh) // 2

        rgb = cv2.cvtColor(self.display_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.create_image(self.off_x, self.off_y, image=self.tk_image, anchor=tk.NW)

        grid_step = max(20, int(40 * self.scale))
        for x in range(self.off_x, self.off_x + nw, grid_step):
            self.canvas.create_line(x, self.off_y, x, self.off_y + nh, fill="#333333")
        for y in range(self.off_y, self.off_y + nh, grid_step):
            self.canvas.create_line(self.off_x, y, self.off_x + nw, y, fill="#333333")

        # Draw measurement lines and labels first
        for seg in self.measure_segments:
            x1, y1 = seg["p1"]["x"], seg["p1"]["y"]
            x2, y2 = seg["p2"]["x"], seg["p2"]["y"]
            c1x, c1y = self.off_x + x1 * self.scale, self.off_y + y1 * self.scale
            c2x, c2y = self.off_x + x2 * self.scale, self.off_y + y2 * self.scale
            self.canvas.create_line(c1x, c1y, c2x, c2y, fill="#00ff66", width=2)
            label = f"{seg['distance_px']:.1f}px"
            if seg.get("distance_mm") is not None:
                label += f" | {seg['distance_mm']:.2f}mm"
            mx, my = (c1x + c2x) / 2.0, (c1y + c2y) / 2.0
            self.canvas.create_text(mx + 6, my - 6, text=label, fill="#00ff66", anchor=tk.SW)

        # Draw marking polygon
        for i, (px, py) in enumerate(self.points_image, start=1):
            cx = self.off_x + px * self.scale
            cy = self.off_y + py * self.scale
            r = 4
            self.canvas.create_rectangle(cx - r, cy - r, cx + r, cy + r, fill="#ff3030", outline="#ffffff")
            self.canvas.create_text(cx + 9, cy - 9, text=str(i), fill="#ffffff", anchor=tk.NW)

        if len(self.points_image) >= 2:
            prev = None
            for px, py in self.points_image:
                cx = self.off_x + px * self.scale
                cy = self.off_y + py * self.scale
                if prev is not None:
                    self.canvas.create_line(prev[0], prev[1], cx, cy, fill="#00e5ff", width=2)
                prev = (cx, cy)

    def on_import_measures(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            pts = data.get("points_image_xy") or []
            self.points_image = [(float(p["x"]), float(p["y"])) for p in pts if "x" in p and "y" in p]
            self.measure_segments = data.get("measurements") or []
            self.comments_text.delete("1.0", tk.END)
            self.comments_text.insert(tk.END, data.get("comments", ""))
            self._refresh_points_box()
            self._redraw()
            self._set_status(f"JSON importado: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Importar", f"No se pudo importar JSON: {exc}")

    def on_export(self) -> None:
        if self.current_path is None:
            messagebox.showinfo("Info", "No hay imagen activa.")
            return

        out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="annotations.json")
        if not out:
            return

        payload = {
            "timestamp": timestamp_iso(),
            "image_file": str(self.current_path),
            "foot": self.foot_var.get(),
            "dpi": float(self.dpi_var.get()) if self.dpi_var.get().strip() else None,
            "points_image_xy": [{"x": float(x), "y": float(y)} for x, y in self.points_image],
            "measurements": self.measure_segments,
            "comments": self.comments_text.get("1.0", tk.END).strip(),
            "metrics_preview": self.metrics,
        }
        try:
            out_path = Path(out)
            save_json(out_path, payload)

            if len(self.points_image) >= 3 and self.display_bgr is not None:
                poly = np.array(self.points_image, dtype=np.int32).reshape((-1, 1, 2))
                base = self.display_bgr.copy()
                mask = np.zeros(base.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                masked = cv2.bitwise_and(base, base, mask=mask)
                x, y, w, h = cv2.boundingRect(poly)

                crop = masked[y : y + h, x : x + w]
                crop_mask = mask[y : y + h, x : x + w]
                crop_path = out_path.with_name(out_path.stem + "_heatmap_crop.png")
                mask_path = out_path.with_name(out_path.stem + "_heatmap_crop_mask.png")
                save_image(crop_path, crop)
                save_image(mask_path, crop_mask)

                payload["crop_export"] = {
                    "crop_file": str(crop_path),
                    "mask_file": str(mask_path),
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                }
                save_json(out_path, payload)

            self._set_status(f"Marcas exportadas: {out_path.name}")
            messagebox.showinfo("OK", f"Archivo guardado:\n{out}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar JSON: {exc}")


def launch_gui() -> None:
    root = tk.Tk()
    FootScanGUI(root)
    root.mainloop()
