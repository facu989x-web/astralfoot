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
        self.root.geometry("1024x768")
        self.root.minsize(980, 700)

        self.current_path: Optional[Path] = None
        self.image_bgr: Optional[np.ndarray] = None
        self.display_bgr: Optional[np.ndarray] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        self.points_image: List[Tuple[float, float]] = []
        self.measure_segments: List[Dict[str, Any]] = []
        self._pending_measure_start: Optional[Tuple[float, float]] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_contact_rel: Optional[np.ndarray] = None


        self.mode_var = tk.StringVar(value="mark")
        self.foot_var = tk.StringVar(value="right")
        self.dpi_var = tk.StringVar(value="300")
        self.show_grid_var = tk.BooleanVar(value=True)
        self.grid_mm_var = tk.StringVar(value="10")
        self.measure_name_var = tk.StringVar(value="largo_pie")
        self.relief_target_var = tk.StringVar(value="0.82")
        self.relief_max_mm_var = tk.StringVar(value="6.0")
        self.relief_gamma_var = tk.StringVar(value="1.35")

        self.fit_scale = 1.0
        self.zoom_factor = 1.0
        self.scale = 1.0
        self.off_x = 0.0
        self.off_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start: Optional[Tuple[float, float]] = None

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        controls_top = ttk.Frame(container)
        controls_top.pack(fill=tk.X, pady=(0, 4))

        ttk.Button(controls_top, text="Importar", command=self.on_import).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Escanear", command=self.on_scan).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Heatmap", command=self.on_analyze).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Realces", command=self.on_relief_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Auto-recorte", command=self.on_auto_crop_noise).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Importar medidas", command=self.on_import_measures).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Limpiar", command=self.on_clear_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_top, text="Exportar", command=self.on_export).pack(side=tk.LEFT, padx=2)

        ttk.Separator(controls_top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(controls_top, text="-", width=2, command=self.on_zoom_out).pack(side=tk.LEFT)
        ttk.Button(controls_top, text="+", width=2, command=self.on_zoom_in).pack(side=tk.LEFT)
        ttk.Button(controls_top, text="Ajustar", command=self.on_zoom_fit).pack(side=tk.LEFT, padx=(4, 0))

        controls_bottom = ttk.Frame(container)
        controls_bottom.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(controls_bottom, text="Pie:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(controls_bottom, textvariable=self.foot_var, values=["left", "right", "auto"], width=7, state="readonly").pack(side=tk.LEFT)

        ttk.Label(controls_bottom, text="DPI:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(controls_bottom, textvariable=self.dpi_var, width=6).pack(side=tk.LEFT)

        ttk.Label(controls_bottom, text="Obj:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(controls_bottom, textvariable=self.relief_target_var, width=5).pack(side=tk.LEFT)
        ttk.Label(controls_bottom, text="Max mm:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(controls_bottom, textvariable=self.relief_max_mm_var, width=5).pack(side=tk.LEFT)
        ttk.Label(controls_bottom, text="γ:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(controls_bottom, textvariable=self.relief_gamma_var, width=4).pack(side=tk.LEFT)

        ttk.Label(controls_bottom, text="Modo:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Radiobutton(controls_bottom, text="Marcar", value="mark", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(controls_bottom, text="Medir", value="measure", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(controls_bottom, text="Manito", value="hand", variable=self.mode_var).pack(side=tk.LEFT)

        ttk.Label(controls_bottom, text="Etiqueta:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Combobox(
            controls_bottom,
            textvariable=self.measure_name_var,
            values=["largo_pie", "ancho_antepie", "ancho_talon", "mediopie", "realce_1", "realce_2"],
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(controls_bottom, text="Grilla", variable=self.show_grid_var, command=self._redraw).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls_bottom, textvariable=self.grid_mm_var, width=4).pack(side=tk.LEFT)
        ttk.Label(controls_bottom, text="mm").pack(side=tk.LEFT)

        main = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=5)
        main.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click_add)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_click_remove)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _e: self._zoom_at(1.12))
        self.canvas.bind("<Button-5>", lambda _e: self._zoom_at(1 / 1.12))
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)

        ttk.Label(right, text="Puntos / Medidas:").pack(anchor="w")
        self.points_text = tk.Text(right, height=11, width=28)
        self.points_text.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(right, text="Comentarios / realces:").pack(anchor="w")
        self.comments_text = tk.Text(right, height=10, width=28)
        self.comments_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(right, textvariable=self.status_var, foreground="#444", wraplength=250, justify=tk.LEFT).pack(anchor="w")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

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
            self.last_mask = None
            self.last_contact_rel = None
            self._reset_view()
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
            self.last_mask = None
            self.last_contact_rel = None
            self._reset_view()
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
            self._set_status("Generando heatmap: 1/4 preprocesando imagen...")
            prep = preprocess_image(self.image_bgr)

            self._set_status("Generando heatmap: 2/4 segmentando huella...")
            seg = segment_footprint(prep["denoised"])

            self._set_status("Generando heatmap: 3/4 calculando métricas y contacto relativo...")
            dpi = float(self.dpi_var.get()) if self.dpi_var.get().strip() else None
            metrics, contact_rel, _ = compute_metrics(seg.mask, prep["corrected"], prep["gray"], foot_hint=self.foot_var.get(), dpi=dpi)

            self._set_status("Generando heatmap: 4/4 renderizando vista...")
            heat_u8 = (contact_rel * 255.0).clip(0, 255).astype("uint8")
            heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
            heat_bgr[seg.mask == 0] = (255, 255, 255)
            self.display_bgr = heat_bgr
            self.last_mask = seg.mask.copy()
            self.last_contact_rel = contact_rel.copy()
            self.metrics = {
                "length_mm": metrics.length_mm,
                "forefoot_mm": metrics.forefoot_width_mm,
                "heel_mm": metrics.heel_width_mm,
                "midfoot_mm": metrics.midfoot_min_width_mm,
                "arch_index": metrics.arch_index_chippaux_smirak,
                "length_px": metrics.length_px,
            }
            length_txt = f"{metrics.length_mm:.1f} mm" if metrics.length_mm is not None else f"{metrics.length_px:.0f} px"
            self._set_status(f"Heatmap listo. Largo aprox: {length_txt}. Podés marcar, medir o abrir 'Realces'.")
            self._redraw()
        except Exception as exc:
            self._set_status("Error durante cálculo de heatmap.")
            messagebox.showerror("Analyze", f"Error al generar heatmap: {exc}")

    def _compute_relief_map(self) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.last_mask is None or self.last_contact_rel is None:
            raise ValueError("Primero generá el heatmap para calcular realces.")

        target = float(self.relief_target_var.get()) if self.relief_target_var.get().strip() else 0.82
        max_mm = float(self.relief_max_mm_var.get()) if self.relief_max_mm_var.get().strip() else 6.0
        gamma = float(self.relief_gamma_var.get()) if self.relief_gamma_var.get().strip() else 1.35

        target = float(np.clip(target, 0.05, 1.0))
        max_mm = float(max(0.5, max_mm))
        gamma = float(np.clip(gamma, 0.4, 3.0))

        relief_map = np.zeros_like(self.last_contact_rel, dtype=np.float32)
        m = self.last_mask > 0
        deficit = np.clip((target - self.last_contact_rel[m]) / max(target, 1e-6), 0.0, 1.0)
        relief_map[m] = max_mm * np.power(deficit, gamma)

        values = relief_map[m]
        stats = {
            "target": target,
            "max_mm": max_mm,
            "gamma": gamma,
            "mean_mm": float(np.mean(values)) if values.size else 0.0,
            "p90_mm": float(np.percentile(values, 90)) if values.size else 0.0,
            "max_out_mm": float(np.max(values)) if values.size else 0.0,
        }
        return relief_map, stats


    def on_relief_preview(self) -> None:
        try:
            self._set_status("Calculando realces sugeridos...")
            relief_map, st = self._compute_relief_map()
            max_ref = max(1e-6, st["max_mm"])
            relief_u8 = (255.0 * np.clip(relief_map / max_ref, 0.0, 1.0)).astype("uint8")
            relief_bgr = cv2.applyColorMap(relief_u8, cv2.COLORMAP_TURBO)
            if self.last_mask is not None:
                relief_bgr[self.last_mask == 0] = (255, 255, 255)
            self.display_bgr = relief_bgr
            self._redraw()
            msg = (
                "Realces calculados con modelo heurístico\n"
                "- rojo (más contacto) => menor realce\n"
                "- celeste (menos contacto) => mayor realce\n\n"
                f"Parámetros: target={st['target']:.2f}, max={st['max_mm']:.2f} mm, gamma={st['gamma']:.2f}\n"
                f"Resumen: media={st['mean_mm']:.2f} mm | p90={st['p90_mm']:.2f} mm | máx={st['max_out_mm']:.2f} mm"
            )
            self._set_status(f"Realces listos. Media {st['mean_mm']:.2f} mm | p90 {st['p90_mm']:.2f} mm.")
            messagebox.showinfo("Realces sugeridos", msg)
        except Exception as exc:
            self._set_status("No se pudo calcular realces.")
            messagebox.showerror("Realces", f"Error al calcular realces: {exc}")

    def on_auto_crop_noise(self) -> None:
        if self.image_bgr is None:
            messagebox.showinfo("Info", "Primero cargá una imagen.")
            return
        try:
            prep = preprocess_image(self.image_bgr)
            seg = segment_footprint(prep["denoised"])
            x, y, w, h = seg.bbox
            margin = max(20, int(max(w, h) * 0.06))
            h_img, w_img = self.image_bgr.shape[:2]
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(w_img, x + w + margin), min(h_img, y + h + margin)

            roi = self.image_bgr[y0:y1, x0:x1].copy()
            roi_mask = seg.mask[y0:y1, x0:x1]
            white_bg = np.full_like(roi, 255)
            white_bg[roi_mask > 0] = roi[roi_mask > 0]

            self.image_bgr = white_bg.copy()
            self.display_bgr = white_bg
            self.points_image.clear()
            self.measure_segments.clear()
            self._pending_measure_start = None
            self._reset_view()
            self._set_status("Auto-recorte aplicado: ruido lateral reducido y fondo blanco.")
            self._refresh_points_box()
            self._redraw()
        except Exception as exc:
            messagebox.showerror("Auto-recorte", f"No se pudo recortar automáticamente: {exc}")

    def on_clear_points(self) -> None:
        self.points_image.clear()
        self.measure_segments.clear()
        self._pending_measure_start = None
        self._refresh_points_box()
        self._redraw()

    def on_click_add(self, event: tk.Event) -> None:
        if self.display_bgr is None:
            return

        if self.mode_var.get() == "hand":
            self._pan_start = (float(event.x), float(event.y))
            self._set_status("Manito activa: arrastrá para mover la imagen.")
            return

        ix, iy = self._canvas_to_image(float(event.x), float(event.y))
        if ix is None:
            return

        if self.mode_var.get() == "measure":
            if self._pending_measure_start is None:
                self._pending_measure_start = (ix, iy)
                self._set_status(f"Medición [{self.measure_name_var.get()}]: elegí punto final.")
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
                        "name": self.measure_name_var.get(),
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

    def on_left_drag(self, event: tk.Event) -> None:
        if self.mode_var.get() != "hand" or self._pan_start is None:
            return
        sx, sy = self._pan_start
        self.pan_x += float(event.x) - sx
        self.pan_y += float(event.y) - sy
        self._pan_start = (float(event.x), float(event.y))
        self._redraw()

    def on_left_release(self, _event: tk.Event) -> None:
        if self.mode_var.get() == "hand":
            self._pan_start = None

    def on_zoom_in(self) -> None:
        self._zoom_at(1.15)

    def on_zoom_out(self) -> None:
        self._zoom_at(1.0 / 1.15)

    def on_zoom_fit(self) -> None:
        self._reset_view()
        self._redraw()

    def on_mouse_wheel(self, event: tk.Event) -> None:
        factor = 1.12 if event.delta > 0 else (1.0 / 1.12)
        self._zoom_at(factor)

    def _zoom_at(self, factor: float) -> None:
        self.zoom_factor = float(np.clip(self.zoom_factor * factor, 0.2, 12.0))
        self._redraw()

    def on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (float(event.x), float(event.y))

    def on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        sx, sy = self._pan_start
        self.pan_x += float(event.x) - sx
        self.pan_y += float(event.y) - sy
        self._pan_start = (float(event.x), float(event.y))
        self._redraw()

    def on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start = None

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
            name = str(seg.get("name", f"medida_{i}"))
            self.points_text.insert(tk.END, f"M{i:02d} [{name}]: {seg.get('distance_px', 0.0):.1f} px{mm_txt}\n")

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.display_bgr is None:
            return

        h, w = self.display_bgr.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.fit_scale = min(cw / w, ch / h)
        self.scale = self.fit_scale * self.zoom_factor
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        self.off_x = (cw - nw) / 2.0 + self.pan_x
        self.off_y = (ch - nh) / 2.0 + self.pan_y

        rgb = cv2.cvtColor(self.display_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.create_image(self.off_x, self.off_y, image=self.tk_image, anchor=tk.NW)

        if self.show_grid_var.get():
            dpi = float(self.dpi_var.get()) if self.dpi_var.get().strip() else 0.0
            grid_mm = float(self.grid_mm_var.get()) if self.grid_mm_var.get().strip() else 10.0
            if dpi > 0 and grid_mm > 0:
                step_px = max(1.0, grid_mm * dpi / 25.4)
                step = max(8, int(step_px * self.scale))
            else:
                step = max(20, int(40 * self.scale))
            for x in range(int(self.off_x), int(self.off_x + nw), step):
                self.canvas.create_line(x, self.off_y, x, self.off_y + nh, fill="#cccccc")
            for y in range(int(self.off_y), int(self.off_y + nh), step):
                self.canvas.create_line(self.off_x, y, self.off_x + nw, y, fill="#cccccc")

        for seg in self.measure_segments:
            x1, y1 = seg["p1"]["x"], seg["p1"]["y"]
            x2, y2 = seg["p2"]["x"], seg["p2"]["y"]
            c1x, c1y = self.off_x + x1 * self.scale, self.off_y + y1 * self.scale
            c2x, c2y = self.off_x + x2 * self.scale, self.off_y + y2 * self.scale
            self.canvas.create_line(c1x, c1y, c2x, c2y, fill="#00ff00", width=3)
            name = str(seg.get("name", ""))
            label = (name + ": ") if name else ""
            label += f"{seg['distance_px']:.1f}px"
            if seg.get("distance_mm") is not None:
                label += f" | {seg['distance_mm']:.2f}mm"
            mx, my = (c1x + c2x) / 2.0, (c1y + c2y) / 2.0
            self.canvas.create_rectangle(mx - 2, my - 16, mx + 180, my + 2, fill="#000000", outline="")
            self.canvas.create_text(mx + 4, my - 4, text=label, fill="#00ff00", anchor=tk.SW)

        for i, (px, py) in enumerate(self.points_image, start=1):
            cx = self.off_x + px * self.scale
            cy = self.off_y + py * self.scale
            r = 4
            self.canvas.create_rectangle(cx - r, cy - r, cx + r, cy + r, fill="#ff3030", outline="#ffffff")
            self.canvas.create_text(cx + 9, cy - 9, text=str(i), fill="#ff3030", anchor=tk.NW)

        if len(self.points_image) >= 2:
            prev = None
            for px, py in self.points_image:
                cx = self.off_x + px * self.scale
                cy = self.off_y + py * self.scale
                if prev is not None:
                    self.canvas.create_line(prev[0], prev[1], cx, cy, fill="#FFD400", width=3)
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
            for i, seg in enumerate(self.measure_segments, start=1):
                if "name" not in seg:
                    seg["name"] = f"medida_{i}"
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
