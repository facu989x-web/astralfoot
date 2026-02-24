"""Simple FootScan GUI for image import/scan, heatmap view and manual markings."""

from __future__ import annotations

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
from footscan.utils import ensure_dir, load_image_any, save_json, timestamp_iso


class FootScanGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FootScan GUI (MVP)")
        self.root.geometry("1200x780")

        self.current_path: Optional[Path] = None
        self.image_bgr: Optional[np.ndarray] = None
        self.display_bgr: Optional[np.ndarray] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.points_canvas: List[Tuple[float, float]] = []
        self.points_image: List[Tuple[float, float]] = []
        self.metrics: Optional[Dict[str, Any]] = None

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
        ttk.Button(controls, text="Limpiar puntos", command=self.on_clear_points).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Exportar marcas", command=self.on_export).pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Pie:").pack(side=tk.LEFT, padx=(18, 4))
        self.foot_var = tk.StringVar(value="right")
        ttk.Combobox(controls, textvariable=self.foot_var, values=["left", "right", "auto"], width=8, state="readonly").pack(side=tk.LEFT)

        ttk.Label(controls, text="DPI:").pack(side=tk.LEFT, padx=(12, 4))
        self.dpi_var = tk.StringVar(value="300")
        ttk.Entry(controls, textvariable=self.dpi_var, width=8).pack(side=tk.LEFT)

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

        ttk.Label(right, text="Puntos marcados (x,y):").pack(anchor="w")
        self.points_text = tk.Text(right, height=12, width=36)
        self.points_text.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(right, text="Comentarios / realces:").pack(anchor="w")
        self.comments_text = tk.Text(right, height=16, width=36)
        self.comments_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(right, textvariable=self.status_var, foreground="#444").pack(anchor="w")

        self.canvas.bind("<Configure>", lambda _e: self._redraw())

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
            self.points_canvas.clear()
            self.points_image.clear()
            self.metrics = None
            self._set_status(f"Imagen cargada: {self.current_path.name}")
            self._redraw()
            self._refresh_points_box()
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
            self.points_canvas.clear()
            self.points_image.clear()
            self.metrics = None
            self._set_status(f"Escaneo cargado: {raw_path.name}")
            self._redraw()
            self._refresh_points_box()
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
            }
            self._set_status("Heatmap generado. Marcá puntos sobre la grilla.")
            self._redraw()
        except Exception as exc:
            messagebox.showerror("Analyze", f"Error al generar heatmap: {exc}")

    def on_clear_points(self) -> None:
        self.points_canvas.clear()
        self.points_image.clear()
        self._refresh_points_box()
        self._redraw()

    def on_click_add(self, event: tk.Event) -> None:
        if self.display_bgr is None:
            return
        x, y = float(event.x), float(event.y)
        ix, iy = self._canvas_to_image(x, y)
        if ix is None:
            return
        self.points_canvas.append((x, y))
        self.points_image.append((ix, iy))
        self._refresh_points_box()
        self._redraw()

    def on_click_remove(self, _event: tk.Event) -> None:
        if self.points_canvas:
            self.points_canvas.pop()
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
            self.points_text.insert(tk.END, f"{i:02d}: ({x:.1f}, {y:.1f})\n")

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
            self.canvas.create_line(x, self.off_y, x, self.off_y + nh, fill="#ffffff22")
        for y in range(self.off_y, self.off_y + nh, grid_step):
            self.canvas.create_line(self.off_x, y, self.off_x + nw, y, fill="#ffffff22")

        for i, (px, py) in enumerate(self.points_image, start=1):
            cx = self.off_x + px * self.scale
            cy = self.off_y + py * self.scale
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#ff3030", outline="#ffffff")
            self.canvas.create_text(cx + 9, cy - 9, text=str(i), fill="#ffffff", anchor=tk.NW)

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
            "comments": self.comments_text.get("1.0", tk.END).strip(),
            "metrics_preview": self.metrics,
        }
        try:
            save_json(Path(out), payload)
            self._set_status(f"Marcas exportadas: {Path(out).name}")
            messagebox.showinfo("OK", f"Archivo guardado:\n{out}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar JSON: {exc}")


def launch_gui() -> None:
    root = tk.Tk()
    FootScanGUI(root)
    root.mainloop()
