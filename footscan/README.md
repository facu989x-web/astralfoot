# FootScan (MVP)

MVP en Python para análisis de **huella plantar por contacto** a partir de imagen de escáner plano.

> ⚠️ Importante: este proyecto mide **contacto/huella** en imágenes escaneadas. **No** es un sistema de presión plantar clínica certificada ni reemplaza evaluación médica.

## Funcionalidades

- CLI con comandos de consola y GUI:
  - `scan`: intenta adquirir imagen del escáner (WIA/TWAIN) y guarda raw.
  - `analyze`: procesa una imagen y genera `overlay.png`, `mask.png`, `heatmap.png`, `results.json`, `report.pdf`.
  - `batch`: procesa una carpeta completa.
  - `calibrate`: crea perfil de calibración `mm_per_px` usando largo real conocido.
  - `gui`: interfaz gráfica para importar/escanear, ver heatmap en grilla y marcar puntos/manual notes.
  - `analyze` y `batch` muestran progreso por etapas en consola para seguimiento.
- Pipeline de visión por computadora:
  1. Escala de grises.
  2. Flat-field correction (normalización de iluminación).
  3. Threshold robusto (Otsu + fallback adaptativo).
  4. Morfología open/close.
  5. Componente conectado más grande.
  6. Recorte automático al ROI del pie para suprimir fondo irrelevante.
  7. Contorno principal y métricas.
- Métricas (px y mm si hay DPI):
  - Largo total.
  - Ancho de antepié.
  - Ancho de talón.
  - Área de contacto.
  - Índice de arco (Chippaux–Smirak).
  - Centroide.
  - Eje principal y ángulo.
- Mapa de intensidad de contacto relativo (ponderado para evitar artefactos de borde y conservar detalle dentro de la planta).
- Control automático de calidad en métricas (banderas `quality_status` y `quality_warnings` en JSON/PDF), con aviso de recorte adaptativo priorizando casos de recorte pesado o mediopié claramente estrecho.
- Bloque `findings` en JSON/PDF con zonas anatómicas (talón/mediopié/antepié/dedos), hallazgos técnicos y acción sugerida (`ok/review/repeat_scan`), incluyendo detección de desbalance zonal/saturación en dedos, severidad (`low/medium/high`) y `review_score` (0-100) con guardas (banda estricta + banda relajada) para reducir falsos `review` en capturas técnicamente limpias y absorber deriva leve del escáner. Además incluye `observations` para notas suaves de seguimiento sin forzar `review` y `subzones` (medial/lateral por talón-mediopié-antepié-dedos) para realces más precisos.
- Regla adaptativa de recorte lateral en mediopié para reducir contaminación por sombras laterales en anchos/arco, con detección automática de basura, agresividad suavizada, salvaguarda anti sobre-recorte, tope de agresión y selector final de suavidad para priorizar mediopié anatómico.

## Estructura

```text
footscan/
  README.md
  requirements.txt
  footscan.py
  footscan/
    __init__.py
    acquire.py
    preprocess.py
    segment.py
    metrics.py
    report.py
    utils.py
  samples/
  outputs/  (se crea automáticamente)
```

## Requisitos

- Python 3.8+
- Windows 10/11 (también puede funcionar en Linux/macOS para modo archivo)

Instalación sugerida:

```bash
cd footscan
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso CLI

### 1) Escaneo (con fallback a archivo)

```bash
python footscan.py scan --dpi 300 --output_dir outputs
```

Si no hay escáner disponible, podés indicar archivo de fallback:

```bash
python footscan.py scan --input "samples/dummy_foot.png" --dpi 300 --output_dir outputs
```

### 2) Análisis de una imagen

```bash
python footscan.py analyze --input "samples/dummy_foot.png" --output_dir outputs --dpi 300 --foot auto --debug
```

Con perfil de calibración (recomendado para precisión real por escáner):

```bash
python footscan.py analyze --input "samples/dummy_foot.png" --output_dir outputs --profile "outputs/scanner_profile.json" --foot auto --debug
```

### 3) Batch por carpeta

```bash
python footscan.py batch --input "samples" --output_dir outputs --dpi 300 --foot auto
```

Al finalizar, `batch` guarda `outputs/batch_summary.json` con conteo de `success/ok/warn/fail`, `warn_trim`, `trim_watch` (casos con `trim_ratio` alto aunque no haya `warn`), promedio de `trim_ratio`, `top_trim_files`, `top_trim_watch_files` y `top_warn_files` para revisión rápida.

### 4) Calibración de escala (mm/px)

Ejemplo: si el largo real del pie medido externamente es 225 mm.

```bash
python footscan.py calibrate --input "samples/dummy_foot.png" --ref_mm 225 --output_profile "outputs/scanner_profile.json" --name "scanner_consultorio"
```

### 5) Calibración manual por puntos Y (talón/dedo)

Si mediste manualmente que el talón está en `y=800 px` y el dedo en `y=6050 px`:

```bash
python footscan.py calibrate-manual --y_heel 800 --y_toe 6050 --ref_mm 225 --dpi 600 --output_profile "outputs/scanner_profile_manual.json" --name "scanner_consultorio_manual" --input "samples/002.png"
```

Los perfiles guardan `source_dpi`; al analizar, si `--profile` y `--dpi` difieren, FootScan ajusta automáticamente `mm_per_px` por el ratio `source_dpi / effective_dpi` para mantener consistencia de escala entre 300/600 DPI.


### 6) GUI para marcar puntos sobre heatmap

```bash
python footscan.py gui
# o lanzamiento directo pensado para empaquetado EXE
python footscan_gui.py
```

La GUI permite:
- importar imagen o intentar escaneo,
- generar heatmap de fondo,
- marcar puntos manuales sobre una cuadrícula,
- agregar comentarios/realces,
- exportar un JSON de anotaciones para fabricación/seguimiento.
- si hay 3+ puntos, exportar también recorte poligonal del heatmap (`*_heatmap_crop.png` + máscara).


### 7) Preparación para EXE (PyInstaller)

Ejemplo rápido (Windows):

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed footscan_gui.py --name FootScanGUI
```

Esto genera un ejecutable de GUI (sin consola) en `dist/FootScanGUI.exe`.

## Salidas

Para una entrada `dummy_foot.png` se generan:

- `outputs/dummy_foot_overlay.png`
- `outputs/dummy_foot_mask.png`
- `outputs/dummy_foot_heatmap.png`
- `outputs/dummy_foot_results.json` (incluye `findings` con contactos zonales, acción sugerida, severidad, `review_score`, `observations` y `subzones`)
- `outputs/dummy_foot_report.pdf`
- además, con `--debug`, imágenes intermedias del pipeline, incluyendo `debug_full_with_roi_box`, `debug_roi_mask_overlay`, `debug_clean_model_mask`, `debug_clean_model_overlay` y `debug_enhanced_map` para comparar máscara base vs modelo limpio y realces por subzonas.
- en GUI podés exportar `annotations.json` con puntos manuales + comentarios sobre el heatmap, y recorte poligonal para usar directo en plantilla.

## Notas de adquisición (WIA/TWAIN)

- `acquire.py` implementa primero modo archivo y deja stubs para WIA/TWAIN.
- Dependencias opcionales:
  - `pywin32` para WIA (vía COM).
  - `twain` para TWAIN.
- Si no están instaladas o no hay hardware compatible, el sistema no se rompe: informa error humano y permite fallback por archivo.

## Limitaciones del MVP

- Segmentación basada en imagen 2D de contacto, sensible a calidad de escaneo.
- Sin calibración clínica ni estimación de presión real.
- `foot auto` usa heurística simple y puede requerir ajuste manual en algunos casos.
- Imágenes muy grandes (ej. 1200 DPI) se redimensionan automáticamente para evitar problemas de memoria, conservando trazabilidad de escala en el JSON.
- El heatmap es relativo a la intensidad de la huella escaneada (más claro en el escaneo tiende a mayor contacto relativo), no una medición de presión certificada.
- La máscara de huella se rellena sobre el contorno principal para evitar "huellas huecas" (contorno con interior negro) que distorsionan el heatmap y el cálculo de área.
- Se aplica recorte automático al bounding box del pie (con margen) para reducir ruido por fondo, ropa o zonas escaneadas sin contacto.
- `results.json` incluye `metadata.adaptive_cleanup` con indicadores de detección de basura (`garbage_ratio`), agresividad de recorte (`trim_aggressiveness`), recuperación anti sobre-recorte (`trim_recovery_applied`/`trim_recovery_level`) y pisos de plausibilidad usados en mediopié (`mid_plausible_floor_px`).
