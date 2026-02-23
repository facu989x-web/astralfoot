# FootScan (MVP)

MVP en Python para análisis de **huella plantar por contacto** a partir de imagen de escáner plano.

> ⚠️ Importante: este proyecto mide **contacto/huella** en imágenes escaneadas. **No** es un sistema de presión plantar clínica certificada ni reemplaza evaluación médica.

## Funcionalidades

- CLI con 3 comandos:
  - `scan`: intenta adquirir imagen del escáner (WIA/TWAIN) y guarda raw.
  - `analyze`: procesa una imagen y genera `overlay.png`, `mask.png`, `heatmap.png`, `results.json`, `report.pdf`.
  - `batch`: procesa una carpeta completa.
- Pipeline de visión por computadora:
  1. Escala de grises.
  2. Flat-field correction (normalización de iluminación).
  3. Threshold robusto (Otsu + fallback adaptativo).
  4. Morfología open/close.
  5. Componente conectado más grande.
  6. Contorno principal y métricas.
- Métricas (px y mm si hay DPI):
  - Largo total.
  - Ancho de antepié.
  - Ancho de talón.
  - Área de contacto.
  - Índice de arco (Chippaux–Smirak).
  - Centroide.
  - Eje principal y ángulo.
- Mapa de intensidad de contacto relativo (ponderado para evitar artefactos de borde y conservar detalle dentro de la planta).

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

### 3) Batch por carpeta

```bash
python footscan.py batch --input "samples" --output_dir outputs --dpi 300 --foot auto
```

## Salidas

Para una entrada `dummy_foot.png` se generan:

- `outputs/dummy_foot_overlay.png`
- `outputs/dummy_foot_mask.png`
- `outputs/dummy_foot_heatmap.png`
- `outputs/dummy_foot_results.json`
- `outputs/dummy_foot_report.pdf`
- además, con `--debug`, imágenes intermedias del pipeline.

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
