# MCC225 — Examen Parcial · Winoground / Evaluating CLIP

**Niels Victor Pacheco Barrios** · Maestría en Ciencias de la Computación · 2026-1
**Tema:** Winoground / Evaluating CLIP · **Cuadernos del curso:** C5, C8, C10.

> **Tesis:** un dual-encoder tipo CLIP puede tener **retrieval (R@K) alto** y aun así
> **fallar el razonamiento composicional** de Winoground (**group score ≈ azar**).
> El retrieval no implica composición. Lo demuestro reutilizando el motor OpenCLIP del
> **Cuaderno 10**, y lo explico con la **fusión profunda (C5)** y la **atención crossmodal (C8)**.

## Resultado principal (verificable en vivo)

Winoground oficial (400 ejemplos), OpenCLIP **ViT-B-32/laion2b**:
**text = 0.347, image = 0.110, group = 0.075** (azar group = 1/6 ≈ 0.167; humano ≈ 0.855),
pero **Recall@5 = 0.67 / R@10 = 0.77**. → *Retrieval alto, composición ≈ azar.*
El scorer está **validado** contra los scores oficiales de CLIP del dataset
(`clip.jsonl`): reproduce exactamente text=0.3075 / image=0.105 / group=0.08
(`python scripts/validate_against_official.py`).

Análisis completo: 3 scores con IC bootstrap, error por tag (Object/Relation/Both),
prueba de "ceguera" a la imagen, contraste R@K vs group y comparación de 3 checkpoints.

| Evidencia | Archivo |
|---|---|
| Scores + entorno | `outputs/metrics/scores.json` |
| IC bootstrap 95% | `outputs/metrics/bootstrap_ci.csv` |
| Comparación de checkpoints | `outputs/metrics/checkpoint_comparison.csv` |
| Retrieval alto vs group bajo | `outputs/metrics/recall_vs_group.json` |
| Figuras publicables | `outputs/figures/*.png` |
| Cuaderno ejecutado | `notebooks/Winoground_Eval_MCC225.ipynb` |
| Respuestas de defensa | `docs/RESPUESTAS_PREGUNTAS.md` · `docs/HOJA_TRAZABILIDAD.md` |

## Reproducción mínima

```bash
# 1) Entorno reproducible (uv + Python 3.12)
make setup            # o: uv venv --python 3.12 .venv && uv pip install -e ".[dev]"

# 2) Datos: set curado offline (siempre) + intento de Winoground real (gated)
make data

# 3) Evaluación -> outputs/metrics/   y   figuras -> outputs/figures/
make run
make figures

# 4) Tests (scorer + métricas)
make test
```

### Winoground oficial (opcional, recomendado)

El benchmark `facebook/winoground` es **gated**. Para usar el dataset real:

1. Acepta la licencia (instantáneo) en https://huggingface.co/datasets/facebook/winoground
2. Autentícate: `huggingface-cli login` (o exporta `HF_TOKEN`).
3. `make data && make run` — el pipeline detecta el acceso y usa el real;
   si no, cae automáticamente al **set curado** (etiquetado en `scores.json: source`).

## Pipeline

```mermaid
flowchart LR
    A[winoground_data.py<br/>real HF · fallback curado] --> B[openclip_utils.py<br/>encode imagen/texto · C10]
    B --> C[winoground_eval.py<br/>text/image/group score]
    C --> D[metrics.py<br/>Recall@K · bootstrap CI]
    C --> E[error_analysis.py<br/>por tag · casos]
    B --> F[blindness_probe.py<br/>¿usa la imagen?]
    D & E & F --> G[outputs/metrics + figures]
    G --> H[notebook · slides · docs]
```

## Mapa a los cuadernos del curso

| Cuaderno | Qué reutilicé / adapté |
|---|---|
| **C10** (OpenCLIP) | Motor de embeddings, similitud coseno, comparación de checkpoints, FAISS → `src/openclip_utils.py`. |
| **C5** (deep fusion) | Marco conceptual: dual-encoder vs fusión profunda; por qué el cross-encoder modela composición → `docs/adr/0001`. |
| **C8** (atención crossmodal) | Interpretabilidad / "¿usa la imagen?" → `src/blindness_probe.py`. |

## Estructura

```
src/         scorer, métricas, motor OpenCLIP, datos, blindness, env logging
scripts/     00_verify_env · 01_prepare_data · 02_run_winoground · 03_make_figures
configs/     checkpoints.yaml · experiment.yaml
notebooks/   Winoground_Eval_MCC225.ipynb (ejecutado)
outputs/     metrics/ · figures/   (evidencia commiteada)
docs/        RESPUESTAS_PREGUNTAS · HOJA_TRAZABILIDAD · INFORME · adr/
slides/      latex/ (Beamer→PDF) · pptx/
tests/       pytest del scorer y métricas
```

## Qué puede romper la ejecución (trazabilidad §9.11)

- **Acceso gated** a Winoground sin licencia aceptada/token → usa set curado (no es error).
- Versiones de `open_clip_torch` / `torch` distintas a las pinneadas en `pyproject.toml`.
- Primera ejecución descarga checkpoints (ViT-L-14 ≈ varios cientos de MB).

## Entorno

Pinneado en `pyproject.toml` (Python 3.10–3.12). Imagen Docker CPU en `Dockerfile`.
La celda inicial del notebook y `scripts/00_verify_env.py` imprimen la **hoja de
trazabilidad** (versiones, git rev, dispositivo).

## Reproducibilidad / CI

GitHub Actions (`.github/workflows/ci.yml`) corre los tests + un *smoke run* del pipeline
sobre el set curado en cada push y **cada noche** (cron).

## Licencia / uso de herramientas generativas

Trabajo individual para el examen MCC225. El código fue desarrollado con asistencia de
herramientas de IA generativa (declarado conforme a la consigna §12); las definiciones de
métrica se verificaron contra el paper de Winoground.
