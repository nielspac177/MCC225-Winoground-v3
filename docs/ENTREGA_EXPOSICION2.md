# Entrega y trazabilidad — Segunda Exposición Académica (Exposición 2)

**Curso:** MCC225 · IA generativa y multimodal · **Autor:** Niels Victor Pacheco Barrios · **Fecha:** 2026-07

Este documento mapea **cada insumo obligatorio** y **cada criterio de la rúbrica**
de la Segunda Exposición Académica a los archivos concretos del repositorio, y da
los comandos exactos para reproducir toda la evidencia.

## 1. Insumos obligatorios → archivo(s)

| # | Insumo obligatorio | Archivo(s) en el repo | Estado |
|---|---|---|---|
| 1 | Presentación PDF (7–8 diapositivas) | `slides/latex/exposicion2_winoground.pdf` (fuente `.tex` junto) | Compilable con `make slides` |
| 2 | Avance técnico (2–3 pp) | `docs/avance_tecnico_MCC225.pdf` (fuente `.tex` junto) | Presente |
| 3 | Repositorio con commits | Este repo (historial git); `README.md` §Exposición 2 | Presente |
| 4 | Cuaderno 14 resuelto + outputs | `notebooks/Cuaderno14_MCC225_resuelto.ipynb` · `outputs/metrics/` · `outputs/tables/` · `outputs/figures/` | Ejecutable con `make avance` |
| 5 | Actividad 5 aplicada | `Actividad5-MCC225.md` · `evaluacion_responsable_mcc225/` | Presente |
| 6 | Evidencia reproducible | `outputs/metrics/` · `outputs/tables/` · `outputs/figures/` | Generada por el pipeline |

## 2. Rúbrica (20 pts) → dónde se evidencia

| Criterio | Pts | Dónde se evidencia |
|---|---:|---|
| Formulación del problema | 3 | `docs/avance_tecnico_MCC225.pdf`; `README.md` (tesis); `slides/latex/exposicion2_winoground.pdf` (slides 1–2) |
| Transformer + atención | 4 | `slides/latex/exposicion2_winoground.pdf` (slide 4, ecuación de atención); `docs/avance_tecnico_MCC225.pdf` (§3); `docs/RESPUESTAS_EXPOSICION2.md` (self vs cross-attention, O(n²)) |
| Arquitecturas multimodales | 4 | `notebooks/Cuaderno14_MCC225_resuelto.ipynb` (BLIP); `src/openclip_utils.py` (dual-encoder CLIP, C10); dual-encoder vs deep fusion (C5) |
| Evidencia del cuaderno | 3 | `notebooks/Cuaderno14_MCC225_resuelto.ipynb` ejecutado + `outputs/tables/` · `outputs/figures/` |
| Actividad 5 | 3 | `Actividad5-MCC225.md` · `evaluacion_responsable_mcc225/` |
| Repositorio / reproducibilidad | 2 | `Makefile` · `Dockerfile` · `docker-compose.yml` · `requirements.txt` · `pyproject.toml` · `.github/workflows/ci.yml` |
| Comunicación | 1 | `slides/latex/exposicion2_winoground.pdf` · `docs/PLAN_EXPOSICION2_MCC225.md` |
| **Total** | **20** | |

## 3. Cómo reproducir

### Opción A — entorno local (uv + Python 3.12)

```bash
make setup      # crea .venv e instala .[dev,notebook]
make avance     # manifest local (120 pares reales) + ejecuta Cuaderno14 headless
make run        # pipeline Winoground -> outputs/metrics
make figures    # figuras -> outputs/figures
make test       # pytest del scorer y métricas
```

Alternativa sin uv/pyproject: `pip install -r requirements.txt`.

### Opción B — Docker (CPU reproducible)

```bash
docker compose run --rm avance      # o: make docker-avance
```

Reproduce todo el flujo de Exposición 2: `build_local_manifest.py 120` →
`nbconvert --execute` del Cuaderno14 → `02_run_winoground.py` → `03_make_figures.py`.

## 4. Documentos de apoyo

- Plan de exposición: `docs/PLAN_EXPOSICION2_MCC225.md`
- Flujograma del pipeline: `docs/FLUJOGRAMA_MCC225.md`
- Actividad 5 (evaluación responsable): `Actividad5-MCC225.md` · `evaluacion_responsable_mcc225/`
- Avance técnico: `docs/avance_tecnico_MCC225.pdf`

## 5. Declaración de uso de IA generativa

Trabajo individual para MCC225. El código y los materiales se desarrollaron con
**asistencia de herramientas de IA generativa**, declarado conforme a la consigna del
curso. Las definiciones de métrica de Winoground se verificaron contra el paper
original y contra los scores oficiales de CLIP (`clip.jsonl`,
`scripts/validate_against_official.py`).
