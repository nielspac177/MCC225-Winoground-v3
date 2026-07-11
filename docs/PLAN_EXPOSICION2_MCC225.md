# Plan de trabajo — Segunda Exposición Académica MCC225

Documento vivo que registra el alcance, las decisiones y los entregables de la
**Segunda Exposición Académica** de MCC225A. El trabajo integrador es la evaluación
de un **dual-encoder CLIP** sobre **Winoground**, extendida con el `Cuaderno14`
(evaluación experimental multimodal) y la `Actividad5` (evaluación responsable).

Autor único: **Niels Victor Pacheco Barrios** · Maestría en Ciencias de la Computación · 2026-1.

## 🎯 Contexto y objetivo

La evaluación pide sustentar el avance formal del trabajo integrador demostrando
dominio de transformer, atención y arquitecturas multimodales, con evidencia
experimental verificable desde el cuaderno, la Actividad5 aplicada y un plan de
cierre. El repositorio es el entregable central: todo debe poder contrastarse con
archivos reales, no solo con capturas.

El proyecto ya existía (evaluación de Winoground con OpenCLIP). Esta exposición
añade tres piezas: el **Cuaderno14 resuelto y ejecutado**, la **Actividad5 aplicada**
y los **entregables en el formato nuevo** (8 diapositivas + avance técnico 2–3 pp).

## 📥 Insumos obligatorios y su ubicación

| Insumo | Archivo en el repo |
|---|---|
| Presentación PDF (7–8 slides) | `slides/latex/exposicion2_winoground.pdf` |
| Avance técnico (2–3 pp) | `docs/avance_tecnico_MCC225.pdf` |
| Cuaderno14 resuelto y ejecutado | `notebooks/Cuaderno14_MCC225_resuelto.ipynb` |
| Actividad5 aplicada | `Actividad5-MCC225.md` + `evaluacion_responsable_mcc225/` |
| Evidencia reproducible | `outputs/metrics/`, `outputs/tables/`, `outputs/figures/` |
| Repositorio con commits verificables | rama `exposicion2-mcc225` |

## 🧠 Decisiones técnicas

1. **Datos del Cuaderno14 = Winoground local.** El cuaderno base usa un dataset
   genérico imagen-texto; se adaptó a los **datos reales del proyecto** construyendo
   `data/raw/manifest_local.csv` con 120 pares reales desde el caché de Winoground
   (`scripts/build_local_manifest.py`). Esto cumple la consigna de Actividad5
   («usar el Cuaderno14 como plantilla adaptable al proyecto») y evita descargas
   pesadas. Ver [ADR 0002](adr/0002-cuaderno14-sobre-winoground.md).
2. **Modelos en safetensors locales.** `transformers>=5` con `torch<2.6` rechaza
   `torch.load` de checkpoints `.bin` (CVE-2025-32434). Se convierten los pesos
   oficiales de CLIP y BLIP a safetensors una sola vez
   (`scripts/convert_models_to_safetensors.py`, `make models`) sin desactivar
   ninguna verificación de seguridad.
3. **Dispositivo MPS.** La ejecución usa Apple Silicon (`mps`) con
   `PYTORCH_ENABLE_MPS_FALLBACK=1` para operaciones no soportadas.
4. **Autoría única.** Commits firmados solo por el estudiante; el uso de asistencia
   de IA generativa se declara una vez en el README/informe conforme a la consigna.

## 🔬 Alcance experimental

- **Cuaderno14** (`Cuaderno14_MCC225_resuelto.ipynb`): retrieval Recall@K, CLIPScore,
  captioning BLIP con BLEU/ROUGE-L/cobertura, degradación visual (blur, gris,
  baja resolución, recorte), perturbación textual (negación, conteo, atributo,
  objeto externo), simulación CapFilt y plantilla de análisis de errores de 10
  categorías anotada a mano.
- **Pipeline Winoground** (ya calculado, `outputs/metrics/`): text=0.3475,
  image=0.110, group=0.075 (azar group=1/6≈0.167; humano≈0.855); R@5=0.667,
  R@10=0.774; error por tag (Relation peor, group=0.047); prueba de ceguera;
  comparación de 3 checkpoints; validación del scorer contra `clip.jsonl` oficial.

## 🗺️ Mapeo a la rúbrica (20 pts)

| Criterio | Pts | Dónde |
|---|---|---|
| Formulación del integrador y tarea | 3 | avance §1 · slide 2 |
| Transformer + atención | 4 | avance §3 · slide 4 |
| Arquitecturas multimodales + comparación | 4 | ADR 0001 · avance §2 · slide 3 |
| Evidencia experimental del cuaderno | 3 | Cuaderno14 resuelto · outputs |
| Actividad5 aplicada | 3 | `evaluacion_responsable_mcc225/` |
| Repositorio, trazabilidad, reproducibilidad | 2 | Docker · Makefile · CI |
| Comunicación y defensa | 1 | `docs/RESPUESTAS_PREGUNTAS.md` |

## 🐳 Reproducción

| Camino | Comando |
|---|---|
| Local | `make setup && make models && make avance && make run && make figures && make test` |
| Docker | `docker compose run --rm avance` |

## ✅ Estado de fases

| Fase | Estado |
|---|---|
| 0. Sincronización de insumos + entorno | ✅ |
| 1. Cuaderno14 resuelto y ejecutado | ⏳ ejecución |
| 2. Actividad5 aplicada | ⏳ |
| 3. Diapositivas (8, Beamer→PDF) | ✅ |
| 4. Avance técnico (2–3 pp) | ✅ |
| 5. Preparación de defensa + QA adversarial | ⏳ |
| 6. Verificación previa + commit | ⏳ |

Ver flujograma completo en [FLUJOGRAMA_MCC225.md](FLUJOGRAMA_MCC225.md).
