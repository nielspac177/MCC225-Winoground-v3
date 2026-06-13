# Hoja de trazabilidad — Examen Parcial MCC225 (§10)

| Elemento | Respuesta del estudiante |
|---|---|
| **Paper / modelo / línea temática** | Winoground (Thrush et al., 2022) + Evaluating CLIP (Agarwal et al., 2021). Razonamiento composicional visio-lingüístico. |
| **URL del repositorio entregado** | https://github.com/nielspac177/MCC225-ExamenParcial-Winoground |
| **Cuadernos usados** | **C10** (OpenCLIP: embeddings, coseno, checkpoints, FAISS) — motor principal. **C5** (dual-encoder vs fusión profunda) y **C8** (atención crossmodal / interpretabilidad) — marco conceptual de contraste. |
| **Notebook / script ejecutado** | `notebooks/Winoground_Eval_MCC225.ipynb` (ejecutado) · `scripts/02_run_winoground.py`. |
| **Celda / función / bloque clave** | `src/winoground_eval.py`: `text_correct`, `image_correct`, `group_correct`, `aggregate`. Matriz 2×2 `sim[caption, imagen]` construida con embeddings normalizados de OpenCLIP. |
| **Resultado obtenido** | Winoground oficial (400 ej.), ViT-B-32/laion2b: **text=0.347, image=0.110, group=0.075** (azar group=0.167; humano≈0.855), pero **Recall@5=0.67 / R@10=0.77**. Alto retrieval + group ≈ azar. Scorer validado contra `clip.jsonl` oficial (text=0.3075/image=0.105/group=0.08). |
| **Métrica / tabla / gráfico / evidencia** | `outputs/figures/scores_vs_chance.png`, `recall_vs_group.png`, `by_tag.png`, `blindness.png`, `checkpoint_comparison.png`; CSVs en `outputs/metrics/`. |
| **Cambio hecho sobre el cuaderno original (C10)** | C10 hacía retrieval/zero-shot estándar. Cambié la **tarea** a pares mínimos composicionales y añadí: (1) scorer Winoground (3 scores oficiales), (2) IC bootstrap, (3) prueba de ceguera por permutación, (4) contraste R@K vs group score, (5) análisis de error por tag. |
| **Limitación encontrada** | El group score mezcla *fallo composicional* con *ítems ambiguos/OOD* (Diwan et al.). El set curado es un proxy sintético (OOD para CLIP). El azar del group es 1/6 (no 1/16). Sensibilidad a empates y normalización. |
| **Mejora propuesta** | Evaluar un **cross-encoder / fusión profunda (C5)** y re-ranking con interacción cruzada (C6); versionar embeddings (DVC); usar el Winoground oficial completo con prompts/checkpoints adicionales. |

## Entorno (auto-generado por `scripts/00_verify_env.py`)

Ejecutar `make setup && .venv/bin/python scripts/00_verify_env.py` imprime versiones,
revisión de git y dispositivo. Snapshot de la última corrida en `outputs/metrics/scores.json`
(campo `environment`): Python 3.12, torch 2.5.1, open_clip 3.3.0, dispositivo MPS (Apple Silicon).

## Orden de ejecución para reproducir (qué celda primero)

1. `make setup` (entorno) → 2. `make data` (datos) → 3. `make run` (métricas) →
4. `make figures` (figuras) → 5. abrir `notebooks/Winoground_Eval_MCC225.ipynb`.
La **celda 1** del notebook imprime esta hoja de trazabilidad (entorno).
