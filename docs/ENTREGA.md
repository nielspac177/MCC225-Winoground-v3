# Checklist de entrega — Examen Parcial MCC225

Mapa de cada requisito del examen al artefacto del repositorio.

## Evidencia mínima (§3)

| Requisito | Dónde |
|---|---|
| 1. Repositorio con respuestas/cuadernos/evidencias | este repo (público) |
| 2. Notebook/script ejecutado | `notebooks/Winoground_Eval_MCC225.ipynb` (con salidas) · `scripts/02_run_winoground.py` |
| 3. Celda/función del resultado principal | `src/winoground_eval.py` (`text/image/group_correct`) |
| 4. Resultado (métrica/tabla/gráfico) | `outputs/metrics/*` · `outputs/figures/*` |
| 5. Relación con el paper asignado | `docs/INFORME.md` (Winoground + Evaluating CLIP) |
| 6. Relación con los 3 cuadernos | C10 (motor), C5/C8 (contraste) — `README.md`, `docs/adr/0001` |
| 7. Limitación técnica | `docs/INFORME.md §4`, `docs/HOJA_TRAZABILIDAD.md` |
| 8. Mejora posible | cross-encoder (C5) + re-ranking (C6) + DVC — `docs/adr/0001` |

## Rúbrica (§11, 20 pts) — cómo se cubre

| Criterio | Pts | Cobertura |
|---|---|---|
| Verificación del resultado | 4 | scorer validado contra `clip.jsonl`; `make run` regenera todo |
| Explicación de ejecución/adaptación de cuadernos | 4 | `README` (mapa C5/C8/C10), notebook, `docs/RESPUESTAS_PREGUNTAS.md §7` |
| Dominio del paper | 3 | `docs/INFORME.md` |
| Mejora de código en vivo | 3 | `docs/RESPUESTAS_PREGUNTAS.md §9` (tabla de acciones) |
| Análisis crítico / reproducibilidad | 3 | bootstrap CI, blindness probe, Docker/uv/CI/seeds |
| Relación con C5–C10 | 2 | `docs/adr/0001`, `README` |
| Claridad de defensa | 1 | `slides/` (Beamer PDF + PPTX), estructura 13 min |

## Hoja de trazabilidad (§10)
`docs/HOJA_TRAZABILIDAD.md` (también impresa por la celda 1 del notebook).

## Para la defensa (13 min, §5)
Slides: `slides/latex/defensa_winoground.pdf` y `slides/pptx/defensa_winoground.pptx`.
Guion de respuestas: `docs/RESPUESTAS_PREGUNTAS.md` (5 preguntas 8.6 + 10 generales + mejoras §9).

## Reproducir desde cero
```bash
make setup && make data && make run && make figures && make test && make validate
```
