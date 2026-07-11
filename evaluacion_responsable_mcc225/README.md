# evaluacion_responsable_mcc225

Índice de entregables de la **Actividad complementaria 5 — Evaluación responsable del proyecto multimodal** (MCC225).

**Proyecto:** Evaluación responsable de CLIP en Winoground.
**Tesis:** un desempeño alto en recuperación (retrieval) no equivale a razonamiento composicional.
**Modelo:** dual-encoder CLIP ViT-B/32 · **Datos:** pares reales de Winoground · **Entorno:** Apple Silicon (`mps`).

Este directorio es un **índice híbrido**: en lugar de duplicar archivos, apunta a los entregables reales que ya viven en el repositorio. La estructura que sugiere la Actividad5 se mapea así:

```text
evaluacion_responsable_mcc225/            (este índice)
├── README.md                             -> este archivo
├── Cuaderno14_MCC225_resuelto.ipynb      -> ../notebooks/Cuaderno14_MCC225_resuelto.ipynb
├── reporte_evaluacion_responsable.md     -> ../reporte_evaluacion_responsable.md
├── results/                              -> ../results/
│   ├── metricas.csv
│   ├── casos_analizados.csv
│   └── ficha_uso_responsable.csv
└── figures/                              -> ../figures/ejemplos_evaluados.png
```

## Entregables (mapeo con la consigna)

| # | Entregable | Ubicación real | Qué contiene |
| --- | --- | --- | --- |
| 1 | Cuaderno 14 resuelto/adaptado | [`../notebooks/Cuaderno14_MCC225_resuelto.ipynb`](../notebooks/Cuaderno14_MCC225_resuelto.ipynb) | Pipeline de evaluación adaptado a datos reales de Winoground (ver [ADR 0002](../docs/adr/0002-cuaderno14-sobre-winoground.md)) |
| 2 | Reporte de evaluación responsable | [`../reporte_evaluacion_responsable.md`](../reporte_evaluacion_responsable.md) | Resuelve las 8 partes de la Actividad5 (ficha, adaptación, resultados, 5 casos, confiabilidad, explicabilidad, sesgo, conclusión) |
| 3 | Tabla de métricas | [`../results/metricas.csv`](../results/metricas.csv) | Métricas principales (retrieval, ablación visual, Winoground, CapFilt, ceguera, captioning) con valores reales |
| 4 | Tabla de 5 casos analizados | [`../results/casos_analizados.csv`](../results/casos_analizados.csv) | 2 aciertos, 2 errores y 1 ambiguo, todos con su origen en `failure_cases.json` |
| 5 | Ficha de uso responsable | [`../results/ficha_uso_responsable.csv`](../results/ficha_uso_responsable.csv) | Sesgos, riesgo, supervisión y uso recomendado (limitado) |
| 6 | Figura con ejemplos evaluados | [`../figures/ejemplos_evaluados.png`](../figures/ejemplos_evaluados.png) | Collage 2×3 de imágenes reales con caption y CLIPScore por caso |
| 7 | Repositorio actualizado | Raíz del repo | Ver evidencia bajo [`../outputs/`](../outputs/) |

## Evidencia reproducible (fuentes de todos los números)

- `../outputs/metrics/metricas_finales.json` — retrieval CLIP vs baseline, CLIPScore, CapFilt, ablación visual.
- `../outputs/metrics/scores.json` — Winoground text=0.3475, image=0.110, group=0.075 (azar group=0.167, humano=0.855).
- `../outputs/metrics/by_tag.csv` — group por categoría (Relation 0.047 peor, Object 0.085, Both 0.269).
- `../outputs/metrics/blindness.json` — prueba de ceguera (group 0.075 → 0.015 al permutar imágenes).
- `../outputs/metrics/failure_cases.json` — 8 casos group=0 con captions y matriz 2×2 (origen de la tabla de 5 casos).
- `../outputs/metrics/recall_vs_group.json` — Recall@5=0.667, Recall@10=0.774 frente a group=0.075.
- `../outputs/tables/perturbacion_textual.csv` — sensibilidad al prompt (negación, conteo, atributo).
- `../outputs/tables/plantilla_analisis_errores.csv` — 29 casos anotados (categoría, severidad, evidencia).
- `../outputs/tables/{comparacion_retrieval,ablacion_visual,metricas_captions}.csv` — tablas de apoyo.
- `../data/raw/manifest_local.csv` y `../data/raw/images_local/` — 120 pares reales e imágenes.

## Reproducir la figura

```bash
.venv/bin/python scripts/generar_figura_ejemplos.py
# -> figures/ejemplos_evaluados.png
```
