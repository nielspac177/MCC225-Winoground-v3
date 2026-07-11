# ADR 0002 — Ejecutar el Cuaderno14 sobre los datos reales del proyecto (Winoground)

- **Estado:** Aceptado
- **Fecha:** 2026-07-11
- **Contexto de la decisión:** Segunda Exposición Académica MCC225A
- **Relacionado con:** [ADR 0001](0001-dual-encoder-vs-deep-fusion.md)

## Contexto

El `Cuaderno14_MCC225.ipynb` es una plantilla experimental de evaluación multimodal.
Su configuración por defecto descarga un dataset genérico de imagen-texto
(`jxie/flickr8k`). La `Actividad5` pide explícitamente usar el Cuaderno14 «como
plantilla experimental adaptable» y evaluar «una parte pequeña del propio sistema»
del estudiante, no un dataset ajeno al proyecto.

El proyecto integrador de este repositorio evalúa un dual-encoder CLIP sobre
**Winoground**, cuyo caché de datos reales ya está disponible localmente
(`data/winoground_cache/`).

## Decisión

Adaptar el Cuaderno14 para que corra sobre **los datos reales del proyecto**
(Winoground) en lugar del dataset genérico:

1. `scripts/build_local_manifest.py` construye `data/raw/manifest_local.csv` con 120
   pares imagen-texto reales tomados del caché de Winoground (cada ejemplo aporta dos
   pares: `image_0↔caption_0` e `image_1↔caption_1`).
2. El cuaderno detecta ese manifiesto local y omite cualquier descarga (rama ya
   prevista en su celda de carga de datos), quedando ejecutable de forma offline y
   reproducible.
3. Se conservan intactas todas las etapas del cuaderno: retrieval, CLIPScore,
   captioning BLIP, degradación visual, perturbación textual, CapFilt y análisis de
   errores.

## Consecuencias

**Positivas**

- La evidencia del cuaderno queda alineada con el proyecto integrador, cumpliendo la
  consigna de la Actividad5 y conectando cuaderno ↔ integrador (la rúbrica penaliza
  no conectarlos).
- Reproducible y offline: sin descargas de datasets pesados; `make avance` regenera
  todo desde el caché local.
- Coherencia de dominio: las mismas imágenes/captions composicionales del proyecto
  aparecen en las métricas de retrieval, en las ablaciones y en el análisis de
  errores.

**Negativas / limitaciones**

- Winoground está diseñado con pares mínimos (dos captions que se diferencian por
  composición); usados como pares imagen-caption sueltos, algunas métricas de
  retrieval son más exigentes que en un dataset de captioning descriptivo. Se
  documenta como limitación en el reporte.
- El tamaño (120 pares) prioriza rapidez de ejecución sobre potencia estadística;
  está dentro del rango 30–100+ que pide la Actividad5 y se declara explícitamente.

## Alternativas consideradas

- **Usar `jxie/flickr8k` tal cual:** descarga pesada y evidencia desconectada del
  proyecto integrador; descartada por la consigna de Actividad5.
- **Set curado sintético (`data/curated/`):** offline e instantáneo, pero con
  imágenes sintéticas de formas; menos defendible como «datos reales».
