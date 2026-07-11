### Reporte individual de evaluación multimodal

#### Datos del estudiante

- **Nombre:** Niels Victor Pacheco Barrios
- **Repositorio:** github.com/nielspac177/MCC225-ExamenParcial-Winoground (rama `exposicion2-mcc225`)
- **Commit final:** ver `git log` de la entrega
- **Fecha:** 2026-07-11

#### Tarea definida

Se evalúa un **dual-encoder CLIP** (`openai/clip-vit-base-patch32`) sobre el
alineamiento imagen–texto. Entrada: una imagen y un caption; salida: un embedding
conjunto por modalidad cuya **similitud coseno** mide la compatibilidad. Sobre esa
base se estudian recuperación (retrieval), CLIPScore, calidad de captions generados
por BLIP, robustez ante degradación visual y perturbación textual, y análisis de
errores. El objetivo es contrastar **retrieval alto vs. razonamiento composicional**,
que es la tesis del trabajo integrador (Winoground).

#### Dataset

**120 pares imagen–texto reales de Winoground** (60 ejemplos, cada uno aporta
`image_0↔caption_0` e `image_1↔caption_1`), construidos desde el caché local con
`scripts/build_local_manifest.py`. Winoground está compuesto por **pares mínimos**:
dos imágenes y dos captions con las mismas palabras reordenadas, que sólo se
distinguen por composición. Etiquetas: Object, Relation, Both. La evaluación de
retrieval usa 100 pares; las ablaciones, subconjuntos de 40–80.

#### Modelos evaluados

- **CLIP** `openai/clip-vit-base-patch32` (151.3 M parámetros), inferencia en `mps`,
  embeddings L2-normalizados, batch 16. Pesos convertidos a `safetensors` local
  (`scripts/convert_models_to_safetensors.py`) por la restricción de
  `transformers>=5` + `torch<2.6` (CVE-2025-32434).
- **BLIP** `Salesforce/blip-image-captioning-base` como generador de captions.
- Semilla global 22514; trazabilidad en `outputs/metrics/metadata_experimento.json`.

#### Baselines

**Captions desplazados**: se rota el emparejamiento imagen–texto para romper la
correspondencia. Es la comparación mínima razonable porque mide cuánto del
desempeño proviene del alineamiento real y no de sesgos del conjunto.

| Sistema | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|---|--:|--:|--:|--:|--:|--:|
| CLIP (pares reales) | 0.44 | 0.87 | 0.97 | 0.39 | 0.82 | 0.95 |
| Baseline (desplazado) | 0.01 | 0.03 | 0.09 | 0.00 | 0.02 | 0.04 |

CLIP supera al baseline por un margen enorme: el alineamiento imagen–texto es real.

#### Métricas

- **Retrieval:** ver tabla anterior; `outputs/tables/comparacion_retrieval.csv`.
- **CLIPScore simplificado (diagonal):** `outputs/tables/clipscore_casos.csv`.
- **Métricas léxicas de captions BLIP (corregidas):** BLEU≈**0.073**, ROUGE-L≈**0.191**,
  cobertura léxica≈**0.263** (`outputs/tables/metricas_captions.csv`). Nota técnica:
  la implementación original filtraba el caption generado dentro de sus propias
  referencias (BLEU/ROUGE=1.0 espurios); se corrigió excluyendo `caption_generado`
  del conjunto de referencias.
- **CapFilt (umbral óptimo 0.210):** precisión 0.887, recall 0.979, accuracy 0.903
  (TP 47, FP 6, FN 1, TN 18); `outputs/tables/capfilt_umbral.csv`.
- **Ablación visual (CLIPScore medio):** original 0.270; desenfoque 0.267; baja
  resolución 0.267; grises 0.259; recorte central 0.262. El retrieval se mantiene
  estable (R@5 ≈ 0.86–0.93): CLIP es **robusto a degradaciones fotométricas leves**.
  `outputs/tables/ablacion_visual.csv`.

#### Análisis de errores

`outputs/tables/plantilla_analisis_errores.csv` (29 casos, mitad peor por CLIPScore
+ muestra aleatoria), anotados con la taxonomía del cuaderno:

| Categoría | Casos |
|---|--:|
| relación espacial | 12 |
| otro (negación, temporal, contraste) | 7 |
| acción (rol agente/paciente) | 4 |
| conteo | 3 |
| atributo | 3 |

Los fallos están dominados por **relación espacial** (izquierda/derecha,
cerca/lejos, encima/debajo) y por el **binding de atributos y roles**: exactamente
lo que un dual-encoder sin cross-attention no resuelve. Ejemplos: «the hurt person
is on the left…» vs. su intercambio reciben similitud casi idéntica.

#### Discusión de limitaciones

- Recall@K y CLIPScore **no capturan composición**: miden presencia de conceptos, no
  su disposición ni el binding. Un sistema puede recuperar bien y aun así fallar el
  par mínimo.
- BLEU/ROUGE-L penalizan a BLIP por no reproducir el léxico composicional específico,
  pero no distinguen entre error semántico y variación de estilo (la cobertura léxica
  es más informativa aquí).
- Tamaño reducido (120 pares) → potencia estadística limitada; los números se
  interpretan como evidencia direccional, no como estimaciones definitivas.
- El baseline desplazado es intencionalmente débil; supera esa cota no implica
  competencia composicional.

#### Conexión conceptual

CLIP (dual-encoder contrastivo) hereda de **BERT** la self-attention por torre, pero
**no** tiene interacción cruzada. Modelos de fusión profunda como **VisualBERT,
UNITER, ViLT** y sobre todo **BLIP/BLIP-2** (Q-Former, cross-attention) y **LLaVA**
(tokens visuales dentro de un LLM) sí modelan la interacción imagen–texto token a
token, y por eso son la vía natural para mejorar composición. El experimento muestra
el techo del dual-encoder y motiva un cross-encoder como paso de cierre.

#### Conclusión

El sistema muestra **alineamiento multimodal parcial**: retrieval alto y robusto,
pero razonamiento composicional cercano al azar (en el pipeline Winoground completo
el group score es 0.075 frente a un azar de 0.167 y un humano de 0.855). La
conclusión central del avance integrador se sostiene con evidencia reproducible:
**tener buen retrieval no garantiza composición**.

#### Declaración de uso de herramientas generativas

Trabajo individual. Se utilizó asistencia de IA generativa para desarrollo de código
y redacción; las definiciones de métrica se verificaron contra el paper de Winoground
y el scorer oficial (`clip.jsonl`), y todos los números provienen de archivos
reproducibles del repositorio.
