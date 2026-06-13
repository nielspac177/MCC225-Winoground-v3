# Informe — Retrieval alto no implica razonamiento composicional: OpenCLIP en Winoground

**Niels Victor Pacheco Barrios** · MCC225, Examen Parcial 2026-1 · Cuadernos C5, C8, C10.

## Resumen

Evalúo si un modelo dual-encoder tipo CLIP, que destaca en recuperación imagen-texto,
razona de forma composicional. Con el motor OpenCLIP del Cuaderno 10 calculo los tres
scores oficiales de Winoground (text, image, group) y los contrasto con el Recall@K de
recuperación sobre el mismo conjunto. El resultado replica el hallazgo central de la
literatura: **el Recall@K es alto mientras el group score se mantiene cerca del azar**.
Es decir, el éxito en retrieval no garantiza comprensión composicional. Interpreto esa brecha
con los conceptos de fusión profunda (C5) y atención crossmodal (C8).

## 1. Introducción

CLIP (Radford et al., 2021) alinea imagen y texto en un espacio común mediante aprendizaje
contrastivo y obtiene fuerte desempeño zero-shot y de recuperación. Winoground (Thrush et al.,
2022) cuestiona si ese desempeño implica *razonamiento composicional*. Presenta pares mínimos
(dos imágenes y dos captions con **las mismas palabras en distinto orden**) donde acertar exige
codificar relaciones y vinculación atributo-objeto, no solo conceptos sueltos. "Evaluating CLIP"
añade que una accuracy mayor no define un modelo "mejor", porque el diseño de clases y prompts introduce
sesgos. **Hipótesis:** un dual-encoder tendrá Recall@K alto y group score ≈ azar.

## 2. Métodos

- **Modelo / motor (C10):** OpenCLIP `ViT-B-32/laion2b` (principal); comparación con
  `ViT-B-16/datacomp_xl` y `ViT-L-14/openai`. Embeddings L2-normalizados, similitud coseno.
- **Datos:** Winoground oficial (400 ejemplos, gated en HF) con *fallback* a un set curado de
  pares mínimos sintéticos (relaciones espaciales y binding color-objeto), 100% offline.
- **Scorer (núcleo):** por ejemplo, matriz 2×2 `sim[caption, imagen]`; definiciones del paper
  (Sec. 3) en `src/winoground_eval.py`. Azar: text=image=1/4, **group=1/6** (text e image no son
  independientes; verificado combinatorialmente y por panel adversarial).
- **Análisis:** IC bootstrap 95% (2000 remuestreos); scores por tag; **prueba de ceguera** por
  permutación de imágenes; Recall@1/5/10 sobre toda la galería; demo FAISS.
- **Reproducibilidad:** `uv`+lockfile, Docker, seeds, `env_logging`, tests (pytest), CI nightly.

## 3. Resultados

Winoground oficial (400 ejemplos), OpenCLIP **ViT-B-32/laion2b** (números exactos en
`outputs/metrics/`, regenerables con `make run`):

- **Scores:** text = **0.348** (IC95% [0.30, 0.40]), image = **0.110** [0.08, 0.14],
  **group = 0.075** [0.05, 0.10]. El text está **por encima** del azar (0.25) y el group
  **por debajo** del azar (1/6 ≈ 0.167). Que group < 1/6 no significa "peor que
  aleatorio" globalmente. Es consecuencia de la **asimetría text ≫ image** (el modelo acierta
  bien una dirección y falla la otra), y el group exige ganar ambas. Queda lejísimos del
  humano (≈0.855).
- **Retrieval vs composición:** sobre el mismo conjunto, **R@5 = 0.67** y **R@10 = 0.77**
  (texto→imagen) frente a **group = 0.075**: el retrieval es alto y la composición falla.
- **Por tag (collapsed):** *Relation* es lo más difícil (group **0.047**, n=233), seguido de
  *Object* 0.085 (n=141) y *Both* 0.27 (n=26). Son fallos de **vinculación**, no de reconocimiento.
- **Prueba de ceguera:** real (text 0.348 / image 0.11 / group 0.075) ≫ control con imágenes
  permutadas (0.135 / 0.035 / 0.015), que **colapsa muy por debajo** del rendimiento real
  (incluso bajo el azar marginal de 0.25 en text, porque la imagen permutada raramente coincide
  con ninguna de las dos captions). El modelo **sí** usa la imagen; su límite es composicional,
  no perceptivo.
- **Checkpoints (sin tendencia monótona con el tamaño):** group 0.075 (B-32/151M), 0.072
  (B-16/datacomp/150M) y 0.085 (L-14/openai/428M). El **text score baja** con el modelo
  mayor: 0.348 (B-32) → 0.298 (B-16) → 0.288 (L-14). Ningún checkpoint se acerca al humano, así que
  escalar tamaño o datos no resuelve la composición.
- **Validación del scorer:** aplicado a los scores oficiales de CLIP del propio dataset
  (`statistics/model_scores/clip.jsonl`) reproduce **exactamente** text=0.3075, image=0.1050,
  group=0.0800 (`scripts/validate_against_official.py`).

> Figuras: `scores_vs_chance.png`, `recall_vs_group.png`, `by_tag.png`, `blindness.png`,
> `checkpoint_comparison.png`, `qualitative_examples.png`.

## 4. Discusión

Un dual-encoder colapsa cada modalidad en un embedding global y compara por coseno. Ese
mecanismo se comporta como *bag-of-concepts* y es insensible al orden y a la relación, justo lo que
Winoground penaliza. La **fusión profunda (C5, MMBT)** y la **atención crossmodal (C8)** permiten
interacción token-a-token y son la vía arquitectónica para abordar la composición. El retrieval
estándar (R@K) no detecta el problema porque la galería es semánticamente diversa, así que las
métricas automáticas resultan necesarias pero **no suficientes**.

**Limitaciones:** el group score mezcla fallo composicional con la dificultad o ambigüedad del ítem
(Diwan et al.). El set curado es OOD para CLIP, y las comparaciones son estrictas y sensibles a
empates. **Mejora futura:** evaluar un cross-encoder real (C5), probar re-ranking con interacción
cruzada (C6), ampliar prompts y checkpoints, y versionar embeddings (DVC).

## 5. Conclusión

El retrieval alto no implica razonamiento composicional. OpenCLIP lo ilustra con un group score ≈
azar pese a un Recall@K alto. El aporte reproducible es el pipeline que mide y contrasta ambos en el
mismo conjunto, conectando C10 (motor), C5 (dual vs deep fusion) y C8 (interpretabilidad).

## Referencias

- Thrush et al. (2022). *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality.* CVPR.
- Diwan et al. (2022). *Why is Winoground Hard?* EMNLP.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML.
- Agarwal et al. (2021). *Evaluating CLIP: Towards Characterization of Broader Capabilities and Downstream Implications.*
