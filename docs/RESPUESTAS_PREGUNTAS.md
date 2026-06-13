# Respuestas técnicas — Defensa MCC225 (Niels V. Pacheco Barrios)

**Tema:** Winoground / Evaluating CLIP · **Cuadernos:** C5 (deep fusion), C8 (atención crossmodal), C10 (OpenCLIP/retrieval/FAISS).

> Resultado principal (verificable en vivo): Winoground oficial (400 ej.), OpenCLIP
> ViT-B-32/laion2b → **text=0.347, image=0.110, group=0.075** (azar group=1/6≈0.167;
> humano≈0.855), pero **R@5=0.67 / R@10=0.77**. Alto retrieval + group ≈ azar.
> Scorer **validado** contra los scores oficiales de CLIP (`clip.jsonl`): reproduce
> text=0.3075/image=0.105/group=0.08. Evidencia en `outputs/metrics/` y `outputs/figures/`.

---

## 8.6 — Mis cinco preguntas clave

### 1. ¿Por qué un buen resultado en retrieval no garantiza razonamiento composicional?

Porque miden cosas distintas. **Retrieval (R@K)** pregunta: dada una caption, ¿está la
imagen correcta entre las K más similares de **toda la galería**? Casi todas las imágenes
de la galería son semánticamente distintas, así que basta con captar el *contenido grueso*
(objetos, escena) para acertar. **Winoground** pregunta algo mucho más fino: dentro de un
**par mínimo** —dos imágenes y dos captions con **las mismas palabras en distinto orden**—
¿asigna el modelo mayor similitud al emparejamiento correcto que al incorrecto? Eso exige
codificar el **orden y las relaciones** (quién está sobre quién, qué color liga a qué objeto),
no solo la bolsa de conceptos.

Un dual-encoder como CLIP (C10) colapsa cada modalidad en **un único embedding global** y
compara por **similitud coseno**. Ese embedding se comporta de forma casi *bag-of-concepts*:
"perro", "césped", "taza" activan dimensiones similares estén como estén compuestos. Por eso
puede tener R@K alto y, a la vez, **group score cercano a 1/6 (azar)**. Mi experimento lo
muestra lado a lado (`recall_vs_group.json`, figura `recall_vs_group.png`). Conexión con **C5**:
la fusión profunda (MMBT) y, en general, un cross-encoder permiten interacción token-a-token y
*sí* pueden modelar la composición; el dual-encoder, por diseño, no.

### 2. ¿Cómo verificaría si el modelo está usando realmente la imagen y no solo pistas textuales?

Con una **prueba de ceguera por permutación** (`src/blindness_probe.py`, figura `blindness.png`).
Reemplazo, para cada ejemplo, sus dos imágenes por las de **otro** ejemplo y recalculo los tres
scores. Si los scores **reales ≫ permutados (≈ azar)**, el resultado depende del contenido visual
⇒ el modelo **sí usa la imagen**. Si fueran parecidos, estaría explotando un sesgo a priori entre
las dos captions (pistas no visuales). En Winoground esto es especialmente importante porque las
dos captions tienen las **mismas palabras**: no hay "atajo" léxico que distinga una de otra sin
mirar la imagen. Complemento: en el **image score** la caption se fija y se eligen imágenes; si el
modelo ignorara la imagen no podría superar el azar en esa dirección. Conexión con **C8**: la
atención crossmodal permite *inspeccionar* qué región/segmento atiende cada token — es la versión
interpretable de "¿está mirando la imagen?".

### 3. ¿Qué tipo de error aparece en Winoground?

Errores de **vinculación composicional**, no de reconocimiento. El modelo reconoce los objetos
correctos pero **falla al ligarlos en la estructura correcta**. Mi `error_analysis.py` los separa
por tag (figura `by_tag.png`, `failure_cases.json`):

- **Relación / orden** ("A a la izquierda de B" vs "B a la izquierda de A", "A sobre B" vs "B sobre A"):
  el modelo da similitud casi idéntica a ambas porque el embedding global no codifica bien la relación espacial.
- **Binding atributo-objeto** ("círculo rojo y cuadrado azul" vs "círculo azul y cuadrado rojo"):
  el modelo "ve" rojo, azul, círculo, cuadrado, pero no a *qué objeto* pertenece cada color.

Típicamente el **text score** y el **image score** caen de forma desigual: hay ejemplos donde acierta
una dirección y falla la otra (por eso el group, que exige ambas, es el más bajo). El paper original
clasifica además en object/relation/both/symbolic/pragmatics/series; el mío reproduce object vs relation.

### 4. ¿Cómo adaptaría el código para evaluar pares imagen-texto con cambios mínimos de composición?

Eso es exactamente lo que hace mi pipeline y lo puedo mostrar en vivo:

1. **Estructura de datos** (`src/winoground_data.py`): cada ejemplo = `(image_0, image_1, caption_0, caption_1)`
   con la convención `caption_0 ↔ image_0`. Para crear un par mínimo nuevo, basta duplicar una caption y
   **permutar dos tokens** (orden o atributo) y aportar las dos imágenes correspondientes.
2. **Scoring** (`src/winoground_eval.py`): construyo la matriz 2×2 `sim[caption, imagen]` con embeddings
   normalizados y aplico `text_correct / image_correct / group_correct`. No hay que tocar el modelo.
3. **Mínimo cambio de código** en vivo: añadir un ejemplo al `examples.jsonl` curado o cambiar `caption_1`
   por otra permutación y re-ejecutar `02_run_winoground.py`. Verás cómo cambian los tres scores.

El motor de embeddings es el de **C10** (`openclip_utils.create_model/encode_*`); lo único "nuevo" frente a
C10 es el **scorer composicional** y la **matriz 2×2 por par**.

### 5. ¿Qué limitación tienen las métricas automáticas en este caso?

- **No miden composición:** R@K y accuracy zero-shot pueden ser altos con un modelo *bag-of-concepts*. Son
  necesarias pero **no suficientes**; Winoground existe justamente para tapar ese hueco.
- **Umbral y empates:** los scores son comparaciones estrictas (`>`); un empate cuenta como fallo. Pequeñas
  diferencias de similitud deciden el resultado, así que las métricas son **sensibles al ruido numérico** y
  a la normalización de embeddings.
- **Azar no trivial:** el azar del group score es **1/6, no 1/16**, porque text e image **no son independientes**
  (comparten las 4 similitudes). Comparar contra el baseline correcto es parte de la métrica.
- **El propio benchmark tiene ruido:** Diwan et al. ("Why is Winoground Hard?") muestran que parte de los
  ejemplos son ambiguos, visualmente difíciles o requieren conocimiento extra; un group score bajo mezcla
  *fallo composicional* con *dificultad del ítem*. Por eso complemento la métrica con análisis por tag y casos
  cualitativos.
- **Evaluating CLIP (Agarwal et al., 2021):** una métrica de "capacidad" más alta no implica "mejor"
  modelo: el diseño de clases y los prompts introducen sesgos que la accuracy oculta.

---

## §7 — Diez preguntas generales

1. **¿Qué resultado concreto puedo reproducir/verificar en vivo?** El cálculo de los tres scores de Winoground
   con OpenCLIP: ejecuto `python scripts/02_run_winoground.py` y muestro `scores.json` + la figura
   `scores_vs_chance.png` (group ≈ azar).
2. **¿Qué cuaderno usé directamente y qué parte adapté?** C10 (OpenCLIP): reutilicé el patrón de
   `create_model_and_transforms`, encode de imagen/texto y similitud coseno (`src/openclip_utils.py`). Lo nuevo
   es el scorer composicional 2×2 y la prueba de ceguera.
3. **¿Qué celdas/funciones/scripts generaron mis resultados?** `scripts/02_run_winoground.py` →
   `src/winoground_eval.py` (`text_correct/image_correct/group_correct/aggregate`) y `src/metrics.py`
   (`recall_at_k`, `bootstrap_ci`). El notebook `notebooks/Winoground_Eval_MCC225.ipynb` los orquesta.
4. **¿Qué cambió respecto al cuaderno original (C10)?** C10 hacía retrieval/zero-shot estándar; yo cambié la
   *tarea* a pares mínimos composicionales y añadí (a) scorer Winoground, (b) IC bootstrap, (c) prueba de
   ceguera, (d) contraste R@K vs group, (e) análisis por tag.
5. **¿Qué evidencia sustenta la afirmación?** Tabla `checkpoint_comparison.csv`, `bootstrap_ci.csv` (IC 95%),
   figuras `scores_vs_chance.png`, `recall_vs_group.png`, `by_tag.png`, `blindness.png`.
6. **¿Qué error/limitación/sesgo observé?** Errores de binding/relación (no de reconocimiento); sensibilidad a
   empates; el set curado es un proxy controlado; el benchmark real tiene ítems ambiguos.
7. **¿Qué pasaría si cambio modelo/prompt/checkpoint/dataset/negativos?** Lo medí: ver `checkpoint_comparison.csv`
   (ViT-B-32 vs B-16 vs L-14). Un checkpoint mayor sube algo los scores pero el group sigue muy por debajo del
   humano; cambiar a pares no-mínimos (retrieval normal) dispara R@K y oculta el problema.
8. **¿Qué parte corresponde a conceptos de clase?** Contrastive dual-encoder y similitud coseno (C10/C6);
   dual vs deep fusion (C5); atención crossmodal / interpretabilidad (C8); zero-shot y FAISS (C10).
9. **¿Qué mejora haría para un pipeline reproducible?** Ya está: `uv` + lockfile, Dockerfile, Makefile, seeds,
   `env_logging`, tests, CI nightly. Siguiente paso: cachear embeddings versionados y DVC para los datos.
10. **¿Qué resultado no puedo asegurar aún y por qué?** Que la **causa** del fallo sea *solo* composicional: el
    group score (0.075) mezcla fallo de composición con ítems ambiguos/visualmente difíciles (Diwan et al. estiman
    que una fracción del benchmark es ruidosa). Tampoco puedo asegurar que un cross-encoder (C5) lo resuelva sin
    evaluarlo: es mi hipótesis y mi mejora propuesta, no un resultado medido. El número de OpenCLIP **sí** lo
    aseguro (validado contra `clip.jsonl`).

---

## §9 — Mejoras de código en vivo (cómo las hago)

| Pedido del docente | Cómo lo hago (archivo / acción) |
|---|---|
| Cambiar una consulta y ver el ranking | `configs/experiment.yaml: faiss_query`; re-corro la demo FAISS (`faiss_demo.csv`). |
| Agregar un prompt alternativo zero-shot | Encapsular la caption en plantillas ("a photo of …") y promediar embeddings (patrón C10). |
| Mostrar top-5 y no top-1 | `recall_at_k(..., ks=(1,5,10))` ya lo calcula; FAISS devuelve top-5. |
| Calcular similitud coseno | `src/openclip_utils.pair_similarity_matrix` / `metrics.cosine_similarity_matrix`. |
| Normalizar embeddings antes de comparar | `encode_*` ya hace `emb / emb.norm(dim=-1)`; lo muestro en el código. |
| Separar correctos e incorrectos | `error_analysis.failure_cases` (group=0) vs el resto. |
| Identificar un hard negative | En cada par mínimo, la imagen/caption *incorrecta* ES el hard negative; la muestro en `failure_cases.json`. |
| Agregar Recall@K | `src/metrics.recall_at_k`. |
| Comparar dos checkpoints | `configs/checkpoints.yaml` + `checkpoint_comparison.csv`. |
| Qué celda ejecutar primero | La celda 1 del notebook (hoja de trazabilidad/entorno) y luego carga de datos. |
| Qué dependencia/ruta puede romper | Acceso gated a Winoground (token HF), versión de `open_clip`/`torch`; documentado en README. |
| Qué parte corresponde al paper | `winoground_eval.py` implementa las definiciones de la Sec. 3 del paper. |
