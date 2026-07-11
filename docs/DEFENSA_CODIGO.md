# Defensa del código — qué abrir y qué decir

Guía para cuando el docente revisa **solo el código**. Objetivo: ubicar el resultado,
explicar qué reutilicé de los cuadernos y qué cambié, y hacer un cambio en vivo.

## Frase de entrada (10 s)
*"El proyecto tiene tres partes: `src/` con la lógica, `scripts/` que la orquesta, y
`tests/` que la verifica. El resultado principal sale de una sola función, el scorer en
`src/winoground_eval.py`. El motor de embeddings es el del Cuaderno 10. Te lo muestro en
ese orden."*

## Recorrido recomendado (abre los archivos en este orden)

### 1. `src/winoground_eval.py` — AQUÍ está el resultado (lo más importante)
*"Este es el bloque que produce mi resultado. Tres funciones. `text_correct` fija la
imagen y mira si gana el caption correcto; `image_correct` fija el caption y mira si gana
la imagen; `group_correct` exige las dos. `aggregate` promedia sobre los 400 ejemplos."*
- Señala: `text_correct`, `image_correct`, `group_correct`, `aggregate`.
- Si preguntan por la convención: *"`sim[c, i]` es la similitud entre el caption c y la
  imagen i; la diagonal es lo correcto."*
- Punto fuerte: *"Seguí el código oficial de Winoground, no la redacción de los papers."*

### 2. `src/openclip_utils.py` — el motor del Cuaderno 10 (qué reutilicé)
*"Esto es lo que tomé del Cuaderno 10 casi tal cual: crear el modelo OpenCLIP, codificar
imágenes y textos a embeddings normalizados, y la similitud coseno."*
- Señala: `create_model`, `encode_images`, `encode_texts`, `pair_similarity_matrix`.
- *"`pair_similarity_matrix` arma la matriz 2×2 que consume el scorer."*

### 3. `scripts/02_run_winoground.py` — el orquestador (cómo se obtiene todo)
*"Este script junta todo: carga datos, codifica con OpenCLIP, arma una matriz 2×2 por
ejemplo en `per_example_sims`, y calcula scores, Recall@K, por tag, ceguera y
checkpoints. Una corrida: `make run`."*
- Señala: `embed_all` (cachea embeddings), `per_example_sims`, y las secciones marcadas
  con comentarios (bootstrap, by_tag, blindness, recall vs group, FAISS).

### 4. `src/metrics.py` — las métricas
*"`recall_at_k` para la recuperación, `bootstrap_ci` para los intervalos de confianza, y
`cosine_similarity_matrix`."*

### 5. `scripts/validate_against_official.py` — la prueba de que el scorer es correcto
*"Aplico mi scorer a los scores oficiales de CLIP que vienen en el dataset y reproduzco
exactamente 0.3075, 0.105 y 0.08. Esto valida la implementación."* (Córrelo si te lo piden.)

### 6. `src/blindness_probe.py` y `src/error_analysis.py` — el análisis crítico
*"`run_blindness_probe` permuta las imágenes para ver si el modelo usa la imagen.
`scores_by_tag` y `failure_cases` separan los errores por tipo."*

### 7. `tests/` — calidad
*"`pytest` corre 16 tests del scorer y las métricas, con casos de verdad conocida."*

## Las tres preguntas que casi seguro hará (sobre código)

**"¿Qué celda/función generó tu resultado?"**
→ `aggregate` en `src/winoground_eval.py`, alimentada por `per_example_sims` en
`scripts/02_run_winoground.py`. La evidencia queda en `outputs/metrics/scores.json`.

**"¿Qué cambiaste respecto al Cuaderno 10?"**
→ *"C10 hacía retrieval y zero-shot estándar. Reusé su motor de embeddings, pero cambié
la tarea a pares mínimos composicionales y agregué cinco cosas nuevas: el scorer de los 3
scores, los intervalos bootstrap, la prueba de ceguera, el contraste R@K vs group, y el
análisis por tag."*

**"¿Qué dependencia/ruta puede romper la ejecución?"**
→ *"El acceso al dataset de Winoground está restringido por licencia: sin token de
Hugging Face cae al set curado. Y las versiones están fijadas en `pyproject.toml` y
`uv.lock`."*

## Cambio de código en vivo (si te lo piden — §9 del examen)

| Te piden... | Qué haces |
|---|---|
| Mostrar top-5 en vez de top-1 | `recall_at_k(..., ks=(1,5,10))` ya lo calcula; o el `faiss_demo` devuelve top-5. |
| Calcular similitud coseno | Señala `cosine_similarity_matrix` en `src/metrics.py`. |
| Normalizar embeddings | Señala `encode_images`/`encode_texts`: ya hacen `emb / emb.norm(...)`. |
| Cambiar una consulta | Edita `faiss_query` en `configs/experiment.yaml` y re-corre. |
| Añadir Recall@K | Ya está en `recall_at_k`; cambia el `recall_ks` del config. |
| Comparar dos checkpoints | Están en `configs/checkpoints.yaml` → `checkpoint_comparison.csv`. |
| Separar correctos/incorrectos | `failure_cases` en `src/error_analysis.py` (group=0). |
| Un hard negative | *"En cada par mínimo, la imagen o caption incorrecta ES el hard negative."* |
| Añadir un par nuevo | Agrega una línea a `data/curated/examples.jsonl` y re-corre `02_run_winoground.py`. |

## Comandos que conviene tener a mano
```bash
make test                                    # 16 tests verdes
python scripts/validate_against_official.py  # reproduce los scores oficiales
python scripts/02_run_winoground.py          # regenera outputs/metrics
```

## Si algo no corre en vivo
No dependas de internet ni de descargas. Ten `make setup` y `make run` ya ejecutados
antes. En vivo solo corre `pytest` y `validate_against_official.py`, que tardan segundos.
