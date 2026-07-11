# Evaluación responsable del proyecto multimodal

## Evaluación responsable de CLIP en Winoground

**Curso:** MCC225 — IA Generativa y Multimodal · **Actividad complementaria 5**
**Tesis del proyecto:** un desempeño alto en recuperación (retrieval) no equivale a razonamiento composicional.
**Modelo evaluado:** dual-encoder CLIP ViT-B/32 · **Datos:** pares reales de Winoground · **Hardware:** Apple Silicon (mps).

> Nota metodológica. Todos los valores de este reporte se leen directamente de la evidencia del repositorio (`outputs/metrics/*.json`, `outputs/tables/*.csv`) generada por el `Cuaderno14_MCC225_resuelto.ipynb` y por el pipeline del proyecto integrador. No se reportan cifras estimadas ni ilustrativas.

---

## Parte 1. Proyecto personal y tarea multimodal

Mi proyecto integrador estudia si un dual-encoder tipo CLIP, que alcanza métricas altas de recuperación imagen-texto, *comprende* realmente la composición de una escena (quién hace qué a quién, qué está sobre qué, cuántos hay). La actividad evalúa responsablemente ese componente multimodal.

| Elemento | Respuesta |
| --- | --- |
| Nombre del proyecto | Evaluación responsable de CLIP en Winoground |
| Problema que aborda | Determinar si el retrieval alto de CLIP implica comprensión composicional, o solo detección de conceptos presentes |
| Dominio de aplicación | Visión-lenguaje / razonamiento composicional imagen-texto |
| Modalidades usadas | Texto + imagen |
| Tarea multimodal evaluada | Emparejamiento / recuperación composicional de pares mínimos (retrieval + contraste Winoground) |
| Usuario previsto | Desarrollador/a de sistemas de búsqueda o moderación imagen-texto que necesita saber hasta dónde puede confiar en CLIP |
| Riesgo principal del sistema | Sobre-confiar en el retrieval (o en el CLIPScore) como si fuera comprensión de la escena |

**¿Qué parte del proyecto se puede evaluar con evidencia en esta etapa?** La capa de emparejamiento imagen-texto: recuperación (Recall@K), CLIPScore por par, la métrica composicional de Winoground (text/image/group score) y un análisis de errores sobre pares mínimos reales. Es un componente acotado, reproducible y directamente ligado a la tesis del proyecto.

---

## Parte 2. Adaptación del Cuaderno 14

El `Cuaderno14_MCC225.ipynb` es una plantilla de evaluación multimodal cuya configuración por defecto descarga un dataset genérico (`jxie/flickr8k`). Siguiendo el [ADR 0002](docs/adr/0002-cuaderno14-sobre-winoground.md), se adaptó para correr sobre **los datos reales del proyecto** (Winoground) de forma offline.

| Elemento | Respuesta |
| --- | --- |
| Modelo usado | CLIP ViT-B/32 (dual-encoder). El pipeline integrador usa además el checkpoint ViT-B-32/laion2b para el group score de Winoground |
| Dataset o subconjunto | 120 pares imagen-texto reales tomados del caché de Winoground (`data/raw/manifest_local.csv`); el integrador evalúa los 400 pares completos |
| Número de ejemplos | 120 pares locales (cuaderno) y 400 pares (integrador). Dentro del rango 30–100+ que exige la actividad |
| Métricas usadas | Recall@{1,5,10} (i2t y t2i), CLIPScore promedio, text/image/group score de Winoground, BLEU/ROUGE-L/cobertura léxica (captioning), y accuracy/precision/recall de CapFilt |
| Baseline usado | Retrieval con captions desplazados (permutados); azar de Winoground (text/image 0.25, group 0.167); prueba de ceguera con imágenes permutadas |
| Hardware o entorno | macOS Apple Silicon, dispositivo `mps`; Python 3.12, PyTorch 2.5.1, open_clip 3.3.0 |
| Parte del Cuaderno 14 reutilizada | Etapas de retrieval, CLIPScore, captioning BLIP, degradación visual, perturbación textual, CapFilt y análisis de errores (intactas) |
| Parte modificada para el proyecto | La celda de carga de datos: detecta el manifiesto local y omite la descarga del dataset genérico |

**¿Qué cambio fue necesario?** Sustituir el origen de datos genérico por el caché real de Winoground mediante `scripts/build_local_manifest.py`, de modo que las mismas imágenes composicionales del proyecto aparecen en el retrieval, las ablaciones y el análisis de errores. La rúbrica penaliza desconectar el cuaderno del proyecto personal; esta adaptación los conecta.

---

## Parte 3. Resultados cuantitativos

Tabla mínima con valores reales (extracto; la tabla completa está en `results/metricas.csv`):

| Experimento | Modelo | Datos | Métrica | Resultado | Interpretación breve |
| --- | --- | --- | ---: | ---: | --- |
| E1 (retrieval) | CLIP ViT-B/32 | 100 pares (de 120) | i2t R@1 / R@5 / R@10 | 0.44 / 0.87 / 0.97 | Recuperación sólida sobre toda la galería (tarea fácil) |
| E1 (retrieval t2i) | CLIP ViT-B/32 | 100 pares (de 120) | t2i R@1 / R@5 / R@10 | 0.39 / 0.82 / 0.95 | Consistente en ambas direcciones |
| E1 (baseline) | Captions desplazados | 100 pares (de 120) | i2t / t2i R@1 | 0.01 / 0.00 | Baseline ≈ azar: valida el protocolo |
| E2 (ablación visual) | CLIP ViT-B/32 | 80 pares degradados | CLIPScore orig / grises / recorte | 0.270 / 0.259 / 0.262 | Robusto a pérdida de color y recorte |
| E3 (Winoground) | CLIP ViT-B/32 laion2b | 400 pares | text / image / group | 0.348 / 0.110 / 0.075 | Group << azar (0.167) y << humano (0.855) |
| E3 (CapFilt) | CLIP (filtro CLIPScore) | 72 captions | accuracy | 0.903 | Filtrar captions buenos/malos es tarea fácil bien resuelta |

**Pregunta 1 — ¿Qué métrica representa mejor el objetivo de la tarea?** El **group score de Winoground** (0.075). Es la única que exige ganar el contraste de pares mínimos, es decir, resolver *simultáneamente* la dirección imagen→texto y texto→imagen. Mide composición, que es exactamente el objeto de estudio del proyecto.

**Pregunta 2 — ¿Qué métrica puede ser insuficiente si se interpreta sola?** El **Recall@K** y el **CLIPScore**. Con toda la galería, R@5 alcanza 0.87 (i2t), pero eso solo indica que el caption correcto está entre los cinco más cercanos: no distingue el par mínimo. De hecho `recall_vs_group.json` muestra R@5=0.667 y R@10=0.774 en la galería de 800, cifras altas que conviven con group=0.075. Además, **BLEU (0.073) y ROUGE-L (0.191)** son engañosamente bajos: penalizan captions correctos con vocabulario distinto al de referencia; un caption acertado puede puntuar 0.0 (p. ej. wg_1_0). Interpretadas solas, estas métricas sobreestiman (Recall/CLIPScore) o subestiman (BLEU/ROUGE) el desempeño real.

**Pregunta 3 — ¿Supera claramente al baseline o solo muestra funcionamiento parcial?** Ambas cosas, según la tarea. En **retrieval** supera nítidamente al baseline de captions desplazados (0.44 vs 0.01 en i2t R@1). En **composición** *no* supera al azar: group score 0.075 está por debajo del azar 0.167. Es un funcionamiento parcial: sabe *qué* hay en la escena, no *cómo* se relaciona.

---

## Parte 4. Análisis de cinco casos

Todos los casos provienen de la evaluación real (`outputs/metrics/failure_cases.json`, con su matriz de similitud 2×2, y `outputs/tables/*.csv` con el CLIPScore). Ninguno es inventado. Figura de apoyo: `figures/ejemplos_evaluados.png`. Tabla completa en `results/casos_analizados.csv`.

| Caso | Entrada | Salida del modelo | Resultado esperado | Tipo | Explicación breve |
| --- | --- | --- | --- | --- | --- |
| 1 | wg_2_0 (imagen_00004) "the masked wrestler hits the unmasked wrestler" | Empareja bien ambos captions con su imagen en text-score (`text_ok=1`); CLIPScore diagonal 0.324 (el más alto) | Asignar cada caption a su imagen | acierto | La máscara es una señal visual distintiva; el reconocimiento de objetos basta |
| 2 | wg_5_0 (imagen_00010) "a bird eats a snake" | Elige la imagen correcta para el caption en image-score (`image_ok=1`); CLIPScore 0.299 | Elegir la imagen dado el caption | acierto | 'bird' y 'snake' son objetos salientes y separables |
| 3 | wg_0 (imagen_00000) "an old person kisses a young person" ↔ "a young person kisses an old person" | Falla text+image (group=0). Matriz [[0.343, 0.329],[0.325, 0.320]]: la diagonal no es máxima | Distinguir quién besa a quién | error | Razonamiento relacional: codifica "dos personas besándose", no la dirección |
| 4 | wg_3 (imagen_00006) "a person watches an animal" ↔ "an animal watches a person" | Falla text+image. Matriz [[0.184, 0.195],[0.185, 0.248]]; CLIPScore 0.211 | Distinguir el rol agente-paciente | error | Bag-of-concepts sin estructura predicado-argumento |
| 5 | wg_8 (imagen_00016) "a tree smashed into a car" ↔ "a car smashed into a tree" | Resuelve text-score (`text_ok=1`) pero falla image-score. Matriz [[0.333, 0.255],[0.316, 0.275]]; CLIPScore 0.301 | Resolver ambas direcciones | ambiguo | Éxito frágil: depende de la dirección evaluada, no hay razonamiento causal robusto |

**¿Cuál fue el error más importante y por qué sería problemático en un uso real?** El **error de razonamiento relacional/rol agente-paciente** (casos 3 y 4). Es el fallo sistémico que define la tesis: el modelo asigna similitudes casi idénticas a las dos ordenaciones de la misma escena. En un uso real —moderación, búsqueda forense, asistencia médica, subtitulado accesible— confundir "A hace X a B" con "B hace X a A" invierte el significado y puede producir decisiones incorrectas con apariencia de alta confianza (CLIPScore aparentemente razonable).

---

## Parte 5. Confiabilidad

Se eligieron dos pruebas: **A. Sensibilidad al prompt** (Prueba A de la consigna) y **B. Caso difícil** (par mínimo Relation).

**Prueba A — Sensibilidad al prompt.** Usando `outputs/tables/perturbacion_textual.csv` (40 imágenes, 5 variantes cada una), se midió el cambio de CLIPScore al perturbar el caption manteniendo/alterando el significado. El modelo es casi insensible a perturbaciones que *deberían* cambiar el significado:

- **Negación** (anteponer "no"): ΔCLIPScore medio **−0.0083** (baja en 35 de 40 casos, pero de forma mínima). Ejemplo: wg_0_0 pasa de 0.2988 a 0.2903. CLIP no "entiende" la negación.
- **Conteo** (anteponer "dos"): Δ medio **−0.0146**.
- **Atributo** (añadir "de color rojo"): Δ medio **−0.0119**.

| Prueba | Entrada original | Variante | Cambio observado | Interpretación |
| --- | --- | --- | --- | --- |
| Prueba 1 (negación) | "an old person kisses a young person" (0.299) | "no an old person kisses a young person" (0.290) | bajo (Δ≈−0.008) | Insensible a la negación: mala señal para confiabilidad semántica |
| Prueba 2 (caso difícil) | Par mínimo Relation wg_0 (kiss) | Contraparte con orden invertido | alto (falla group; diagonal no máxima) | En pares mínimos el comportamiento es ≈ azar |

**Prueba B — Caso difícil.** Los pares mínimos de la categoría *Relation* son el escenario adverso natural. El desglose por tag (`by_tag.csv`) confirma que *Relation* es la categoría peor: group **0.047** (n=233), frente a *Object* 0.085 y *Both* 0.269. Es decir, cuanto más depende el par de la relación, peor rinde.

**Clasificación final: Parcialmente confiable (confiable solo en condiciones controladas).**

**Justificación.** El sistema es robusto y claramente superior al azar en *recuperación* (R@1 i2t 0.44 vs baseline 0.01) y estable ante degradaciones visuales (CLIPScore cae ≤0.011 al perder color). Pero es casi insensible a perturbaciones textuales que invierten el significado (negación Δ≈−0.008) y su razonamiento composicional cae por debajo del azar (group 0.075 < 0.167). Por tanto es confiable como recuperador o ranker grueso bajo supervisión, y no confiable como decisor de significado composicional. Esa dualidad es precisamente "parcialmente confiable / confiable solo en condiciones controladas".

---

## Parte 6. Explicabilidad

Dos casos explicados con evidencia observable (no racionalización posterior).

| Caso | Evidencia visual o textual | Explicación propuesta | Límite de la explicación |
| --- | --- | --- | --- |
| Caso 1 (prueba de ceguera) | `blindness.json`: group real 0.075; al **permutar las imágenes** cae a 0.015 (text 0.348→0.135, image 0.110→0.035) | El modelo **sí usa el contenido de la imagen**: si se apoyara en pistas no visuales, permutar no cambiaría el resultado. Pero el efecto es débil (0.075 es de por sí bajo) | Prueba que la imagen influye, no que la *comprenda*; no dice *qué* región usa ni por qué falla la composición |
| Caso 2 (error por tag) | `by_tag.csv`: group por categoría — Relation 0.047, Object 0.085, Both 0.269 | El fallo se concentra donde la respuesta depende de la **relación** entre entidades; el modelo resuelve mejor cuando basta reconocer objetos (Both/Object) | Es una correlación agregada por categoría; explica *dónde* falla, no el mecanismo interno en cada par |

**Pregunta 1 — ¿La explicación se basa en evidencia observable o solo racionaliza?** En evidencia observable y cuantificada: una ablación controlada (permutar imágenes) y una estratificación por categoría anotada, ambas con números reales, no una interpretación a posteriori de una salida aislada.

**Pregunta 2 — ¿La explicación ayuda a detectar un error del sistema?** Sí. La prueba de ceguera detecta que el uso de la imagen es débil (riesgo de "ceguera parcial"), y el desglose por tag permite anticipar que cualquier consulta relacional será poco fiable, señalando de antemano dónde hará falta supervisión humana.

---

## Parte 7. Sesgo y uso responsable

| Aspecto | Respuesta breve |
| --- | --- |
| Posible sesgo visual | Entrenado en pares imagen-texto de la web (LAION); hereda su distribución (escenas frecuentes, estética occidental) y rinde peor en escenas inusuales |
| Posible sesgo lingüístico | Encoder de texto en inglés; no evaluado en español, donde el desempeño se degradaría de forma no medida |
| Posible sesgo cultural o de dominio | Fotografía web genérica; no cubre dominios especializados (médico, satelital, documental) ni contextos culturales específicos |
| Riesgo principal si se usa mal | Tratar un CLIPScore alto o un buen Recall@K como comprensión de la escena (group 0.075 << humano 0.855) |
| Supervisión humana necesaria | Obligatoria en decisiones que dependan de relaciones, roles, conteo o negación |
| Uso recomendado | **Limitado** |
| Justificación del uso recomendado | Confiable como recuperador/ranker grueso y filtro de calidad (CapFilt accuracy 0.903), no como razonador composicional; solo en condiciones controladas con verificación humana |

**¿Qué afirmación sería irresponsable hacer sobre este sistema?** Afirmar que **"CLIP entiende la relación espacial de la escena"** (o "comprende quién hace qué a quién"). La evidencia lo contradice: group score 0.075 (bajo el azar 0.167), Relation el tag peor (0.047) e insensibilidad a la negación. Sería confundir recuperación con comprensión —justo el error que el proyecto busca evitar.

---

## Parte 8. Conclusión técnica

Este trabajo evaluó de forma responsable un componente acotado de mi proyecto integrador: el **dual-encoder CLIP ViT-B/32** como sistema de **emparejamiento y recuperación composicional imagen-texto**, usando el `Cuaderno14_MCC225` adaptado (ADR 0002) para correr sobre **datos reales de Winoground** —120 pares locales en el cuaderno y 400 pares en el pipeline integrador—, en Apple Silicon (`mps`). La pregunta guía fue si un retrieval alto equivale a razonamiento composicional; la respuesta, sostenida por la evidencia, es que no.

Los resultados principales son consistentes y, deliberadamente, no seleccionados solo entre éxitos. En **recuperación**, CLIP es sólido: imagen→texto R@1=0.44, R@5=0.87, R@10=0.97; texto→imagen 0.39/0.82/0.95; muy por encima del baseline de captions desplazados (0.01/0.00 en R@1). Es además robusto a degradaciones visuales: el CLIPScore promedio cae de 0.270 (original) a 0.259 (grises) o 0.262 (recorte). Y como filtro de calidad de captions (CapFilt) alcanza accuracy 0.903. Sin embargo, en la métrica que sí exige composición, el **group score de Winoground es 0.075**, por debajo del azar (0.167) y muy lejos del humano (0.855); el image score (0.110) está muy por debajo del azar (0.25). El desglose por categoría muestra que *Relation* es lo peor (0.047).

El **error más importante** es el de razonamiento relacional / rol agente-paciente: ante pares mínimos como "an old person kisses a young person" frente a su inversión, la matriz de similitud es casi simétrica ([[0.343, 0.329],[0.325, 0.320]]) y el modelo no distingue la dirección. Es el fallo que, en un uso real, invertiría el significado de una escena con apariencia de alta confianza.

Sobre **confiabilidad**, la clasificación es *parcialmente confiable / confiable solo en condiciones controladas*. Dos pruebas lo sostienen: la sensibilidad al prompt muestra que negar el caption apenas cambia el CLIPScore (Δ≈−0.008), y la prueba de ceguera (`blindness.json`) muestra que, aunque el modelo sí usa la imagen (group 0.075→0.015 al permutarla), la usa débilmente. El sistema es fiable para *recuperar*, no para *comprender*.

La **limitación que debe comunicarse obligatoriamente** es doble: (1) el tamaño reducido (120/400 pares) prioriza reproducibilidad sobre potencia estadística, por lo que las cifras son indicativas; y (2) un CLIPScore o un Recall@K altos no deben leerse como comprensión composicional —esa es la trampa central que el proyecto documenta.

En cuanto a **condiciones de uso**: el sistema puede emplearse como recuperador o ranker grueso y como filtro de calidad de captions, siempre bajo supervisión humana y en inglés. No debe usarse como decisor autónomo cuando la tarea dependa de relaciones espaciales, roles, conteo o negación. Sería irresponsable afirmar que "CLIP entiende la relación espacial". Ejecutar el modelo no es evaluarlo; evaluarlo no es declararlo confiable; y declarar confiabilidad exige, como aquí, evidencia, límites explícitos y responsabilidad sobre el uso.

---

## Declaración de uso de IA generativa

En la elaboración de este entregable se utilizó un asistente de IA generativa (Claude, Anthropic) como apoyo para **organizar y redactar** el reporte, estructurar las tablas y generar el script de la figura (`scripts/generar_figura_ejemplos.py`). **Todos los valores numéricos** (Recall@K, CLIPScore, text/image/group score, matrices de similitud, deltas de perturbación, CapFilt, ablaciones) fueron **leídos directamente de la evidencia experimental real** del repositorio (`outputs/metrics/*.json`, `outputs/tables/*.csv`) generada por el `Cuaderno14_MCC225_resuelto.ipynb` y el pipeline del proyecto; no fueron inventados ni estimados por el asistente. La interpretación técnica, la selección de casos, la clasificación de confiabilidad y las conclusiones fueron revisadas y son responsabilidad del autor. La figura se generó ejecutando el script sobre imágenes reales de `data/raw/images_local/`.
