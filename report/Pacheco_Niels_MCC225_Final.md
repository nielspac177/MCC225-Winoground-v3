---
title: "Recuperar no es componer: auditoría del razonamiento visio-lingüístico de CLIP en Winoground"
author: "Niels Victor Pacheco Barrios"
date: "MCC225 — IA Generativa y Aprendizaje Multimodal · Período 2026-1"
lang: es
geometry: margin=2.3cm
fontsize: 10pt
---

**Repositorio:** <https://github.com/nielspac177/MCC225-Winoground-v3> · tag `final-mcc225`

---

## 1. Problema, modalidades y pregunta experimental

Los modelos visión-lenguaje contrastivos se evalúan casi siempre mediante métricas de recuperación como Recall@K. Esas métricas resumen la capacidad del modelo para encontrar el elemento correcto dentro de una galería grande y heterogénea, una tarea que puede resolverse reconociendo el tema de la escena. El riesgo metodológico es que un valor alto de Recall@K se interprete como evidencia de comprensión del lenguaje cuando en realidad solo acredita reconocimiento de vocabulario.

El problema es genuinamente multimodal porque ninguna de las dos modalidades basta por separado. La modalidad visual aporta la identidad de los objetos, sus atributos y su disposición espacial; la modalidad textual aporta la estructura predicado-argumento que asigna roles a esos objetos. Un sistema que solo procese imágenes no puede saber quién besa a quién, y uno que solo procese texto no puede verificar si la escena descrita es la que aparece. La composicionalidad vive precisamente en la intersección: exige ligar cada entidad visual con su rol sintáctico.

Winoground (Thrush et al., CVPR 2022) aísla esa capacidad mediante un diseño de par mínimo. Cada uno de sus 400 ejemplos contiene dos imágenes y dos captions que comparten **exactamente el mismo conjunto de palabras en distinto orden**. Bajo esa construcción, cualquier modelo que trate la oración como una bolsa de palabras asigna necesariamente la misma puntuación a ambos captions, de modo que el atajo léxico no solo es insuficiente: es inútil por construcción. Lo único que discrimina es el orden, es decir, la composición.

La pregunta experimental de este trabajo es, por tanto, directa y falsable: **¿el buen desempeño de CLIP en recuperación implica razonamiento composicional?** Si lo implicara, un modelo con Recall@5 en torno a 0.70 debería resolver los pares mínimos muy por encima del nivel de azar. La unidad de análisis es el par mínimo completo, no la imagen ni el caption aislados, y la salida esperada es una decisión binaria por ejemplo bajo cada una de las métricas oficiales.

## 2. Línea base, modelo principal y diseño experimental

El modelo principal es OpenCLIP ViT-B/32 preentrenado en LAION-2B, un dual-encoder que proyecta imagen y texto por separado a un espacio común y los compara por similitud coseno. Es la arquitectura canónica del paradigma contrastivo, y su principal restricción es justamente lo que la vuelve útil aquí: al no existir interacción token a token entre modalidades, cualquier fallo composicional es atribuible a una representación global que no puede codificar roles.

La línea base de comparación mínima razonable es el **nivel de azar analítico** de cada métrica, y no un modelo alternativo. En una tarea de emparejamiento con dos alternativas, el azar no es un artefacto empírico sino una cantidad derivable, y compararse contra él responde exactamente a la pregunta planteada: si el modelo no supera el azar, no está usando ninguna información útil para la tarea, con independencia de cuán bien recupere en otros contextos. Como control adicional se incluye un baseline de captions desplazados, que confirma que el protocolo no genera aciertos espurios.

El diseño experimental compara tres condiciones sobre los mismos 400 ejemplos. La primera contrasta **recuperación frente a composición**, midiendo Recall@K sobre la galería completa de 800 imágenes y 800 captions, y enfrentando ese resultado al group score sobre los mismos datos. La segunda es una **prueba de ceguera** que permuta aleatoriamente las imágenes entre ejemplos para descartar que los aciertos provengan de pistas no visuales. La tercera es una **auditoría de la métrica**, que evalúa las mismas similitudes bajo el criterio oficial y bajo un criterio alternativo de emparejamiento, y que constituye la contribución metodológica principal de este informe.

## 3. Dataset, particiones, métricas y configuración reproducible

El dataset es el benchmark oficial `facebook/winoground` en su revisión `b400e173`, distribuido como un parquet de 367 MB con las imágenes embebidas. El acceso está restringido: exige aceptar un acuerdo de licencia que limita el uso a investigación no comercial. Las imágenes proceden de Getty Images y por ello **no se redistribuyen** en el repositorio; solo se versionan los embeddings derivados y las matrices de similitud, que no permiten reconstruir las imágenes originales. La procedencia completa y las restricciones están documentadas en `data/MANIFIESTO.md`.

No hay particiones de entrenamiento, validación y prueba porque no se entrena ni se ajusta ningún modelo: la evaluación es enteramente *zero-shot* sobre los 400 ejemplos, que constituyen la totalidad del benchmark. Esta ausencia de particiones elimina cualquier posibilidad de fuga de información entre conjuntos, pero introduce una limitación distinta que se discute en la sección 8: al ser 400 ejemplos la población completa y no una muestra, la inferencia hacia "pares mínimos composicionales en general" descansa en un supuesto de intercambiabilidad que el benchmark no garantiza.

Las métricas son las tres oficiales, definidas sobre la matriz de similitud $\mathrm{sim}[c][i]$ de cada ejemplo. El **text score** fija la imagen y exige elegir el caption correcto en ambas direcciones; el **image score** fija el caption y exige elegir la imagen; el **group score** exige ambas condiciones simultáneamente. Sus niveles de azar son 0.25, 0.25 y 1/6 respectivamente. Conviene detenerse en el último: no es 1/16, porque text e image se calculan sobre las mismas cuatro similitudes y por tanto no son independientes. De las 4! = 24 ordenaciones equiprobables de cuatro valores distintos, solo cuatro sitúan ambas diagonales por encima de ambas antidiagonales, de donde 4/24 = 1/6.

A estas tres se añade **GroupMatch** (Zhu et al., ICLR 2026), que pregunta si el emparejamiento correcto maximiza la similitud total, es decir, si $s_{00} + s_{11} > s_{01} + s_{10}$. Su azar es 1/2, por lo que sus valores **no son comparables** con los del group score y siempre se reportan junto a su nivel de azar.

La configuración es reproducible mediante semilla fija en 42, 2000 rondas de bootstrap, y un registro completo de versiones, hardware y hash de commit en `results/configuracion_experimental.json`. La suite de pruebas comprende 50 tests, y el scorer se valida externamente contra `statistics/model_scores/clip.jsonl`, el archivo de puntuaciones por ejemplo que publica el propio dataset.

## 4. Resultados principales

La Tabla 1 recoge el resultado central. Bajo el criterio oficial, CLIP ViT-B/32 obtiene un text score de 0.3475, que supera el azar, pero un image score de 0.1100 y un group score de 0.0750, ambos **por debajo** del azar correspondiente y a enorme distancia del acuerdo humano de 0.855. Ninguno de los tres checkpoints evaluados difiere significativamente de los otros: el test exacto de McNemar sobre pares discordantes arroja valores de p de 1.00, 0.61 y 0.52.

**Tabla 1.** Resultados sobre los 400 pares oficiales, con intervalos bootstrap de percentil (2000 rondas). El asterisco marca los casos en que el límite inferior del intervalo supera el azar.

| Modelo | text | image | group | GroupMatch |
|---|---|---|---|---|
| CLIP ViT-B/32 (laion2b) | 0.3475 [0.300, 0.398]\* | 0.1100 [0.080, 0.142] | 0.0750 [0.050, 0.102] | 0.6750 [0.630, 0.720]\* |
| CLIP ViT-B/16 (datacomp_xl) | 0.2975 [0.253, 0.343]\* | 0.0875 [0.060, 0.115] | 0.0725 [0.050, 0.098] | 0.6425 [0.595, 0.688]\* |
| CLIP ViT-L/14 (openai) | 0.2875 [0.245, 0.333] | 0.1100 [0.083, 0.142] | 0.0850 [0.060, 0.113] | 0.6650 [0.618, 0.710]\* |
| *Azar* | *0.2500* | *0.2500* | *0.1667* | *0.5000* |
| *Humano (Thrush et al.)* | *0.8950* | *0.8850* | *0.8550* | *no recomputable* |

El contraste entre recuperación y composición es nítido. Sobre la misma galería de 800 elementos, CLIP alcanza un Recall@5 de 0.701 en dirección imagen a texto y 0.668 en dirección texto a imagen, mientras que su group score es de 0.075. La distancia entre ambas cifras, casi un orden de magnitud, responde negativamente a la pregunta experimental: **el buen desempeño en recuperación no implica razonamiento composicional**. La prueba de ceguera confirma que ese 0.075 no es un artefacto del protocolo: al permutar las imágenes entre ejemplos el group score cae a 0.015 y el text score de 0.3475 a 0.135, de modo que el modelo sí utiliza el contenido visual, aunque lo utilice mal.

El desglose por tag localiza el fallo. Los ejemplos etiquetados como `Relation`, que son 233 de los 400, obtienen un group score de 0.047, la mitad que los 141 ejemplos de tipo `Object`, con 0.085. Es decir, la mayoría del benchmark corresponde precisamente al tipo de ejemplo que peor se resuelve.

![**Figura 1.** Scores con intervalo de confianza del 95 % frente al azar y al acuerdo humano. Cada panel tiene su propio nivel de azar; el eje vertical es compartido para que la comparación entre paneles sea honesta, a costa de comprimir los paneles B y C. El panel D omite deliberadamente la referencia humana: el acuerdo humano se midió bajo el criterio oficial y no es recomputable bajo GroupMatch, porque el paper original publicó tasas de acierto y no las cuatro similitudes por ejemplo.](../outputs/figures/informe_fig1_metricas.png)

El hallazgo que más matiza la conclusión aparece en el panel D. Las mismas cuatro similitudes que producen un group score de 0.075 producen un GroupMatch de 0.675, claramente por encima de su azar de 0.5. La discrepancia no es marginal: **240 de los 400 ejemplos fallan el group score pero aciertan GroupMatch**. La explicación es que el group score exige que cada elemento diagonal sea el máximo de su fila *y* de su columna —cuatro comparaciones simultáneas—, mientras que GroupMatch solo pregunta si la asignación correcta es la mejor globalmente. Son preguntas distintas, y la brecha entre ambas cuantifica cuánto de lo que se reporta como fallo composicional es en realidad exigencia del criterio de medida.

![**Figura 2.** Panel A: recuperación sobre la galería completa frente a composición sobre pares mínimos, con los mismos datos y el mismo modelo. La línea de azar se dibuja solo sobre la barra del group score, porque el azar de Recall@K sobre una galería de 800 es de orden 1/800 y no 1/6. Panel B: distribución del margen que decide cada una de las 1600 comparaciones. No hay ningún empate exacto, pero la mediana del margen es de 0.0154 frente a una desviación típica de las similitudes de 0.0583.](../outputs/figures/informe_fig2_tesis.png)

El panel B de la Figura 2 aporta el segundo matiz. El scorer oficial decide con desigualdad estricta, lo que contabiliza cualquier empate exacto como fallo sin dejar rastro. La medición muestra que ese riesgo no se materializa —hay **cero empates exactos** en los tres checkpoints— pero también que las decisiones no son holgadas: la mitad de las comparaciones se resuelve por un margen inferior a 0.0154 sobre similitudes que recorren el rango [0.027, 0.430]. Introducir una tolerancia de apenas 0.001 convierte 43 decisiones de text en indeterminadas y hace caer el score de 0.3475 a 0.2925. La convención de desigualdad estricta resulta entonces empíricamente inocua, pero el score es frágil ante cualquier cambio de preprocesado que altere la tercera cifra decimal.

## 5. Dos aciertos, dos errores y un caso ambiguo

Los dos aciertos de mayor margen muestran algo que la media no deja ver: que **CLIP acierta precisamente cuando la tarea no exige componer**. En el ejemplo 364, *"some are parking in a train"* frente a *"some are training in a park"*, el intercambio de orden convierte un tren en un parque: cambia qué objetos aparecen en la escena, no cómo se relacionan. La similitud diagonal alcanza 0.2855 y 0.2867 frente a antidiagonales de 0.0268 y 0.1421, un margen de 0.4033. El ejemplo 133, *"there is a split banana"* frente a *"there is a banana split"*, opera igual: un plátano partido y un postre son objetos distintos. En ambos casos el reconocimiento de objetos basta, y por eso el modelo destaca.

Los dos errores exhiben el patrón inverso. En el ejemplo 380, *"two kids on a pool floatie and one kid swimming"* frente a *"one kid on a pool floatie and two kids swimming"*, los objetos son idénticos —niños, flotador, agua— y solo cambian las cantidades asignadas a cada rol. El modelo falla text e image con un margen negativo de −0.0556. En el ejemplo 98, *"someone on the ground is spraying water towards a vehicle"* frente a *"someone is on a vehicle spraying water towards the ground"*, los elementos vuelven a coincidir y solo se invierte la relación espacial entre persona, vehículo y suelo. Ambos pertenecen a la categoría donde el reconocimiento de vocabulario no aporta nada.

El caso ambiguo es el ejemplo 288, cuya matriz de similitud vale [[0.1093, 0.2155], [0.1181, 0.2244]]. El modelo falla el group score, pero acierta GroupMatch por un margen de +0.0001, esencialmente el lanzamiento de una moneda. La inspección de la matriz revela algo más interesante que la propia indeterminación: **la imagen 1 puntúa más alto que la imagen 0 para los dos captions**, lo que indica que la decisión está dominada por una preferencia global por una de las imágenes y no por ningún juicio composicional. Este caso ilustra por qué un acierto bajo GroupMatch no debe leerse automáticamente como comprensión, y por qué reportar una sola métrica —cualquiera de las dos— produciría una conclusión engañosa.

## 6. Comparación controlada

Se realizaron dos comparaciones controladas. La primera contrasta el mismo modelo bajo **dos criterios de medida** manteniendo fijas las similitudes, el modelo y los datos; su resultado es la brecha de 0.075 frente a 0.675 ya descrita. Es una ablación de la métrica, no del modelo, y aísla limpiamente la contribución del criterio de evaluación al resultado publicado.

La segunda compara **tres checkpoints** que varían en tamaño y datos de preentrenamiento. Su resultado principal es negativo y conviene enunciarlo con precisión: ninguna diferencia entre ellos es estadísticamente distinguible. Un análisis de potencia sobre el diseño pareado indica que, con n = 400, $\alpha = 0.05$ y potencia 0.80, la **diferencia mínima detectable en group score es de 5.3 puntos porcentuales**, mientras que la mayor diferencia observada entre checkpoints es de 1.25 puntos. Es decir, aunque las variables estuvieran desconfundidas, el benchmark carece de potencia para resolverlas. Este resultado tiene una implicación que excede al presente trabajo: buena parte de la literatura que ordena modelos en Winoground por diferencias de uno o dos puntos está reportando ruido de muestreo.

Estaba prevista una tercera comparación, más ambiciosa y directamente pertinente al mecanismo, que no pudo completarse y se declara como tal en la sección siguiente.

## 7. Limitaciones y amenazas a la validez

**La afirmación mecanística no está demostrada.** La evidencia reunida es correlacional: se observa que un modelo sin atención cruzada falla la composición. Pero CLIP difiere de un cross-encoder en arquitectura, objetivo de entrenamiento, datos y escala simultáneamente, y cualquiera de esos factores podría ser la causa. Sostener que el fallo se debe *a* la ausencia de atención cruzada exigiría variar solo esa dimensión. El experimento está implementado en `src/blip_utils.py` mediante `BlipForImageTextRetrieval`, que expone dos cabezas sobre el mismo cuerpo —similitud coseno entre proyecciones separadas frente a atención cruzada real entre tokens de texto y parches de imagen—, de modo que mismos pesos, mismo preprocesado y mismos datos dejan la atención cruzada como única variable. No pudo ejecutarse dentro del plazo por una limitación del entorno de cómputo, documentada en `results/configuracion_experimental.json`. Hasta cerrarlo, la conclusión sobre el mecanismo permanece como hipótesis.

**Potencia estadística.** Con 400 ejemplos, diferencias menores a 5.3 puntos en group score no son resolubles. Esto no compromete la conclusión principal, cuyo efecto es de casi un orden de magnitud, pero sí invalida cualquier ranking entre checkpoints.

**El benchmark no mide únicamente composición.** Diwan et al. (EMNLP 2022) reanotaron los 400 ítems y encontraron que 38 requieren detectar rasgos visuales sutiles, 56 contienen imágenes inusuales, 50 captions difíciles de analizar sintácticamente y 46 son ambiguos. Una fracción del fallo observado es ruido del dataset y no incapacidad del modelo. Filtrar esos ítems y re-medir es la extensión natural de este trabajo.

**Variables confundidas.** Los tres checkpoints varían simultáneamente en tamaño y corpus de preentrenamiento, por lo que ninguna diferencia sería atribuible a un factor concreto aunque fuera significativa.

**Validez externa.** Los resultados se refieren a 400 pares en inglés de dominio fotográfico. No autorizan a extrapolar a otros idiomas, y en particular no dicen nada sobre lenguas con orden de palabras más flexible o marcado morfológico explícito de roles, donde la tarea podría ser estructuralmente distinta.

**Fragilidad numérica.** Como muestra el panel B de la Figura 2, las decisiones descansan sobre márgenes diminutos, de modo que dos implementaciones que difieran en el redimensionado de imágenes podrían producir scores apreciablemente distintos sin que ninguna esté equivocada.

Como **trabajo futuro**, la prioridad es cerrar la ablación del cross-encoder, que es la única que convierte la hipótesis en experimento. En segundo lugar, aplicar el filtro de ambigüedad de Diwan et al. para separar el fallo del modelo del ruido del benchmark. En tercer lugar, y como línea propia, caracterizar qué fracción de los 400 pares mínimos admite un equivalente en español: la construcción "mismas palabras, distinto orden" no se preserva bajo traducción, y determinar cuándo sí lo hace es en sí mismo un resultado sobre la relación entre orden de palabras y composicionalidad.

## 8. Relación con los temas del curso

El proyecto articula seis bloques del programa. De **alineamiento** proviene el motor experimental: aprendizaje contrastivo, dual encoders, OpenCLIP, evaluación *zero-shot*, retrieval cruzado y las métricas Recall@K y MRR, todos ellos trabajados en los Cuadernos 9 y 10 y reutilizados aquí como línea base de la que el proyecto se separa deliberadamente. El bloque de **fundamentos** aporta la distinción entre fusión temprana, intermedia y tardía, que es el eje sobre el que se plantea la hipótesis mecanística: CLIP es fusión tardía extrema, con un único punto de contacto entre modalidades al final del cómputo.

En **Transformers y VLM** aparece la distinción entre self-attention y cross-attention, que aquí deja de ser una definición para convertirse en la variable independiente del experimento diseñado con BLIP. El bloque de **evaluación** aporta el propio Winoground, el análisis por tags, el análisis de error, la robustez y la confiabilidad; el informe extiende ese bloque al incorporar la discusión sobre la métrica misma como objeto de análisis y no solo como instrumento. La **ingeniería experimental** del curso se refleja en las semillas fijas, el registro de metadatos, Docker, la integración continua, los tests y las salidas auditables. Finalmente, el bloque de **evaluación responsable** se refleja en la decisión de no redistribuir las imágenes, en la declaración explícita del carácter *gated* del dataset y en el criterio, sostenido a lo largo del informe, de acotar cada afirmación a lo que la evidencia disponible sostiene.

---

### Referencias

Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela, D. y Ross, C. (2022). *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*. CVPR 2022. arXiv:2204.03162

Diwan, A., Berry, L., Choi, E., Harwath, D. y Mahowald, K. (2022). *Why is Winoground Hard? Investigating Failures in Visuolinguistic Compositionality*. EMNLP 2022. arXiv:2211.00768

Zhu, Y., Zhang, J. y Tang, F. (2025). *Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models*. ICLR 2026. arXiv:2510.07632

Li, J., Li, D., Xiong, C. y Hoi, S. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. ICML 2022. arXiv:2201.12086

Cherti, M. et al. (2023). *Reproducible scaling laws for contrastive language-image learning* (OpenCLIP). CVPR 2023. arXiv:2212.07143
