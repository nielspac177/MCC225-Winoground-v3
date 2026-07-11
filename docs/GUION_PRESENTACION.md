# Guion de presentación — 15 minutos (25 diapositivas)

Defensa: Winoground / Evaluating CLIP · Niels V. Pacheco Barrios · Cuadernos C5, C8, C10.
Tiempo total ~15 min: ~11 min de exposición + ~3 min de demo en vivo + margen.

> Lo que **dices** está en *cursiva*. Habla en primera persona, calmado, mirando a cámara.

---

## Bloque 0 — Apertura (0:00–1:00)

**Slide 1 (Portada).** *"Mi tema es Winoground y Evaluating CLIP. Voy a defender una idea: que un modelo recupere bien una imagen no prueba que entienda la escena. Lo medí reutilizando el Cuaderno 10."*

**Slide 2 (Agenda).** *"En 15 minutos: el problema, cómo lo medí, el resultado, una verificación en vivo, y por qué pasa más limitaciones."* (10 s, no te detengas.)

## Bloque 1 — El problema (1:00–4:00)

**Slide 3 (Dos papers).** *"Winoground pregunta si estos modelos razonan de forma composicional o solo reconocen objetos. Evaluating CLIP recuerda que una accuracy alta no significa un modelo mejor. Los dos dicen lo mismo: la métrica puede engañar."*

**Slide 4 (Par mínimo).** *"El corazón de Winoground es el par mínimo: dos imágenes y dos frases con las mismas palabras en distinto orden. 'An old person kisses a young person' contra 'a young person kisses an old person'. Para acertar hay que saber quién besa a quién, no solo detectar 'old', 'young', 'kiss'."*

**Slide 5 (Tres métricas).** *"Para cada ejemplo armo una matriz dos por dos de similitud. El text score fija la imagen y pregunta si gana el caption correcto; el image score fija el caption y pregunta si gana la imagen; el group score pide las dos a la vez."*

**Slide 6 (Azar 1/6).** *"Un detalle que un examinador puede preguntar: el azar del group no es un dieciseisavo, es un sexto. Text e image no son independientes, comparten las cuatro similitudes. De 24 ordenaciones, solo 4 dan group correcto."*

**Slide 7 (La pregunta).** *"Entonces, la pregunta concreta: ¿un buen Recall@K implica composición? Voy a mostrar que no."*

## Bloque 2 — Cómo lo medí, Cuaderno 10 (4:00–6:30)

**Slide 8 (Cómo funciona CLIP).** *"CLIP es un dual-encoder: dos torres, una para imagen y otra para texto, cada una colapsa en un vector y se comparan por coseno. Es exactamente el motor del Cuaderno 10, que reutilicé tal cual."*

**Slide 9 (Por qué falla).** *"El problema es que un vector global se comporta como bolsa de conceptos: no codifica quién está con quién. Aquí entran C5 y C8: la fusión profunda y la atención crossmodal sí dejan interactuar los tokens."*

**Slide 10 (Pipeline).** *"Mi pipeline: cargo los datos, codifico con OpenCLIP, aplico el scorer, y mido Recall@K, error por tag y una prueba de ceguera. Todo determinista y con tests."*

**Slide 11 (Scorer, código).** *"Este es el bloque que produce el resultado. Tres funciones cortas. Importante: seguí el código oficial de Winoground, no la redacción de los papers, que suena al revés."*

**Slide 12 (Validación).** *"Y para probar que mi scorer está bien, lo apliqué a los scores oficiales de CLIP que vienen en el dataset. Reproduce exactamente 0.3075, 0.105 y 0.08. Coincidencia perfecta."* (Esta slide te blinda; dilo con seguridad.)

## Bloque 3 — Resultados (6:30–10:00)

**Slide 13 (Resultado principal).** *"Este es el resultado con 400 ejemplos reales. Text 0.35, por encima del azar. Pero group 0.075, por debajo del azar de 0.167 y lejísimos del humano, que es 0.86."*

**Slide 14 (Cómo leerlo).** *"Ojo con interpretarlo: que el group esté bajo el azar no es 'peor que aleatorio'. Es la asimetría: acierta una dirección y falla la otra, y el group exige ganar las dos. El intervalo de confianza queda entero bajo el azar."*

**Slide 15 (Retrieval vs group).** *"Y esta es la evidencia de mi tesis. Mismo modelo, mismo conjunto: Recall@5 de 0.67 y Recall@10 de 0.77, pero group de 0.075. El retrieval usa toda la galería, que es fácil; el group exige ganar el par mínimo."*

**Slide 16 (Por tag).** *"¿Qué tipo de error? Lo separé por tag. Relation es lo más difícil. Son errores de vinculación, de quién con quién, no de reconocer objetos."*

**Slide 17 (Ceguera).** *"¿Y usa la imagen o adivina por el texto? Permuté las imágenes entre ejemplos: los scores caen a casi cero. Sí usa la imagen; su límite es composicional, no perceptivo. Es la versión cuantitativa de la interpretabilidad de C8."*

**Slide 18 (Checkpoints).** *"Comparé tres checkpoints. Un modelo más grande no arregla la composición: el group no sube de forma clara y el text incluso baja."*

**Slide 19 (Ejemplos).** *"Y aquí pares concretos que el modelo confunde: reconoce los objetos, pero no su composición."*

## Bloque 4 — Verificación en vivo (10:00–13:00)

**Slide 20 (Reproducibilidad).** *"Todo es reproducible: entorno fijo con uv y lock, Docker, 16 tests y CI. El dataset gated no lo redistribuyo, por licencia."* (30 s, luego cambias a la pantalla del notebook.)

**Slide 21 (Demo).** Cambia a la terminal/notebook y haz, en este orden:
1. Notebook, **celda 1**: *"Aquí imprimo la hoja de trazabilidad: versiones, git, dispositivo."*
2. **Celda del scorer**: *"Tomo un par, armo la matriz dos por dos, y aquí salen text, image y group."*
3. Terminal: `python scripts/validate_against_official.py` → *"Y esto confirma que mi scorer reproduce los números oficiales."*
4. (Si piden mejora en vivo) cambia `faiss_query` en `configs/experiment.yaml` o añade un par y re-ejecuta.

## Bloque 5 — Cierre (13:00–15:00)

**Slide 22 (Limitaciones).** *"Soy crítico con mi propio resultado: el group bajo mezcla fallo composicional con ítems ambiguos del benchmark. Las métricas son sensibles a empates. Y mi set curado offline es un proxy sintético."*

**Slide 23 (Mejora).** *"La mejora natural es evaluar un cross-encoder o fusión profunda, que es C5, con re-ranking de C6. Eso sí podría componer. Es mi hipótesis, todavía no un resultado medido."*

**Slide 24 (Conclusión).** *"En resumen: usé C10 como motor, C5 y C8 para explicar el porqué, y demostré que el retrieval alto no implica razonamiento composicional. Lo medí, lo validé y lo dejé reproducible."*

**Slide 25 (Gracias).** *"Gracias. Quedo atento a sus preguntas."*

---

## Si el tiempo aprieta (versión 12 min)
Salta las slides 6 (azar 1/6), 14 (cómo leerlo) y 18 (checkpoints); menciónalas solo si preguntan. El núcleo irrenunciable: 4, 5, 11, 12, 13, 15, 17, 21.

## Repreguntas frecuentes (ten la respuesta lista)
- *"¿El azar del group?"* → 1/6, porque text e image no son independientes.
- *"¿Cómo sabes que el scorer está bien?"* → reproduce clip.jsonl oficial (slide 12).
- *"¿Está usando la imagen?"* → prueba de ceguera (slide 17).
- *"¿Por qué Relation es peor que Both?"* → en 'Both' las dos imágenes difieren más, hay más señal visual.
- *"¿Un modelo más grande lo arregla?"* → no, slide 18.
