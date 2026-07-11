# -*- coding: utf-8 -*-
"""Narración en español, una entrada por página del PDF de slides (30 páginas).

El orden sigue al Beamer (metropolis intercala una página por cada \\section).
Texto pensado para leerse con TTS: números escritos, pausas naturales.
"""

NARRATION = [
    # 1 Portada
    "Hola. Soy Niels Pacheco y esta es mi defensa para el examen parcial del curso "
    "de inteligencia artificial generativa y multimodal. Mi tema es Winoground y "
    "Evaluating CLIP, con los cuadernos cinco, ocho y diez. La idea que voy a "
    "defender es simple: que un modelo recupere bien una imagen no prueba que "
    "entienda la escena.",
    # 2 Agenda
    "En los próximos minutos recorro cinco cosas. El problema. Cómo lo medí "
    "reutilizando el motor OpenCLIP del cuaderno diez. El resultado principal. Una "
    "verificación en vivo. Y por qué ocurre, junto con sus limitaciones y una mejora.",
    # 3 Sección: El problema
    "Primer bloque: el problema.",
    # 4 Dos papers
    "Mi tema combina dos papers. Winoground, de Thrush y colegas en dos mil "
    "veintidós, pregunta si los modelos de visión y lenguaje razonan de forma "
    "composicional, o si solo reconocen objetos sueltos. Evaluating CLIP, de Agarwal "
    "y colegas, advierte que una exactitud más alta no significa un modelo mejor, "
    "porque el diseño de las clases y de los prompts puede esconder sesgos. Los dos "
    "comparten una misma idea: una buena métrica puede ocultar que el modelo, en "
    "realidad, no entiende.",
    # 5 Par mínimo
    "El corazón de Winoground es el par mínimo. Son dos imágenes y dos descripciones "
    "que usan exactamente las mismas palabras, solo que en distinto orden. Por "
    "ejemplo: una persona mayor besa a una persona joven, contra, una persona joven "
    "besa a una persona mayor. El vocabulario es idéntico. Para distinguirlas hay que "
    "entender quién hace qué a quién, es decir, composición, y no solo reconocer las "
    "palabras mayor, joven y besar. El conjunto tiene cuatrocientos ejemplos, "
    "etiquetados según el tipo de cambio: objeto, relación, o ambos.",
    # 6 Tres métricas
    "Para cada ejemplo construyo una matriz dos por dos con la similitud entre cada "
    "descripción y cada imagen. Con ella defino tres métricas. El text score fija la "
    "imagen y pregunta si gana la descripción correcta. El image score fija la "
    "descripción y pregunta si gana la imagen correcta. Y el group score exige "
    "acertar las dos cosas a la vez.",
    # 7 Azar 1/6
    "Un detalle fino que conviene tener claro: el azar del group score no es un "
    "dieciseisavo, es un sexto. La razón es que el text score y el image score no son "
    "independientes; comparten las mismas cuatro similitudes. Si uno cuenta las "
    "veinticuatro ordenaciones posibles de cuatro valores, solo cuatro cumplen la "
    "condición del group. Cuatro entre veinticuatro es un sexto, aproximadamente cero "
    "punto diecisiete.",
    # 8 Tesis
    "Con eso, la pregunta concreta de mi trabajo es la siguiente: ¿un buen Recall, es "
    "decir, buena recuperación, implica razonamiento composicional? Voy a mostrar que "
    "no.",
    # 9 Sección: Cómo lo medí
    "Segundo bloque: cómo lo medí, reutilizando el cuaderno diez.",
    # 10 Cómo funciona CLIP
    "CLIP es un dual-encoder. Tiene dos torres separadas: una codifica la imagen y "
    "otra codifica el texto, y cada una resume su entrada en un único vector. Luego "
    "compara esos vectores con similitud coseno. Se entrena de forma contrastiva, "
    "acercando los pares correctos y alejando los incorrectos. Esto es exactamente el "
    "motor del cuaderno diez con OpenCLIP, que reutilicé tal cual: crear el modelo, "
    "codificar imagen y texto, y calcular la similitud coseno.",
    # 11 Por qué falla
    "¿Por qué un dual-encoder falla la composición? Porque un único vector global se "
    "comporta como una bolsa de conceptos. Las palabras mayor, joven y besar activan "
    "dimensiones parecidas sin importar cómo estén compuestas. No hay interacción "
    "token a token entre las palabras y las regiones de la imagen. Aquí conecto con "
    "los cuadernos cinco y ocho: la fusión profunda y la atención crossmodal sí "
    "permiten esa interacción, y son la vía natural para la composición.",
    # 12 Pipeline
    "Mi pipeline tiene cinco piezas. Primero, la carga de datos, con el Winoground "
    "real desde Hugging Face, o un set curado offline. Segundo, el motor OpenCLIP que "
    "produce embeddings normalizados. Tercero, el scorer, que arma la matriz dos por "
    "dos y calcula los tres scores. Cuarto, las métricas: Recall e intervalos de "
    "confianza por bootstrap. Y quinto, la prueba de ceguera y el análisis de error "
    "por categoría. Todo es determinista, con semillas, versionado, y con tests.",
    # 13 Scorer código
    "Este es el bloque que produce el resultado principal. Son tres funciones cortas. "
    "El text correct compara dentro de cada columna; el image correct, dentro de cada "
    "fila; y el group correct exige las dos. Un punto importante: seguí el código "
    "oficial de Winoground, y no la redacción verbal de los papers de seguimiento, "
    "que suena al revés.",
    # 14 Validación
    "¿Cómo sé que mi scorer está bien? Lo apliqué a los scores oficiales de CLIP que "
    "vienen incluidos en el propio dataset. Reproduce exactamente cero punto treinta y "
    "cinco centésimas en text, cero punto diez en image, y cero punto cero ocho en "
    "group. Coincidencia perfecta. Esto blinda cualquier duda sobre si mi "
    "implementación es correcta.",
    # 15 Sección: Resultados
    "Tercer bloque: resultados.",
    # 16 Resultado principal
    "Este es el resultado con los cuatrocientos ejemplos reales y el modelo ViT base, "
    "entrenado en laion. El text score es cero punto treinta y cinco, por encima del "
    "azar. Pero el group score es cero punto cero setenta y cinco, por debajo del azar "
    "de cero punto diecisiete, y lejísimos del desempeño humano, que ronda el cero "
    "punto ochenta y seis.",
    # 17 Cómo leerlo
    "Hay que interpretarlo con cuidado. Que el group quede por debajo del azar no "
    "significa que el modelo sea peor que aleatorio en general. Es la consecuencia de "
    "una asimetría: acierta bastante una dirección, el text, y falla la otra, el "
    "image; y el group exige ganar las dos a la vez. El intervalo de confianza del "
    "group, calculado por bootstrap, queda entero por debajo del azar.",
    # 18 Retrieval vs group
    "Y esta es la evidencia central de mi tesis. Sobre el mismo modelo y el mismo "
    "conjunto, el Recall en los cinco primeros es cero punto sesenta y siete, y en los "
    "diez primeros, cero punto setenta y siete. Son valores altos. Pero el group score "
    "es solo cero punto cero setenta y cinco. La diferencia es que el retrieval usa "
    "toda la galería, que es fácil, mientras que el group exige ganar el contraste del "
    "par mínimo, que es difícil.",
    # 19 Por tag
    "¿Qué tipo de error aparece? Lo separé por tipo de cambio. La relación es lo más "
    "difícil, con un group de apenas cero punto cero cuarenta y siete. Son errores de "
    "vinculación, de quién va con quién, y no de reconocer objetos. La categoría "
    "ambos sube un poco, porque ahí las dos imágenes son más distintas entre sí y hay "
    "más señal visual.",
    # 20 Ceguera
    "¿El modelo realmente usa la imagen, o adivina por el texto? Para comprobarlo, "
    "permuté las imágenes entre ejemplos y recalculé. Los scores caen a casi cero. "
    "Eso confirma que sí usa el contenido visual; su límite es composicional, no "
    "perceptivo. Es la versión cuantitativa de la pregunta de interpretabilidad del "
    "cuaderno ocho.",
    # 21 Checkpoints
    "¿Y si uso un modelo más grande? Comparé tres checkpoints. La conclusión es que "
    "escalar el tamaño no arregla la composición: el group no mejora de forma clara, y "
    "el text incluso baja con el modelo más grande.",
    # 22 Ejemplos cualitativos
    "Aquí muestro pares concretos que el modelo confunde. Reconoce los objetos, pero "
    "no logra ligarlos en la composición correcta.",
    # 23 Sección: Cierre
    "Cuarto bloque: cierre, reproducibilidad, y crítica.",
    # 24 Reproducibilidad
    "Todo el trabajo es reproducible. Uso uv con un archivo de bloqueo, una imagen "
    "Docker, semillas globales, un registro del entorno, dieciséis tests, y la "
    "validación contra los scores oficiales. Hay integración continua en GitHub "
    "Actions que corre en cada cambio y cada noche. El dataset con licencia no lo "
    "redistribuyo; si no hay acceso, el pipeline cae automáticamente al set curado.",
    # 25 Demo en vivo
    "En vivo mostraré, en orden: la celda de trazabilidad del notebook, la celda del "
    "scorer que imprime la matriz dos por dos, el archivo de scores con la figura "
    "principal, y la ejecución del script de validación que reproduce los números "
    "oficiales. Lo pesado va pre-ejecutado; en vivo solo corro lo que tarda segundos.",
    # 26 Limitaciones
    "Soy crítico con mi propio resultado. El group bajo mezcla el fallo composicional "
    "con ítems que son ambiguos o difíciles en el propio benchmark. Las métricas son "
    "comparaciones estrictas, sensibles a empates y a la normalización. Y el Recall y "
    "la exactitud no miden composición: son necesarios, pero no suficientes.",
    # 27 Mejoras
    "La mejora natural es evaluar un cross-encoder, o fusión profunda, que es el "
    "cuaderno cinco, con re-ranking de interacción cruzada del cuaderno seis. Esa es "
    "la arquitectura que sí puede componer. Es mi hipótesis; todavía no es un "
    "resultado medido.",
    # 28 Conclusión
    "En resumen: usé el cuaderno diez como motor, y los cuadernos cinco y ocho para "
    "explicar por qué ocurre. Demostré, midiendo y validando, que un retrieval alto no "
    "implica razonamiento composicional, y lo dejé todo reproducible.",
    # 29 Gracias
    "Gracias por su atención. Quedo atento a sus preguntas.",
    # 30 Backup referencias
    "Como referencia: Winoground, de Thrush y colegas; Why is Winoground Hard, de "
    "Diwan y colegas; CLIP, de Radford y colegas; y Evaluating CLIP, de Agarwal y "
    "colegas.",
]
