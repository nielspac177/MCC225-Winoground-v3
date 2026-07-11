### Actividad complementaria

### Evaluación responsable del proyecto multimodal

#### Propósito

Esta actividad complementa el `Cuaderno14_MCC225.ipynb` y permite que cada estudiante deje organizada una base técnica para evaluar responsablemente su proyecto multimodal.

El objetivo es conectar el proyecto personal con una evaluación breve, trazable y responsable. No se busca construir otro proyecto ni ejecutar una evaluación extensa. Se busca interpretar con rigor los resultados obtenidos, identificar errores, discutir confiabilidad, explicar algunos casos y reconocer limitaciones de uso.

La actividad debe resolverse usando el proyecto de interés ya definido por el estudiante y tomando el `Cuaderno14_MCC225.ipynb` como plantilla experimental adaptable.

El resultado de esta actividad podrá servir posteriormente como insumo para una presentación, sustentación o informe más amplio del proyecto.

#### Idea central

El estudiante debe responder la siguiente pregunta:

#### Pregunta guía

- ¿Cómo evalúo responsablemente el sistema multimodal asociado a mi proyecto?

### Relación entre proyecto, Cuaderno 14 y actividad

#### Proyecto personal

- Define el dominio, el problema, los usuarios previstos, la motivación y el posible contexto de uso.

#### Cuaderno 14

- Aporta el pipeline técnico de evaluación multimodal: modelo, datos, métricas, resultados, baseline y ejemplos.

#### Actividad complementaria

- Aporta la capa crítica: análisis de error, confiabilidad, explicabilidad, sesgo, limitaciones y uso responsable.

#### Producto posterior

- La evidencia generada en esta actividad puede usarse más adelante para construir una presentación, un informe técnico o una defensa académica del proyecto.

### Casos permitidos

#### Caso A. Proyecto con componente multimodal claro

Aplica si el proyecto trabaja con imagen y texto, captioning, VQA, recuperación imagen-texto, documentos visuales, RAG multimodal, clasificación multimodal, audio, video u otra combinación de modalidades.

En este caso, el estudiante debe evaluar una parte pequeña de su propio sistema.

#### Caso B. Proyecto todavía no implementado

Aplica si el proyecto está definido, pero aún no tiene un prototipo completo.

En este caso, el estudiante puede evaluar un prototipo mínimo relacionado con su idea usando CLIP, BLIP, OpenCLIP, un VLM disponible o una herramienta equivalente.

#### Caso C. Proyecto principalmente textual

Aplica si el proyecto original no es claramente multimodal.

En este caso, el estudiante debe agregar una conexión multimodal mínima. Por ejemplo, puede evaluar imágenes, diagramas, capturas, documentos con texto visual, pares imagen-texto o entradas visuales asociadas al dominio del proyecto.

### Carga mínima esperada

#### Tamaño de evaluación

Entre 30 y 100 ejemplos.

#### Métricas

Entre 2 y 3 métricas principales.

#### Casos cualitativos

Cinco casos reales:

- Dos aciertos claros.
- Dos errores claros.
- Un caso ambiguo.

#### Pruebas de confiabilidad

- Dos pruebas breves.

#### Explicabilidad

- Dos casos explicados con evidencia visual o textual.

#### Reporte

- Entre 3 y 5 páginas.

### Producto esperado

- El estudiante debe entregar un reporte breve en Markdown o PDF y mantener evidencia mínima en su repositorio personal. Además del cuaderno 14 resuelto.

#### Estructura sugerida del repositorio

```text
evaluacion_responsable_mcc225/
├── README.md
├── Cuaderno14_MCC225_resuelto.ipynb
├── reporte_evaluacion_responsable.md
├── results/
│   ├── metricas.csv
│   ├── casos_analizados.csv
│   └── ficha_uso_responsable.csv
└── figures/
    └── ejemplos_evaluados.png
```

### Parte 1. Proyecto personal y tarea multimodal

#### Objetivo

Explicar brevemente el proyecto personal y qué componente multimodal se evaluará.

#### Ficha mínima

| Elemento                     | Respuesta                                                                   |
| ---------------------------- | --------------------------------------------------------------------------- |
| Nombre del proyecto          | completar                                                                   |
| Problema que aborda          | completar                                                                   |
| Dominio de aplicación        | completar                                                                   |
| Modalidades usadas           | texto, imagen, audio, video u otra                                          |
| Tarea multimodal evaluada    | retrieval, captioning, VQA, clasificación, grounding, RAG multimodal u otra |
| Usuario previsto             | completar                                                                   |
| Riesgo principal del sistema | completar                                                                   |

#### Pregunta breve

- ¿Qué parte del proyecto se puede evaluar con evidencia en esta etapa del curso?

### Parte 2. Adaptación del Cuaderno 14

#### Objetivo

- Indicar qué partes del `Cuaderno14_MCC225.ipynb` fueron usadas o adaptadas.

#### Ficha mínima

| Elemento                          | Respuesta |
| --------------------------------- | --------- |
| Modelo usado                      | completar |
| Dataset o subconjunto             | completar |
| Número de ejemplos                | completar |
| Métricas usadas                   | completar |
| Baseline usado                    | completar |
| Hardware o entorno                | completar |
| Parte del Cuaderno 14 reutilizada | completar |
| Parte modificada para el proyecto | completar |

#### Pregunta breve

- ¿Qué cambio fue necesario para adaptar el Cuaderno 14 al proyecto personal?

### Parte 3. Resultados cuantitativos

#### Objetivo

- Reportar los resultados principales sin extender innecesariamente el análisis.

#### Tabla mínima

| Experimento | Modelo    |     Datos | Métrica   | Resultado | Interpretación breve |
| ----------- | --------- | --------: | --------- | --------: | -------------------- |
| E1          | completar | completar | completar | completar | completar            |
| E2          | completar | completar | completar | completar | completar            |
| E3          | completar | completar | completar | completar | completar            |

#### Preguntas breves

#### Pregunta 1

- ¿Qué métrica representa mejor el objetivo de la tarea?

#### Pregunta 2

- ¿Qué métrica puede ser insuficiente si se interpreta sola?

#### Pregunta 3

- ¿El sistema supera claramente al baseline o solo muestra funcionamiento parcial?

### Parte 4. Análisis de cinco casos

#### Objetivo

- Analizar pocos casos, pero con cuidado. No se aceptan casos inventados. Todos deben salir del experimento realizado.

#### Casos obligatorios

- Dos aciertos claros.
- Dos errores claros.
- Un caso ambiguo.

#### Tabla mínima

| Caso | Entrada   | Salida del modelo | Resultado esperado | Tipo de caso | Explicación breve |
| ---- | --------- | ----------------- | ------------------ | ------------ | ----------------- |
| 1    | completar | completar         | completar          | acierto      | completar         |
| 2    | completar | completar         | completar          | acierto      | completar         |
| 3    | completar | completar         | completar          | error        | completar         |
| 4    | completar | completar         | completar          | error        | completar         |
| 5    | completar | completar         | completar          | ambiguo      | completar         |

#### Tipos de error sugeridos

- Error perceptual.
- Error de alineamiento imagen-texto.
- Error de OCR.
- Error de razonamiento espacial.
- Error de conteo.
- Alucinación visual.
- Error por ambigüedad.
- Error por sesgo del dato.
- Error de la métrica.

#### Pregunta breve

- ¿Cuál fue el error más importante y por qué sería problemático en un uso real?

### Parte 5. Confiabilidad

#### Objetivo

- Evaluar si el sistema mantiene un comportamiento razonable en condiciones pequeñas, pero no triviales.

El estudiante debe escoger solo dos pruebas.

#### Prueba A. Sensibilidad al prompt

- Modificar una consulta, caption o pregunta manteniendo el mismo significado aproximado.

#### Prueba B. Caso difícil

- Usar una entrada con ambigüedad, varios objetos, texto visible, baja calidad, ruido o escena poco común.

#### Prueba C. Comparación con baseline

- Comparar contra un resultado aleatorio, léxico o una configuración más simple.

#### Tabla mínima

| Prueba   | Entrada original | Variante  | Cambio observado   | Interpretación |
| -------- | ---------------- | --------- | ------------------ | -------------- |
| Prueba 1 | completar        | completar | bajo, medio o alto | completar      |
| Prueba 2 | completar        | completar | bajo, medio o alto | completar      |

#### Clasificación final de confiabilidad

Elegir una sola opción:

- Confiable solo en condiciones controladas.
- Parcialmente confiable.
- No confiable para uso real.

#### Justificación

- Escribir un párrafo breve que justifique la clasificación elegida.

### Parte 6. Explicabilidad

#### Objetivo

- Explicar dos casos usando evidencia observable. La explicación no debe ser una justificación decorativa.

#### Tabla mínima

| Caso   | Evidencia visual o textual | Explicación propuesta | Límite de la explicación |
| ------ | -------------------------- | --------------------- | ------------------------ |
| Caso 1 | completar                  | completar             | completar                |
| Caso 2 | completar                  | completar             | completar                |

#### Preguntas breves

#### Pregunta 1

- ¿La explicación se basa en evidencia observable o solo racionaliza la salida después de verla?

#### Pregunta 2

- ¿La explicación ayuda a detectar un error del sistema?

### Parte 7. Sesgo y uso responsable

#### Objetivo

- Discutir riesgos posibles de manera honesta. No se exige demostrar estadísticamente un sesgo si el tamaño del experimento no lo permite.

#### Ficha mínima

| Aspecto                             | Respuesta breve                      |
| ----------------------------------- | ------------------------------------ |
| Posible sesgo visual                | completar                            |
| Posible sesgo lingüístico           | completar                            |
| Posible sesgo cultural o de dominio | completar                            |
| Riesgo principal si se usa mal      | completar                            |
| Supervisión humana necesaria        | completar                            |
| Uso recomendado                     | permitido, limitado o no recomendado |
| Justificación del uso recomendado   | completar                            |

#### Pregunta breve

- ¿Qué afirmación sería irresponsable hacer sobre este sistema?

### Parte 8. Conclusión técnica del reporte

#### Objetivo

- Redactar una conclusión breve que sintetice la evaluación realizada y deje lista la interpretación principal del proyecto.

#### Extensión

Entre 450 y 600 palabras.

#### Debe responder

- Qué proyecto se evaluó.
- Qué tarea multimodal se evaluó.
- Qué resultados principales se obtuvieron.
- Qué error fue más importante.
- Qué tan confiable parece el sistema.
- Qué limitación debe comunicarse obligatoriamente.
- En qué condiciones podría usarse o no usarse.

### Entregables

#### Entregable 1

- Cuaderno 14 resuelto o adaptado al proyecto.

#### Entregable 2

- Reporte breve de evaluación responsable.

#### Entregable 3

- Tabla de métricas.

#### Entregable 4

- Tabla de cinco casos analizados.

#### Entregable 5

Ficha de uso responsable.

#### Entregable 6

- Una figura, captura o collage con ejemplos evaluados.

#### Entregable 7

- Repositorio personal actualizado.

#### Penalizaciones

Se penalizará si:

- Solo se muestran ejemplos exitosos.
- No se indica el modelo o dataset usado.
- No se incluye evidencia reproducible.
- Se afirma que el sistema es confiable sin justificarlo.
- Se confunde una demostración visual con una evaluación.
- Se presentan conclusiones más fuertes que los resultados.
- Se entrega evidencia sin relación clara con el proyecto personal.
- No se presenta un repositorio y solo cuadernos dispersos o solo presentaciones.
- No se reportan limitaciones.

### Evidencia mínima que debe quedar lista

Al finalizar la actividad, el estudiante debe dejar preparados los siguientes elementos:

- Tabla de métricas.
- Tabla de casos analizados.
- Ficha de uso responsable.
- Conclusión técnica.
- Figura o captura con ejemplos.
- Repositorio ordenado.
- Cuaderno ejecutado o adaptado.

Estos elementos podrán ser reutilizados posteriormente para una presentación, sustentación o informe más amplio.

### Cierre

Esta actividad no busca demostrar que el sistema funciona perfectamente. Busca que el estudiante pueda defender, con evidencia limitada pero honesta, qué tan bien funciona, dónde falla, qué tan confiable es y bajo qué condiciones sería irresponsable usarlo.

La diferencia central es la siguiente:

- Ejecutar un modelo no equivale a evaluarlo.
- Evaluar un modelo no equivale a declararlo confiable.
- Declarar confiabilidad exige evidencia, límites explícitos y responsabilidad sobre el uso.
