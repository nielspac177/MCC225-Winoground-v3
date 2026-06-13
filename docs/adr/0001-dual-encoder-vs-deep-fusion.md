# ADR 0001 — Evaluar Winoground con un dual-encoder (CLIP) y contrastarlo con fusión profunda

- **Estado:** Aceptado
- **Fecha:** 2026-06-13
- **Contexto del curso:** MCC225 — Examen Parcial. Tema: *Winoground / Evaluating CLIP*. Cuadernos C5, C8, C10.

## Contexto

Winoground mide razonamiento composicional con pares mínimos: dos imágenes y dos
captions que comparten exactamente las mismas palabras en distinto orden. Tenemos que
decidir con qué arquitectura producir la evidencia principal y cómo conectar el
resultado con los conceptos de los cuadernos obligatorios.

El curso ofrece dos familias:

1. **Dual-encoder (CLIP / OpenCLIP, C10):** codifica imagen y texto por separado en
   un embedding global y los compara por similitud coseno. Es barato, escala bien y
   funciona para retrieval, pero no modela la interacción token-a-token entre modalidades.
2. **Fusión profunda / cross-encoder (MMBT en C5, atención crossmodal en C8):**
   deja que los tokens visuales y textuales interactúen en capas conjuntas. Gana
   expresividad para composición a cambio de mayor costo, y no escala a retrieval masivo.

## Decisión

Usar **OpenCLIP (dual-encoder) como sujeto de evaluación principal** sobre Winoground,
reutilizando el motor del **Cuaderno 10** (embeddings, similitud coseno, comparación de
checkpoints, FAISS). El scorer implementa los tres scores oficiales (text/image/group).
La **fusión profunda (C5)** y la **atención crossmodal (C8)** se usan como **marco
conceptual de contraste**: explican *por qué* el dual-encoder falla la composición y
qué arquitectura la abordaría.

## Justificación

- El experimento es **trazable y reproducible en CPU/MPS** (solo inferencia, con modelos
  pequeños), lo que cumple el énfasis del examen en reproducibilidad mínima.
- Permite la evidencia central de la tesis: **alto Recall@K + group score ≈ azar**, que
  es el fenómeno que Winoground expone y que enlaza las tres preguntas clave del
  estudiante.
- Conecta C10 (motor), C5 (dual vs deep fusion) y C8 (¿usa la imagen? /
  interpretabilidad), tal como pide la rúbrica para la relación con C5–C10.

## Consecuencias

- **Positivas:** pipeline ligero, determinista, defendible en vivo; figuras claras;
  comparación de checkpoints directa.
- **Negativas / limitaciones:** no entrenamos ni evaluamos un cross-encoder real sobre
  Winoground, que sería el siguiente paso. El set curado offline es un *proxy* controlado
  y no el benchmark oficial. Dejamos ambas cosas documentadas como limitación y mejora futura.

## Alternativas descartadas

- **Entrenar un MMBT/cross-encoder y evaluarlo en Winoground:** demasiado costoso para
  el alcance del examen (requiere datos y GPU); se deja como mejora propuesta.
- **Usar solo accuracy de retrieval (R@K):** descartada porque es precisamente la
  métrica que *oculta* el fallo composicional; se incluye solo como contraste.
