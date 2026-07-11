# Plan de exposición — 20 minutos

Combina **slides** (para hablar) + **notebook ejecutado** (para verificar en vivo).
Cubre toda la rúbrica. Tiempos aproximados; deja 1-2 min de colchón para preguntas.

| Tiempo | Qué haces | Material |
|---|---|---|
| **0:00–2:30** | **Problema.** Tema (Winoground + Evaluating CLIP), el par mínimo, las 3 métricas, azar 1/6, y la tesis: ¿retrieval implica composición? | Slides 1–7 |
| **2:30–5:00** | **Método.** CLIP es dual-encoder (motor C10); por qué falla la composición (bolsa de conceptos → C5/C8); el pipeline; el scorer; la validación. | Slides 8–12 |
| **5:00–8:30** | **Resultados.** Los 3 scores (group≈azar), R@K vs group, por tag, ceguera, checkpoints. | Slides 13–19 |
| **8:30–12:30** | **Verificación EN VIVO** (cambias a la pantalla del notebook). Corre: celda 1 (entorno), **celda 4 (scorer → matriz 2×2)**, **celda 6 (validación → 0.3075/0.105/0.08)**, y muestra celda 5 (scores + figura). | Notebook |
| **12:30–15:30** | **Entrenamiento + crítica (tu carta fuerte).** Notebook sección 12: explica el re-ranker (interacción imagen-texto = mejora C5/C6), muestra la **validación cruzada** y di la conclusión honesta: *"las barras de error se solapan, con 400 ejemplos no puedo afirmar que supere a CLIP; es prueba de concepto del método."* | Notebook §12 |
| **15:30–18:00** | **Defensa crítica.** Limitaciones (ruido del benchmark, Diwan et al.; empates; azar 1/6 no 1/16) y mejora (cross-encoder real, re-ranking C6, más datos). | Slides 22–23 |
| **18:00–20:00** | **Cierre** (conexión C5/C8/C10 + tesis) y **preguntas**. | Slide 24 |

## Reglas para el demo en vivo
- **No** corras el notebook entero ni `make run` (lento). Ya está **pre-ejecutado**.
- En vivo solo re-corre lo que tarda segundos: **celda 4 (scorer)** y **celda 6 (validación)**.
- La sección 12 (CV) tarda ~1 min; muéstrala pre-ejecutada y re-córrela solo si lo piden.
- Ten listo `make setup` y `make run` ya corridos, y el `.venv` activo.

## Si te aprietan el tiempo (versión 15 min)
Salta slides 6 (azar), 18 (checkpoints) y la sección 12 del notebook; menciónalas solo si preguntan.
Lo irrenunciable: par mínimo, scorer, validación, R@K vs group, ceguera, limitaciones.

## Las 3 frases que debes clavar
1. *"Recall@5 de 0.67 pero group score de 0.075: el retrieval no implica composición."*
2. *"Mi scorer reproduce exactamente los números oficiales de CLIP — está validado."*
3. *"Probé la mejora con validación cruzada y, honestamente, 400 ejemplos no alcanzan para afirmar que gana; es prueba de concepto."*
