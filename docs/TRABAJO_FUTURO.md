# Trabajo futuro — hacia el trabajo final

Todo el trabajo futuro se deriva del hallazgo central del avance: el dual-encoder CLIP
**recupera bien pero no compone** (Winoground *group* = 0.075 < azar 0.167 < humano 0.855).
El trabajo final debe **cerrar o explicar** esa brecha. Las líneas están priorizadas y con
esfuerzo estimado (S = bajo, M = medio, L = alto).

## 🎯 Línea principal: cross-encoder profundo

| Qué | Detalle | Pregunta que responde | Esfuerzo |
|---|---|---|---|
| Cross-encoder real | Evaluar un modelo con **cross-attention** real (BLIP-2 / Q-Former, MMBT o fusión profunda C5–C6), más allá del *re-ranker-lite* ya probado (folds ≈ 0.11 / 0.01 / 0.03 / 0.06) | ¿La interacción imagen–texto token a token sube el *group score* por encima del azar? | **L** |

Es la contribución que cierra la narrativa *"retrieval ≠ composición → la solución es la
atención cruzada"* y la que ya anticipa la slide 8 (plan de cierre) y el §7 del avance técnico.

## 📊 Reforzar la evidencia

| Línea | Detalle | Pregunta | Esfuerzo |
|---|---|---|---|
| Escala estadística | Los 400 pares completos + más checkpoints, IC *bootstrap* y test de significancia vs azar | ¿El 0.075 es robusto o ruido de muestreo? | S |
| Hard-negative training | Fine-tuning contrastivo estilo **NegCLIP** (negativos composicionales) | ¿Se puede *entrenar* la composición sin cambiar la arquitectura? | M |
| Baseline superior | Comparar contra un **VLM moderno** (LLaVA / GPT-4V) como techo | ¿Es límite del dual-encoder o del paradigma contrastivo? | M |

## 🔍 Interpretabilidad y confiabilidad (extiende la Actividad 5)

- **Mapas de atención cruzada** en el cross-encoder: visualizar dónde mira el modelo al
  resolver un par mínimo, y contrastarlo con la prueba de ceguera actual. (M)
- **Perturbaciones sistemáticas**: negación, conteo y orden espacial como curvas de
  sensibilidad, no casos sueltos. (S)
- **Ambigüedad del benchmark** (Diwan et al., 2022): anotar/filtrar los pares ambiguos de
  Winoground y re-medir. Parte del *group* bajo puede ser ruido del dataset, no del modelo;
  es un punto honesto y sofisticado para la defensa. (M)

## 🛠️ Ingeniería y reproducibilidad

- Versionar embeddings con **DVC**, añadir una *dataset card* y ampliar el CI (ya hay Docker,
  tests y CI base). (S)

---

## Riesgos y límites que el trabajo final debe declarar

- **Tamaño de muestra:** 120 pares locales / 400 oficiales → potencia estadística limitada.
- **Sesgos:** entrenamiento en LAION (web, inglés), posibles sesgos culturales y de dominio.
- **Costo del cross-encoder:** O(pares) en inferencia vs. el dual-encoder cacheable.
- **Confiabilidad actual:** *parcialmente confiable / solo en condiciones controladas*; afirmar
  comprensión composicional sería irresponsable con la evidencia presente.

> **En una frase (para la defensa):** el avance demuestra el **límite** del dual-encoder; el
> trabajo final mide si un **cross-encoder con atención cruzada** cierra la brecha
> composicional, con evaluación a mayor escala e interpretabilidad de la atención.
