# ADR 0003 — Tratamiento de empates y métrica alternativa de emparejamiento

- **Estado:** Aceptado
- **Fecha:** 2026-08-01
- **Contexto del curso:** MCC225 — Examen Final, Proyecto 5. Requisitos §11.2.2 y §11.5.2.

## Contexto

El scorer oficial de Winoground (Thrush et al., 2022) decide con `>` estricto:

```python
text_correct  = s(c0,i0) > s(c1,i0) and s(c1,i1) > s(c0,i1)
image_correct = s(c0,i0) > s(c0,i1) and s(c1,i1) > s(c1,i0)
group_correct = text_correct and image_correct
```

Esto tiene dos consecuencias que el examen pide abordar de frente.

**Primera: los empates se contabilizan como fallos, en silencio.** Si dos similitudes
coinciden exactamente, `>` devuelve `False` y el ejemplo suma cero. Pero "el modelo prefiere
la respuesta equivocada" y "el modelo no expresa ninguna preferencia" son estados
epistémicamente distintos, y el score los colapsa. Con similitudes coseno en `float32`, los
empates exactos son raros pero no imposibles; y bajo cualquier tolerancia numérica razonable
dejan de ser raros. Sin medirlos no se puede afirmar que usar `>` sea inocuo: hay que
mostrar el conteo.

**Segunda: `group_score` es una condición muy fuerte.** Exige que cada elemento diagonal
sea el máximo de su fila *y* de su columna — cuatro comparaciones simultáneas. Zhu et al.
(ICLR 2026, arXiv:2510.07632) señalan que esa exigencia no es la pregunta natural de
emparejamiento, y que castiga a modelos que sí resuelven la asignación global.

## Decisión

**1. Un scorer de tres estados, con el comportamiento oficial como valor por defecto.**

Se añaden `text_status`, `image_status`, `group_status` y `group_match_status`, que devuelven
`correcto`, `incorrecto` o `empate`, con una tolerancia `atol` configurable (por defecto
`0.0`). La regla de combinación es:

> Un fallo estricto domina. En ausencia de fallo, un empate deja el caso abierto.

Es decir: si alguna comparación falla → `incorrecto`; si ninguna falla pero alguna empata →
`empate`; si todas aciertan → `correcto`. Un empate nunca rescata a un fallo.

**2. Tres convenciones explícitas para convertir un empate en número**, vía `tie_policy`:

| Política | Empate vale | Interpretación |
|---|---|---|
| `fail` (por defecto) | 0.0 | Convención oficial. Conservadora: la indeterminación no es mérito. |
| `half` | 0.5 | Un empate es equivalente a decidir al azar entre dos opciones. |
| `pass` | 1.0 | Cota superior: máximo posible si todos los empates se resolvieran a favor. |

Reportar `fail` y `pass` acota el score verdadero por debajo y por arriba sin tener que
elegir una convención.

**3. `GroupMatch` como métrica complementaria**, no sustituta:

```
group_match(sim) = (s00 + s11) > (s01 + s10)
```

Para k=2 esto pregunta si el emparejamiento correcto es el de mayor similitud total. El azar
sube de **1/6 a 1/2**, por lo que los valores de las dos métricas no son comparables entre sí
y siempre se reportan con su nivel de azar al lado.

**4. Validación estricta de entrada.** `_as_matrix` rechaza matrices que no sean 2×2 y
cualquier `NaN` o infinito. Motivo: un `NaN` hace que toda comparación `>` devuelva `False`,
produciendo un cero indistinguible de un fallo genuino del modelo — el peor tipo de error
silencioso en un pipeline de evaluación.

## Justificación

- **Compatibilidad demostrada, no prometida.** Un test (`test_estados_coinciden_con_el_scorer_oficial_sin_tolerancia`)
  compara los dos caminos sobre 200 matrices aleatorias y exige
  `(status == correcto) is scorer_oficial(sim)` para las tres métricas. Los 16 tests
  originales pasan sin modificación.
- **GroupMatch no reemplaza al group score, lo contextualiza.** Son preguntas distintas:
  `group_score` mide si el modelo ordena bien *cada comparación por separado*; `GroupMatch`,
  si acierta la *asignación global*. La brecha entre ambos es informativa por sí misma —
  separa "el modelo no compone" de "la métrica no lo deja demostrar".
- **La implicación es verificable.** `group_score = 1 ⟹ GroupMatch = 1` (si ambas diagonales
  superan a ambas antidiagonales, su suma también lo hace). Un test lo comprueba sobre 500
  matrices aleatorias. Lo contrario no se cumple, y ese es exactamente el conjunto de
  ejemplos donde las dos métricas discrepan.

## Consecuencias

- **Positivas:** el informe puede reportar cuántos ejemplos quedan indeterminados y acotar el
  score entre dos convenciones; queda cubierto §11.2.2; y `GroupMatch` da la respuesta
  concreta a §11.4.7 ("¿qué alternativa sería razonable?").
- **Negativas:** la firma de `per_example_scores` y `aggregate` crece con dos parámetros
  opcionales, y `per_example.csv` gana cinco columnas. Es deuda de interfaz aceptada a cambio
  de no perder información.
- **Riesgo declarado:** `atol` es un parámetro con efecto directo sobre el resultado. Se fija
  en `0.0` por defecto y cualquier valor distinto debe reportarse junto al score — por eso
  `as_dict()` incluye `atol` y `tie_policy` en su salida.

## Alternativas descartadas

- **Contar el empate como medio acierto sin más.** Descartada como *default*: cambia los
  números publicados frente al scorer oficial y rompería la comparación con la literatura.
  Se ofrece como opción (`tie_policy="half"`), no como comportamiento base.
- **Sustituir `group_score` por `GroupMatch`.** Descartada: el azar cambia de 1/6 a 1/2 y los
  valores dejan de ser comparables con los tres años de resultados publicados. Se reportan
  ambas.
- **Añadir ruido para romper empates.** Descartada por indefendible: introduce
  no-determinismo en un pipeline cuyo valor principal es ser reproducible.
