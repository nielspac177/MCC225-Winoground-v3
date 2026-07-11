# Banco de defensa — Segunda Exposición Académica

Preguntas probables del docente y respuestas técnicas breves, ancladas en evidencia
real del repositorio. Complementa `docs/RESPUESTAS_PREGUNTAS.md` (Examen Parcial 1).

## 🧠 Transformer y atención

**¿Qué es la atención producto-punto escalada y qué hacen Q, K, V?**
Cada token se proyecta en query, key y value. La compatibilidad es
`softmax(QKᵀ/√d)·V`; el `√d` evita que los productos crezcan con la dimensión y
saturen el softmax. Q pregunta, K indexa, V aporta el contenido que se promedia.

**¿Self-attention vs cross-attention y dónde aparece en tu modelo?**
CLIP usa **self-attention dentro de cada torre** (imagen y texto por separado). No
tiene **cross-attention** entre modalidades: las torres sólo se encuentran al final
por similitud coseno. Esa ausencia es la razón mecanística de que la composición
quede cerca del azar (group=0.075).

**¿Costo computacional de la atención al crecer los tokens?**
O(n²·d) en memoria y cómputo por el mapa n×n. Con parches de imagen y tokens de
texto, n crece rápido; por eso los modelos de fusión profunda son más caros que un
dual-encoder, que codifica cada modalidad una sola vez y reutiliza embeddings.

## 🔀 Arquitectura y comparación

**¿Por qué dual-encoder y no un cross-encoder?**
El dual-encoder (CLIP) es barato e ideal para **retrieval**: precomputa embeddings e
indexa (FAISS). Un cross-encoder / fusión profunda (BLIP-2, ViLT) modela interacción
token a token → mejor **composición y grounding**, pero cuesta O(pares) en inferencia.
Elegí dual-encoder porque el objetivo del avance es medir el techo de retrieval y
mostrar su límite composicional. Ver `docs/adr/0001` y `docs/adr/0002`.

**¿Qué ganarías con un cross-encoder?**
Capacidad de resolver pares mínimos (relación espacial, binding de atributos) que el
dual-encoder no separa. Es mi acción de cierre propuesta.

## 📊 Evidencia experimental (Cuaderno 14)

**¿Qué datos y modelo usaste?**
120 pares reales de Winoground (adaptación del Cuaderno14 al proyecto, ADR 0002),
CLIP `ViT-B/32` en `mps`, pesos en safetensors local. Baseline: captions desplazados.

**¿Qué resultados obtuviste?**
Retrieval alto (i2t R@5=0.87) y muy por encima del baseline (R@5=0.03); robusto a
degradación visual (CLIPScore 0.26–0.27). Captions BLIP con BLEU≈0.073 (genéricos).
En el pipeline Winoground completo (400 pares): text=0.3475, image=0.110,
group=0.075 (azar 0.167, humano 0.855).

**Detectaste un problema en el cuaderno, ¿cuál?**
Sí: la métrica de captions incluía el caption generado dentro de sus propias
referencias → BLEU/ROUGE=1.0 espurios (fuga). Lo corregí excluyendo `caption_generado`;
los valores reales bajaron a BLEU≈0.073, coherentes con captions genéricos.

## 🔎 Errores, confiabilidad y sesgo (Actividad 5)

**¿Cuál es el error más importante?**
Relación espacial (12/29 casos anotados; tag Relation el peor en Winoground,
group=0.047). El modelo reconoce objetos pero no su disposición ni el rol
agente/paciente.

**¿Qué tan confiable es el sistema?**
**Parcialmente confiable / sólo en condiciones controladas.** Robusto en retrieval y
degradación, pero casi insensible a la negación (Δ≈−0.008) y con composición bajo el
azar. Uso recomendado: limitado, con supervisión humana.

**¿Cómo pruebas que usa la imagen (explicabilidad)?**
Prueba de ceguera: al permutar las imágenes, el group score cae 0.075→0.015. Usa la
imagen, pero débilmente para composición.

**¿Qué afirmación sería irresponsable?**
Decir que "CLIP entiende la relación espacial" o que es "confiable": la evidencia
muestra composición cercana al azar.

## 🗃️ Repositorio y reproducibilidad

**¿Cómo se reproduce todo?**
`make setup && make avance && make run && make figures && make test`, o
`docker compose run --rm avance`. Evidencia versionada en `outputs/`; trazabilidad en
`docs/HOJA_TRAZABILIDAD.md` y `docs/ENTREGA_EXPOSICION2.md`.

**¿Declaraste el uso de IA generativa?**
Sí, en README y en los reportes, conforme a la consigna.
