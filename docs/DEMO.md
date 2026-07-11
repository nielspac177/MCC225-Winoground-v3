# Demo en vivo — CLIP sobre pares mínimos de Winoground

Demo reproducible y **offline** (checkpoint y datos cacheados) para la defensa. En
~15 s muestra la tesis del avance: CLIP asigna similitudes casi idénticas a los dos
captions de un par mínimo, por lo que **falla la composición (group)** aunque el
retrieval sea alto.

## ▶️ Ejecutar

```bash
make demo
# o directamente, con más ejemplos:
python scripts/demo_winoground.py --n 8 --retrieval
```

Requisitos: entorno instalado (`make setup`) y el checkpoint `ViT-B-32/laion2b` en
caché (se descarga la primera vez; el demo usa el mismo motor que la evidencia
committeada en `outputs/`).

## 🗣️ Qué mostrar y decir (60–90 s)

1. **La matriz 2×2.** Para cada par mínimo se imprime la similitud coseno
   `caption × imagen`. Señala que la diferencia entre columnas (Δ) es diminuta
   (~0.01–0.02): CLIP casi no distingue cuál imagen corresponde a cuál caption.
2. **Los veredictos `text / image / group`.** Usan el scorer oficial de Winoground
   (`src/winoground_eval.py`, validado contra `clip.jsonl`). El `group` casi siempre
   sale ✗.
3. **El resumen.** `group 0/5 (0%)` frente a un azar de ~17 % y un humano de ~85 %.
   Aquí se enuncia la conclusión: *retrieval alto ≠ composición*.
4. **Mini-demo de retrieval (`--retrieval`).** La misma query recupera su imagen
   correcta en el top-1, pero su par mínimo queda casi empatado: el problema no es
   percibir objetos, es la composición.

## 🔁 Plan B (si falla algo en vivo)

- Si no hay red / checkpoint: el demo cae al set curado (`source=curated`) y funciona
  igual con formas sintéticas.
- Alternativa sin ejecutar nada: abrir `outputs/figures/scores_vs_chance.png` y
  `outputs/metrics/scores.json`, y el notebook ejecutado
  `notebooks/Cuaderno14_MCC225_resuelto.ipynb`.
