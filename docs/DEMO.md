# Demo en vivo — CLIP sobre pares mínimos de Winoground

Demo reproducible y **offline** (checkpoint y datos cacheados) para la defensa. En
~15 s muestra la tesis del avance: CLIP asigna similitudes casi idénticas a los dos
captions de un par mínimo, por lo que **falla la composición (group)** aunque el
retrieval sea alto.

## ▶️ Ejecutar

```bash
make demo-app       # APP HTML interactiva (offline) -> abrir en el navegador  ⭐ recomendada
make demo           # versión terminal (matriz 2x2 + veredictos en texto)
make demo-visual    # versión VISUAL: paneles con imágenes reales + heatmap + GIF animado
# o directamente:
python scripts/demo_winoground.py --n 8 --retrieval
python scripts/demo_visual.py --n 6      # -> outputs/demo/panel_*.png + demo_winoground.gif
```

La versión visual (`make demo-visual`) genera, por cada par mínimo, un panel con las **dos
imágenes reales**, sus captions, el **mapa de calor 2×2** de similitud y los veredictos
`text / image / group` con ✓/✗, y ensambla un **GIF animado** (`outputs/demo/demo_winoground.gif`)
ideal para proyectar o incrustar en las diapositivas.

Requisitos: entorno instalado (`make setup`) y el checkpoint `ViT-B-32/laion2b` en
caché (se descarga la primera vez; el demo usa el mismo motor que la evidencia
committeada en `outputs/`).

### App HTML para presentar (abrir y mostrar)

`make demo-app` genera `outputs/demo/demo_app.html`: un **único archivo autónomo**
(imágenes embebidas, sin servidor, offline) que se abre en el navegador y permite
navegar los pares mínimos con ← / →, con el heatmap 2×2, los veredictos ✓/✗ y un
marcador acumulado. Ábrelo así:

```bash
open outputs/demo/demo_app.html                    # navegador por defecto (macOS)
open -a "Google Chrome" outputs/demo/demo_app.html # o Chrome / Safari
```

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
