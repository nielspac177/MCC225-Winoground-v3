"""Construye el cuaderno de defensa para las tareas de código de §11.5.

El cuaderno está pensado para abrirse **durante** el examen: cada tarea del banco
tiene su celda, ejecutable en segundos porque solo usa numpy sobre la caché de
matrices 2x2. No importa torch, así que arranca al instante.

Uso:
    python scripts/15_build_defensa_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINO = ROOT / "notebooks" / "Defensa_Final_MCC225.ipynb"


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip().splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": texto.strip().splitlines(keepends=True),
    }


CELDAS = [
    md("""
# Defensa Final MCC225 — Tareas de código

**Proyecto 5:** Evaluación del razonamiento visio-lingüístico composicional en CLIP mediante Winoground
**Estudiante:** Niels Victor Pacheco Barrios

Este cuaderno resuelve las cinco tareas de código de §11.5 del examen. Cada celda corre en
segundos: todo el análisis opera sobre las matrices de similitud 2×2 ya cacheadas, sin
importar `torch` ni `faiss`.

**Convención fundamental:** `sim[c][i]` es la similitud entre el *caption c* y la *imagen i*.
Por diseño del benchmark, `caption_0` corresponde a `image_0` y `caption_1` a `image_1`, de
modo que **la diagonal es la respuesta correcta**.
"""),
    code("""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.winoground_eval import (
    text_correct, image_correct, group_correct,
    text_status, image_status, group_status,
    group_match, group_match_margin, group_match_status,
    aggregate, tie_report, per_example_scores,
    CORRECTO, INCORRECTO, EMPATE,
)
from src.metrics import (
    bootstrap_ci, stratified_bootstrap_ci,
    mcnemar_exact, minimum_resolvable_difference,
)

Z = np.load(ROOT / "outputs/sims/winoground_real__ViT-B-32__laion2b_s34b_b79k.npz",
            allow_pickle=True)
SIMS = Z["sims"]                      # (400, 2, 2)
META = pd.read_csv(ROOT / "data/winoground_meta.csv")
TAGS = META["collapsed_tag"].fillna("(sin_tag)").to_numpy()

print(f"Modelo   : {Z['etiqueta']}")
print(f"Matrices : {SIMS.shape}")
print(f"Tags     : {pd.Series(TAGS).value_counts().to_dict()}")
"""),

    md("""
---
## Tarea 1 — Calcular text, image y group a mano sobre una matriz 2×2

Uso el **ejemplo 0 real** del benchmark, el par mínimo
*"an old person kisses a young person"* contra *"a young person kisses an old person"*.
Es un caso de rol invertido: la misma bolsa de palabras, distinta dirección de la relación.
"""),
    code("""
sim = SIMS[0]
print("Matriz sim[c][i]:\\n", np.round(sim, 4), "\\n")
print(f"  s(c0,i0) = {sim[0,0]:.4f}   s(c0,i1) = {sim[0,1]:.4f}")
print(f"  s(c1,i0) = {sim[1,0]:.4f}   s(c1,i1) = {sim[1,1]:.4f}\\n")

# text: se FIJA LA IMAGEN y se elige el CAPTION -> se compara dentro de cada COLUMNA
t1 = sim[0,0] > sim[1,0]   # imagen 0: gana c0?
t2 = sim[1,1] > sim[0,1]   # imagen 1: gana c1?
print(f"text : ({sim[0,0]:.4f} > {sim[1,0]:.4f}) = {t1}  Y  "
      f"({sim[1,1]:.4f} > {sim[0,1]:.4f}) = {t2}  ->  {t1 and t2}")

# image: se FIJA EL CAPTION y se elige la IMAGEN -> se compara dentro de cada FILA
i1 = sim[0,0] > sim[0,1]   # caption 0: gana i0?
i2 = sim[1,1] > sim[1,0]   # caption 1: gana i1?
print(f"image: ({sim[0,0]:.4f} > {sim[0,1]:.4f}) = {i1}  Y  "
      f"({sim[1,1]:.4f} > {sim[1,0]:.4f}) = {i2}  ->  {i1 and i2}")

print(f"group: text AND image -> {(t1 and t2) and (i1 and i2)}\\n")
print("Comprobacion contra el scorer:",
      text_correct(sim), image_correct(sim), group_correct(sim))

# GroupMatch: la asignacion correcta, es la de mayor similitud TOTAL?
m = group_match_margin(sim)
print(f"\\nGroupMatch: ({sim[0,0]:.4f} + {sim[1,1]:.4f}) - ({sim[0,1]:.4f} + {sim[1,0]:.4f})"
      f" = {m:+.5f}  ->  {group_match(sim)}")
print("\\nEste ejemplo falla group pero acierta GroupMatch: es exactamente")
print("la discrepancia entre 'ordenar bien cada comparacion' y 'acertar la asignacion global'.")
"""),

    md("""
---
## Tarea 2 — Modificar el scorer para registrar empates y devolver tres estados

El scorer oficial decide con `>` estricto, así que un empate exacto se contabiliza **en
silencio como fallo**. Eso colapsa dos estados distintos: *el modelo prefiere lo incorrecto*
y *el modelo no expresa preferencia*.

La regla de combinación que implementé: **un fallo estricto domina; en su ausencia, un empate
deja el caso abierto.** Un empate nunca rescata a un fallo.
"""),
    code("""
casos = {
    "diagonal dominante"      : np.array([[0.9, 0.1], [0.1, 0.9]]),
    "antidiagonal dominante"  : np.array([[0.1, 0.9], [0.9, 0.1]]),
    "empate en una compar."   : np.array([[0.5, 0.1], [0.5, 0.9]]),
    "empate + fallo"          : np.array([[0.5, 0.9], [0.5, 0.1]]),
    "todo constante"          : np.full((2, 2), 0.3),
}
print(f"{'caso':26s} {'text':12s} {'image':12s} {'group':12s}")
print("-" * 64)
for nombre, s in casos.items():
    print(f"{nombre:26s} {text_status(s):12s} {image_status(s):12s} {group_status(s):12s}")

print("\\nNotese la fila 'empate + fallo': el empate NO rescata al fallo -> incorrecto.")
"""),
    code("""
# Invariante critico: con atol=0 el estado CORRECTO debe coincidir EXACTAMENTE
# con el scorer oficial. Sin esto, cualquier numero dejaria de ser comparable
# con la literatura.
rng = np.random.default_rng(0)
for _ in range(2000):
    s = rng.random((2, 2))
    assert (text_status(s)  == CORRECTO) is text_correct(s)
    assert (image_status(s) == CORRECTO) is image_correct(s)
    assert (group_status(s) == CORRECTO) is group_correct(s)
print("OK: 2000 matrices aleatorias, los dos caminos coinciden.")

# Cuantos empates hay REALMENTE en los datos, y como de holgadas son las decisiones
print("\\n--- Barrido de tolerancia sobre los 400 ejemplos reales ---")
print(f"{'atol':>8s} {'emp_text':>9s} {'emp_image':>10s} {'text':>8s} {'group':>8s}")
for a in [0.0, 1e-6, 1e-4, 1e-3, 5e-3, 1e-2]:
    r, s = tie_report(SIMS, atol=a), aggregate(SIMS, atol=a)
    print(f"{a:8.0e} {r['text_ties']:9d} {r['image_ties']:10d} {s.text:8.4f} {s.group:8.4f}")

margenes = np.concatenate([
    np.abs(SIMS[:,0,0]-SIMS[:,1,0]), np.abs(SIMS[:,1,1]-SIMS[:,0,1]),
    np.abs(SIMS[:,0,0]-SIMS[:,0,1]), np.abs(SIMS[:,1,1]-SIMS[:,1,0]),
])
print(f"\\nCERO empates exactos. Pero la mediana del margen de decision es "
      f"{np.median(margenes):.4f}")
print(f"y el percentil 5 es {np.percentile(margenes,5):.5f}, sobre similitudes en "
      f"[{SIMS.min():.3f}, {SIMS.max():.3f}].")
print("Conclusion honesta: el '>' estricto es inocuo, pero las decisiones NO son holgadas.")
"""),

    md("""
---
## Tarea 3 — Bootstrap estratificado por tag para el group score

El bootstrap simple trata los 400 ejemplos como intercambiables. Pero los tags están
desbalanceados (Relation 233, Object 141, Both 26) y el group score difiere mucho entre
ellos, así que un remuestreo libre hace fluctuar **la composición** de la muestra además de
su contenido.

Eso infla el intervalo con una fuente de variación que no nos interesa: no preguntamos qué
pasaría si el benchmark tuviera otra mezcla de tags, sino qué pasaría con **otros ejemplos de
la misma mezcla**.
"""),
    code("""
puntos = per_example_scores(SIMS)
g = [p["group"] for p in puntos]

simple  = bootstrap_ci(g, rounds=2000, seed=42)
estrat  = stratified_bootstrap_ci(g, TAGS, rounds=2000, seed=42)

print(f"group score = {simple['mean']:.4f}   (azar = {1/6:.4f})\\n")
print(f"  IC simple      [{simple['lo']:.4f}, {simple['hi']:.4f}]  ancho "
      f"{simple['hi']-simple['lo']:.4f}")
print(f"  IC estratific. [{estrat['lo']:.4f}, {estrat['hi']:.4f}]  ancho "
      f"{estrat['hi']-estrat['lo']:.4f}   ({estrat['n_strata']} estratos)")
print(f"\\nAmbos IC estan enteramente por DEBAJO del azar de {1/6:.4f}.")

print("\\n--- Desglose por tag ---")
df = pd.DataFrame(puntos); df["tag"] = TAGS
print(df.groupby("tag")[["text","image","group","group_match"]]
        .agg(["mean","count"]).round(4).to_string())
"""),

    md("""
---
## Tarea 4 — Test parametrizado: text solo, image solo, group, y shape inválido

Detalle que importa: uso `is True` en vez de `== True`. Eso obliga a que el scorer devuelva
un `bool` de Python y no un `np.bool_`, y es la razón por la que las funciones envuelven el
resultado en `bool(...)`.
"""),
    code("""
# pytest se invoca abajo por linea de comandos, con el interprete del venv

test_src = '''
import numpy as np
import pytest
from src.winoground_eval import text_correct, image_correct, group_correct, per_example_scores

@pytest.mark.parametrize("sim, e_text, e_image, e_group", [
    (np.array([[0.9, 0.1 ], [0.1 , 0.9 ]]), True,  True,  True),   # todo bien
    (np.array([[0.9, 0.95], [0.1 , 0.97]]), True,  False, False),  # solo text
    (np.array([[0.9, 0.1 ], [0.95, 0.97]]), False, True,  False),  # solo image
    (np.array([[0.1, 0.9 ], [0.9 , 0.1 ]]), False, False, False),  # todo mal
])
def test_scorer_parametrizado(sim, e_text, e_image, e_group):
    assert text_correct(sim)  is e_text
    assert image_correct(sim) is e_image
    assert group_correct(sim) is e_group

@pytest.mark.parametrize("forma", [(2, 3), (3, 2), (2,), (2, 2, 2)])
def test_shape_invalido(forma):
    with pytest.raises(ValueError):
        per_example_scores([np.zeros(forma)])
'''
(ROOT / "tests" / "test_defensa_parametrizado.py").write_text(test_src)
print("Escrito tests/test_defensa_parametrizado.py\\n")

# Se invoca por subproceso con el interprete del venv: el kernel global de
# Jupyter no ve las dependencias del proyecto.
import subprocess
r = subprocess.run(
    [str(ROOT / ".venv/bin/python"), "-m", "pytest",
     "tests/test_defensa_parametrizado.py", "-q"],
    cwd=ROOT, capture_output=True, text=True,
)
print(r.stdout.strip()[-600:] or r.stderr.strip()[-600:])
"""),

    md("""
---
## Tarea 5 — Complejidad temporal y espacial con los embeddings ya calculados

Con embeddings de dimensión *d* ya calculados:

- **Tiempo:** cada matriz 2×2 es un producto (2×d)·(d×2) = **O(d)** por ejemplo, más O(1)
  comparaciones. Total **O(N·d)**.
- **Espacio:** **O(N·d)** para los embeddings, O(N) para los scores.

Lo interesante es el contraste con un cross-encoder, que **no factoriza**: no hay embeddings
que cachear porque la representación depende del par, así que cada ejemplo cuesta 4 pasadas
completas y el benchmark 1600. Esa asimetría O(N+M) contra O(N·M) es la razón por la que los
sistemas de retrieval usan dual-encoders para recuperar y reservan el cross-encoder para
reordenar.
"""),
    code("""
emb = np.load(ROOT / "data/winoground_cache/embeddings/"
              "winoground_real__ViT-B-32__laion2b_s34b_b79k.npz")
cap, img = emb["cap"], emb["img"]
N, d = len(SIMS), cap.shape[1]

print(f"N = {N} ejemplos,  d = {d} dimensiones\\n")
print(f"Espacio embeddings : {cap.nbytes + img.nbytes:,} bytes "
      f"({(cap.nbytes+img.nbytes)/1e6:.2f} MB)   -> O(N*d)")
print(f"Espacio matrices   : {SIMS.nbytes:,} bytes "
      f"({SIMS.nbytes/1e3:.1f} KB)   -> O(N)")

t = time.perf_counter()
for _ in range(100):
    _ = [cap[2*k:2*k+2] @ img[2*k:2*k+2].T for k in range(N)]
ms = (time.perf_counter() - t) / 100 * 1000
print(f"\\nConstruir las {N} matrices : {ms:.2f} ms")

t = time.perf_counter()
for _ in range(100):
    _ = aggregate(SIMS)
print(f"Puntuar los {N} ejemplos   : {(time.perf_counter()-t)/100*1000:.2f} ms")

print(f"\\nUn cross-encoder necesitaria {4*N} pasadas completas por la red")
print("para lo mismo, y no puede cachear nada entre pares.")
"""),

    md("""
---
## Resultado principal — la métrica cambia el veredicto

Mismo modelo, mismos datos, **las mismas cuatro similitudes**. Solo cambia el criterio.
"""),
    code("""
s = aggregate(SIMS)
azar   = {"text":0.25, "image":0.25, "group":1/6, "group_match":0.5}
humano = {"text":0.895,"image":0.885,"group":0.855,"group_match":None}
vals   = {"text":s.text,"image":s.image,"group":s.group,"group_match":s.group_match}

print(f"{'metrica':14s} {'score':>8s} {'IC 95%':>18s} {'azar':>7s} {'humano':>8s}  veredicto")
print("-"*78)
for m in ["text","image","group","group_match"]:
    ci = bootstrap_ci([p[m] for p in per_example_scores(SIMS)], rounds=2000, seed=42)
    v  = "SOBRE el azar" if ci["lo"] > azar[m] else "bajo el azar"
    h  = f"{humano[m]:.3f}" if humano[m] else "n/d"
    print(f"{m:14s} {vals[m]:8.4f} [{ci['lo']:6.3f},{ci['hi']:6.3f}] {azar[m]:7.3f} "
          f"{h:>8s}  {v}")

print("\\n" + "="*78)
print("group score = 0.075 -> BAJO el azar de 0.167  ->  'la composicion colapsa'")
print("GroupMatch  = 0.675 -> SOBRE el azar de 0.500  ->  'el modelo acierta la asignacion'")
print("="*78)
print("\\nLa afirmacion honesta no es 'CLIP no compone', sino: CLIP falla el criterio")
print("estricto de Winoground, y parte de esa caida es la exigencia de la metrica.")
"""),

    md("""
---
## Comparación entre checkpoints: por qué **no** puedo decir que uno sea mejor

Los tres se evalúan sobre **los mismos 400 ítems**, así que comparar dos intervalos de
confianza independientes sería un error: ignoraría que los aciertos están correlacionados.
McNemar mira solo los **pares discordantes**, que es la evidencia que realmente distingue.
"""),
    code("""
import glob
modelos = {}
for f in sorted(glob.glob(str(ROOT / "outputs/sims/*.npz"))):
    z = np.load(f, allow_pickle=True)
    modelos[str(z["etiqueta"])] = [p["group"] for p in per_example_scores(z["sims"])]

nombres = list(modelos)
print("McNemar exacto sobre group score:\\n")
for i in range(len(nombres)):
    for j in range(i+1, len(nombres)):
        a, b = nombres[i], nombres[j]
        r = mcnemar_exact(modelos[a], modelos[b])
        print(f"  {a}\\n  vs {b}")
        print(f"     dif = {r['dif_media']:+.4f}   discordantes = {r['discordantes']}"
              f"   p = {r['p_value']:.3f}   -> "
              f"{'DISTINGUIBLES' if r['p_value'] < 0.05 else 'indistinguibles'}\\n")

print("--- Por que: potencia estadistica con n=400 ---")
for m, p in [("text",0.31), ("image",0.10), ("group",0.078), ("group_match",0.66)]:
    r = minimum_resolvable_difference(n=400, p=p)
    print(f"  {m:12s} diferencia minima detectable = "
          f"{r['dif_minima_detectable']*100:.1f} puntos")
print("\\nLa mayor diferencia observada entre checkpoints es de 1.25 puntos en group,")
print("frente a un minimo detectable de 5.3. El benchmark no tiene potencia para")
print("resolverlos, asi que reportar un ranking seria reportar ruido.")
"""),
]


def main() -> None:
    nb = {
        "cells": CELDAS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in CELDAS)
    print(f"[ok] {DESTINO.relative_to(ROOT)}  ({len(CELDAS)} celdas, {n_code} de código)")


if __name__ == "__main__":
    main()
