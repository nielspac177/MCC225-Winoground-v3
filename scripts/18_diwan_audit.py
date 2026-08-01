"""Auditoría factorial: ¿cuánto de la brecha composicional es métrica, dataset o ruido?

Cruza las tres correcciones conocidas al resultado de Winoground, que hasta ahora
se han publicado por separado:

  1. **Métrica.** group score (azar 1/6) frente a GroupMatch (azar 1/2).
     Zhu et al., ICLR 2026.
  2. **Ambigüedad del dataset.** Diwan et al. (EMNLP 2022) reanotaron los 400
     ítems y encontraron que muchos no miden composición sino sentido común,
     resolución de correferencia o percepción de detalles finos. Filtrarlos
     separa el fallo del modelo del ruido del benchmark.
  3. **Ruido de muestreo.** Intervalos bootstrap y diferencia mínima detectable.

Nadie ha cruzado las tres. La pregunta que responde este script es si el
diagnóstico "los modelos no componen" sobrevive cuando se aplican a la vez.

Niveles de limpieza del dataset, de menos a más estricto:

| Nivel | Qué excluye | Justificación |
|---|---|---|
| `todos` | nada | los 400 oficiales |
| `sin_ambiguos` | `Ambiguously Correct` | ítems donde ambos captions describen ambas imágenes; el fallo no es del modelo |
| `pares_minimos` | + `Non Minimal` | ítems que no son pares mínimos reales, así que no aíslan el orden |
| `sin_confusores` | + `Visually Difficult`, `Unusual Image`, `Unusual Text`, `Complex Reasoning` | deja solo los 172 sin etiqueta: composición y nada más |

Salidas en outputs/metrics/:
  diwan_factorial.csv        modelo x métrica x nivel de limpieza, con IC
  diwan_por_etiqueta.csv     score por cada etiqueta individual de Diwan
  diwan_estabilidad.csv      Kendall tau entre rankings de modelos por condición

Uso:
    python scripts/18_diwan_audit.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import bootstrap_ci, minimum_resolvable_difference  # noqa: E402
from src.winoground_eval import per_example_scores  # noqa: E402

SIMS = ROOT / "outputs" / "sims"
MET = ROOT / "outputs" / "metrics"
DIWAN = ROOT / "data" / "anotaciones" / "diwan_new_tag_assignments.json"

RONDAS, SEMILLA = 2000, 42
METRICAS = ["text", "image", "group", "group_match"]
AZAR = {"text": 0.25, "image": 0.25, "group": 1 / 6, "group_match": 0.5}

# Cada nivel excluye acumulativamente las etiquetas listadas.
NIVELES = {
    "todos": [],
    "sin_ambiguos": ["Ambiguously Correct"],
    "pares_minimos": ["Ambiguously Correct", "Non Minimal"],
    "sin_confusores": ["Ambiguously Correct", "Non Minimal", "Visually Difficult",
                       "Unusual Image", "Unusual Text", "Complex Reasoning"],
}

ORDEN = [
    "winoground_real__ViT-B-32__laion2b_s34b_b79k",
    "winoground_real__ViT-B-16__datacomp_xl_s13b_b90k",
    "winoground_real__ViT-L-14-quickgelu__openai",
    "winoground_real__BLIP-base__itc",
    "winoground_real__BLIP-base__itm",
]


def cargar_modelos() -> dict:
    hallados = {p.stem: p for p in SIMS.glob("*.npz")}
    claves = [k for k in ORDEN if k in hallados]
    claves += sorted(k for k in hallados if k not in ORDEN)
    modelos = {}
    for k in claves:
        z = np.load(hallados[k], allow_pickle=True)
        modelos[str(z["etiqueta"])] = z["sims"]
    if not modelos:
        raise SystemExit(f"no hay matrices en {SIMS}")
    return modelos


def cargar_diwan() -> dict:
    if not DIWAN.exists():
        raise SystemExit(
            f"faltan las anotaciones de Diwan en {DIWAN}.\n"
            "Descárgalas con:\n  curl -sL -o data/anotaciones/diwan_new_tag_assignments.json "
            "https://raw.githubusercontent.com/ajd12342/why-winoground-hard/main/"
            "new_tag_assignments.json"
        )
    d = json.loads(DIWAN.read_text())
    if len(d) != 400:
        raise ValueError(f"se esperaban 400 anotaciones, hay {len(d)}")
    return {int(k): v for k, v in d.items()}


def mascara(anot: dict, excluir: list[str]) -> np.ndarray:
    """Máscara booleana de los ejemplos que NO llevan ninguna etiqueta excluida."""
    return np.array([not (set(anot[i]) & set(excluir)) for i in range(400)])


def tabla_factorial(modelos: dict, anot: dict) -> pd.DataFrame:
    filas = []
    for etiqueta, sims in modelos.items():
        puntos = pd.DataFrame(per_example_scores(sims))
        for nivel, excluir in NIVELES.items():
            m = mascara(anot, excluir)
            sub = puntos[m]
            for met in METRICAS:
                ci = bootstrap_ci(sub[met].tolist(), rounds=RONDAS, seed=SEMILLA)
                filas.append({
                    "modelo": etiqueta, "nivel": nivel, "n": int(m.sum()),
                    "metrica": met, "score": ci["mean"],
                    "ic_lo": ci["lo"], "ic_hi": ci["hi"],
                    "azar": AZAR[met],
                    "supera_azar": bool(ci["lo"] > AZAR[met]),
                    # Distancia al azar en unidades de la propia métrica: permite
                    # comparar group y GroupMatch pese a tener azares distintos.
                    "exceso_sobre_azar": ci["mean"] - AZAR[met],
                })
    return pd.DataFrame(filas)


def tabla_por_etiqueta(modelos: dict, anot: dict) -> pd.DataFrame:
    etiquetas = sorted({t for v in anot.values() for t in v})
    filas = []
    for modelo, sims in modelos.items():
        puntos = pd.DataFrame(per_example_scores(sims))
        for et in etiquetas + ["(sin etiqueta)"]:
            if et == "(sin etiqueta)":
                m = np.array([not anot[i] for i in range(400)])
            else:
                m = np.array([et in anot[i] for i in range(400)])
            if m.sum() == 0:
                continue
            sub = puntos[m]
            fila = {"modelo": modelo, "etiqueta_diwan": et, "n": int(m.sum())}
            for met in METRICAS:
                fila[met] = float(sub[met].mean())
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_estabilidad(modelos: dict, anot: dict) -> pd.DataFrame:
    """¿Sobrevive el ranking de modelos a cambiar la métrica o limpiar el dataset?

    Se calcula Kendall tau entre el ranking de referencia (group score sobre los
    400) y el ranking bajo cada condición. Tau = 1 significa orden idéntico;
    valores bajos o negativos significan que el ranking publicado depende de una
    elección metodológica y no del mérito de los modelos.
    """
    from scipy.stats import kendalltau

    puntos = {m: pd.DataFrame(per_example_scores(s)) for m, s in modelos.items()}
    nombres = list(modelos)
    if len(nombres) < 3:
        return pd.DataFrame([{"aviso": "se necesitan >=3 modelos para un tau informativo"}])

    def ranking(met: str, nivel: str) -> list[float]:
        m = mascara(anot, NIVELES[nivel])
        return [float(puntos[n][m][met].mean()) for n in nombres]

    ref = ranking("group", "todos")
    filas = []
    for met, nivel in itertools.product(METRICAS, NIVELES):
        if (met, nivel) == ("group", "todos"):
            continue
        act = ranking(met, nivel)
        tau, p = kendalltau(ref, act)
        filas.append({
            "metrica": met, "nivel": nivel,
            "kendall_tau": float(tau), "p_value": float(p),
            "ranking": " > ".join(
                n for _, n in sorted(zip(act, nombres), reverse=True)
            ),
        })
    return pd.DataFrame(filas)


def main() -> None:
    modelos = cargar_modelos()
    anot = cargar_diwan()
    print(f"[diwan] {len(modelos)} modelos, 400 anotaciones")
    for nivel, excl in NIVELES.items():
        print(f"  nivel {nivel:16s} n={int(mascara(anot, excl).sum()):3d}"
              f"  (excluye: {', '.join(excl) or 'nada'})")

    fac = tabla_factorial(modelos, anot)
    eti = tabla_por_etiqueta(modelos, anot)
    est = tabla_estabilidad(modelos, anot)
    for nombre, df in [("diwan_factorial.csv", fac),
                       ("diwan_por_etiqueta.csv", eti),
                       ("diwan_estabilidad.csv", est)]:
        df.to_csv(MET / nombre, index=False)
        print(f"[diwan] {nombre:26s} {len(df):4d} filas")

    print("\n=== Efecto de limpiar el dataset (group score) ===")
    g = fac[fac["metrica"] == "group"]
    for modelo in g["modelo"].unique():
        s = g[g["modelo"] == modelo].set_index("nivel")
        partes = [f"{niv}={s.loc[niv, 'score']:.4f}(n={int(s.loc[niv, 'n'])})"
                  for niv in NIVELES]
        print(f"  {modelo:34s} " + "  ".join(partes))

    print("\n=== Estabilidad del ranking frente al group score sobre los 400 ===")
    if "kendall_tau" in est.columns:
        for _, r in est[est["nivel"] == "todos"].iterrows():
            print(f"  cambiar métrica a {r['metrica']:12s} tau={r['kendall_tau']:+.3f}"
                  f"  ranking: {r['ranking']}")
        sub = est[(est["metrica"] == "group") & (est["nivel"] != "todos")]
        for _, r in sub.iterrows():
            print(f"  limpiar a {r['nivel']:16s}    tau={r['kendall_tau']:+.3f}"
                  f"  ranking: {r['ranking']}")

    n_limpio = int(mascara(anot, NIVELES["sin_confusores"]).sum())
    md = minimum_resolvable_difference(n=n_limpio, p=0.08)
    print(f"\n=== Potencia sobre el subconjunto limpio (n={n_limpio}) ===")
    print(f"  diferencia mínima detectable en group: "
          f"{md['dif_minima_detectable'] * 100:.1f} puntos "
          f"(frente a 5.3 con los 400)")


if __name__ == "__main__":
    main()
