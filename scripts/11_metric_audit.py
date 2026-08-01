"""Auditoría de la métrica de Winoground: empates, márgenes, IC y potencia.

Consume la caché de matrices 2x2 producida por `10_export_sims.py` y
`12_run_blip.py`. No importa torch ni faiss: todo el análisis es numpy sobre
cuatro números por ejemplo, y corre en segundos.

Cubre cuatro exigencias del Examen Final:
  §11.2.2  intervalos de confianza y tratamiento de empates
  §11.4.7  cómo trata el scorer los empates y qué alternativa sería razonable
  §11.5.3  bootstrap estratificado por tag
  §11.4.9  qué variable permanece confundida al comparar checkpoints

Salidas en outputs/metrics/:
  audit_modelos.csv    scores + IC simple y estratificado por modelo y métrica
  audit_empates.csv    barrido de tolerancia: cuántas decisiones son casi-empate
  audit_margenes.csv   percentiles del margen de decisión
  audit_por_tag.csv    desglose modelo x tag x métrica
  audit_pareado.csv    McNemar exacto + IC bootstrap pareado entre modelos
  audit_potencia.csv   diferencia mínima detectable con n=400

Uso:
    python scripts/11_metric_audit.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    bootstrap_ci,
    mcnemar_exact,
    minimum_resolvable_difference,
    paired_bootstrap_diff,
    stratified_bootstrap_ci,
)
from src.winoground_eval import (  # noqa: E402
    aggregate,
    group_match_margin,
    per_example_scores,
    tie_report,
)

SIMS = ROOT / "outputs" / "sims"
OUT = ROOT / "outputs" / "metrics"
META = ROOT / "data" / "winoground_meta.csv"

RONDAS = 2000
SEMILLA = 42

METRICAS = ["text", "image", "group", "group_match"]
AZAR = {"text": 0.25, "image": 0.25, "group": 1 / 6, "group_match": 0.5}
# Referencia humana del paper original. No existe para GroupMatch: el paper
# publicó acuerdo humano bajo el scorer oficial, y no las cuatro similitudes
# por ejemplo, así que ese valor NO puede recomputarse bajo la métrica nueva.
HUMANO = {"text": 0.895, "image": 0.885, "group": 0.855, "group_match": np.nan}

# Orden canónico de presentación: CLIP de menor a mayor, luego BLIP.
ORDEN = [
    "winoground_real__ViT-B-32__laion2b_s34b_b79k",
    "winoground_real__ViT-B-16__datacomp_xl_s13b_b90k",
    "winoground_real__ViT-L-14-quickgelu__openai",
    "winoground_real__BLIP-base__itc",
    "winoground_real__BLIP-base__itm",
]

TOLERANCIAS = [0.0, 1e-6, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
PERCENTILES = [1, 5, 10, 25, 50, 75, 90]


def cargar_modelos() -> dict:
    """Lee la caché de sims respetando el orden canónico."""
    encontrados = {p.stem: p for p in SIMS.glob("*.npz")}
    claves = [k for k in ORDEN if k in encontrados]
    claves += sorted(k for k in encontrados if k not in ORDEN)
    modelos = {}
    for k in claves:
        z = np.load(encontrados[k], allow_pickle=True)
        modelos[k] = {"sims": z["sims"], "etiqueta": str(z["etiqueta"])}
    if not modelos:
        raise SystemExit(
            f"No hay matrices en {SIMS}. Ejecuta antes scripts/10_export_sims.py"
        )
    return modelos


def cargar_tags() -> np.ndarray | None:
    if not META.exists():
        return None
    return pd.read_csv(META)["collapsed_tag"].fillna("(sin_tag)").to_numpy()


def tabla_modelos(modelos: dict, tags: np.ndarray | None) -> pd.DataFrame:
    filas = []
    for clave, m in modelos.items():
        puntos = per_example_scores(m["sims"])
        for met in METRICAS:
            v = [p[met] for p in puntos]
            ci = bootstrap_ci(v, rounds=RONDAS, seed=SEMILLA)
            fila = {
                "modelo": m["etiqueta"],
                "clave": clave,
                "metrica": met,
                "score": ci["mean"],
                "ic_lo": ci["lo"],
                "ic_hi": ci["hi"],
                "azar": AZAR[met],
                "humano": HUMANO[met],
                "supera_azar": bool(ci["lo"] > AZAR[met]),
            }
            if tags is not None:
                est = stratified_bootstrap_ci(v, tags, rounds=RONDAS, seed=SEMILLA)
                fila["ic_estrat_lo"] = est["lo"]
                fila["ic_estrat_hi"] = est["hi"]
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_empates(modelos: dict) -> pd.DataFrame:
    filas = []
    for m in modelos.values():
        for atol in TOLERANCIAS:
            r = tie_report(m["sims"], atol=atol)
            s = aggregate(m["sims"], atol=atol, tie_policy="fail")
            s_pass = aggregate(m["sims"], atol=atol, tie_policy="pass")
            filas.append({
                "modelo": m["etiqueta"], "atol": atol,
                "empates_text": r["text_ties"], "empates_image": r["image_ties"],
                "empates_group": r["group_ties"],
                "empates_group_match": r["group_match_ties"],
                # Cota inferior y superior del score verdadero bajo esa tolerancia.
                "group_cota_inf": s.group, "group_cota_sup": s_pass.group,
            })
    return pd.DataFrame(filas)


def tabla_margenes(modelos: dict) -> pd.DataFrame:
    """Distribución del margen que decide cada comparación.

    Es la respuesta sustantiva a "¿los empates importan?". Que no haya empates
    EXACTOS no implica que las decisiones sean holgadas: si la mitad se decide
    por milésimas, el score es frágil ante cualquier cambio de preprocesado.
    """
    filas = []
    for m in modelos.values():
        s = m["sims"]
        # Las cuatro comparaciones que usan text_score e image_score.
        margenes = np.concatenate([
            np.abs(s[:, 0, 0] - s[:, 1, 0]), np.abs(s[:, 1, 1] - s[:, 0, 1]),
            np.abs(s[:, 0, 0] - s[:, 0, 1]), np.abs(s[:, 1, 1] - s[:, 1, 0]),
        ])
        gm = np.array([abs(group_match_margin(x)) for x in s])
        fila = {
            "modelo": m["etiqueta"],
            "sim_min": float(s.min()), "sim_max": float(s.max()),
            "sim_desv": float(s.std()),
            "margen_medio": float(margenes.mean()),
            # Cuán pequeño es el margen típico frente a la escala de las
            # similitudes: si es <<1, las decisiones son marginales.
            "margen_mediano_sobre_desv": float(np.median(margenes) / s.std()),
            "margen_gm_mediano": float(np.median(gm)),
        }
        for q in PERCENTILES:
            fila[f"p{q}"] = float(np.percentile(margenes, q))
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_por_tag(modelos: dict, tags: np.ndarray) -> pd.DataFrame:
    filas = []
    for m in modelos.values():
        puntos = per_example_scores(m["sims"])
        df = pd.DataFrame(puntos)
        df["tag"] = tags
        for tag, g in df.groupby("tag"):
            fila = {"modelo": m["etiqueta"], "tag": tag, "n": len(g)}
            for met in METRICAS:
                fila[met] = float(g[met].mean())
            filas.append(fila)
    return pd.DataFrame(filas).sort_values(["modelo", "tag"])


def tabla_pareada(modelos: dict) -> pd.DataFrame:
    """Compara cada par de modelos sobre LOS MISMOS 400 ejemplos."""
    puntos = {k: per_example_scores(m["sims"]) for k, m in modelos.items()}
    filas = []
    for a, b in itertools.combinations(modelos, 2):
        for met in METRICAS:
            va = [p[met] for p in puntos[a]]
            vb = [p[met] for p in puntos[b]]
            mc = mcnemar_exact(va, vb)
            pb = paired_bootstrap_diff(va, vb, rounds=RONDAS, seed=SEMILLA)
            filas.append({
                "modelo_a": modelos[a]["etiqueta"],
                "modelo_b": modelos[b]["etiqueta"],
                "metrica": met,
                "score_a": float(np.mean(va)), "score_b": float(np.mean(vb)),
                "diferencia": pb["dif"], "dif_ic_lo": pb["lo"], "dif_ic_hi": pb["hi"],
                "discordantes": mc["discordantes"],
                "p_mcnemar": mc["p_value"],
                # El IC que cruza cero y un p alto dicen lo mismo por dos vías.
                "distinguibles": bool(pb["lo"] > 0 or pb["hi"] < 0),
            })
    return pd.DataFrame(filas)


def tabla_potencia(modelos: dict) -> pd.DataFrame:
    """Diferencia mínima detectable, anclada en los scores realmente observados."""
    filas = []
    for met in METRICAS:
        base = float(np.mean([
            aggregate(m["sims"]).as_dict()[f"{met}_score"] for m in modelos.values()
        ]))
        base = min(max(base, 0.01), 0.99)
        r = minimum_resolvable_difference(n=400, p=base)
        filas.append({
            "metrica": met, "score_base_medio": base,
            "n": r["n"], "alpha": r["alpha"], "potencia": r["power"],
            "dif_minima_detectable": r["dif_minima_detectable"],
        })
    return pd.DataFrame(filas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    modelos = cargar_modelos()
    tags = cargar_tags()
    print(f"[audit] {len(modelos)} modelos, tags={'sí' if tags is not None else 'no'}")

    salidas = {
        "audit_modelos.csv": tabla_modelos(modelos, tags),
        "audit_empates.csv": tabla_empates(modelos),
        "audit_margenes.csv": tabla_margenes(modelos),
        "audit_pareado.csv": tabla_pareada(modelos),
        "audit_potencia.csv": tabla_potencia(modelos),
    }
    if tags is not None:
        salidas["audit_por_tag.csv"] = tabla_por_tag(modelos, tags)

    for nombre, df in salidas.items():
        df.to_csv(OUT / nombre, index=False)
        print(f"[audit] {nombre:24s} {len(df):4d} filas")

    # Resumen legible en consola: la tabla que va al informe.
    print("\n=== Scores con IC bootstrap del 95 % ===")
    piv = salidas["audit_modelos.csv"]
    for modelo in piv["modelo"].unique():
        sub = piv[piv["modelo"] == modelo].set_index("metrica")
        partes = [
            f"{met}={sub.loc[met, 'score']:.4f} "
            f"[{sub.loc[met, 'ic_lo']:.3f},{sub.loc[met, 'ic_hi']:.3f}]"
            f"{'*' if sub.loc[met, 'supera_azar'] else ' '}"
            for met in METRICAS
        ]
        print(f"  {modelo:34s} " + "  ".join(partes))
    print("  (* = el IC inferior supera el azar)")

    print("\n=== Diferencia mínima detectable (n=400, alfa=0.05, potencia=0.80) ===")
    for _, r in salidas["audit_potencia.csv"].iterrows():
        print(f"  {r['metrica']:12s} {r['dif_minima_detectable'] * 100:5.1f} puntos "
              f"(score base {r['score_base_medio'] * 100:.1f} %)")


if __name__ == "__main__":
    main()
