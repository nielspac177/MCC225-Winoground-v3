"""Figuras del informe final. Solo matplotlib sobre los CSV de outputs/metrics/.

No importa torch ni faiss, así que corre en segundos. Mantiene la paleta
Okabe-Ito (segura para daltonismo) ya establecida en scripts/03_make_figures.py.

Salidas en outputs/figures/:
  informe_fig1_metricas.png   scores con IC 95 % frente al azar y al humano
  informe_fig2_tesis.png      retrieval vs composición, y márgenes de decisión

Uso:
    python scripts/16_figuras_informe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MET = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"

# Paleta Okabe-Ito, idéntica a la de scripts/03_make_figures.py.
C_TEXT, C_IMAGE, C_GROUP, C_GM = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_CHANCE, C_HUMAN = "#999999", "#D55E00"

plt.rcParams.update({
    "figure.dpi": 300, "font.size": 9, "axes.grid": True,
    "axes.axisbelow": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False,
})

METRICAS = ["text", "image", "group", "group_match"]
ETIQ = {"text": "text", "image": "image", "group": "group", "group_match": "GroupMatch"}
COLOR = {"text": C_TEXT, "image": C_IMAGE, "group": C_GROUP, "group_match": C_GM}
AZAR = {"text": 0.25, "image": 0.25, "group": 1 / 6, "group_match": 0.5}
HUMANO = {"text": 0.895, "image": 0.885, "group": 0.855}


def fig1_metricas() -> None:
    """El resultado central: la métrica decide el veredicto.

    Cada panel es una métrica con su PROPIO nivel de azar, porque group score y
    GroupMatch tienen azares distintos (1/6 y 1/2) y superponerlos en un solo eje
    invitaría a compararlos directamente, que es justo el error a evitar.
    """
    df = pd.read_csv(MET / "audit_modelos.csv")
    modelos = df["modelo"].unique().tolist()

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, met in zip(axes, METRICAS):
        sub = df[df["metrica"] == met].set_index("modelo").loc[modelos]
        x = np.arange(len(modelos))
        yerr = np.vstack([sub["score"] - sub["ic_lo"], sub["ic_hi"] - sub["score"]])
        ax.bar(x, sub["score"], color=COLOR[met], width=0.62,
               yerr=yerr, capsize=4, ecolor="#333333", error_kw={"lw": 1.1})
        ax.axhline(AZAR[met], ls="--", lw=1.4, color=C_CHANCE,
                   label=f"azar = {AZAR[met]:.3f}")
        if met in HUMANO:
            ax.axhline(HUMANO[met], ls=":", lw=1.4, color=C_HUMAN,
                       label=f"humano = {HUMANO[met]:.3f}")
        # La etiqueta va por encima del extremo SUPERIOR del intervalo, no de la
        # barra: si no, se solapa con el bigote cuando el IC es ancho.
        for xi, v, hi in zip(x, sub["score"], sub["ic_hi"]):
            ax.text(xi, hi + 0.03, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(f"{chr(65 + METRICAS.index(met))}. {ETIQ[met]}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("CLIP ", "").replace(" (", "\n(") for m in modelos],
                           fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    axes[0].set_ylabel("score (IC 95 % bootstrap)")
    fig.suptitle(
        "Winoground, 400 pares: el veredicto depende de la métrica. "
        "group = 0.075 (bajo el azar) vs GroupMatch = 0.675 (sobre el azar)",
        fontsize=9.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG / "informe_fig1_metricas.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] informe_fig1_metricas.png")


def fig2_tesis() -> None:
    """Dos paneles: la tesis del proyecto, y por qué el score es frágil."""
    fig, (izq, der) = plt.subplots(1, 2, figsize=(10, 3.6))

    # --- Panel A: recuperación alta contra composición nula ---------------- #
    rec = json.loads((MET / "recall_vs_group.json").read_text())
    i2t, t2i = rec["image_to_text_recall"], rec["text_to_image_recall"]
    etiquetas = ["R@1", "R@5", "R@10", "group\nscore"]
    valores = [i2t["R@1"], i2t["R@5"], i2t["R@10"], rec["winoground_group_score"]]
    colores = [C_TEXT] * 3 + [C_GROUP]
    barras = izq.bar(etiquetas, valores, color=colores, width=0.62)
    for b, v in zip(barras, valores):
        izq.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=8.5, fontweight="bold")
    # La línea de azar se dibuja SOLO sobre la barra del group score. Trazarla a
    # lo ancho del panel invitaría a leerla como referencia común, pero el azar
    # de R@K sobre una galería de 800 es ~1/800, no 1/6.
    izq.plot([2.62, 3.38], [1 / 6, 1 / 6], ls="--", lw=1.6, color=C_CHANCE,
             label="azar del group score (1/6)")
    izq.set_ylim(0, 1.0)
    izq.set_ylabel("acierto")
    izq.set_title("A. Recuperar no es componer\n(imagen a texto, galería de 800)",
                  fontsize=9, fontweight="bold")
    izq.legend(fontsize=7.5)

    # --- Panel B: distribución del margen que decide cada comparación ------ #
    z = np.load(ROOT / "outputs/sims/winoground_real__ViT-B-32__laion2b_s34b_b79k.npz",
                allow_pickle=True)
    s = z["sims"]
    margenes = np.concatenate([
        np.abs(s[:, 0, 0] - s[:, 1, 0]), np.abs(s[:, 1, 1] - s[:, 0, 1]),
        np.abs(s[:, 0, 0] - s[:, 0, 1]), np.abs(s[:, 1, 1] - s[:, 1, 0]),
    ])
    der.hist(margenes, bins=60, color=C_IMAGE, edgecolor="white", linewidth=0.4)
    mediana = float(np.median(margenes))
    der.axvline(mediana, ls="--", lw=1.5, color="#333333",
                label=f"mediana = {mediana:.4f}")
    der.axvline(float(s.std()), ls=":", lw=1.5, color=C_HUMAN,
                label=f"desv. de las similitudes = {s.std():.4f}")
    der.set_xlim(0, 0.10)
    der.set_xlabel("margen que decide cada comparación")
    der.set_ylabel("frecuencia")
    der.set_title("B. No hay empates, pero el margen es diminuto\n"
                  "(1600 comparaciones; cero empates exactos)",
                  fontsize=9, fontweight="bold")
    der.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIG / "informe_fig2_tesis.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] informe_fig2_tesis.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_metricas()
    fig2_tesis()


if __name__ == "__main__":
    main()
