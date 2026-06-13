#!/usr/bin/env python3
"""Genera figuras publicables a partir de outputs/metrics/.

Figuras (outputs/figures/):
  scores_vs_chance.png       3 scores con IC bootstrap vs azar vs humano
  recall_vs_group.png        Recall@K (alto) vs group score (≈azar) — la tesis
  by_tag.png                 scores por tag
  blindness.png              real vs imágenes permutadas
  checkpoint_comparison.png  group score por checkpoint
  qualitative_examples.png   pares de imágenes con group=0
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

# Paleta colorblind-safe (Okabe-Ito)
C_TEXT, C_IMAGE, C_GROUP = "#0072B2", "#E69F00", "#009E73"
C_CHANCE, C_HUMAN = "#999999", "#D55E00"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.3})


def fig_scores_vs_chance():
    ci = pd.read_csv(MET / "bootstrap_ci.csv").set_index("metric")
    scores = json.loads((MET / "scores.json").read_text())
    human = scores["human_reference"]
    metrics = ["text", "image", "group"]
    means = [ci.loc[m, "mean"] for m in metrics]
    los = [ci.loc[m, "mean"] - ci.loc[m, "lo"] for m in metrics]
    his = [ci.loc[m, "hi"] - ci.loc[m, "mean"] for m in metrics]
    chance = [0.25, 0.25, 1 / 6]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    ax.bar(x, means, yerr=[los, his], capsize=6,
           color=[C_TEXT, C_IMAGE, C_GROUP], alpha=0.9, width=0.6,
           label="OpenCLIP (IC 95%)")
    for xi, ch in zip(x, chance):
        ax.hlines(ch, xi - 0.32, xi + 0.32, color=C_CHANCE, ls="--", lw=2)
    for xi, m in zip(x, metrics):
        ax.hlines(human[m], xi - 0.32, xi + 0.32, color=C_HUMAN, ls=":", lw=2)
    ax.plot([], [], ls="--", color=C_CHANCE, label="Azar")
    ax.plot([], [], ls=":", color=C_HUMAN, label="Humano (paper)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Text score", "Image score", "Group score"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Winoground · {scores['primary_checkpoint']} · fuente={scores['source']}")
    ax.legend(loc="upper right", framealpha=0.95)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.02, f"{m:.2f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "scores_vs_chance.png")
    plt.close(fig)


def fig_recall_vs_group():
    rg = json.loads((MET / "recall_vs_group.json").read_text())
    t2i = rg["text_to_image_recall"]
    ks = [k for k in t2i if k.startswith("R@")]
    vals = [t2i[k] for k in ks]
    labels = ks + ["Group\nscore", "Azar\ngroup"]
    heights = vals + [rg["winoground_group_score"], rg["chance_group"]]
    colors = [C_TEXT] * len(ks) + [C_GROUP, C_CHANCE]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, heights, color=colors, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Métrica")
    ax.set_title("Retrieval alto ≠ razonamiento composicional\n(R@K usa toda la galería; group score exige el par mínimo)")
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "recall_vs_group.png")
    plt.close(fig)


def fig_by_tag():
    df = pd.read_csv(MET / "by_tag.csv")
    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w, df["text"], w, label="text", color=C_TEXT)
    ax.bar(x, df["image"], w, label="image", color=C_IMAGE)
    ax.bar(x + w, df["group"], w, label="group", color=C_GROUP)
    ax.axhline(1 / 6, ls="--", color=C_CHANCE, lw=1.5, label="azar group")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(df["tag"], df["n"])])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Scores por tipo de cambio composicional (tag)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "by_tag.png")
    plt.close(fig)


def fig_blindness():
    b = json.loads((MET / "blindness.json").read_text())
    metrics = ["text_score", "image_score", "group_score"]
    real = [b["real"][m] for m in metrics]
    perm = [b["permuted_images"][m] for m in metrics]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, real, w, label="imágenes reales", color=C_GROUP)
    ax.bar(x + w / 2, perm, w, label="imágenes permutadas (control)", color=C_CHANCE)
    ax.set_xticks(x)
    ax.set_xticklabels(["text", "image", "group"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Prueba de ceguera: ¿el modelo usa la imagen?\n(real ≫ permutado ⇒ sí usa el contenido visual)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "blindness.png")
    plt.close(fig)


def fig_checkpoint_comparison():
    df = pd.read_csv(MET / "checkpoint_comparison.csv")
    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, df["text_score"], w, label="text", color=C_TEXT)
    ax.bar(x, df["image_score"], w, label="image", color=C_IMAGE)
    ax.bar(x + w, df["group_score"], w, label="group", color=C_GROUP)
    ax.axhline(1 / 6, ls="--", color=C_CHANCE, lw=1.5, label="azar group")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparación de checkpoints OpenCLIP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "checkpoint_comparison.png")
    plt.close(fig)


def fig_qualitative():
    from src.winoground_data import load_dataset
    cases = json.loads((MET / "failure_cases.json").read_text())
    if not cases:
        return
    examples, _ = load_dataset(prefer_real=json.loads((MET / "scores.json").read_text())["source"] == "winoground_real")
    by_id = {e.id: e for e in examples}
    cases = [c for c in cases if c["id"] in by_id][:3]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 2, figsize=(7, 3.3 * len(cases)))
    if len(cases) == 1:
        axes = axes.reshape(1, 2)
    for row, c in enumerate(cases):
        ex = by_id[c["id"]]
        for col, (im, cap) in enumerate([(ex.image_0, ex.caption_0), (ex.image_1, ex.caption_1)]):
            ax = axes[row, col]
            ax.imshow(im)
            ax.set_title(f"img{col} · «{cap}»", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(f"{c['id']}\nfalló: {c['failed']}", fontsize=8)
    fig.suptitle("Casos de error (group=0): pares mínimos que el modelo confunde", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "qualitative_examples.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_scores_vs_chance()
    fig_recall_vs_group()
    fig_by_tag()
    fig_blindness()
    fig_checkpoint_comparison()
    fig_qualitative()
    print(f"[figuras] generadas en {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
