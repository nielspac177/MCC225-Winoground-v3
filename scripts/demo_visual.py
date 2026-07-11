"""Demo VISUAL para la exposición — paneles con imágenes reales + heatmap 2x2.

Para cada par mínimo de Winoground genera un panel que muestra las dos imágenes
reales, sus dos captions, el mapa de calor 2x2 de similitud coseno (caption x
imagen) y los veredictos text/image/group con ✓/✗. Guarda un PNG por ejemplo en
outputs/demo/ y ensambla un GIF animado que cicla los ejemplos. Todo offline.

Uso:
    python scripts/demo_visual.py            # 6 ejemplos -> PNGs + GIF
    python scripts/demo_visual.py --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.openclip_utils import create_model, encode_images, encode_texts, get_device
from src.winoground_data import load_dataset
from src.winoground_eval import text_correct, image_correct, group_correct

OUT = ROOT / "outputs" / "demo"
GREEN, RED = "#2ca02c", "#d62728"


def wrap(s: str, width: int = 34) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def badge(ax, x, ok, label):
    ax.text(x, 0.5, f"{label}\n{'✓' if ok else '✗'}", ha="center", va="center",
            fontsize=15, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc=GREEN if ok else RED, ec="none"),
            transform=ax.transAxes)


def panel(ex, sim, t_ok, i_ok, g_ok, idx, total, tally):
    fig = plt.figure(figsize=(11, 5.2), dpi=120)
    fig.suptitle(f"Par mínimo Winoground  ·  tag={ex.tag}  ·  ejemplo {idx}/{total}",
                 fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.15], width_ratios=[1, 1, 1.1],
                          hspace=0.35, wspace=0.28)

    for col, (img, cap) in enumerate([(ex.image_0, ex.caption_0), (ex.image_1, ex.caption_1)]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img.convert("RGB")); ax.axis("off")
        ax.set_title(f"imagen {col}", fontsize=10, color="#444")
        ax.text(0.5, -0.06, wrap(cap), transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#111")

    axh = fig.add_subplot(gs[0, 2])
    im = axh.imshow(sim, cmap="viridis", vmin=sim.min() - 0.02, vmax=sim.max() + 0.02)
    axh.set_xticks([0, 1], ["img 0", "img 1"]); axh.set_yticks([0, 1], ["cap 0", "cap 1"])
    axh.set_title("similitud coseno", fontsize=10)
    for r in range(2):
        for c in range(2):
            best = (c == np.argmax(sim[r]))
            axh.text(c, r, f"{sim[r, c]:.3f}", ha="center", va="center",
                     color="white" if not best else "yellow",
                     fontweight="bold" if best else "normal", fontsize=11)
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)

    for col, (ok, lab) in enumerate([(t_ok, "text"), (i_ok, "image"), (g_ok, "group")]):
        axb = fig.add_subplot(gs[1, col]); axb.axis("off")
        badge(axb, 0.5, ok, lab)

    delta = abs(sim[0, 0] - sim[0, 1])
    fig.text(0.5, 0.02,
             f"Δ(cap0·img0 − cap0·img1) = {delta:.3f}  (pequeño ⇒ CLIP casi no distingue)   |   "
             f"acumulado  text {tally[0]}/{idx}  image {tally[1]}/{idx}  group {tally[2]}/{idx}",
             ha="center", fontsize=9, color="#333")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"panel_{idx:02d}.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    print("Cargando modelo y datos (offline si hay caché)…", flush=True)
    device = get_device()
    model, preprocess, tokenizer, device = create_model("ViT-B-32", "laion2b_s34b_b79k", device=device)
    examples, source = load_dataset(prefer_real=True)
    subset = examples[: args.n]
    print(f"Fuente: {source} · {len(subset)} ejemplos · device={device}", flush=True)

    paths, tally = [], [0, 0, 0]
    for k, ex in enumerate(subset, 1):
        imgs = encode_images(model, preprocess, [ex.image_0, ex.image_1], device)
        txts = encode_texts(model, tokenizer, [ex.caption_0, ex.caption_1], device)
        sim = txts @ imgs.T
        t_ok, i_ok, g_ok = text_correct(sim), image_correct(sim), group_correct(sim)
        tally[0] += t_ok; tally[1] += i_ok; tally[2] += g_ok
        paths.append(panel(ex, sim, t_ok, i_ok, g_ok, k, len(subset), tally))
        print(f"  panel {k}/{len(subset)}  text={t_ok} image={i_ok} group={g_ok}", flush=True)

    # GIF animado
    frames = [Image.open(p).convert("RGB") for p in paths]
    w = min(f.width for f in frames); frames = [f.resize((w, int(f.height * w / f.width))) for f in frames]
    gif = OUT / "demo_winoground.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=2200, loop=0)
    n = len(subset)
    print(f"\nListo. {n} paneles en {OUT}/  y  GIF -> {gif}")
    print(f"Resumen: text {tally[0]}/{n} · image {tally[1]}/{n} · group {tally[2]}/{n} "
          f"(azar group≈17%, humano≈85%).")


if __name__ == "__main__":
    main()
