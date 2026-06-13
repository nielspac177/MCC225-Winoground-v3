#!/usr/bin/env python3
"""Prepara los datos de evaluación.

1) Intenta cachear el Winoground oficial (gated; requiere licencia aceptada).
2) SIEMPRE (re)genera un set curado offline de pares mínimos composicionales,
   determinista, con imágenes PIL: binding color-objeto y relaciones espaciales.
   Mismo formato/escala que consume el scorer.

Uso:  python scripts/01_prepare_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CURATED = ROOT / "data" / "curated"
IMG_DIR = CURATED / "images"
CANVAS = 224  # cuadrado: sin recorte por el center-crop de CLIP

COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 170, 70),
    "yellow": (235, 200, 40),
}


def _draw_shape(d: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, r: int, color):
    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "star":
        pts = []
        import math
        for k in range(10):
            ang = math.pi / 2 + k * math.pi / 5
            rad = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy - rad * math.sin(ang)))
        d.polygon(pts, fill=color)
    else:
        raise ValueError(shape)


def render(objects) -> Image.Image:
    """objects: lista de dicts {shape, color, cx, cy, r}."""
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    d = ImageDraw.Draw(img)
    for o in objects:
        _draw_shape(d, o["shape"], o["cx"], o["cy"], o["r"], COLORS[o["color"]])
    return img


# Posiciones canónicas
L, R = 64, 160          # izquierda / derecha
TOP, BOT = 64, 160      # arriba / abajo
RAD = 42


def build_examples():
    """Devuelve lista de (id, objs_img0, objs_img1, caption_0, caption_1, tag)."""
    ex = []

    # --- Familia A: relación espacial izquierda/derecha (relation swap) ---
    spatial_lr = [
        ("circle", "square"),
        ("triangle", "circle"),
        ("star", "square"),
        ("square", "triangle"),
    ]
    for a, b in spatial_lr:
        # image_0: a a la izquierda, b a la derecha   (caption_0)
        img0 = [dict(shape=a, color="red", cx=L, cy=112, r=RAD),
                dict(shape=b, color="blue", cx=R, cy=112, r=RAD)]
        # image_1: b a la izquierda, a a la derecha   (caption_1)
        img1 = [dict(shape=b, color="blue", cx=L, cy=112, r=RAD),
                dict(shape=a, color="red", cx=R, cy=112, r=RAD)]
        cap0 = f"the {a} is to the left of the {b}"
        cap1 = f"the {b} is to the left of the {a}"
        ex.append((f"lr_{a}_{b}", img0, img1, cap0, cap1, "relation"))

    # --- Familia B: relación espacial arriba/abajo (relation swap) ---
    spatial_ab = [
        ("circle", "square"),
        ("triangle", "star"),
        ("square", "circle"),
        ("star", "triangle"),
    ]
    for a, b in spatial_ab:
        img0 = [dict(shape=a, color="green", cx=112, cy=TOP, r=RAD),
                dict(shape=b, color="yellow", cx=112, cy=BOT, r=RAD)]
        img1 = [dict(shape=b, color="yellow", cx=112, cy=TOP, r=RAD),
                dict(shape=a, color="green", cx=112, cy=BOT, r=RAD)]
        cap0 = f"the {a} is above the {b}"
        cap1 = f"the {b} is above the {a}"
        ex.append((f"ab_{a}_{b}", img0, img1, cap0, cap1, "relation"))

    # --- Familia C: binding color-objeto (object/attribute swap) ---
    binding = [
        ("circle", "square", "red", "blue"),
        ("triangle", "circle", "green", "yellow"),
        ("star", "square", "blue", "red"),
        ("square", "triangle", "yellow", "green"),
        ("circle", "triangle", "red", "green"),
        ("star", "circle", "blue", "yellow"),
        ("square", "star", "green", "red"),
        ("triangle", "square", "yellow", "blue"),
    ]
    for s1, s2, c1, c2 in binding:
        # image_0: s1 es c1, s2 es c2     (caption_0)
        img0 = [dict(shape=s1, color=c1, cx=L, cy=112, r=RAD),
                dict(shape=s2, color=c2, cx=R, cy=112, r=RAD)]
        # image_1: s1 es c2, s2 es c1     (caption_1) — colores intercambiados
        img1 = [dict(shape=s1, color=c2, cx=L, cy=112, r=RAD),
                dict(shape=s2, color=c1, cx=R, cy=112, r=RAD)]
        cap0 = f"a {c1} {s1} and a {c2} {s2}"
        cap1 = f"a {c2} {s1} and a {c1} {s2}"
        ex.append((f"bind_{s1}_{s2}_{c1}_{c2}", img0, img1, cap0, cap1, "object"))

    return ex


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    manifest_path = CURATED / "examples.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for (eid, o0, o1, c0, c1, tag) in examples:
            p0 = f"images/{eid}_img0.png"
            p1 = f"images/{eid}_img1.png"
            render(o0).save(CURATED / p0)
            render(o1).save(CURATED / p1)
            fh.write(json.dumps({
                "id": eid, "image_0": p0, "image_1": p1,
                "caption_0": c0, "caption_1": c1, "tag": tag,
            }, ensure_ascii=False) + "\n")
    print(f"[curated] {len(examples)} ejemplos -> {manifest_path}")
    by_tag = {}
    for e in examples:
        by_tag[e[5]] = by_tag.get(e[5], 0) + 1
    print(f"[curated] por tag: {by_tag}")

    # Intentar cachear el Winoground real
    print("[real] intentando cachear facebook/winoground ...")
    try:
        from src.winoground_data import load_winoground_real
        real = load_winoground_real()
        print(f"[real] OK: {len(real)} ejemplos cacheados.")
    except Exception as exc:  # noqa: BLE001
        print(f"[real] no disponible ({type(exc).__name__}): "
              f"acepta la licencia en https://huggingface.co/datasets/facebook/winoground")


if __name__ == "__main__":
    main()
