"""Demo en vivo para la exposición — CLIP sobre pares mínimos de Winoground.

Muestra, en segundos y offline (embeddings/checkpoint cacheados), la tesis del
avance: CLIP asigna similitudes casi iguales a ambos captions de un par mínimo, por
lo que **falla la composición (group)** aunque el retrieval sea alto. Reutiliza el
motor y el scorer ya validados del repositorio.

Uso:
    python scripts/demo_winoground.py            # 5 pares ilustrativos
    python scripts/demo_winoground.py --n 8      # más ejemplos
    python scripts/demo_winoground.py --retrieval  # + mini-demo de recuperación
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.openclip_utils import create_model, encode_images, encode_texts, get_device
from src.winoground_data import load_dataset
from src.winoground_eval import text_correct, image_correct, group_correct

BOLD, DIM, GRN, RED, YEL, RST = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def tick(ok: bool) -> str:
    return f"{GRN}✓{RST}" if ok else f"{RED}✗{RST}"


def show_pair(ex, model, preprocess, tokenizer, device) -> tuple[bool, bool, bool]:
    imgs = encode_images(model, preprocess, [ex.image_0, ex.image_1], device)
    txts = encode_texts(model, tokenizer, [ex.caption_0, ex.caption_1], device)
    # fila = caption, columna = imagen
    sim = txts @ imgs.T  # 2x2
    t_ok, i_ok, g_ok = text_correct(sim), image_correct(sim), group_correct(sim)
    print(f"\n{BOLD}[{ex.id}] tag={ex.tag}{RST}")
    print(f"  C0: {ex.caption_0}")
    print(f"  C1: {ex.caption_1}")
    print(f"  {DIM}similitud coseno (fila=caption, col=imagen){RST}")
    print(f"            img0     img1")
    print(f"    C0    {sim[0,0]:+.3f}   {sim[0,1]:+.3f}")
    print(f"    C1    {sim[1,0]:+.3f}   {sim[1,1]:+.3f}")
    diff = abs(sim[0, 0] - sim[0, 1])
    print(f"  {DIM}Δ(C0·img0 − C0·img1) = {diff:.3f}  (pequeño ⇒ CLIP casi no distingue){RST}")
    print(f"  text {tick(t_ok)}   image {tick(i_ok)}   group {tick(g_ok)}")
    return t_ok, i_ok, g_ok


def retrieval_demo(examples, model, preprocess, tokenizer, device, k=3):
    print(f"\n{BOLD}== Mini-demo de recuperación (retrieval) =={RST}")
    gallery_imgs, labels = [], []
    for ex in examples:
        gallery_imgs += [ex.image_0, ex.image_1]
        labels += [ex.caption_0, ex.caption_1]
    img_emb = encode_images(model, preprocess, gallery_imgs, device)
    query = labels[0]
    q = encode_texts(model, tokenizer, [query], device)[0]
    scores = img_emb @ q
    order = np.argsort(-scores)[:k]
    print(f"  query: \"{query}\"")
    for rank, idx in enumerate(order, 1):
        hit = "  <- correcta" if idx == 0 else ""
        print(f"    top{rank}: img#{idx}  sim={scores[idx]:+.3f}  «{labels[idx][:52]}»{hit}")
    print(f"  {DIM}El retrieval funciona; la dificultad está en el par mínimo (group).{RST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="número de pares mínimos a mostrar")
    ap.add_argument("--retrieval", action="store_true", help="añade mini-demo de recuperación")
    args = ap.parse_args()

    print(f"{BOLD}Demo — CLIP en pares mínimos de Winoground{RST}")
    print(f"{DIM}Cargando modelo (ViT-B-32/laion2b) y datos…{RST}")
    device = get_device()
    model, preprocess, tokenizer, device = create_model(
        "ViT-B-32", "laion2b_s34b_b79k", device=device)
    examples, source = load_dataset(prefer_real=True)
    print(f"{DIM}Fuente de datos: {source} · {len(examples)} ejemplos · device={device}{RST}")

    subset = examples[: args.n]
    t = i = g = 0
    for ex in subset:
        to, io, go = show_pair(ex, model, preprocess, tokenizer, device)
        t += to; i += io; g += go
    n = len(subset)
    print(f"\n{BOLD}Resumen sobre {n} pares:{RST} "
          f"text {t}/{n} ({t/n:.0%}) · image {i}/{n} ({i/n:.0%}) · "
          f"{YEL}group {g}/{n} ({g/n:.0%}){RST}")
    print(f"{DIM}Azar: text/image=25%, group≈17%. Humano group≈85%.{RST}")
    print(f"{BOLD}Conclusión:{RST} similitudes casi iguales por caption ⇒ "
          f"retrieval alto pero composición ≈ azar.")

    if args.retrieval:
        retrieval_demo(subset, model, preprocess, tokenizer, device)


if __name__ == "__main__":
    main()
