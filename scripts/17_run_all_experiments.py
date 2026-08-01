"""Ejecuta en UN SOLO proceso los dos experimentos que faltan.

En este entorno cada arranque con torch cuesta varios minutos (los imports se
bloquean en `read()` de disco). Agrupar los dos experimentos en un proceso paga
ese coste una vez en lugar de dos.

  Experimento A (§11.2.1, §11.2.3) — Ablación del cross-encoder.
      BLIP con `use_itm_head=False` (ITC, dual-encoder) frente a `True` (ITM,
      atención cruzada real). Mismos pesos, mismo preprocesado, mismos datos:
      la única variable que cambia es la atención cruzada.

  Experimento B (§11.2.5) — Robustez visual frente a sensibilidad semántica.
      Sobre CLIP ViT-B/32. El eje semántico perturba el TEXTO preservando el
      vocabulario (negación, relación espacial, cuantificador) y mide si el
      modelo sigue prefiriendo el caption original. El eje visual degrada la
      IMAGEN preservando el contenido (grises, recorte, baja resolución,
      desenfoque) y mide cuánto se mueve el puntaje.

      La tesis a contrastar: un modelo composicional debería ser INSENSIBLE a la
      degradación visual y MUY SENSIBLE al cambio semántico. Si ocurre lo
      contrario, está reconociendo vocabulario y no estructura.

Uso:
    python scripts/17_run_all_experiments.py [--limite N] [--saltar-blip]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIMS = ROOT / "outputs" / "sims"
MET = ROOT / "outputs" / "metrics"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Experimento A — cross-encoder                                                 #
# --------------------------------------------------------------------------- #
def experimento_blip(ejemplos) -> None:
    from src import blip_utils as bu
    from src.winoground_eval import aggregate

    for use_itm, sufijo, etiqueta in [
        (False, "itc", "BLIP-base ITC (dual-encoder)"),
        (True, "itm", "BLIP-base ITM (cross-attention)"),
    ]:
        t = time.time()
        log(f"A. {etiqueta}")
        sims, n_params = bu.winoground_sims(ejemplos, use_itm_head=use_itm, verbose=False)
        s = aggregate(sims)
        seg = time.time() - t
        destino = SIMS / f"winoground_real__BLIP-base__{sufijo}.npz"
        np.savez_compressed(destino, sims=sims, etiqueta=etiqueta,
                            n_params=np.int64(n_params), segundos=np.float64(seg))
        log(f"   text={s.text:.4f} image={s.image:.4f} group={s.group:.4f} "
            f"group_match={s.group_match:.4f}  ({seg:.0f}s, {n_params / 1e6:.0f}M params)")


# --------------------------------------------------------------------------- #
# Experimento B — perturbaciones                                                #
# --------------------------------------------------------------------------- #
def experimento_perturbaciones(ejemplos) -> None:
    import torch

    from src import openclip_utils as oc
    from src.perturbations import (
        SEMANTIC_PERTURBATIONS,
        VISUAL_PERTURBATIONS,
        perturbation_coverage,
    )
    from src.winoground_eval import aggregate

    log("B. cargando CLIP ViT-B/32 (laion2b)")
    modelo, preproc, tok, device = oc.create_model("ViT-B-32", "laion2b_s34b_b79k")

    captions = [c for ej in ejemplos for c in (ej.caption_0, ej.caption_1)]
    cobertura = perturbation_coverage(captions)
    log("   cobertura de las perturbaciones semánticas:")
    for r in cobertura:
        log(f"     {r['perturbacion']:14s} {r['aplicables']:3d}/{r['total']} "
            f"({r['cobertura'] * 100:.1f} %)")

    # ---- Eje semántico -------------------------------------------------- #
    # Para cada par (imagen, caption correcto) se genera el caption perturbado
    # y se pregunta si el modelo sigue prefiriendo el original. Azar = 0.5.
    filas_sem = []
    for nombre, fn in SEMANTIC_PERTURBATIONS.items():
        imgs, orig, pert = [], [], []
        for ej in ejemplos:
            for img, cap in ((ej.image_0, ej.caption_0), (ej.image_1, ej.caption_1)):
                p = fn(cap)
                if p is not None:
                    imgs.append(img); orig.append(cap); pert.append(p)
        if not imgs:
            continue
        fi = oc.encode_images(modelo, preproc, imgs, device)
        fo = oc.encode_texts(modelo, tok, orig, device)
        fp = oc.encode_texts(modelo, tok, pert, device)
        s_o = np.sum(fi * fo, axis=1)
        s_p = np.sum(fi * fp, axis=1)
        filas_sem.append({
            "eje": "semantico", "perturbacion": nombre, "n": len(imgs),
            "preferencia_original": float(np.mean(s_o > s_p)),
            "azar": 0.5,
            "delta_medio": float(np.mean(s_o - s_p)),
            "delta_abs_medio": float(np.mean(np.abs(s_o - s_p))),
        })
        log(f"   semántico/{nombre:14s} n={len(imgs):3d} "
            f"prefiere el original {filas_sem[-1]['preferencia_original']:.3f} "
            f"(azar 0.500)  Δ={filas_sem[-1]['delta_medio']:+.4f}")

    # ---- Eje visual ------------------------------------------------------ #
    # Se recalcula la matriz 2x2 completa con las imágenes degradadas. Si el
    # modelo entiende la escena, los scores no deberían moverse mucho.
    caps = [c for ej in ejemplos for c in (ej.caption_0, ej.caption_1)]
    f_cap = oc.encode_texts(modelo, tok, caps, device)
    imgs_orig = [im for ej in ejemplos for im in (ej.image_0, ej.image_1)]
    f_img_orig = oc.encode_images(modelo, preproc, imgs_orig, device)

    def sims_de(f_img: np.ndarray) -> list:
        return [f_cap[2 * k:2 * k + 2] @ f_img[2 * k:2 * k + 2].T
                for k in range(len(ejemplos))]

    base = aggregate(sims_de(f_img_orig))
    clipscore_base = float(np.mean(np.sum(f_cap * f_img_orig, axis=1)))
    filas_vis = [{
        "eje": "visual", "perturbacion": "original", "n": len(ejemplos),
        "text": base.text, "image": base.image, "group": base.group,
        "group_match": base.group_match, "clipscore": clipscore_base,
        "delta_clipscore": 0.0, "delta_group": 0.0,
    }]
    for nombre, fn in VISUAL_PERTURBATIONS.items():
        f_img = oc.encode_images(modelo, preproc, [fn(im) for im in imgs_orig], device)
        s = aggregate(sims_de(f_img))
        cs = float(np.mean(np.sum(f_cap * f_img, axis=1)))
        filas_vis.append({
            "eje": "visual", "perturbacion": nombre, "n": len(ejemplos),
            "text": s.text, "image": s.image, "group": s.group,
            "group_match": s.group_match, "clipscore": cs,
            "delta_clipscore": cs - clipscore_base,
            "delta_group": s.group - base.group,
        })
        log(f"   visual/{nombre:18s} group={s.group:.4f} "
            f"(Δ={s.group - base.group:+.4f})  CLIPScore Δ={cs - clipscore_base:+.4f}")

    import pandas as pd
    pd.DataFrame(filas_sem).to_csv(MET / "perturbaciones_semanticas.csv", index=False)
    pd.DataFrame(filas_vis).to_csv(MET / "perturbaciones_visuales.csv", index=False)
    pd.DataFrame(cobertura).to_csv(MET / "perturbaciones_cobertura.csv", index=False)
    log("   escritos perturbaciones_{semanticas,visuales,cobertura}.csv")

    del modelo
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0,
                    help="usar solo los primeros N ejemplos (0 = todos)")
    ap.add_argument("--saltar-blip", action="store_true")
    ap.add_argument("--saltar-perturbaciones", action="store_true")
    args = ap.parse_args()

    SIMS.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)

    log("cargando Winoground")
    from src.winoground_data import load_dataset
    ejemplos, fuente = load_dataset(prefer_real=True)
    if fuente != "winoground_real":
        raise SystemExit("se requiere el benchmark oficial; el set curado no sirve aquí")
    if args.limite:
        ejemplos = ejemplos[:args.limite]
    log(f"fuente={fuente}  N={len(ejemplos)}")

    if not args.saltar_perturbaciones:
        experimento_perturbaciones(ejemplos)
    if not args.saltar_blip:
        experimento_blip(ejemplos)

    log("TERMINADO")


if __name__ == "__main__":
    main()
