"""Prueba de 'ceguera': ¿el modelo usa realmente la imagen o solo pistas textuales?

(Responde la pregunta 8.6.2 y conecta con la interpretabilidad de C8.)

Idea: si los scores reales caen a nivel de azar cuando reemplazamos las imágenes
de cada ejemplo por imágenes de OTROS ejemplos (control por permutación), entonces
el modelo está usando el contenido visual. Si NO caen, estaría explotando regularidades
textuales o sesgos a priori entre las dos captions.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .winoground_eval import aggregate


def permuted_image_control(caption_feats_list: List[np.ndarray],
                           image_feats_list: List[np.ndarray],
                           seed: int = 42) -> List[np.ndarray]:
    """Devuelve matrices de similitud usando, para cada ejemplo, las imágenes de
    OTRO ejemplo (permutación derangement-aproximada). caption_feats_list[k] e
    image_feats_list[k] son arrays (2, d)."""
    n = len(image_feats_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    # evitar que algún ejemplo se empareje consigo mismo
    for k in range(n):
        if perm[k] == k:
            swap_with = (k + 1) % n
            perm[k], perm[swap_with] = perm[swap_with], perm[k]
    sims = []
    for k in range(n):
        cf = caption_feats_list[k]
        imf = image_feats_list[perm[k]]
        sims.append(cf @ imf.T)
    return sims


def real_sims(caption_feats_list: List[np.ndarray],
              image_feats_list: List[np.ndarray]) -> List[np.ndarray]:
    return [cf @ imf.T for cf, imf in zip(caption_feats_list, image_feats_list)]


def run_blindness_probe(caption_feats_list, image_feats_list, seed: int = 42) -> Dict:
    real = aggregate(real_sims(caption_feats_list, image_feats_list))
    perm = aggregate(permuted_image_control(caption_feats_list, image_feats_list, seed=seed))
    return {
        "real": real.as_dict(),
        "permuted_images": perm.as_dict(),
        "interpretation": (
            "Si 'real' >> 'permuted_images' (≈ azar), el modelo SÍ usa el contenido "
            "de la imagen. Si fueran similares, estaría apoyándose en pistas no visuales."
        ),
    }
