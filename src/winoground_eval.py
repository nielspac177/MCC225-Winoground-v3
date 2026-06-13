"""Scorer oficial de Winoground (Thrush et al., 2022).

Cada ejemplo de Winoground tiene DOS captions (c0, c1) y DOS imágenes (i0, i1).
Las captions comparten el mismo conjunto de palabras en distinto orden, de modo
que distinguirlas exige razonamiento composicional y no solo solapamiento léxico.

Convención de la matriz de similitud por ejemplo:

    sim[c, i] = similitud(caption_c, imagen_i)   con  c, i ∈ {0, 1}

es decir:
    sim[0, 0] = s(c0, i0)   sim[0, 1] = s(c0, i1)
    sim[1, 0] = s(c1, i0)   sim[1, 1] = s(c1, i1)

Definiciones (código OFICIAL de Winoground, Thrush et al. 2022 — validadas
numéricamente contra `statistics/model_scores/clip.jsonl` del propio dataset, que
reproduce text=0.3075, image=0.1050, group=0.0800 para CLIP ViT-B/32):

    text_score  = 1  ⇔  s(c0,i0) > s(c1,i0)  Y  s(c1,i1) > s(c0,i1)
                 (fijada la IMAGEN, el modelo elige el CAPTION correcto)
    image_score = 1  ⇔  s(c0,i0) > s(c0,i1)  Y  s(c1,i1) > s(c1,i0)
                 (fijado el CAPTION, el modelo elige la IMAGEN correcta)
    group_score = 1  ⇔  text_score == 1  Y  image_score == 1

OJO: el nombre es por la entidad que se *selecciona* (text_score → se elige el texto).
La redacción verbal de algunos papers de seguimiento (Diwan et al.) suena invertida;
la referencia canónica es el código oficial + los scores de clip.jsonl.

Azar: text = image = 1/4,  group = 1/6 (≈16.67%).
  El group score NO es 1/16: text e image NO son independientes porque comparten
  las mismas 4 similitudes. group=1 exige que ambas diagonales s(c0,i0) y s(c1,i1)
  superen a ambas off-diagonales s(c0,i1) y s(c1,i0); de las 4!=24 ordenaciones de
  4 valores distintos, solo 4 cumplen (2 diagonales arriba × 2 órdenes internos) ⇒ 4/24 = 1/6.
Humanos (paper): text ≈ 89.5, image ≈ 88.5, group ≈ 85.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def text_correct(sim: np.ndarray) -> bool:
    """text_score: fijada cada imagen, ¿el caption correcto puntúa más alto?
    (código oficial: c0_i0 > c1_i0 and c1_i1 > c0_i1)."""
    sim = np.asarray(sim, dtype=float)
    return bool(sim[0, 0] > sim[1, 0] and sim[1, 1] > sim[0, 1])


def image_correct(sim: np.ndarray) -> bool:
    """image_score: fijado cada caption, ¿la imagen correcta puntúa más alto?
    (código oficial: c0_i0 > c0_i1 and c1_i1 > c1_i0)."""
    sim = np.asarray(sim, dtype=float)
    return bool(sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0])


def group_correct(sim: np.ndarray) -> bool:
    """group_score = text_score AND image_score."""
    return text_correct(sim) and image_correct(sim)


@dataclass
class WinogroundScores:
    """Promedios agregados sobre N ejemplos (en [0, 1])."""

    n: int
    text: float
    image: float
    group: float

    def as_dict(self) -> dict:
        return {
            "n_examples": self.n,
            "text_score": self.text,
            "image_score": self.image,
            "group_score": self.group,
            "chance_text": 0.25,
            "chance_image": 0.25,
            "chance_group": 1.0 / 6.0,
        }


def per_example_scores(sims: Iterable[np.ndarray]) -> List[dict]:
    """Devuelve text/image/group (0/1) por cada matriz 2x2 de la secuencia."""
    rows: List[dict] = []
    for k, sim in enumerate(sims):
        sim = np.asarray(sim, dtype=float)
        if sim.shape != (2, 2):
            raise ValueError(f"ejemplo {k}: se esperaba matriz 2x2, se obtuvo {sim.shape}")
        t = int(text_correct(sim))
        i = int(image_correct(sim))
        rows.append({"index": k, "text": t, "image": i, "group": int(t and i)})
    return rows


def aggregate(sims: Iterable[np.ndarray]) -> WinogroundScores:
    """Calcula los tres scores promediados sobre todos los ejemplos."""
    rows = per_example_scores(sims)
    if not rows:
        return WinogroundScores(0, 0.0, 0.0, 0.0)
    n = len(rows)
    text = sum(r["text"] for r in rows) / n
    image = sum(r["image"] for r in rows) / n
    group = sum(r["group"] for r in rows) / n
    return WinogroundScores(n=n, text=text, image=image, group=group)
