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


# --------------------------------------------------------------------------- #
# Tres estados y empates (§11.2.2 del Examen Final)                             #
# --------------------------------------------------------------------------- #
# El scorer oficial usa `>` estricto, de modo que un empate exacto entre dos
# similitudes se contabiliza silenciosamente como fallo. Eso es una decisión
# defendible (es el código de referencia), pero oculta información: no es lo
# mismo que el modelo prefiera la respuesta equivocada a que no exprese ninguna
# preferencia. Aquí se separan los dos casos sin alterar el valor por defecto.

CORRECTO = "correcto"
INCORRECTO = "incorrecto"
EMPATE = "empate"

TIE_POLICIES = ("fail", "half", "pass")


def _as_matrix(sim: np.ndarray) -> np.ndarray:
    """Valida forma (2,2) y ausencia de NaN/inf, y devuelve float64.

    Un NaN convertiría toda comparación `>` en False, produciendo un fallo
    silencioso indistinguible de un error real del modelo.
    """
    sim = np.asarray(sim, dtype=float)
    if sim.shape != (2, 2):
        raise ValueError(f"se esperaba matriz 2x2, se obtuvo {sim.shape}")
    if not np.all(np.isfinite(sim)):
        raise ValueError("la matriz de similitud contiene NaN o infinitos")
    return sim


def _check_atol(atol: float) -> float:
    if atol < 0:
        raise ValueError(f"atol debe ser >= 0, se recibió {atol}")
    return float(atol)


def _compare(a: float, b: float, atol: float) -> int:
    """+1 si a supera a b, -1 si b supera a a, 0 si empatan dentro de `atol`."""
    if a - b > atol:
        return 1
    if b - a > atol:
        return -1
    return 0


def _combine(*comparisons: int) -> str:
    """Un fallo estricto domina; en su ausencia, un empate deja el caso abierto."""
    if any(c < 0 for c in comparisons):
        return INCORRECTO
    if any(c == 0 for c in comparisons):
        return EMPATE
    return CORRECTO


def text_status(sim: np.ndarray, atol: float = 0.0) -> str:
    """Estado de text_score en tres valores. Fija la imagen, compara captions."""
    sim = _as_matrix(sim)
    atol = _check_atol(atol)
    return _combine(
        _compare(sim[0, 0], sim[1, 0], atol),   # imagen 0: ¿gana c0?
        _compare(sim[1, 1], sim[0, 1], atol),   # imagen 1: ¿gana c1?
    )


def image_status(sim: np.ndarray, atol: float = 0.0) -> str:
    """Estado de image_score en tres valores. Fija el caption, compara imágenes."""
    sim = _as_matrix(sim)
    atol = _check_atol(atol)
    return _combine(
        _compare(sim[0, 0], sim[0, 1], atol),   # caption 0: ¿gana i0?
        _compare(sim[1, 1], sim[1, 0], atol),   # caption 1: ¿gana i1?
    )


def group_status(sim: np.ndarray, atol: float = 0.0) -> str:
    """Estado de group_score: combina text e image con la misma regla."""
    t, i = text_status(sim, atol), image_status(sim, atol)
    if INCORRECTO in (t, i):
        return INCORRECTO
    if EMPATE in (t, i):
        return EMPATE
    return CORRECTO


# --------------------------------------------------------------------------- #
# GroupMatch (Zhu et al., ICLR 2026, arXiv:2510.07632)                          #
# --------------------------------------------------------------------------- #
# El group score exige que cada elemento diagonal sea el máximo de su fila Y de
# su columna. GroupMatch relaja eso a la pregunta natural de emparejamiento:
# ¿es el emparejamiento correcto el de mayor similitud total? Para k=2 se
# reduce a s00 + s11 > s01 + s10, y el azar sube de 1/6 a 1/2.
#
# No es un scorer "mejor": mide otra cosa. group score pregunta si el modelo
# ordena bien CADA comparación por separado; GroupMatch, si acierta la
# asignación global. Reportar ambos es lo que separa un modelo que no compone
# de una métrica que no lo deja demostrarlo.

def group_match_margin(sim: np.ndarray) -> float:
    """Margen de GroupMatch: (s00 + s11) - (s01 + s10). Positivo = acierta."""
    sim = _as_matrix(sim)
    return float(sim[0, 0] + sim[1, 1] - sim[0, 1] - sim[1, 0])


def group_match(sim: np.ndarray) -> bool:
    """GroupMatch: ¿el emparejamiento correcto maximiza la similitud total?"""
    return bool(group_match_margin(sim) > 0)


def group_match_status(sim: np.ndarray, atol: float = 0.0) -> str:
    """Estado de GroupMatch en tres valores (margen nulo => empate)."""
    atol = _check_atol(atol)
    return _combine(_compare(group_match_margin(sim), 0.0, atol))


def _score(status: str, tie_policy: str) -> float:
    """Traduce un estado a un valor numérico según la convención de empates."""
    if tie_policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy debe ser una de {TIE_POLICIES}, se recibió {tie_policy!r}")
    if status == CORRECTO:
        return 1.0
    if status == INCORRECTO:
        return 0.0
    return {"fail": 0.0, "half": 0.5, "pass": 1.0}[tie_policy]


def tie_report(sims: Iterable[np.ndarray], atol: float = 0.0) -> dict:
    """Cuenta cuántos ejemplos quedan indeterminados por empate en cada métrica.

    Es la evidencia que pide §11.2.2: sin este conteo no se puede afirmar que
    el uso de `>` estricto sea inocuo.
    """
    n = t = i = g = gm = any_ = 0
    for sim in sims:
        n += 1
        st, si = text_status(sim, atol), image_status(sim, atol)
        sg, sgm = group_status(sim, atol), group_match_status(sim, atol)
        t += st == EMPATE
        i += si == EMPATE
        g += sg == EMPATE
        gm += sgm == EMPATE
        any_ += EMPATE in (st, si, sgm)
    return {
        "n_examples": n,
        "atol": atol,
        "text_ties": t,
        "image_ties": i,
        "group_ties": g,
        "group_match_ties": gm,
        "any_tie": any_,
    }


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
    group_match: float = 0.0
    tie_policy: str = "fail"
    atol: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_examples": self.n,
            "text_score": self.text,
            "image_score": self.image,
            "group_score": self.group,
            "group_match_score": self.group_match,
            "chance_text": 0.25,
            "chance_image": 0.25,
            "chance_group": 1.0 / 6.0,
            # GroupMatch sólo exige acertar la asignación global: 1 de 2! permutaciones.
            "chance_group_match": 0.5,
            "tie_policy": self.tie_policy,
            "atol": self.atol,
        }


def per_example_scores(
    sims: Iterable[np.ndarray], atol: float = 0.0, tie_policy: str = "fail"
) -> List[dict]:
    """Puntajes por ejemplo, con el estado de tres valores junto a cada número.

    Las columnas `*_status` conservan la distinción entre fallo y empate que el
    puntaje numérico necesariamente pierde.
    """
    _check_atol(atol)
    _score(CORRECTO, tie_policy)  # valida tie_policy antes de iterar
    rows: List[dict] = []
    for k, sim in enumerate(sims):
        try:
            sim = _as_matrix(sim)
        except ValueError as exc:
            raise ValueError(f"ejemplo {k}: {exc}") from exc
        st = text_status(sim, atol)
        si = image_status(sim, atol)
        sg = group_status(sim, atol)
        sgm = group_match_status(sim, atol)
        rows.append(
            {
                "index": k,
                "text": _score(st, tie_policy),
                "image": _score(si, tie_policy),
                "group": _score(sg, tie_policy),
                "group_match": _score(sgm, tie_policy),
                "text_status": st,
                "image_status": si,
                "group_status": sg,
                "group_match_status": sgm,
                "group_match_margin": group_match_margin(sim),
            }
        )
    return rows


def aggregate(
    sims: Iterable[np.ndarray], atol: float = 0.0, tie_policy: str = "fail"
) -> WinogroundScores:
    """Promedia los scores sobre todos los ejemplos.

    Los valores por defecto (`atol=0.0`, `tie_policy="fail"`) reproducen
    exactamente el scorer oficial de Thrush et al. (2022).
    """
    rows = per_example_scores(sims, atol=atol, tie_policy=tie_policy)
    if not rows:
        return WinogroundScores(0, 0.0, 0.0, 0.0, 0.0, tie_policy, atol)
    n = len(rows)
    return WinogroundScores(
        n=n,
        text=sum(r["text"] for r in rows) / n,
        image=sum(r["image"] for r in rows) / n,
        group=sum(r["group"] for r in rows) / n,
        group_match=sum(r["group_match"] for r in rows) / n,
        tie_policy=tie_policy,
        atol=atol,
    )
