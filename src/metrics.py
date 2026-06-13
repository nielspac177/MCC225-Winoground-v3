"""Métricas de similitud y recuperación reutilizadas del Cuaderno 10 (OpenCLIP).

Incluye similitud coseno (sobre embeddings ya normalizados), Recall@K para
recuperación texto→imagen / imagen→texto, y un intervalo de confianza bootstrap
para métricas binarias por ejemplo (p. ej. los scores de Winoground).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz de similitud coseno (N, M). Normaliza por seguridad."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def recall_at_k(sim: np.ndarray, positive_index: Sequence[int], ks=(1, 5, 10)) -> Dict[str, float]:
    """Recall@K para recuperación: por cada fila (consulta), ¿está el positivo
    entre los K columnas de mayor similitud?

    sim[q, d] = similitud de la consulta q con el documento d.
    positive_index[q] = índice de la columna correcta para la consulta q.
    """
    sim = np.asarray(sim, dtype=float)
    n, gallery = sim.shape
    # rank: número de documentos con score estrictamente mayor que el positivo (+1)
    ranks = np.empty(n, dtype=int)
    for q in range(n):
        pos = positive_index[q]
        pos_score = sim[q, pos]
        ranks[q] = int(np.sum(sim[q] > pos_score)) + 1
    out: Dict[str, float] = {"gallery_size": int(gallery)}
    for k in ks:
        # R@K es trivial (≈1) si k ≥ tamaño de galería; se omite para una
        # comparación honesta R@K vs group score.
        if k >= gallery:
            continue
        out[f"R@{k}"] = float(np.mean(ranks <= k))
    out["median_rank"] = float(np.median(ranks))
    out["mrr"] = float(np.mean(1.0 / ranks))
    return out


def bootstrap_ci(
    binary_values: Sequence[int],
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """IC bootstrap (percentil) para la media de una métrica binaria por ejemplo.

    Devuelve la media observada y los límites inferior/superior al (1-alpha)."""
    vals = np.asarray(binary_values, dtype=float)
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "rounds": rounds, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(rounds)
    for r in range(rounds):
        idx = rng.integers(0, n, size=n)
        means[r] = vals[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(vals.mean()), "lo": lo, "hi": hi, "rounds": rounds, "n": n}
