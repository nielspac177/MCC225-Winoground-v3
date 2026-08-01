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


def stratified_bootstrap_ci(
    binary_values: Sequence[int],
    strata: Sequence[str],
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """IC bootstrap remuestreando DENTRO de cada estrato (§11.5.3 del examen).

    El bootstrap simple trata los 400 ejemplos como intercambiables. Pero los
    tags de Winoground no están balanceados (Relation 233, Object 141, Both 26)
    y el group score difiere mucho entre ellos, de modo que un remuestreo libre
    hace fluctuar la *composición* de la muestra además de su contenido. Eso
    infla el intervalo con una fuente de variación que no nos interesa: no nos
    preguntamos qué pasaría si el benchmark tuviera otra mezcla de tags, sino
    qué pasaría con otros ejemplos de la misma mezcla.

    Fijando el tamaño de cada estrato, el intervalo refleja solo la
    incertidumbre dentro de cada tipo de ejemplo.
    """
    vals = np.asarray(binary_values, dtype=float)
    strata_arr = np.asarray(strata)
    if len(vals) != len(strata_arr):
        raise ValueError(
            f"longitudes distintas: {len(vals)} valores vs {len(strata_arr)} estratos"
        )
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "rounds": rounds, "n": 0, "n_strata": 0}
    grupos = [np.flatnonzero(strata_arr == s) for s in np.unique(strata_arr)]
    rng = np.random.default_rng(seed)
    means = np.empty(rounds)
    for r in range(rounds):
        muestra = np.concatenate(
            [g[rng.integers(0, len(g), size=len(g))] for g in grupos]
        )
        means[r] = vals[muestra].mean()
    return {
        "mean": float(vals.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "rounds": rounds,
        "n": n,
        "n_strata": len(grupos),
    }


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Dict[str, float]:
    """Test exacto de McNemar para dos modelos sobre LOS MISMOS ejemplos.

    Comparar dos modelos con dos intervalos de confianza independientes es un
    error cuando ambos se evalúan sobre los mismos 400 ítems: ignora que los
    aciertos están correlacionados (los ejemplos fáciles lo son para ambos).
    McNemar mira solo los pares discordantes — donde uno acierta y el otro
    falla — que es exactamente la evidencia que distingue a los modelos.

    Bajo H0 (misma tasa de acierto), cada discordancia es una moneda justa, así
    que b01 ~ Binomial(b01 + b10, 1/2). Se usa la versión exacta y no la
    aproximación chi-cuadrado porque con n=400 el número de discordancias suele
    ser pequeño.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"formas distintas: {a_arr.shape} vs {b_arr.shape}")
    solo_a = int(np.sum((a_arr == 1) & (b_arr == 0)))
    solo_b = int(np.sum((a_arr == 0) & (b_arr == 1)))
    discordantes = solo_a + solo_b
    if discordantes == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest

        p = float(binomtest(solo_a, discordantes, 0.5).pvalue)
    return {
        "n": int(a_arr.size),
        "solo_a": solo_a,
        "solo_b": solo_b,
        "discordantes": discordantes,
        "dif_media": float(a_arr.mean() - b_arr.mean()),
        "p_value": p,
    }


def paired_bootstrap_diff(
    a: Sequence[int],
    b: Sequence[int],
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """IC bootstrap de la diferencia entre dos modelos, remuestreando ejemplos.

    Complementa a `mcnemar_exact`: este da el tamaño del efecto con su
    incertidumbre, aquel da la significancia. Se remuestrean los ÍNDICES una
    sola vez por ronda y se aplican a ambos modelos, preservando el
    emparejamiento.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"formas distintas: {a_arr.shape} vs {b_arr.shape}")
    n = a_arr.size
    if n == 0:
        return {"dif": 0.0, "lo": 0.0, "hi": 0.0, "rounds": rounds, "n": 0}
    rng = np.random.default_rng(seed)
    difs = np.empty(rounds)
    for r in range(rounds):
        idx = rng.integers(0, n, size=n)
        difs[r] = a_arr[idx].mean() - b_arr[idx].mean()
    return {
        "dif": float(a_arr.mean() - b_arr.mean()),
        "lo": float(np.quantile(difs, alpha / 2)),
        "hi": float(np.quantile(difs, 1 - alpha / 2)),
        "rounds": rounds,
        "n": n,
    }


def minimum_resolvable_difference(
    n: int = 400, p: float = 0.10, alpha: float = 0.05, power: float = 0.80
) -> Dict[str, float]:
    """Diferencia mínima detectable entre dos modelos pareados en n ejemplos.

    Responde a una pregunta que la literatura de Winoground casi nunca hace:
    ¿cuándo *no* podemos distinguir dos modelos? Con n=400 y group scores en
    torno al 8 %, las diferencias de uno o dos puntos que se reportan como
    mejoras están dentro del ruido de muestreo.

    Aproximación normal para dos proporciones pareadas, asumiendo que la
    proporción de pares discordantes es `2p(1-p)` (equivalente a independencia
    entre los dos modelos). Es una cota **optimista**: modelos correlacionados
    tienen menos discordancias y por tanto menos potencia, no más.
    """
    from scipy.stats import norm

    if not 0 < p < 1:
        raise ValueError(f"p debe estar en (0,1), se recibió {p}")
    if n <= 0:
        raise ValueError(f"n debe ser positivo, se recibió {n}")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    psi = 2 * p * (1 - p)  # proporción esperada de pares discordantes
    delta = (z_a + z_b) * np.sqrt(psi / n)
    return {
        "n": n,
        "p_base": p,
        "alpha": alpha,
        "power": power,
        "prop_discordantes": float(psi),
        "dif_minima_detectable": float(delta),
    }
