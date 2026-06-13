"""Tests de métricas de recuperación y bootstrap."""
import numpy as np
import pytest

from src.metrics import bootstrap_ci, cosine_similarity_matrix, recall_at_k


def test_cosine_identity():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_matrix(a, a)
    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(0.0)


def test_cosine_is_scale_invariant():
    a = np.array([[3.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    assert cosine_similarity_matrix(a, b)[0, 0] == pytest.approx(1.0)


def test_recall_perfect():
    # Diagonal dominante -> el positivo siempre es top-1.
    sim = np.eye(6) + 1e-3
    pos = list(range(6))
    r = recall_at_k(sim, pos, ks=(1, 3))
    assert r["R@1"] == pytest.approx(1.0)
    assert r["R@3"] == pytest.approx(1.0)
    assert r["median_rank"] == pytest.approx(1.0)
    assert r["gallery_size"] == 6


def test_recall_skips_trivial_k():
    # k >= tamaño de galería se omite (sería ≈1 trivialmente).
    sim = np.eye(4)
    r = recall_at_k(sim, list(range(4)), ks=(1, 4, 10))
    assert "R@1" in r and "R@4" not in r and "R@10" not in r


def test_recall_worst():
    # Positivo es el de menor score en cada fila -> rank = N.
    sim = np.array([[0.1, 0.9, 0.8], [0.2, 0.1, 0.9], [0.9, 0.8, 0.1]])
    pos = [0, 1, 2]
    r = recall_at_k(sim, pos, ks=(1, 2))
    assert r["R@1"] == pytest.approx(0.0)
    assert r["R@2"] == pytest.approx(0.0)


def test_bootstrap_ci_all_ones():
    ci = bootstrap_ci([1, 1, 1, 1], rounds=500, seed=0)
    assert ci["mean"] == pytest.approx(1.0)
    assert ci["lo"] == pytest.approx(1.0)
    assert ci["hi"] == pytest.approx(1.0)


def test_bootstrap_ci_bounds_order():
    vals = [0, 1, 0, 1, 1, 0, 1, 0]
    ci = bootstrap_ci(vals, rounds=1000, seed=1)
    assert 0.0 <= ci["lo"] <= ci["mean"] <= ci["hi"] <= 1.0
    assert ci["mean"] == pytest.approx(0.5)
