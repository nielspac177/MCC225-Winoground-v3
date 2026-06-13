"""TDD del scorer de Winoground: casos de verdad conocida construidos a mano.

Convención: sim[caption, imagen], con sim[0,0]=s(c0,i0), sim[1,1]=s(c1,i1)."""
import numpy as np
import pytest

from src.winoground_eval import (
    aggregate,
    group_correct,
    image_correct,
    per_example_scores,
    text_correct,
)


def test_perfect_diagonal_passes_all():
    # Diagonal dominante: el modelo acierta caption e imagen correctas.
    sim = np.array([[0.9, 0.1],
                    [0.2, 0.8]])
    assert text_correct(sim) is True
    assert image_correct(sim) is True
    assert group_correct(sim) is True


def test_fully_wrong_fails_all():
    sim = np.array([[0.1, 0.9],
                    [0.8, 0.2]])
    assert text_correct(sim) is False
    assert image_correct(sim) is False
    assert group_correct(sim) is False


def test_text_correct_but_image_wrong():
    # Defs oficiales: text=(s00>s10 y s11>s01); image=(s00>s01 y s11>s10).
    sim = np.array([[0.9, 0.95],
                    [0.1, 0.97]])
    # text: 0.9>0.1 ✓ y 0.97>0.95 ✓ -> True
    # image: 0.9>0.95 ✗ -> False
    assert text_correct(sim) is True
    assert image_correct(sim) is False
    assert group_correct(sim) is False


def test_image_correct_but_text_wrong():
    sim = np.array([[0.9, 0.1],
                    [0.95, 0.97]])
    # image: 0.9>0.1 ✓ y 0.97>0.95 ✓ -> True
    # text: 0.9>0.95 ✗ -> False
    assert text_correct(sim) is False
    assert image_correct(sim) is True
    assert group_correct(sim) is False


def test_group_requires_both():
    sim_good = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert group_correct(sim_good) == (text_correct(sim_good) and image_correct(sim_good))


def test_aggregate_means():
    sims = [
        np.array([[0.9, 0.1], [0.2, 0.8]]),   # group=1
        np.array([[0.1, 0.9], [0.8, 0.2]]),   # group=0, text=0, image=0
        np.array([[0.9, 0.95], [0.1, 0.97]]), # text=1, image=0
    ]
    s = aggregate(sims)
    assert s.n == 3
    assert s.text == pytest.approx(2 / 3)
    assert s.image == pytest.approx(1 / 3)
    assert s.group == pytest.approx(1 / 3)


def test_per_example_shape_validation():
    with pytest.raises(ValueError):
        per_example_scores([np.zeros((2, 3))])


def test_empty_aggregate():
    s = aggregate([])
    assert s.n == 0 and s.group == 0.0


def test_chance_levels_in_dict():
    s = aggregate([np.array([[0.9, 0.1], [0.2, 0.8]])])
    d = s.as_dict()
    assert d["chance_text"] == 0.25
    assert d["chance_group"] == pytest.approx(1 / 6)  # text e image NO son independientes
