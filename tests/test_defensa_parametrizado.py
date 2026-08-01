
import numpy as np
import pytest
from src.winoground_eval import text_correct, image_correct, group_correct, per_example_scores

@pytest.mark.parametrize("sim, e_text, e_image, e_group", [
    (np.array([[0.9, 0.1 ], [0.1 , 0.9 ]]), True,  True,  True),   # todo bien
    (np.array([[0.9, 0.95], [0.1 , 0.97]]), True,  False, False),  # solo text
    (np.array([[0.9, 0.1 ], [0.95, 0.97]]), False, True,  False),  # solo image
    (np.array([[0.1, 0.9 ], [0.9 , 0.1 ]]), False, False, False),  # todo mal
])
def test_scorer_parametrizado(sim, e_text, e_image, e_group):
    assert text_correct(sim)  is e_text
    assert image_correct(sim) is e_image
    assert group_correct(sim) is e_group

@pytest.mark.parametrize("forma", [(2, 3), (3, 2), (2,), (2, 2, 2)])
def test_shape_invalido(forma):
    with pytest.raises(ValueError):
        per_example_scores([np.zeros(forma)])
