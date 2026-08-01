"""Tests de las perturbaciones semánticas y visuales (§11.2.5).

El requisito de diseño que se verifica aquí es el de **par mínimo**: cada
perturbación textual debe alterar un solo token, para que el contraste sea
atribuible al cambio semántico y no a la longitud o la fluidez. Cuando una
regla no aplica, debe devolver None en vez de forzar una paráfrasis.
"""
import numpy as np
import pytest
from PIL import Image

from src.perturbations import (
    RELATION_ANTONYMS,
    SEMANTIC_PERTURBATIONS,
    VISUAL_PERTURBATIONS,
    blur,
    center_crop,
    low_resolution,
    negate,
    perturbation_coverage,
    swap_quantifier,
    swap_relation,
    to_grayscale,
)


def imagen(w=64, h=48, color=(200, 30, 30)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


# --------------------------------------------------------------------------- #
# Negación                                                                      #
# --------------------------------------------------------------------------- #
def test_negacion_inserta_not_tras_el_auxiliar():
    assert negate("a bottle is in water") == "a bottle is not in water"
    assert negate("there are more skiers than snowboarders") == \
        "there are not more skiers than snowboarders"


def test_negacion_devuelve_none_sin_auxiliar():
    """Sin cópula no hay negación de un solo token; se declara, no se fuerza."""
    assert negate("a dog chases a cat") is None
    assert negate("an old person kisses a young person") is None


def test_negacion_usa_el_primer_auxiliar():
    r = negate("the dog is brown and the cat is black")
    assert r == "the dog is not brown and the cat is black"


def test_negacion_anade_exactamente_un_token():
    original = "a bottle is in water"
    assert len(negate(original).split()) == len(original.split()) + 1


# --------------------------------------------------------------------------- #
# Relación espacial                                                             #
# --------------------------------------------------------------------------- #
def test_swap_relation_invierte_la_primera_relacion():
    assert swap_relation("a brown dog is on a white couch") == \
        "a brown dog is under a white couch"
    assert swap_relation("the red book is above the yellow book") == \
        "the red book is below the yellow book"


def test_swap_relation_devuelve_none_sin_relacion_espacial():
    assert swap_relation("a dog chases a cat") is None


def test_swap_relation_preserva_la_longitud():
    """Cambiar una palabra por su antónimo no altera el número de tokens."""
    original = "a brown dog is on a white couch"
    assert len(swap_relation(original).split()) == len(original.split())


def test_antonimos_de_relacion_son_simetricos():
    """Aplicar la inversión dos veces devuelve la palabra original."""
    for a, b in RELATION_ANTONYMS.items():
        assert RELATION_ANTONYMS[b] == a


def test_swap_relation_respeta_mayuscula_inicial():
    assert swap_relation("Above the table there is a lamp").startswith("Below")


def test_swap_relation_no_toca_subcadenas():
    """'onion' contiene 'on' pero no es una relación espacial."""
    assert swap_relation("an onion and a potato") is None


# --------------------------------------------------------------------------- #
# Cuantificador                                                                 #
# --------------------------------------------------------------------------- #
def test_swap_quantifier():
    assert swap_quantifier("there are more skiers than snowboarders") == \
        "there are fewer skiers than snowboarders"
    assert swap_quantifier("there is less orange juice than milk") == \
        "there is more orange juice than milk"


def test_swap_quantifier_devuelve_none_sin_cuantificador():
    assert swap_quantifier("a bottle is in water") is None


# --------------------------------------------------------------------------- #
# Cobertura                                                                     #
# --------------------------------------------------------------------------- #
def test_cobertura_cuenta_los_aplicables():
    captions = [
        "a bottle is in water",              # negación sí, relación no
        "a dog chases a cat",                # ninguna
        "a brown dog is on a white couch",   # negación sí, relación sí
    ]
    filas = {r["perturbacion"]: r for r in perturbation_coverage(captions)}
    assert filas["negacion"]["aplicables"] == 2
    assert filas["relacion"]["aplicables"] == 1
    assert filas["cuantificador"]["aplicables"] == 0
    assert filas["negacion"]["cobertura"] == pytest.approx(2 / 3)


def test_cobertura_con_lista_vacia_no_divide_por_cero():
    for r in perturbation_coverage([]):
        assert r["cobertura"] == 0.0


def test_todas_las_perturbaciones_semanticas_aceptan_cualquier_texto():
    """Ninguna debe lanzar excepción: devuelven None cuando no aplican."""
    for nombre, fn in SEMANTIC_PERTURBATIONS.items():
        for texto in ["", "x", "a dog", "IS", "the more the merrier"]:
            r = fn(texto)
            assert r is None or isinstance(r, str), nombre


# --------------------------------------------------------------------------- #
# Perturbaciones visuales                                                       #
# --------------------------------------------------------------------------- #
def test_grises_elimina_el_color_conservando_el_tamano():
    im = imagen()
    g = to_grayscale(im)
    assert g.size == im.size and g.mode == "RGB"
    a = np.asarray(g)
    # En una imagen en gris los tres canales coinciden.
    assert np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])


def test_recorte_central_reduce_el_tamano():
    im = imagen(100, 50)
    assert center_crop(im, 0.5).size == (50, 25)


def test_recorte_completo_no_cambia_nada():
    im = imagen(100, 50)
    assert center_crop(im, 1.0).size == im.size


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
def test_recorte_con_fraccion_invalida(frac):
    with pytest.raises(ValueError):
        center_crop(imagen(), frac)


def test_baja_resolucion_conserva_el_tamano_y_pierde_detalle():
    """El tamaño en píxeles no cambia; lo que se pierde es el detalle fino."""
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    baja = low_resolution(im, factor=8)
    assert baja.size == im.size
    assert np.asarray(baja).std() < np.asarray(im).std()


def test_baja_resolucion_factor_uno_es_casi_identidad():
    assert low_resolution(imagen(), factor=1).size == imagen().size


def test_factor_invalido():
    with pytest.raises(ValueError):
        low_resolution(imagen(), factor=0)


def test_desenfoque_conserva_el_tamano():
    im = imagen()
    assert blur(im, 2.0).size == im.size


def test_radio_negativo_es_invalido():
    with pytest.raises(ValueError):
        blur(imagen(), -1.0)


def test_todas_las_visuales_devuelven_una_imagen_rgb_valida():
    im = imagen(128, 96)
    for nombre, fn in VISUAL_PERTURBATIONS.items():
        r = fn(im)
        assert isinstance(r, Image.Image), nombre
        assert r.size[0] > 0 and r.size[1] > 0, nombre
