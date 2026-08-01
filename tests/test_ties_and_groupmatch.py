"""Tests del tratamiento de empates y de la métrica GroupMatch.

Cubre dos exigencias del Examen Final (§11.2.2 y §11.5.2):
  - registrar empates en las cuatro similitudes y devolver tres estados;
  - ofrecer una alternativa razonable al group score estricto.

Invariante de compatibilidad: con `atol=0.0` y `tie_policy="fail"` el scorer
nuevo debe coincidir exactamente con el oficial de Thrush et al. (2022).
"""
import numpy as np
import pytest

from src.winoground_eval import (
    CORRECTO,
    EMPATE,
    INCORRECTO,
    aggregate,
    group_correct,
    group_match,
    group_match_status,
    group_status,
    image_correct,
    image_status,
    per_example_scores,
    text_correct,
    text_status,
    tie_report,
)

# Convención: sim[c, i] = similitud(caption_c, imagen_i).

# Diagonal dominante: todo correcto bajo cualquier métrica.
PERFECTA = np.array([[0.9, 0.1], [0.1, 0.9]])

# Antidiagonal dominante: todo incorrecto.
INVERTIDA = np.array([[0.1, 0.9], [0.9, 0.1]])


# --------------------------------------------------------------------------- #
# Tres estados                                                                  #
# --------------------------------------------------------------------------- #
def test_estados_en_matriz_perfecta():
    """Sin empates y con diagonal dominante, los tres estados son 'correcto'."""
    assert text_status(PERFECTA) == CORRECTO
    assert image_status(PERFECTA) == CORRECTO
    assert group_status(PERFECTA) == CORRECTO


def test_estados_en_matriz_invertida():
    """Antidiagonal dominante: las dos comparaciones fallan estrictamente."""
    assert text_status(INVERTIDA) == INCORRECTO
    assert image_status(INVERTIDA) == INCORRECTO
    assert group_status(INVERTIDA) == INCORRECTO


def test_empate_exacto_en_text():
    """s(c0,i0) == s(c1,i0) es un empate real, no un acierto ni un fallo.

    text exige c0_i0 > c1_i0 Y c1_i1 > c0_i1. Aquí la primera comparación
    empata y la segunda es correcta: no hay fallo estricto => EMPATE.
    """
    sim = np.array([[0.5, 0.1], [0.5, 0.9]])
    assert text_status(sim) == EMPATE
    # image sí queda determinado: 0.5 > 0.1 y 0.9 > 0.5.
    assert image_status(sim) == CORRECTO


def test_empate_no_enmascara_un_fallo_estricto():
    """Si una comparación empata pero la otra falla, el resultado es INCORRECTO.

    Un empate nunca puede rescatar a un fallo: el veredicto más fuerte manda.
    """
    sim = np.array([[0.5, 0.9], [0.5, 0.1]])
    # text: c0_i0(0.5) vs c1_i0(0.5) -> empate ; c1_i1(0.1) vs c0_i1(0.9) -> fallo
    assert text_status(sim) == INCORRECTO


def test_group_es_incorrecto_si_cualquiera_lo_es():
    """group combina text e image: un INCORRECTO domina sobre un EMPATE."""
    # text empata, image falla.
    sim = np.array([[0.5, 0.9], [0.5, 0.9]])
    assert text_status(sim) == EMPATE
    assert image_status(sim) == INCORRECTO
    assert group_status(sim) == INCORRECTO


def test_group_es_empate_si_no_hay_fallo_y_hay_empate():
    """Sin ningún fallo estricto y con al menos un empate => EMPATE."""
    sim = np.array([[0.5, 0.1], [0.5, 0.9]])
    assert text_status(sim) == EMPATE
    assert image_status(sim) == CORRECTO
    assert group_status(sim) == EMPATE


def test_matriz_totalmente_constante_es_empate():
    """Cuatro similitudes idénticas: el modelo no expresa ninguna preferencia."""
    sim = np.full((2, 2), 0.3)
    assert text_status(sim) == EMPATE
    assert image_status(sim) == EMPATE
    assert group_status(sim) == EMPATE


# --------------------------------------------------------------------------- #
# Tolerancia numérica                                                           #
# --------------------------------------------------------------------------- #
def test_atol_convierte_diferencia_minuscula_en_empate():
    """Una diferencia de 1e-6 es ruido de punto flotante, no una preferencia."""
    sim = np.array([[0.500001, 0.1], [0.500000, 0.9]])
    # Con atol=0 (oficial) la diferencia cuenta como preferencia estricta.
    assert text_status(sim, atol=0.0) == CORRECTO
    # Con atol=1e-4 pasa a ser un empate.
    assert text_status(sim, atol=1e-4) == EMPATE


def test_atol_no_afecta_diferencias_grandes():
    """Una tolerancia razonable no debe alterar veredictos claros."""
    assert text_status(PERFECTA, atol=1e-3) == CORRECTO
    assert image_status(INVERTIDA, atol=1e-3) == INCORRECTO


def test_atol_negativo_es_invalido():
    with pytest.raises(ValueError):
        text_status(PERFECTA, atol=-1e-6)


# --------------------------------------------------------------------------- #
# Compatibilidad con el scorer oficial                                          #
# --------------------------------------------------------------------------- #
def test_estados_coinciden_con_el_scorer_oficial_sin_tolerancia():
    """Invariante crítico: atol=0 => CORRECTO ⇔ la función booleana oficial."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        sim = rng.random((2, 2))
        assert (text_status(sim) == CORRECTO) is text_correct(sim)
        assert (image_status(sim) == CORRECTO) is image_correct(sim)
        assert (group_status(sim) == CORRECTO) is group_correct(sim)


def test_aggregate_por_defecto_no_cambia():
    """Los valores por defecto reproducen el comportamiento histórico."""
    sims = [PERFECTA, INVERTIDA, PERFECTA]
    s = aggregate(sims)
    assert s.text == pytest.approx(2 / 3)
    assert s.image == pytest.approx(2 / 3)
    assert s.group == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Políticas de empate                                                           #
# --------------------------------------------------------------------------- #
def test_politicas_de_empate_acotan_el_score():
    """fail <= half <= pass: las tres convenciones ordenan el mismo resultado."""
    # Un ejemplo correcto y uno con empate en text.
    sims = [PERFECTA, np.array([[0.5, 0.1], [0.5, 0.9]])]
    fail = aggregate(sims, tie_policy="fail")
    half = aggregate(sims, tie_policy="half")
    pas = aggregate(sims, tie_policy="pass")
    assert fail.text == pytest.approx(0.5)
    assert half.text == pytest.approx(0.75)
    assert pas.text == pytest.approx(1.0)
    assert fail.text <= half.text <= pas.text


def test_politica_desconocida_es_invalida():
    with pytest.raises(ValueError):
        aggregate([PERFECTA], tie_policy="inventada")


def test_tie_report_cuenta_empates_por_metrica():
    """El reporte de empates es la evidencia que pide §11.2.2."""
    sims = [PERFECTA, np.full((2, 2), 0.3), INVERTIDA]
    rep = tie_report(sims)
    assert rep["n_examples"] == 3
    assert rep["text_ties"] == 1
    assert rep["image_ties"] == 1
    assert rep["group_ties"] == 1
    assert rep["any_tie"] == 1


def test_tie_report_sin_empates():
    rep = tie_report([PERFECTA, INVERTIDA])
    assert rep["text_ties"] == 0
    assert rep["any_tie"] == 0


# --------------------------------------------------------------------------- #
# GroupMatch (Zhu et al., ICLR 2026)                                            #
# --------------------------------------------------------------------------- #
def test_group_match_en_matriz_perfecta():
    """0.9+0.9 > 0.1+0.1: el emparejamiento correcto es el mejor global."""
    assert group_match(PERFECTA) is True


def test_group_match_en_matriz_invertida():
    assert group_match(INVERTIDA) is False


def test_group_match_es_mas_laxo_que_group_score():
    """Caso que separa las dos métricas: falla group pero pasa GroupMatch.

    sim = [[0.6, 0.5], [0.5, 0.4]]
      group: c0_i0(0.6) > c0_i1(0.5) OK, pero c1_i1(0.4) < c1_i0(0.5) => falla.
      GroupMatch: 0.6+0.4 = 1.0  >  0.5+0.5 = 1.0  => FALSO por igualdad.
    Se usa un caso con margen positivo real.
    """
    sim = np.array([[0.7, 0.5], [0.5, 0.4]])
    # group score falla: c1_i1 (0.4) no supera a c1_i0 (0.5).
    assert group_correct(sim) is False
    # GroupMatch pasa: 0.7 + 0.4 = 1.1 > 0.5 + 0.5 = 1.0.
    assert group_match(sim) is True


def test_group_match_implicado_por_group_score():
    """Si group score acierta, GroupMatch también. La implicación es estricta.

    group=1 exige que ambas diagonales superen a ambas antidiagonales, luego
    su suma también las supera.
    """
    rng = np.random.default_rng(7)
    for _ in range(500):
        sim = rng.random((2, 2))
        if group_correct(sim):
            assert group_match(sim) is True


def test_group_match_status_registra_el_empate():
    """Margen exactamente cero: no hay emparejamiento preferido."""
    sim = np.array([[0.6, 0.5], [0.5, 0.4]])  # 1.0 vs 1.0
    assert group_match_status(sim) == EMPATE


def test_group_match_status_respeta_atol():
    sim = np.array([[0.6, 0.5], [0.5, 0.400001]])  # margen 1e-6
    assert group_match_status(sim, atol=0.0) == CORRECTO
    assert group_match_status(sim, atol=1e-4) == EMPATE


def test_group_match_en_as_dict_de_aggregate():
    """aggregate expone group_match y su azar (1/2, no 1/6)."""
    d = aggregate([PERFECTA, INVERTIDA]).as_dict()
    assert d["group_match_score"] == pytest.approx(0.5)
    assert d["chance_group_match"] == pytest.approx(0.5)
    assert d["chance_group"] == pytest.approx(1 / 6)


# --------------------------------------------------------------------------- #
# Validación de forma                                                           #
# --------------------------------------------------------------------------- #
def test_per_example_scores_incluye_estados_y_group_match():
    filas = per_example_scores([PERFECTA])
    f = filas[0]
    assert f["text"] == 1 and f["image"] == 1 and f["group"] == 1
    assert f["group_match"] == 1
    assert f["text_status"] == CORRECTO
    assert f["group_match_status"] == CORRECTO


def test_group_match_valida_la_forma():
    with pytest.raises(ValueError):
        group_match(np.zeros((2, 3)))


def test_matriz_con_nan_es_rechazada():
    """Un NaN silencioso convertiría cualquier comparación en False."""
    sim = np.array([[np.nan, 0.1], [0.1, 0.9]])
    with pytest.raises(ValueError):
        text_status(sim)
