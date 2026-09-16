"""Valor estimado, desacuerdo con el mercado y eleccion 1X2.

Estos tests protegen tres errores REALES encontrados en produccion, no casos
inventados. Los tres producian numeros que parecian correctos:

  1. X2 se calculaba como 1 - P(1X), que es P(gana el visitante), no
     P(empate o gana el visitante). 18 de 90 predicciones activas publicaban
     hasta 25 puntos de menos.
  2. La cuota y la probabilidad del mercado se cogian SIEMPRE del lado OVER,
     aunque la seleccion publicada fuese UNDER. La diferencia con el mercado
     salia con el signo cambiado.
  3. Una cuota americana imposible (-9.0) se convertia a decimal 12.1 e
     inventaba una ganancia enorme donde solo habia un dato roto.
"""
from __future__ import annotations

import pytest

from shared import valor
from SOCCER.markets.predict import (MARGEN_IGUALADO, UMBRAL_GANADOR,
                                    _eleccion_1x2, _lado_mercado)


# --------------------------------------------------------------------------
# Eleccion 1X2
# --------------------------------------------------------------------------

def test_x2_es_empate_mas_visitante_no_uno_menos_1x():
    """El bug original, con numeros reales de Fenerbahce vs AS Roma."""
    ph, pd_, pa = 0.246, 0.247, 0.507
    sel, prob, tipo = _eleccion_1x2("Fenerbahce", "AS Roma", ph, pd_, pa)
    assert sel == "AS Roma o empate"
    assert prob == pytest.approx(pd_ + pa)          # 75.4 %
    assert prob != pytest.approx(1 - (ph + pd_))    # 50.7 %, lo que se publicaba
    assert prob > 0.75


def test_un_favorito_claro_se_publica_como_ganador():
    ph = UMBRAL_GANADOR + 0.05
    sel, prob, tipo = _eleccion_1x2("Barcelona", "Getafe", ph, 0.12, 1 - ph - 0.12)
    assert sel == "Barcelona gana"
    assert prob == pytest.approx(ph)
    assert tipo == "ganador"


def test_el_visitante_favorito_tambien_puede_ser_ganador_directo():
    pa = UMBRAL_GANADOR + 0.03
    sel, prob, tipo = _eleccion_1x2("Getafe", "Real Madrid", 0.15, 0.1, pa)
    assert sel == "Real Madrid gana" and tipo == "ganador"
    assert prob == pytest.approx(pa)


def test_un_partido_igualado_si_usa_la_doble_oportunidad():
    """Es justo donde cubrir el empate paga: nadie domina."""
    ph, pd_, pa = 0.40, 0.27, 0.33
    assert abs(ph - pa) < MARGEN_IGUALADO
    sel, prob, tipo = _eleccion_1x2("Sevilla", "Valencia", ph, pd_, pa)
    assert sel == "Sevilla o empate"
    assert prob == pytest.approx(ph + pd_)
    assert tipo == "doble_igualado"


def test_justo_por_debajo_del_umbral_no_se_arriesga_al_ganador():
    ph = UMBRAL_GANADOR - 0.001
    sel, _, tipo = _eleccion_1x2("A", "B", ph, 0.2, 1 - ph - 0.2)
    assert sel == "A o empate" and tipo != "ganador"


def test_gana_el_lado_mas_probable_de_los_dos():
    ph, pd_, pa = 0.20, 0.25, 0.55        # 1X = .45, X2 = .80
    sel, prob, _ = _eleccion_1x2("A", "B", ph, pd_, pa)
    assert sel == "B o empate" and prob == pytest.approx(0.80)


# --------------------------------------------------------------------------
# El lado del mercado tiene que coincidir con la seleccion
# --------------------------------------------------------------------------

def test_un_under_no_se_compara_contra_el_precio_del_over():
    casa = _lado_mercado("totals", "UNDER 2.5", "A", "B")
    assert casa("Under") and not casa("Over")


def test_un_over_sigue_cogiendo_el_over():
    casa = _lado_mercado("totals", "OVER 2.5", "A", "B")
    assert casa("Over") and not casa("Under")


def test_el_no_de_ambos_marcan_coge_el_no():
    casa = _lado_mercado("btts", "NO", "A", "B")
    assert casa("No") and not casa("Yes")


def test_la_doble_oportunidad_no_tiene_precio_en_h2h():
    """'gana o empata' no es ninguna de las tres vias: mejor sin precio que con
    el precio equivocado."""
    assert _lado_mercado("h2h", "Barcelona o empate", "Barcelona", "Getafe") is None


def test_el_ganador_directo_si_coge_su_precio():
    casa = _lado_mercado("h2h", "Barcelona gana", "Barcelona", "Getafe")
    assert casa("Barcelona") and not casa("Getafe")


# --------------------------------------------------------------------------
# Valor
# --------------------------------------------------------------------------

def test_el_ev_sale_de_la_cuota_real():
    v = valor.evaluar(0.60, 0.55, 2.0)
    assert v["ev"] == pytest.approx(0.20)
    assert v["ev_pct"] == pytest.approx(20.0)


def test_nunca_se_declara_validado():
    """La regla del proyecto: sin muestra no hay valor demostrado."""
    assert valor.evaluar(0.9, 0.4, 3.0)["validado"] is False


def test_sin_cuota_no_hay_ev_y_se_dice():
    v = valor.evaluar(0.60, 0.55, None)
    assert v["ev"] is None
    assert "no se puede calcular valor" in v["aviso"]


def test_una_cuota_americana_imposible_se_rechaza():
    """-9.0 aparece en los datos y daria decimal 12.1: ganancia inventada."""
    assert valor.decimal_desde_americana(-9.0) is None
    assert valor.evaluar(0.6, None, -9.0)["ev"] is None


def test_americana_normal_se_convierte_bien():
    assert valor.decimal_desde_americana(-110) == pytest.approx(1.909, abs=1e-3)
    assert valor.decimal_desde_americana(+150) == pytest.approx(2.5)


def test_lados_opuestos_es_contra_el_mercado():
    v = valor.evaluar(0.58, 0.42)
    assert v["contra_mercado"] is True
    v2 = valor.evaluar(0.58, 0.54)
    assert v2["contra_mercado"] is False


def test_una_diferencia_menor_que_el_margen_no_es_desacuerdo():
    """2 pp es menos que la comision tipica de la casa: no significa nada."""
    assert valor.evaluar(0.52, 0.50)["tramo"] == "de acuerdo"
    assert valor.evaluar(0.65, 0.50)["tramo"] == "fuerte"


def test_los_que_no_tienen_cuota_van_al_final_no_como_ev_cero():
    items = [{"v": valor.evaluar(0.5, None, None)},
             {"v": valor.evaluar(0.5, None, 1.5)},      # EV -0.25
             {"v": valor.evaluar(0.5, None, 3.0)}]      # EV +0.50
    orden = valor.ordenar_por_valor(items, lambda x: x["v"])
    assert [x["v"]["ev"] for x in orden] == [0.5, -0.25, None]
