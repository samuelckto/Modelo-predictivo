"""Tests del gating de mercados secundarios.

Comprueban la regla que impide publicar un modelo que no le gana a su baseline,
y que la decision se toma en el bloque de SELECCION y no mirando el holdout.
"""
from __future__ import annotations

import pytest

from CHAT import api, gating


def test_el_margen_minimo_no_es_cero():
    """Sin margen, una diferencia en el quinto decimal 'ganaria'."""
    assert gating.MARGEN_MINIMO > 0


def test_un_empate_lo_gana_el_baseline():
    sel = {"modelo": 0.6900, "baseline": 0.6901}      # ventaja de 0.0001
    r = gating._decidir("x", sel, None, "cfg")
    assert r["estado"] == "no_pick"
    assert r["permite_publicar_probabilidad"] is False


def test_una_ventaja_clara_si_publica():
    sel = {"modelo": 0.6300, "baseline": 0.6700}
    r = gating._decidir("x", sel, {"modelo": 0.64, "baseline": 0.66}, "cfg")
    assert r["estado"] == "projection"
    assert r["permite_publicar_probabilidad"] is True
    # Nunca 'ok': sin cuota historica no se puede demostrar ventaja economica.
    assert r["estado"] != "ok"


def test_decide_en_seleccion_aunque_el_holdout_diga_lo_contrario():
    """El holdout se reporta, no decide. Es exactamente para lo que existe."""
    sel = {"modelo": 0.6950, "baseline": 0.6940}      # el modelo pierde
    hold = {"modelo": 0.6800, "baseline": 0.6950}     # el modelo gana
    r = gating._decidir("x", sel, hold, "cfg")
    assert r["estado"] == "no_pick"
    assert r["holdout"]["ganancia"] > 0               # se reporta igualmente


def test_sin_walkforward_es_insufficient_data():
    r = gating._decidir("x", None, None, None)
    assert r["estado"] == "insufficient_data"


# --------------------------------------------------------------------------
# Estado real medido sobre los walk-forward que hay en disco
# --------------------------------------------------------------------------

def test_el_gating_real_devuelve_un_estado_valido():
    from CHAT.registry import ESTADOS
    for mid, r in gating.todos().items():
        if mid == "version":
            continue
        assert r["estado"] in ESTADOS, (mid, r["estado"])
        assert r["motivo"]


def test_un_mercado_bloqueado_no_devuelve_ninguna_probabilidad():
    """La comprobacion que importa: si el gating dice no, la API no da numeros."""
    for mid in ("mlb.f5_total", "mlb.runs_dist", "soccer.cards"):
        g = (gating.carreras("f5") if mid == "mlb.f5_total" else
             gating.carreras("completo") if mid == "mlb.runs_dist" else
             gating.tarjetas())
        if g.get("permite_publicar_probabilidad"):
            continue
        d = api.get_distribution("cualquiera", mid)
        assert d["disponible"] is False
        for k in ("expected_runs", "expected_cards", "lineas", "ambos_reciben"):
            assert k not in d, f"{mid} filtro numeros estando bloqueado: {k}"
