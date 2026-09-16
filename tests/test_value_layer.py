"""Tests de la capa de valor: calibracion, ROI, discrepancia y tracking.

Casi todos comprueban que el sistema se NIEGA a afirmar algo sin respaldo. Es
lo mas facil de romper sin darse cuenta: basta con que un cambio deje pasar un
default optimista y el sistema empieza a vender ventaja donde solo hay ruido.
"""
from __future__ import annotations

import numpy as np
import pytest

from shared import calibrators as C
from shared import discrepancy as D
from shared import roi as R


# --------------------------------------------------------------------------
# ROI: conversion de cuotas
# --------------------------------------------------------------------------

def test_americana_a_decimal_en_los_dos_signos():
    assert R.to_decimal(american=-110) == pytest.approx(1.909091, abs=1e-6)
    assert R.to_decimal(american=+150) == pytest.approx(2.5, abs=1e-9)
    assert R.to_decimal(american=+100) == pytest.approx(2.0, abs=1e-9)


def test_americana_imposible_se_rechaza():
    """Entre -100 y +100 no existe ningun precio americano.

    Los datos reales traen valores asi (un -9.0 en un handicap de tenis) y
    convertirlos daria cuota 12.1, o sea una ganancia inventada del 1100 %.
    """
    for mala in (-9.0, 0, 50, -99.9, 99):
        assert R.to_decimal(american=mala) is None, mala


def test_ida_y_vuelta_decimal_americana():
    for d in (1.5, 1.91, 2.0, 3.75, 11.0):
        assert R.to_decimal(decimal=R.to_decimal(
            american=R.to_american(d))) == pytest.approx(d, abs=0.01)


def test_decimal_invalida():
    assert R.to_decimal(decimal=1.0) is None
    assert R.to_decimal(decimal=0.5) is None
    assert R.to_decimal(decimal="hola") is None


# --------------------------------------------------------------------------
# ROI: liquidacion
# --------------------------------------------------------------------------

def test_profit_de_cada_resultado():
    assert R.profit("win", decimal=2.5) == pytest.approx(1.5)
    assert R.profit("loss", decimal=2.5) == pytest.approx(-1.0)
    assert R.profit("push", decimal=2.5) == 0.0
    assert R.profit("void", decimal=2.5) == 0.0


def test_win_sin_cuota_no_inventa_ganancia():
    assert R.profit("win") is None
    assert R.profit("loss") is None


def test_push_y_void_se_cuentan_distinto():
    """Push ocurrio; void no. El void no debe entrar como apuesta hecha."""
    t = R.liquidar([{"result": "push", "odds_decimal": 2.0},
                    {"result": "void", "odds_decimal": 2.0}])
    assert t["pushes"] == 1
    assert t["voids"] == 1
    assert t["n_contabilizadas"] == 1        # solo el push


def test_roi_conocido():
    ap = [{"result": "win", "odds_decimal": 2.0}] * 6 + \
         [{"result": "loss", "odds_decimal": 2.0}] * 4
    t = R.liquidar(ap)
    assert t["profit_unidades"] == pytest.approx(2.0)
    assert t["roi"] == pytest.approx(0.2)
    assert t["yield_"] == t["roi"]
    assert t["win_rate"] == pytest.approx(0.6)


def test_muestra_corta_se_advierte():
    t = R.liquidar([{"result": "win", "odds_decimal": 2.0}] * 5)
    assert "azar" in t["advertencia"]
    assert "NO demuestra ventaja" in t["advertencia"]


def test_sin_cuota_no_hay_roi():
    t = R.liquidar([{"result": "win"}, {"result": "loss"}])
    assert t["roi"] is None
    assert t["sin_cuota"] == 2
    assert "no se puede calcular ROI" in t["advertencia"]


# --------------------------------------------------------------------------
# Discrepancia
# --------------------------------------------------------------------------

def test_buckets_de_gap():
    assert D.bucket(1.0) == "0-3 pp"
    assert D.bucket(-4.0) == "3-5 pp"          # se usa el valor absoluto
    assert D.bucket(9.9) == "8-10 pp"
    assert D.bucket(40) == "15+ pp"
    assert D.bucket(None) is None


def test_gap_pequeno_es_alineado_no_desacuerdo():
    """Por debajo de medio overround no se discrepa del mercado: de su comision."""
    r = D.evaluar(model_cal=0.66, market=0.65, overround=0.05)
    assert r["FINAL_STATUS"] == "ALIGNED"


def test_modelo_por_encima_del_mercado_no_es_valor():
    """La regla central de todo el modulo."""
    r = D.evaluar(model_cal=0.64, market=0.53, overround=0.04)
    assert r["FINAL_STATUS"] == "UNVALIDATED"
    assert r["direccion"] == "MODEL > MARKET"
    assert "NO como valor" in r["explicacion"]
    assert r["FINAL_STATUS"] != "VALIDATED VALUE"


def test_sin_evidencia_nunca_hay_validated_value():
    for gap in (0.06, 0.12, 0.30):
        r = D.evaluar(model_cal=0.5 + gap, market=0.5, overround=0.04)
        assert r["FINAL_STATUS"] != "VALIDATED VALUE"


def test_con_evidencia_positiva_si_se_valida():
    ev = {"buckets": [{"bucket": "10-15 pp", "n": 500, "roi": 0.08,
                       "roi_ic95": (0.02, 0.14)}]}
    r = D.evaluar(model_cal=0.62, market=0.50, overround=0.04, evidencia=ev)
    assert r["FINAL_STATUS"] == "VALIDATED VALUE"


def test_evidencia_con_cero_dentro_del_intervalo_no_valida():
    ev = {"buckets": [{"bucket": "10-15 pp", "n": 500, "roi": 0.03,
                       "roi_ic95": (-0.05, 0.11)}]}
    r = D.evaluar(model_cal=0.62, market=0.50, overround=0.04, evidencia=ev)
    assert r["FINAL_STATUS"] == "UNVALIDATED"
    assert "el cero esta dentro" in r["explicacion"]


def test_muestra_insuficiente_no_valida_aunque_el_roi_sea_alto():
    ev = {"buckets": [{"bucket": "15+ pp", "n": 12, "roi": 0.9,
                       "roi_ic95": (0.4, 1.4)}]}
    r = D.evaluar(model_cal=0.75, market=0.50, overround=0.04, evidencia=ev)
    assert r["FINAL_STATUS"] == "UNVALIDATED"
    assert "12 casos" in r["explicacion"]


def test_sin_mercado_no_se_inventa_comparacion():
    r = D.evaluar(model_cal=0.7, market=None)
    assert r["FINAL_STATUS"] == "NO MARKET"
    assert r["gap_pp"] is None


def test_veredicto_sin_evidencia_lo_dice_claro():
    bt = D.backtest_buckets([])
    v = D.veredicto(bt)
    assert v["con_evidencia_de_valor"] == []
    assert "NO SE HA DEMOSTRADO VENTAJA RENTABLE" in v["conclusion"]


def test_backtest_separa_los_tramos():
    regs = ([{"gap_pp": 1.0, "result": "win", "odds_decimal": 2.0, "probability": 0.6}] * 3
            + [{"gap_pp": 12.0, "result": "loss", "odds_decimal": 2.0, "probability": 0.7}] * 2)
    bt = D.backtest_buckets(regs)
    por = {b["bucket"]: b for b in bt["buckets"]}
    assert por["0-3 pp"]["n"] == 3 and por["0-3 pp"]["wins"] == 3
    assert por["10-15 pp"]["n"] == 2 and por["10-15 pp"]["losses"] == 2
    assert bt["estado"] == "insufficient_data"


# --------------------------------------------------------------------------
# Calibracion
# --------------------------------------------------------------------------

def test_aplicar_ninguna_no_toca_la_probabilidad():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(C.apply(None, p), p)
    assert np.allclose(C.apply({"metodo": "ninguna"}, p), p)


def test_muestra_corta_no_se_calibra():
    r = C.seleccionar(np.random.rand(50), np.random.randint(0, 2, 50))
    assert r["elegido"] == "ninguna"
    assert r["estado"] == "insufficient_data"
    assert "muestra insuficiente" in r["motivo"]


def test_un_modelo_ya_calibrado_no_se_toca():
    """Si las probabilidades ya son correctas, ningun calibrador debe ganar."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    r = C.seleccionar(p, y)
    assert r["elegido"] == "ninguna"
    assert "ningun calibrador mejora" in r["motivo"]


def test_un_modelo_sobreconfiado_si_se_corrige():
    """Exceso de confianza: el calibrador debe encoger hacia el centro."""
    rng = np.random.default_rng(11)
    p_real = rng.uniform(0.35, 0.65, 6000)
    y = (rng.uniform(size=6000) < p_real).astype(float)
    # se infla la confianza estirando el logit
    z = np.log(p_real / (1 - p_real)) * 2.5
    p_infl = 1 / (1 + np.exp(-z))
    r = C.seleccionar(p_infl, y)
    assert r["elegido"] != "ninguna"
    assert r["holdout"]["mejora_log_loss"] > 0


def test_los_tres_bloques_son_disjuntos():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.1, 0.9, 2000)
    y = (rng.uniform(size=2000) < p).astype(float)
    r = C.seleccionar(p, y)
    b = r["bloques"]
    assert b["ajuste"] + b["seleccion"] + b["holdout"] == 2000
    assert min(b.values()) > 0


def test_platt_e_isotonica_son_serializables_a_json():
    import json
    rng = np.random.default_rng(5)
    p = rng.uniform(0.1, 0.9, 1000)
    y = (rng.uniform(size=1000) < p).astype(float)
    for fn in (C.fit_platt, C.fit_isotonic, C.fit_beta):
        cal = fn(p, y)
        assert cal is not None
        json.dumps(cal)                      # no debe llevar objetos de sklearn
        assert 0 < C.apply(cal, 0.7) < 1
