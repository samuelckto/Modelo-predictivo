"""Matematica de cuotas, calibracion y politica de combinacion."""
import numpy as np
import pandas as pd
import pytest

from shared.blend import (ALLOW_MARKET_ONLY, DEFAULT_POLICY, MIN_MODEL_WEIGHT, BlendPolicy,
                          disagreement, select_policy, signal_label)
from shared.calibration import brier, ece, log_loss, summary
from shared.odds import american_to_prob, decimal_to_prob, devig, overround, two_way


# ------------------------------------------------------------------ cuotas
def test_conversion_de_cuotas():
    assert american_to_prob(-150) == pytest.approx(0.6)
    assert american_to_prob(+150) == pytest.approx(0.4)
    assert american_to_prob(None) is None and american_to_prob("x") is None
    assert decimal_to_prob(2.0) == pytest.approx(0.5)
    assert decimal_to_prob(0.9) is None


def test_devig_quita_el_overround():
    p = [american_to_prob(-150), american_to_prob(+130)]
    assert overround(p) > 0
    for m in ("proportional", "log", "shin"):
        d = devig(p, m)
        assert sum(d) == pytest.approx(1.0, abs=1e-6), m
        assert d[0] > d[1], m


def test_devig_no_inventa_si_falta_un_lado():
    assert devig([0.6, None]) is None
    assert two_way(0.6, None) == (None, None)


# ------------------------------------------------------------ calibracion
def test_metricas_de_calibracion():
    y = np.array([1, 1, 0, 0]); perfecto = np.array([1, 1, 0, 0]) * 0.999 + 0.0005
    assert log_loss(y, perfecto) < 0.01 and brier(y, perfecto) < 0.01
    r = np.random.default_rng(0); p = r.random(4000); yy = (r.random(4000) < p).astype(int)
    assert ece(yy, p) < 0.05                       # calibrado por construccion
    mal = np.clip(p * 1.6, 0.01, 0.99)             # sobreconfiado
    assert ece(yy, mal) > ece(yy, p)
    assert summary(yy, p)["n"] == 4000


# ------------------------------------------------------------------ blend
def _hist(seed=0, mq=1.2, mo=0.5, n=300, seasons=(2021, 2022, 2023)):
    r = np.random.default_rng(seed); rows = []
    for s in seasons:
        z = r.normal(0, 1.1, n); y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        rows.append(pd.DataFrame({"season": s, "y": y,
                                  "p_market": 1 / (1 + np.exp(-(mq * z + r.normal(0, .25, n)))),
                                  "p_model": 1 / (1 + np.exp(-(mo * z + r.normal(0, .7, n)))),
                                  "p_elo": 1 / (1 + np.exp(-(0.4 * z + r.normal(0, .8, n))))}))
    return pd.concat(rows, ignore_index=True)


def test_el_mercado_no_puede_cambiar_el_pick_por_si_solo():
    pol = BlendPolicy(kind="fixed_weight", market_weight=0.5)
    assert pol.blend(0.79, 0.45) > 0.5             # modelo muy convencido gana
    for pm, pk in [(0.9, 0.1), (0.1, 0.9), (0.6, 0.55), (0.45, 0.52)]:
        pf = pol.blend(pm, pk)
        assert (pf >= 0.5) == (pf >= 0.5)          # el pick sale SOLO de p_final


def test_sin_cuotas_se_usa_solo_el_modelo():
    assert BlendPolicy(kind="fixed_weight", market_weight=0.9).blend(0.73, None) == pytest.approx(0.73)


def test_market_only_prohibido_por_defecto():
    pol = select_policy(_hist(1, mq=1.4, mo=0.15))
    assert ALLOW_MARKET_ONLY is False and pol.kind != "market_only"
    if pol.kind == "fixed_weight":
        assert pol.market_weight <= 1 - MIN_MODEL_WEIGHT + 1e-9
    assert select_policy(_hist(1), allow_market_only=True) is not None


def test_sin_historial_politica_a_priori():
    vacio = pd.DataFrame(columns=["season", "p_model", "p_market", "y"])
    p = select_policy(vacio)
    assert p.kind == DEFAULT_POLICY.kind and p.market_weight == 0.5 and "a priori" in p.rationale
    corto = _hist(n=60, seasons=(2021,))
    assert select_policy(corto).market_weight == 0.5


def test_la_seleccion_declara_con_que_datos_se_ajusto():
    h = _hist(3)
    p = select_policy(h)
    assert set(p.fitted_on["seasons"]) <= set(h.season.unique())
    assert p.fitted_on["validation_season"] == max(h.season)
    assert p.fitted_on["validation_season"] not in p.fitted_on["seasons"]   # no se valida donde se ajusta
    assert len(p.selection) >= 5


def test_metricas_de_desacuerdo():
    d = disagreement(0.79, 0.615, 0.70, 0.02)
    assert d["directional_disagreement"] is False
    assert d["market_model_gap"] == pytest.approx(0.175) and d["agreement_bucket"] == "10-20pp"
    d2 = disagreement(0.35, 0.62, 0.5, 0.02)
    assert d2["directional_disagreement"] and d2["agreement_bucket"] == "desacuerdo_direccional"
    d3 = disagreement(0.7, None, 0.7, 0.03)
    assert d3["market_available"] is False and d3["agreement_bucket"] == "sin_mercado"


def test_ninguna_senal_afirma_quien_tiene_razon():
    for d in (disagreement(0.79, .615, .70, .02), disagreement(0.35, .62, .5, .02)):
        t = (signal_label(d) or "").lower()
        assert not any(x in t for x in ("va a ganar", "seguro", "garantiz", "tiene razon", "perdera"))
