"""Data leakage: informacion posterior al momento de la prediccion NUNCA puede usarse.

Escenario del enunciado: partido 20:00, prediccion 12:00. Un lineup confirmado a
las 17:00, una lesion de las 16:00, cuotas de las 19:30 y clima de las 18:00 no
pueden aparecer en las features de esa prediccion.
"""
import pandas as pd
import pytest

from shared.asof import (LeakageError, asof_join, assert_no_leakage, audit_frame,
                         filter_available, latest_before)

PRED = pd.Timestamp("2026-04-10 12:00")


def _eventos():
    return pd.DataFrame([
        {"game_id": 1, "kind": "lineup",  "state": "projected", "available_at": "2026-04-10 09:00"},
        {"game_id": 1, "kind": "lineup",  "state": "confirmed", "available_at": "2026-04-10 17:00"},
        {"game_id": 1, "kind": "injury",  "state": "IL-10",     "available_at": "2026-04-10 16:00"},
        {"game_id": 1, "kind": "odds",    "state": "close",     "available_at": "2026-04-10 19:30"},
        {"game_id": 1, "kind": "weather", "state": "obs",       "available_at": "2026-04-10 18:00"},
        {"game_id": 1, "kind": "odds",    "state": "open",      "available_at": "2026-04-09 22:00"},
    ])


def test_solo_sobrevive_lo_anterior_a_la_prediccion():
    d = filter_available(_eventos(), PRED)
    assert set(d.state) == {"projected", "open"}
    assert "confirmed" not in set(d.state)


def test_el_lineup_confirmado_posterior_no_entra():
    d = latest_before(_eventos()[lambda x: x.kind == "lineup"], PRED, ["game_id"])
    assert len(d) == 1 and d.iloc[0].state == "projected"


def test_las_cuotas_de_cierre_no_entran_en_una_prediccion_previa():
    d = filter_available(_eventos()[lambda x: x.kind == "odds"], PRED)
    assert set(d.state) == {"open"}


def test_timestamp_exacto_se_considera_no_disponible():
    df = pd.DataFrame([{"game_id": 1, "available_at": PRED}])
    assert len(filter_available(df, PRED)) == 0


def test_fila_sin_timestamp_se_descarta():
    df = pd.DataFrame([{"game_id": 1, "available_at": None},
                       {"game_id": 1, "available_at": "2026-04-10 08:00"}])
    assert len(filter_available(df, PRED)) == 1


def test_assert_no_leakage_detecta_la_violacion():
    with pytest.raises(LeakageError):
        assert_no_leakage(_eventos(), PRED, label="eventos")
    assert_no_leakage(filter_available(_eventos(), PRED), PRED)   # no lanza


def test_cutoff_invalido_no_pasa_en_silencio():
    with pytest.raises(LeakageError):
        filter_available(_eventos(), None)


def test_asof_no_usa_la_fila_del_propio_partido():
    juegos = pd.DataFrame([{"team": "LAD", "ts": "2026-04-10 12:00"},
                           {"team": "LAD", "ts": "2026-04-12 12:00"}])
    stats = pd.DataFrame([{"team": "LAD", "ts": "2026-04-10 12:00", "wrc": 999},
                          {"team": "LAD", "ts": "2026-04-08 12:00", "wrc": 105}])
    out = asof_join(juegos, stats, on="ts", by="team")
    assert out.iloc[0].wrc == 105          # NO toma el 999 del mismo instante
    assert out.iloc[1].wrc == 999          # dos dias despues ya es pasado: es valido


def test_auditoria_posterior_cuenta_violaciones():
    df = _eventos().assign(prediction_timestamp=PRED)
    a = audit_frame(df, "prediction_timestamp", ["available_at"])
    assert a["rows"] == 6 and a["total_violations"] == 4
    limpio = filter_available(_eventos(), PRED).assign(prediction_timestamp=PRED)
    assert audit_frame(limpio, "prediction_timestamp", ["available_at"])["total_violations"] == 0
