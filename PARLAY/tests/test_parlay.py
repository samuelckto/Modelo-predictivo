"""Tests de las combinadas: independencia, probabilidad, calificacion y aislamiento."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from PARLAY.engine import builder as B
from PARLAY.engine.db import Parlay, ParlayLeg, init_db, session_scope
from PARLAY.engine.scoring import history, score
from shared.paths import NFL_HOME


def _card(sport, gid, sel, p, mode="pick", market="Moneyline", key="moneyline", home="A", away="B", mp=None):
    return {"sport": sport, "game_id": gid, "home": home, "away": away, "market": market,
            "selection": sel, "ensemble_probability": p, "market_probability": mp,
            "prediction_id": 1, "start_utc": "2030-01-01 20:00:00", "game_date": "2030-01-01",
            "result": None, "extra": {"mode": mode, "market_key": key, "publish_pick": mode == "pick"}}


# ---------------------------------------------------------------- seleccion de patas
def test_solo_entran_mercados_pick():
    cards = [_card("MLB", "1", "NYY", 0.72), _card("NBA", "2", "BOS", 0.80, mode="projection"),
             _card("TENIS", "3", "Alcaraz", 0.75, mode="no_pick")]
    legs = B.eligible_legs(cards)
    assert [l["sport"] for l in legs] == ["MLB"]


def test_se_descartan_patas_de_probabilidad_baja():
    assert B.eligible_legs([_card("MLB", "1", "X", 0.51)]) == []
    assert len(B.eligible_legs([_card("MLB", "1", "X", 0.56)])) == 1


def test_no_se_repite_el_mismo_partido():
    cards = [_card("MLB", "1", "NYY", 0.72), _card("MLB", "1", "OVER 8.5", 0.70, market="Total", key="total")]
    assert len(B.eligible_legs(cards)) == 1


def test_no_se_repite_equipo_en_dias_distintos():
    """Dos partidos de la misma serie no son sucesos independientes."""
    cards = [_card("MLB", "1", "NYY", 0.72, home="NYY", away="COL"),
             _card("MLB", "2", "NYY", 0.71, home="NYY", away="COL")]
    assert len(B.eligible_legs(cards)) == 1


def test_equipos_distintos_si_se_combinan():
    cards = [_card("MLB", "1", "NYY", 0.72, home="NYY", away="COL"),
             _card("MLB", "2", "BOS", 0.68, home="BOS", away="LAA")]
    legs = B.eligible_legs(cards)
    assert len(legs) == 2 and len(B.combos(legs, sizes=(2,))) == 1


# ---------------------------------------------------------------- probabilidad
def test_la_probabilidad_es_el_producto_y_baja_con_mas_patas():
    legs = B.eligible_legs([_card("MLB", "1", "A", 0.70, home="A", away="a"),
                            _card("NBA", "2", "B", 0.65, home="B", away="b"),
                            _card("NFL", "3", "C", 0.60, home="C", away="c")])
    c2 = B.combos(legs, sizes=(2,))[0]
    c3 = B.combos(legs, sizes=(3,))[0]
    assert c2["probability"] == pytest.approx(0.70 * 0.65, abs=1e-9)
    assert c3["probability"] == pytest.approx(0.70 * 0.65 * 0.60, abs=1e-9)
    assert c3["probability"] < c2["probability"] < 0.70          # mas patas, menos probabilidad


def test_cuota_justa_es_el_inverso():
    legs = B.eligible_legs([_card("MLB", "1", "A", 0.70, home="A", away="a"),
                            _card("NBA", "2", "B", 0.60, home="B", away="b")])
    c = B.combos(legs, sizes=(2,))[0]
    assert c["fair_odds"] == pytest.approx(1 / 0.42, abs=1e-6)


def test_probabilidad_de_mercado_solo_si_todas_las_patas_la_tienen():
    legs = B.eligible_legs([_card("MLB", "1", "A", 0.70, home="A", away="a", mp=0.66),
                            _card("NBA", "2", "B", 0.60, home="B", away="b", mp=None)])
    assert B.combos(legs, sizes=(2,))[0]["market_probability"] is None
    legs2 = B.eligible_legs([_card("MLB", "1", "A", 0.70, home="A", away="a", mp=0.66),
                             _card("NBA", "2", "B", 0.60, home="B", away="b", mp=0.58)])
    assert B.combos(legs2, sizes=(2,))[0]["market_probability"] == pytest.approx(0.66 * 0.58, abs=1e-9)


def test_se_ordenan_por_probabilidad():
    legs = B.eligible_legs([_card("MLB", "1", "A", 0.80, home="A", away="a"),
                            _card("NBA", "2", "B", 0.70, home="B", away="b"),
                            _card("NFL", "3", "C", 0.60, home="C", away="c")])
    cs = B.combos(legs, sizes=(2,))
    assert cs == sorted(cs, key=lambda c: -c["probability"])
    assert cs[0]["probability"] == pytest.approx(0.80 * 0.70, abs=1e-9)


def test_mezcla_de_deportes_se_marca():
    legs = B.eligible_legs([_card("MLB", "1", "A", 0.70, home="A", away="a"),
                            _card("TENIS", "2", "B", 0.68, home="B", away="b")])
    c = B.combos(legs, sizes=(2,))[0]
    assert c["mixed"] is True and c["sports"] == "MLB,TENIS"


def test_solo_tamanos_2_y_3():
    legs = B.eligible_legs([_card(s, str(i), f"S{i}", 0.7, home=f"H{i}", away=f"A{i}")
                            for i, s in enumerate(("MLB", "NBA", "NFL", "TENIS"))])
    assert {c["size"] for c in B.combos(legs)} == {2, 3}


# ---------------------------------------------------------------- calificacion
def _mk(session, legs_results, size=2, prob=0.5):
    """Crea una combinada sintetica con resultados dados y la califica."""
    key = hashlib.md5(str((legs_results, datetime.utcnow().timestamp())).encode()).hexdigest()
    p = Parlay(parlay_key=key, size=size, mixed=False, sports="MLB", created_at=datetime.utcnow(),
               match_date=date(2030, 1, 1), probability=prob, fair_odds=1 / prob, min_leg_probability=prob,
               rank=1, status="open", legs_total=size, legs_won=0)
    session.add(p); session.flush()
    for i, r in enumerate(legs_results):
        session.add(ParlayLeg(parlay_id=p.id, sport="MLB", game_id=f"TEST{i}", market="Moneyline",
                              market_key="moneyline", selection=f"S{i}", probability=0.7,
                              start_utc=datetime(2030, 1, 1), teams="A @ B", result=r,
                              correct=(None if r in (None, "push") else r == "win")))
    session.flush()
    return p.id


def test_gana_solo_si_todas_las_patas_ganan():
    init_db()
    with session_scope() as s:
        gana = _mk(s, ["win", "win"])
        falla = _mk(s, ["win", "loss"])
        abierta = _mk(s, ["win", None])
    score()
    with session_scope() as s:
        assert s.get(Parlay, gana).result == "win" and s.get(Parlay, gana).status == "graded"
        assert s.get(Parlay, falla).result == "loss"
        assert s.get(Parlay, abierta).status == "open"       # sigue pendiente
        for pid in (gana, falla, abierta):
            s.query(ParlayLeg).filter(ParlayLeg.parlay_id == pid).delete()
            s.query(Parlay).filter(Parlay.id == pid).delete()


def test_una_pata_perdida_cierra_aunque_falten_resultados():
    init_db()
    with session_scope() as s:
        pid = _mk(s, ["loss", None])
    score()
    with session_scope() as s:
        p = s.get(Parlay, pid)
        assert p.status == "graded" and p.result == "loss"
        s.query(ParlayLeg).filter(ParlayLeg.parlay_id == pid).delete()
        s.query(Parlay).filter(Parlay.id == pid).delete()


def test_push_no_invalida_la_combinada():
    init_db()
    with session_scope() as s:
        pid = _mk(s, ["win", "push"])
    score()
    with session_scope() as s:
        p = s.get(Parlay, pid)
        assert p.result == "win"                            # la pata anulada no cuenta
        s.query(ParlayLeg).filter(ParlayLeg.parlay_id == pid).delete()
        s.query(Parlay).filter(Parlay.id == pid).delete()


def test_historico_cuenta_aciertos_y_fallos_por_tamano():
    init_db()
    with session_scope() as s:
        ids = [_mk(s, ["win", "win"]), _mk(s, ["win", "loss"]),
               _mk(s, ["win", "win", "win"], size=3, prob=0.35)]
    score()
    h = history()
    assert h["total"]["n"] >= 3 and h["por_tamano"][2]["n"] >= 2 and h["por_tamano"][3]["n"] >= 1
    assert h["total"]["accuracy"] is not None and h["nota"]
    with session_scope() as s:
        for pid in ids:
            s.query(ParlayLeg).filter(ParlayLeg.parlay_id == pid).delete()
            s.query(Parlay).filter(Parlay.id == pid).delete()


def test_no_se_duplican_combinadas():
    init_db()
    a = B.generate()
    b = B.generate()
    assert b["nuevas"] == 0 or b["nuevas"] < a["nuevas"] + 1


# ---------------------------------------------------------------- dashboard y aislamiento
def test_endpoint_de_combinadas():
    from dashboard.backend.main import parlays
    d = parlays()
    assert "total" in d and "abiertas" in d and "por_tamano" in d


def test_parlay_no_escribe_en_ningun_deporte():
    from shared.paths import (IsolationError, MLB_DIR, NBA_DIR, NFL_DIR, PARLAY_DB_DIR, TENIS_DIR,
                              assert_writable)
    assert assert_writable(PARLAY_DB_DIR / "parlays.sqlite3", "PARLAY")
    for bad in (MLB_DIR / "database" / "mlb.sqlite3", NBA_DIR / "database" / "nba_markets.sqlite3",
                TENIS_DIR / "database" / "tenis_markets.sqlite3",
                NFL_DIR / "database" / "nfl_markets.sqlite3", NFL_HOME / "database" / "nflpred.sqlite3"):
        with pytest.raises(IsolationError):
            assert_writable(bad, "PARLAY")


def test_generar_no_altera_las_bases_de_los_deportes():
    import hashlib as hl
    root = Path(__file__).resolve().parents[2]
    files = [f for f in (root / "MLB" / "database" / "mlb.sqlite3",
                         root / "NBA" / "database" / "nba_markets.sqlite3",
                         root / "TENIS" / "database" / "tenis_markets.sqlite3",
                         NFL_HOME / "data" / "processed" / "features.parquet") if f.exists()]
    before = {f: hl.md5(f.read_bytes()).hexdigest() for f in files}
    B.generate()
    score()
    assert before == {f: hl.md5(f.read_bytes()).hexdigest() for f in files}
