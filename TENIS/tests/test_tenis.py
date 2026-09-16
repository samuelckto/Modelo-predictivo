"""Tests del modulo TENIS. No tocan NFL, MLB, NBA ni el motor NFL original."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from TENIS.markets import predict as P
from TENIS.markets.db import (TenisOdds, TenisPrediction, TenisSchedule, init_db, session_scope)
from TENIS.markets.features import FEATURES_PARQUET, ID_COLS, TARGET_COLS, families
from TENIS.markets.gating import decide
from TENIS.markets.ingest import parse_score
from TENIS.markets.odds import best_of, consensus, match_player, movement, surface_of, tour_of
from TENIS.markets.pointmodel import (game_prob, match_distribution, p_cover_games, p_over_games,
                                      push_prob, solve_serve_probs, tiebreak_prob)
from shared.odds import american_to_prob
from shared.paths import NFL_HOME, TENIS_MODELS_DIR, TENIS_OUT_DIR

needs_data = pytest.mark.skipif(not FEATURES_PARQUET.exists(), reason="sin features de tenis")
needs_research = pytest.mark.skipif(not (TENIS_OUT_DIR / "research_lines.json").exists(), reason="sin investigacion")
needs_models = pytest.mark.skipif(not (TENIS_MODELS_DIR / "ATP_winner_v1.joblib").exists(), reason="sin modelos")


# ------------------------------------------------------------- modelo punto a punto
def test_juego_al_saque_es_monotono_y_conocido():
    assert game_prob(0.5) == pytest.approx(0.5, abs=1e-9)
    assert game_prob(0.62) == pytest.approx(0.7759, abs=1e-3)      # valor clasico del tenis masculino
    assert game_prob(0.7) > game_prob(0.62) > game_prob(0.55)


def test_tiebreak_simetrico():
    assert tiebreak_prob(0.62, 0.62) == pytest.approx(0.5, abs=1e-6)
    assert tiebreak_prob(0.70, 0.60) > 0.5 > tiebreak_prob(0.60, 0.70)


def test_partido_simetrico_y_distribucion_valida():
    d = match_distribution(0.62, 0.62, 3)
    assert d["p_win"] == pytest.approx(0.5, abs=1e-6)
    assert sum(d["games_total"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(d["games_margin"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 18 <= d["exp_games"] <= 30


def test_best_of_5_amplifica_la_ventaja_y_alarga_el_partido():
    a = match_distribution(0.66, 0.60, 3)
    b = match_distribution(0.66, 0.60, 5)
    assert b["p_win"] > a["p_win"]
    assert b["exp_games"] > a["exp_games"] * 1.4


def test_probabilidades_de_linea_coherentes():
    d = match_distribution(0.66, 0.60, 3)
    assert p_over_games(d, 0) == pytest.approx(1.0, abs=1e-6)
    assert p_over_games(d, 100) == pytest.approx(0.0, abs=1e-9)
    assert p_over_games(d, 20.5) > p_over_games(d, 25.5)
    assert p_cover_games(d, 0) > p_cover_games(d, -6.5)             # dar juegos es mas dificil
    assert p_cover_games(d, 6.5) > p_cover_games(d, 0)


def test_push_solo_con_linea_entera():
    d = match_distribution(0.64, 0.62, 3)
    assert push_prob(d, 22.5, "total") == 0.0
    assert push_prob(d, 22, "total") > 0.0
    assert push_prob(d, 3.5, "handicap") == 0.0


def test_solver_invierte_la_probabilidad():
    for target in (0.55, 0.70, 0.85):
        p1, p2 = solve_serve_probs(target, 0.62, 3)
        assert match_distribution(p1, p2, 3)["p_win"] == pytest.approx(target, abs=0.01)


# ------------------------------------------------------------- marcadores y calificacion
def test_lectura_de_marcadores():
    assert parse_score("6-4 6-3", False) == (12, 7, 2, 0)
    assert parse_score("7-6(5) 6-7(3) 7-5", False) == (20, 18, 2, 1)
    assert parse_score("W/O", False) == (None, None, None, None)
    assert parse_score("6-2 RET", True)[0] == 6


def _grade(market, line, games_total, margin, p_side1):
    if market == "winner":
        return "win" if ((margin > 0) == (p_side1 >= .5)) else "loss"
    if market == "total_games":
        if games_total == line:
            return "push"
        return "win" if ((games_total > line) == (p_side1 >= .5)) else "loss"
    cover = margin + line
    if cover == 0:
        return "push"
    return "win" if ((cover > 0) == (p_side1 >= .5)) else "loss"


def test_calificacion_ganador():
    assert _grade("winner", None, 22, 4, 0.7) == "win"
    assert _grade("winner", None, 22, -4, 0.7) == "loss"
    assert _grade("winner", None, 22, -4, 0.3) == "win"


def test_calificacion_total_over_under_y_push():
    assert _grade("total_games", 22.5, 23, 3, 0.6) == "win"
    assert _grade("total_games", 22.5, 21, 3, 0.6) == "loss"
    assert _grade("total_games", 22.5, 21, 3, 0.4) == "win"
    assert _grade("total_games", 22, 22, 2, 0.6) == "push"


def test_calificacion_handicap_ambos_signos_y_cero():
    assert _grade("handicap_games", -3.5, 24, 5, 0.6) == "win"       # gana por 5, da 3.5
    assert _grade("handicap_games", -3.5, 24, 3, 0.6) == "loss"
    assert _grade("handicap_games", 3.5, 24, -2, 0.6) == "win"       # pierde por 2, recibe 3.5
    assert _grade("handicap_games", 0.0, 24, 0, 0.6) == "push"
    assert _grade("handicap_games", -4.0, 24, 4, 0.6) == "push"


def test_linea_de_referencia_termina_en_medio():
    assert P._ref_line(22.3) == 22.5 and P._ref_line(-3.8) == -3.5


# ------------------------------------------------------------- cuotas y calendario
def test_circuito_superficie_y_sets_del_torneo():
    assert tour_of("tennis_atp_wimbledon") == "ATP" and tour_of("tennis_wta_us_open") == "WTA"
    assert surface_of("ATP Roland Garros") == "Clay" and surface_of("WTA Wimbledon") == "Grass"
    assert best_of("tennis_atp_us_open", "ATP") == 5 and best_of("tennis_wta_us_open", "WTA") == 3


@needs_data
def test_emparejado_de_jugadores():
    init_db()
    with session_scope() as s:
        pid, name = match_player(s, "ATP", "Carlos Alcaraz")
        assert pid and name == "Carlos Alcaraz"
        assert match_player(s, "ATP", "Jugador Que No Existe")[0] is None


def test_no_vig_y_cuotas_posteriores_excluidas():
    init_db()
    eid = "TEST_TENIS_ODDS"
    now = datetime(2030, 5, 1, 12)
    with session_scope() as s:
        s.query(TenisOdds).filter(TenisOdds.event_id == eid).delete()
        for sel, price, line, when in (("p1", -200, None, now - timedelta(hours=4)),
                                       ("p2", +170, None, now - timedelta(hours=4)),
                                       ("p1", -250, None, now - timedelta(hours=1)),
                                       ("p2", +200, None, now - timedelta(hours=1)),
                                       ("p1", -400, None, now + timedelta(hours=1))):
            s.add(TenisOdds(event_id=eid, bookmaker="bk", market="winner", selection=sel, line=line,
                            price_american=price, implied_prob=american_to_prob(price), available_at=when))
        s.flush()
        c = consensus(s, eid, "winner", now)
        esperado = american_to_prob(-250) / (american_to_prob(-250) + american_to_prob(200))
        assert c["p1"] == pytest.approx(esperado, abs=1e-9)
        assert consensus(s, eid, "winner", now - timedelta(hours=5)) is None
        assert movement(s, eid, "winner")["movimiento_pp"] > 0
        s.query(TenisOdds).filter(TenisOdds.event_id == eid).delete()


# ------------------------------------------------------------- leakage
@needs_data
def test_ninguna_feature_es_resultado_ni_mercado():
    X = pd.read_parquet(FEATURES_PARQUET)
    for name, cols in families(X).items():
        assert not (set(cols) & TARGET_COLS), name
        assert not (set(cols) & ID_COLS), name
        assert not any(any(k in c for k in ("odds", "line", "market", "score", "minutes"))
                       for c in cols), name


@needs_data
def test_las_fuentes_son_anteriores_al_dia_del_partido():
    X = pd.read_parquet(FEATURES_PARQUET)
    assert (X.max_source_available_at < X.cutoff_utc).all()


@needs_data
def test_p1_no_es_el_ganador():
    """p1/p2 se asignan por id, no por resultado: si no, el modelo aprenderia el resultado."""
    X = pd.read_parquet(FEATURES_PARQUET)
    assert 0.45 < X.p1_win.mean() < 0.55
    assert (X.p1_id < X.p2_id).all()


@needs_data
def test_el_primer_partido_de_un_jugador_no_tiene_historial():
    """El DEBUT de cada jugador (como p1 o como p2) no puede traer estadisticas."""
    X = pd.read_parquet(FEATURES_PARQUET)
    largo = pd.concat([
        X[["date", "match_id", "p1_id", "p1_y1_n"]].rename(columns={"p1_id": "pid", "p1_y1_n": "n"}),
        X[["date", "match_id", "p2_id", "p2_y1_n"]].rename(columns={"p2_id": "pid", "p2_y1_n": "n"})],
        ignore_index=True).sort_values(["date", "match_id"])
    debut = largo.drop_duplicates("pid", keep="first")
    assert len(debut) > 5000
    assert debut.n.isna().mean() > 0.95


@needs_research
def test_walk_forward_no_usa_la_temporada_de_prueba():
    r = json.loads((TENIS_OUT_DIR / "research.json").read_text())
    assert r["select"] == [2015, 2016, 2017, 2018, 2019]
    assert min(r["holdout"]) > max(r["select"])
    d = pd.read_parquet(TENIS_OUT_DIR / "preds_ATP.parquet")
    assert d.season.min() >= 2015 and d.p_elo_model.notna().all()


# ------------------------------------------------------------- gating y calibracion
@needs_research
def test_estados_coherentes_con_la_evidencia():
    g = decide()
    for tour in ("ATP", "WTA"):
        w = g[tour]["winner"]
        if w["publish_pick"]:
            assert w["picks"]["z_vs_50"] >= 1.96 and w["picks"]["z_vs_elo"] >= 1.64
            assert w["holdout"]["ece"] < 0.05 and w["threshold"] is not None
        for m in ("total_games", "handicap_games"):
            assert g[tour][m]["publish_pick"] is False
            if g[tour][m]["mode"] == "projection":
                h = g[tour][m]["holdout"]
                assert h["mae"] < h.get("mae_media", h.get("mae_cero"))


@needs_research
def test_mercado_bloqueado_cuando_no_mejora_al_baseline():
    g = decide()
    for tour in ("ATP", "WTA"):
        b = g[tour]["total_games"]
        h = b["holdout"]
        if b["mode"] == "blocked":
            assert h["mae"] >= h["mae_media"] - 0.10


@needs_research
def test_calibracion_por_buckets_medida():
    g = decide()
    for tour in ("ATP", "WTA"):
        b = g[tour]["winner"]["buckets"]
        assert set(b) == {"50-55", "55-60", "60-65", "65-70", "70-100"}
        assert sum(x["n"] for x in b.values()) > 5000


@needs_research
def test_umbral_sale_de_la_validacion():
    for tour in ("ATP", "WTA"):
        for T, u in decide()[tour]["winner"]["umbrales"].items():
            if u["umbral"] is not None and u["candidatos"]:
                assert u["candidatos"][str(u["umbral"])]["z"] >= 1.96


@needs_models
def test_el_calibrador_no_invierte_el_lado():
    import joblib
    for tour in ("ATP", "WTA"):
        art = joblib.load(TENIS_MODELS_DIR / f"{tour}_winner_v1.joblib")
        for p in (0.2, 0.45, 0.55, 0.8):
            q = P._calibrate(p, art)
            assert (q >= 0.5) == (p >= 0.5) or abs(p - 0.5) < 0.02


# ------------------------------------------------------------- extremo a extremo
@needs_models
def test_prediccion_completa_versionada_y_calificada():
    X = pd.read_parquet(FEATURES_PARQUET)
    atp = X[(X.tour == "ATP") & X.p1_y1_spw.notna()].iloc[-1]
    init_db()
    eid = "TEST_TENIS_E2E"
    with session_scope() as s:
        s.query(TenisPrediction).filter(TenisPrediction.event_id == eid).delete()
        s.query(TenisOdds).filter(TenisOdds.event_id == eid).delete()
        s.query(TenisSchedule).filter(TenisSchedule.event_id == eid).delete()
        sc = TenisSchedule(event_id=eid, tour="ATP", sport_key="tennis_atp_test", tourney_name="Test Open",
                           surface="Hard", start_utc=datetime.utcnow() + timedelta(days=1),
                           p1_name=atp.p1_name, p2_name=atp.p2_name, p1_id=atp.p1_id, p2_id=atp.p2_id,
                           best_of=3, ingested_at=datetime.utcnow())
        s.add(sc); s.flush()
        r = P.predict_upcoming(s, 3, "test", X=X)
        assert r["predictions"] >= 1
        rows = s.execute(select(TenisPrediction).where(TenisPrediction.event_id == eid,
                                                       TenisPrediction.status != "superseded")).scalars().all()
        mk = {p.market_type for p in rows}
        assert "winner" in mk and mk <= {"winner", "total_games", "handicap_games"}
        for p in rows:
            assert p.model_version.startswith("TENIS_") and p.prediction_timestamp
            assert p.cutoff_timestamp.date() <= p.match_date
            assert 0.5 <= p.pick_probability <= 1.0
            assert p.status_market in ("pick", "no_pick", "projection")
            if p.market_type != "winner":
                assert p.status_market == "projection" and p.line is not None
                assert 0.35 <= p.p_serve_1 <= 0.85 and 0.35 <= p.p_serve_2 <= 0.85
        # sin cambios -> no duplica
        assert P.predict_upcoming(s, 3, "test", X=X)["predictions"] == 0
        # cambia la linea -> nueva version
        now = datetime.utcnow()
        for sel, price, line in (("over", -110, 30.5), ("under", -110, 30.5)):
            s.add(TenisOdds(event_id=eid, bookmaker="bk", market="total_games", selection=sel, line=line,
                            price_american=price, implied_prob=american_to_prob(price),
                            available_at=now - timedelta(minutes=5)))
        s.flush()
        r3 = P.predict_upcoming(s, 3, "test", X=X)
        tg = s.execute(select(TenisPrediction).where(TenisPrediction.event_id == eid,
                                                     TenisPrediction.market_type == "total_games")
                       .order_by(TenisPrediction.version)).scalars().all()
        if len(tg) >= 2:
            assert tg[0].status == "superseded" and tg[-1].line == 30.5 and tg[-1].version == len(tg)
        s.query(TenisPrediction).filter(TenisPrediction.event_id == eid).delete()
        s.query(TenisOdds).filter(TenisOdds.event_id == eid).delete()
        s.query(TenisSchedule).filter(TenisSchedule.event_id == eid).delete()


@needs_models
def test_dashboard_top_picks_performance_health():
    from dashboard.backend.main import (data_health, games, markets, performance, tenis_health,
                                        tenis_markets, top_picks)
    m = markets()["TENIS"]
    assert m["ATP"]["winner"]["mode"] in ("pick", "projection")
    assert m["WTA"]["total_games"]["publish_pick"] is False
    assert tenis_markets()["ATP"]["handicap_games"]["mode"] in ("projection", "blocked")
    h = tenis_health()
    assert h["circuitos"]["ATP"]["partidos"] > 50000
    assert data_health()["tenis"]["estado_global"] in ("OK", "WARNING")
    assert "ATP" in performance("TENIS")["TENIS"]["markets"]
    assert games("TENIS", "custom", "2020-01-01", "2030-12-31")["warnings"] == []
    t = top_picks("all", "custom", "2020-01-01", "2030-12-31", limit=50)
    for pk in t["picks"]:
        if pk["sport"] == "TENIS":
            assert pk["validated_pick"] == ((pk.get("extra") or {}).get("mode") == "pick")


def test_calificacion_con_resultado_en_vivo():
    """El archivo esta congelado: sin la fuente en vivo nada nuevo se calificaria."""
    from TENIS.markets.db import TenisResult
    init_db()
    eid = "TEST_TENIS_RESULT"
    with session_scope() as s:
        s.query(TenisPrediction).filter(TenisPrediction.event_id == eid).delete()
        s.query(TenisResult).filter(TenisResult.event_id == eid).delete()
        base = dict(event_id=eid, tour="ATP", season=2026, match_date=date(2026, 9, 7),
                    start_utc=datetime(2026, 9, 7, 20), tourney_name="Test", surface="Hard", best_of=3,
                    p1_name="Jugador Uno", p2_name="Jugador Dos", model_version="TENIS_ATP_WINNER_v1",
                    prediction_timestamp=datetime.utcnow(), cutoff_timestamp=datetime(2026, 9, 7),
                    status="published", version=1)
        s.add(TenisPrediction(**base, market_type="winner", final_probability=0.7,
                              pick_probability=0.7, selection="Jugador Uno", status_market="pick"))
        s.add(TenisPrediction(**base, market_type="total_games", final_probability=0.6, line=21.5,
                              pick_probability=0.6, selection="OVER 21.5", status_market="projection"))
        s.add(TenisPrediction(**base, market_type="handicap_games", final_probability=0.6, line=-3.5,
                              pick_probability=0.6, selection="Jugador Uno -3.5", status_market="projection"))
        # resultado real: gana el 1 por 12-8 juegos (20 en total, margen +4)
        s.add(TenisResult(event_id=eid, tour="ATP", match_date=date(2026, 9, 7), p1_name="Jugador Uno",
                          p2_name="Jugador Dos", winner_name="Jugador Uno", p1_games=12, p2_games=8,
                          games_total=20, games_margin=4, sets_p1=2, sets_p2=0, status="final",
                          source="test", fetched_at=datetime.utcnow()))
        s.flush()
        P.score(s, pd.read_parquet(FEATURES_PARQUET))
        rows = {p.market_type: p for p in s.execute(select(TenisPrediction).where(
            TenisPrediction.event_id == eid)).scalars()}
        assert rows["winner"].result == "win"                 # acerto el ganador
        assert rows["total_games"].result == "loss"           # 20 juegos < 21.5, se eligio OVER
        assert rows["handicap_games"].result == "win"         # gano por 4 > 3.5
        assert rows["winner"].actual_value == 1.0 and rows["total_games"].actual_value == 20.0
        s.query(TenisPrediction).filter(TenisPrediction.event_id == eid).delete()
        s.query(TenisResult).filter(TenisResult.event_id == eid).delete()


def test_emparejado_de_nombres_por_apellido():
    from TENIS.markets.results import _apellido
    assert _apellido("Aryna Sabalenka") == "sabalenka"
    assert _apellido("Botic Van De Zandschulp") == "zandschulp"
    assert _apellido("Félix Auger-Aliassime") == "augeraliassime"


# ------------------------------------------------------------- aislamiento
def test_tenis_no_importa_los_demas_deportes():
    import re
    for f in (Path(__file__).resolve().parents[1] / "markets").glob("*.py"):
        code = "\n".join(l for l in f.read_text().splitlines() if not l.strip().startswith(("#", '"""')))
        assert not re.search(r"\b(from|import) (MLB|NFL|NBA)\b|MLB_|NFL_HOME|NBA_", code), f.name


def test_tenis_escribe_solo_en_su_arbol():
    from shared.paths import (IsolationError, MLB_DIR, NBA_DIR, NFL_DIR, TENIS_DB_DIR, assert_writable)
    assert assert_writable(TENIS_DB_DIR / "tenis_markets.sqlite3", "TENIS")
    for bad in (MLB_DIR / "database" / "mlb.sqlite3", NBA_DIR / "database" / "nba_markets.sqlite3",
                NFL_DIR / "database" / "nfl_markets.sqlite3", NFL_HOME / "database" / "nflpred.sqlite3"):
        with pytest.raises(IsolationError):
            assert_writable(bad, "TENIS")


def test_los_demas_deportes_no_cambian_tras_un_ciclo_de_tenis():
    root = Path(__file__).resolve().parents[2]
    files = [f for f in (NFL_HOME / "data" / "processed" / "features.parquet",
                         root / "MLB" / "database" / "mlb.sqlite3",
                         root / "NBA" / "database" / "nba_markets.sqlite3") if f.exists()]
    before = {f: hashlib.md5(f.read_bytes()).hexdigest() for f in files}
    from TENIS.markets.pipeline import refresh_results
    refresh_results()
    assert before == {f: hashlib.md5(f.read_bytes()).hexdigest() for f in files}
