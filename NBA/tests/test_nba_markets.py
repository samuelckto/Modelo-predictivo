"""Tests del modulo NBA. No tocan NFL, MLB ni el motor NFL original."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from NBA.markets import predict as P
from NBA.markets.db import NbaGame, NbaOdds, NbaPrediction, NbaTeam, init_db, session_scope
from NBA.markets.features import FEATURES_PARQUET, ID_COLS, TARGET_COLS, build, families
from NBA.markets.gating import decide
from NBA.markets.odds import abbr_from_name, consensus, movement
from shared.odds import american_to_prob
from shared.paths import NBA_MODELS_DIR, NBA_OUT_DIR, NFL_HOME

needs_data = pytest.mark.skipif(not FEATURES_PARQUET.exists(), reason="sin features NBA")
needs_research = pytest.mark.skipif(not (NBA_OUT_DIR / "research_lines.json").exists(), reason="sin investigacion")
needs_models = pytest.mark.skipif(not (NBA_MODELS_DIR / "moneyline_v1.joblib").exists(), reason="sin modelos")


# ---------------------------------------------------------------- reglas de calificacion (1-11)
def _grade(market, line, hp, ap, p_home):
    """Replica exacta de predict.score sobre un caso sintetico."""
    if market == "moneyline":
        return "win" if ((hp - ap > 0) == (p_home >= 0.5)) else "loss"
    if market == "spread":
        cv = (hp - ap) + line
        if cv == 0:
            return "push"
        return "win" if ((cv > 0) == (p_home >= 0.5)) else "loss"
    tot = hp + ap
    if tot == line:
        return "push"
    return "win" if ((tot > line) == (p_home >= 0.5)) else "loss"


def test_moneyline_gana_y_pierde():
    assert _grade("moneyline", None, 110, 100, 0.6) == "win"
    assert _grade("moneyline", None, 100, 110, 0.6) == "loss"
    assert _grade("moneyline", None, 100, 110, 0.4) == "win"


def test_spread_linea_negativa_favorito_local():
    # BOS -4.5: local debe ganar por mas de 4.5
    assert _grade("spread", -4.5, 110, 100, 0.6) == "win"
    assert _grade("spread", -4.5, 104, 100, 0.6) == "loss"
    assert _grade("spread", -4.5, 104, 100, 0.4) == "win"     # se eligio visitante +4.5


def test_spread_linea_positiva_underdog_local():
    # LAL +4.5 (local underdog): local cubre si pierde por menos de 4.5 o gana
    assert _grade("spread", 4.5, 100, 103, 0.6) == "win"
    assert _grade("spread", 4.5, 100, 108, 0.6) == "loss"


def test_spread_linea_cero_pick_em_y_push():
    assert _grade("spread", 0.0, 100, 100, 0.6) == "push"
    assert _grade("spread", 0.0, 101, 100, 0.6) == "win"


def test_spread_push_en_linea_entera():
    assert _grade("spread", -3.0, 103, 100, 0.6) == "push"
    assert _grade("spread", -3.0, 104, 100, 0.6) == "win"


def test_lineas_medias_nunca_push():
    for hp, ap in ((110, 100), (100, 110), (105, 105)):
        assert _grade("spread", -3.5, hp, ap, 0.5) != "push"
        assert _grade("total", 224.5, hp, ap, 0.5) != "push"


def test_total_over_under_y_push():
    assert _grade("total", 224.5, 115, 112, 0.6) == "win"     # 227 > 224.5 over
    assert _grade("total", 224.5, 110, 110, 0.6) == "loss"    # 220 under, se eligio over
    assert _grade("total", 224.5, 110, 110, 0.4) == "win"
    assert _grade("total", 220.0, 110, 110, 0.6) == "push"


def test_etiqueta_spread_ambos_lados():
    line = -4.5
    assert f"BOS {line:+g}" == "BOS -4.5" and f"LAL {-line:+g}" == "LAL +4.5"


def test_linea_de_referencia_en_medio():
    assert P._ref(224.2) == 224.5 and P._ref(-3.1) == -3.5 and P._ref(2.9) == 2.5


# ---------------------------------------------------------------- no-vig, cuotas, movimiento (12, 25)
def test_abreviaturas_odds_api():
    assert abbr_from_name("Boston Celtics") == "BOS" and abbr_from_name("Portland Trail Blazers") == "POR"
    assert abbr_from_name("Equipo X") is None


def test_no_vig_y_exclusion_de_cuotas_posteriores():
    init_db()
    gid = "TEST_NBA_ODDS"
    now = datetime(2030, 1, 1, 12)
    with session_scope() as s:
        s.query(NbaOdds).filter(NbaOdds.game_id == gid).delete()
        for sel, price, line, when in (("home", -150, -3.5, now - timedelta(hours=5)),
                                       ("away", +130, 3.5, now - timedelta(hours=5)),
                                       ("home", -170, -4.5, now - timedelta(hours=1)),
                                       ("away", +145, 4.5, now - timedelta(hours=1)),
                                       ("home", -200, -6.5, now + timedelta(hours=1))):     # posterior
            s.add(NbaOdds(game_id=gid, bookmaker="bk", market="spread", selection=sel, line=line,
                          price_american=price, implied_prob=american_to_prob(price), available_at=when))
        s.flush()
        c = consensus(s, gid, "spread", now)
        assert c["line"] == -4.5 and 0.5 < c["p_home"] < 0.7 and abs(
            c["p_home"] - american_to_prob(-170) / (american_to_prob(-170) + american_to_prob(145))) < 1e-9
        assert consensus(s, gid, "spread", now - timedelta(hours=6)) is None
        mv = movement(s, gid, "spread")
        assert mv["apertura"] == -3.5 and mv["movimiento"] == -3.0      # incluye el snapshot posterior: es historial
        s.query(NbaOdds).filter(NbaOdds.game_id == gid).delete()


# ---------------------------------------------------------------- leakage (13-17)
@needs_data
def test_ninguna_feature_es_resultado_ni_mercado():
    X = pd.read_parquet(FEATURES_PARQUET)
    for name, cols in families(X).items():
        assert not (set(cols) & TARGET_COLS), name
        assert not (set(cols) & ID_COLS), name
        assert not any("odds" in c or "line" in c or "market" in c for c in cols), name


@needs_data
def test_logs_disponibles_solo_antes_del_dia_del_partido():
    """available_at de cada log = fin del dia; el corte es 00:00 del dia -> nunca entra el mismo dia."""
    X = pd.read_parquet(FEATURES_PARQUET)
    assert (X.max_source_available_at < X.cutoff_utc).all()
    assert (X.cutoff_utc == X.game_date).all()


@needs_data
def test_truncacion_las_features_no_cambian_al_quitar_el_futuro():
    """Se reconstruyen features con la base truncada en una fecha y deben coincidir
    con las del parquet completo para los partidos anteriores a esa fecha."""
    X = pd.read_parquet(FEATURES_PARQUET)
    cut = pd.Timestamp("2024-01-15")
    from NBA.markets import features as F
    from NBA.markets.db import session_scope as ss
    # truncar en memoria: monkeypatch de la consulta de partidos
    orig_build = F.build
    with ss() as s:
        games = s.execute(select(NbaGame)).scalars().all()
        ids_after = {g.game_id for g in games if pd.Timestamp(g.game_date) >= cut}
    import NBA.markets.features as FM
    real_select = FM.select

    class _Trunc:
        def __init__(self, s): self.s = s
        def execute(self, q):
            res = self.s.execute(q)
            class R:
                def __init__(self, rows): self.rows = rows
                def scalars(self): return iter(self.rows)
            return R([r for r in res.scalars() if getattr(r, "game_id", None) not in ids_after])
    from contextlib import contextmanager
    @contextmanager
    def fake_scope():
        with ss() as s:
            yield _Trunc(s)
    FM.session_scope, keep = fake_scope, FM.session_scope
    try:
        Xt = FM.build(save=False)
    finally:
        FM.session_scope = keep
    cols = [c for c in families(X)["strength"] + families(X)["elo"] + families(X)["rest"]]
    a = X[X.game_date < cut].set_index("game_id")[cols].sort_index()
    b = Xt.set_index("game_id")[cols].sort_index()
    common = a.index.intersection(b.index)
    assert len(common) > 5000
    diff = (a.loc[common] - b.loc[common]).abs().max().max()
    assert diff < 1e-9, f"las features cambian al quitar el futuro: {diff}"


@needs_research
def test_walk_forward_reproducible_y_sin_temporada_de_prueba():
    from NBA.markets.features import numeric
    from NBA.markets.research import reg_algos
    r = json.loads((NBA_OUT_DIR / "research.json").read_text())["targets"]["spread"]
    d = pd.read_parquet(NBA_OUT_DIR / "preds_spread.parquet")
    assert d.season.min() >= 2019
    X = pd.read_parquet(FEATURES_PARQUET); X = X[X.season_type == "Regular Season"]
    T = 2024
    tr = X[(X.season < T) & X.margin.notna()]
    m = reg_algos()[r["algoritmo_elegido"]](); m.fit(numeric(tr, r["features"]), tr.margin.values)
    te = X[X.season == T].head(8)
    ref = d[d.season == T].set_index("game_id").loc[te.game_id.values, "pred"].values
    assert np.allclose(m.predict(numeric(te, r["features"])), ref, atol=1e-6)


# ---------------------------------------------------------------- calibracion, buckets, gating (19-22)
@needs_research
def test_calibracion_por_buckets_medida_en_holdout():
    g = decide()
    for m in ("moneyline", "spread", "total"):
        b = g[m]["buckets"]
        assert set(b) == {"50-55", "55-60", "60-65", "65-70", "70-100"}
        assert sum(x["n"] for x in b.values()) > 1000


@needs_research
def test_estados_de_mercado_coherentes_con_la_evidencia():
    g = decide()
    ml = g["moneyline"]
    if ml["publish_pick"]:
        assert ml["picks"]["z_vs_50"] >= 1.96 and ml["holdout_calibrado"]["ece"] < 0.05 and ml["threshold"] is not None
    for m in ("spread", "total"):
        assert g[m]["publish_pick"] is False and g[m]["mode"] in ("projection", "blocked")
        if g[m]["mode"] == "projection":
            assert g[m]["holdout"]["modelo"]["mae"] < g[m]["holdout"]["media"]["mae"]


@needs_research
def test_umbral_no_pick_sale_de_la_validacion_no_de_un_numero_fijo():
    g = decide()["moneyline"]
    for T, u in g["umbrales"].items():
        if u["umbral"] is not None:
            c = u["candidatos"][str(u["umbral"])]
            assert c["z"] >= 1.96 and c["n"] >= 100


def test_mercado_bloqueado_no_genera_predicciones(monkeypatch, tmp_path):
    """Si gating bloquea un mercado, predict lo salta aunque exista el modelo."""
    import NBA.markets.predict as PR
    blocked = {m: {"enabled": False, "publish_pick": False, "mode": "blocked", "reason": "test"} for m in ("moneyline", "spread", "total")}
    monkeypatch.setattr(PR, "decide", lambda: blocked)
    X = pd.read_parquet(FEATURES_PARQUET) if FEATURES_PARQUET.exists() else None
    if X is None:
        pytest.skip("sin features")
    fake = X.head(1).copy()
    fake["home_points"] = np.nan; fake["away_points"] = np.nan
    fake["game_date"] = pd.Timestamp(date.today() + timedelta(days=1)); fake["game_id"] = "TEST_BLOCKED"
    init_db()
    with session_scope() as s:
        r = PR.predict_upcoming(s, 3, "test", X=fake)
        assert r["predictions"] == 0 and len(r["skipped"]) == 3


# ---------------------------------------------------------------- extremo a extremo con partido sintetico (21, 23, 24, 26-30)
@needs_models
def test_prediccion_versionada_unica_y_cambio_de_linea():
    X = pd.read_parquet(FEATURES_PARQUET)
    played = X[X.home_points.notna()]
    base = played[played.season == played.season.max()].iloc[[-1]].copy()
    gid = "TEST_NBA_E2E"
    base["game_id"] = gid; base["home_points"] = np.nan; base["away_points"] = np.nan
    base["game_date"] = pd.Timestamp(date.today() + timedelta(days=1)); base["start_utc"] = pd.NaT
    init_db()
    with session_scope() as s:
        s.query(NbaPrediction).filter(NbaPrediction.game_id == gid).delete()
        s.query(NbaOdds).filter(NbaOdds.game_id == gid).delete()
        r = P.predict_upcoming(s, 3, "test", X=base)
        assert r["predictions"] == 3
        rows = s.execute(select(NbaPrediction).where(NbaPrediction.game_id == gid, NbaPrediction.status != "superseded")).scalars().all()
        assert {p.market_type for p in rows} == {"moneyline", "spread", "total"}
        for p in rows:
            assert p.model_version.startswith("NBA_") and p.prediction_timestamp and p.cutoff_timestamp < datetime.combine(p.game_date, datetime.min.time()) + timedelta(seconds=1)
            assert 0.5 <= p.pick_probability <= 1 and p.status_market in ("pick", "no_pick", "projection")
            if p.market_type != "moneyline":
                assert p.status_market == "projection" and p.line is not None and "referencia" in p.line_source
        # misma prediccion otra vez: no duplica
        r2 = P.predict_upcoming(s, 3, "test", X=base)
        assert r2["predictions"] == 0
        # llega el mercado con otra linea -> nueva version, la anterior queda superseded
        now = datetime.utcnow()
        for sel, price, line in (("home", -110, -12.5), ("away", -110, 12.5)):
            s.add(NbaOdds(game_id=gid, bookmaker="bk", market="spread", selection=sel, line=line,
                          price_american=price, implied_prob=american_to_prob(price), available_at=now - timedelta(minutes=5)))
        s.flush()
        r3 = P.predict_upcoming(s, 3, "test", X=base)
        assert r3["revisions"] >= 1
        sp = s.execute(select(NbaPrediction).where(NbaPrediction.game_id == gid, NbaPrediction.market_type == "spread")
                       .order_by(NbaPrediction.version)).scalars().all()
        assert len(sp) == 2 and sp[0].status == "superseded" and sp[1].version == 2 and sp[1].line == -12.5
        assert sp[1].line_source == "the_odds_api" and sp[1].market_probability is not None
        # calificacion con resultado sintetico: local gana 120-100 -> ML segun lado, spread -12.5 cubre, total 220
        fin = base.copy(); fin["home_points"] = 120.0; fin["away_points"] = 100.0
        r4 = P.score(s, fin)
        assert r4["scored"] == 3
        for p in s.execute(select(NbaPrediction).where(NbaPrediction.game_id == gid, NbaPrediction.status != "superseded")).scalars().all():
            assert p.result in ("win", "loss", "push") and p.actual_value is not None
        s.query(NbaPrediction).filter(NbaPrediction.game_id == gid).delete()
        s.query(NbaOdds).filter(NbaOdds.game_id == gid).delete()


@needs_models
def test_dashboard_top_picks_performance_health_y_api():
    from dashboard.backend.main import data_health, games, markets, nba_health, nba_markets, performance, top_picks
    m = markets()["NBA"]
    assert m["moneyline"]["mode"] in ("pick", "projection", "blocked") and m["spread"]["publish_pick"] is False
    assert nba_markets()["total"]["mode"] in ("projection", "blocked")
    h = nba_health()
    assert h["partidos"] > 10000 and h["mercados"]["moneyline"]["modelo"] == "NBA_ML_v1"
    assert data_health()["nba"]["estado_global"] in ("OK", "WARNING")
    p = performance("NBA")["NBA"]["markets"]
    assert "2024" in p["moneyline"]["por_temporada"]
    g = games("NBA", "custom", "2020-01-01", "2030-12-31")
    assert g["warnings"] == []
    t = top_picks("all", "custom", "2020-01-01", "2030-12-31", limit=50)
    for pk in t["picks"]:
        if pk["sport"] == "NBA":
            assert pk["validated_pick"] == ((pk.get("extra") or {}).get("mode") == "pick")


# ---------------------------------------------------------------- aislamiento (31-34)
def test_nba_no_importa_nfl_ni_mlb():
    import re
    for f in (Path(__file__).resolve().parents[1] / "markets").glob("*.py"):
        code = "\n".join(l for l in f.read_text().splitlines() if not l.strip().startswith(("#", '"""')))
        assert not re.search(r"\bfrom (MLB|NFL)\b|\bimport (MLB|NFL)\b|MLB_|NFL_HOME|nflpred", code), f.name


def test_nba_escribe_solo_en_su_arbol():
    from shared.paths import IsolationError, NBA_DB_DIR, MLB_DIR, NFL_DIR, assert_writable
    assert assert_writable(NBA_DB_DIR / "nba_markets.sqlite3", "NBA")
    for bad in (MLB_DIR / "database" / "mlb.sqlite3", NFL_DIR / "database" / "nfl_markets.sqlite3",
                NFL_HOME / "database" / "nflpred.sqlite3"):
        with pytest.raises(IsolationError):
            assert_writable(bad, "NBA")


def test_motor_nfl_y_base_mlb_intactos_tras_un_ciclo_nba():
    files = [f for f in (NFL_HOME / "data" / "processed" / "features.parquet",
                         Path(__file__).resolve().parents[2] / "MLB" / "database" / "mlb.sqlite3") if f.exists()]
    before = {f: hashlib.md5(f.read_bytes()).hexdigest() for f in files}
    from NBA.markets.pipeline import refresh_results
    refresh_results()
    assert before == {f: hashlib.md5(f.read_bytes()).hexdigest() for f in files}
