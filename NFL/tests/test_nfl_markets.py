"""Tests de los mercados NFL propios (total y spread). Nada aqui toca el motor NFL
original ni la base MLB."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

import NFL.adapter as nfl
from NFL.markets import predict as P
from NFL.markets.data import MARKET_COLS, TARGET_COLS, families, load
from NFL.markets.db import NflMarketPrediction, NflOdds, init_db, session_scope
from NFL.markets.gating import OUT, decide
from NFL.markets.odds import abbr_from_name, consensus
from shared.paths import NFL_HOME, NFL_MARKETS_MODELS_DIR

needs_nfl = pytest.mark.skipif(not nfl.available()[0], reason="motor NFL no instalado")
needs_research = pytest.mark.skipif(not (OUT / "research_lines.json").exists(), reason="sin investigacion")
needs_model = pytest.mark.skipif(not (NFL_MARKETS_MODELS_DIR / "total_v1.joblib").exists(), reason="sin modelo")


# ---------------------------------------------------------------- calificacion (logica pura)
def _grade(market, line, home, away, p_final):
    """Reproduce la regla de score() sobre un partido sintetico."""
    actual = home + away if market == "total" else home - away
    if actual == line:
        return "push"
    over = actual > line
    return "win" if over == (p_final >= 0.5) else "loss"


def test_total_over_gana_si_se_pasa():
    assert _grade("total", 47.5, 27, 24, 0.6) == "win"      # 51 > 47.5, se eligio over


def test_total_under_gana_si_no_se_pasa():
    assert _grade("total", 47.5, 20, 17, 0.4) == "win"      # 37 < 47.5, se eligio under


def test_total_push_en_linea_entera():
    assert _grade("total", 47.0, 27, 20, 0.6) == "push"


def test_linea_media_nunca_da_push():
    for h, a in ((27, 20), (20, 27), (24, 24)):
        assert _grade("total", 47.5, h, a, 0.5) != "push"
        assert _grade("spread", 3.5, h, a, 0.5) != "push"


def test_spread_favorito_negativo_cubre():
    # linea +3.5 = local favorito por 3.5 (convencion nflverse); local gana por 7 -> cubre
    assert _grade("spread", 3.5, 27, 20, 0.6) == "win"
    assert _grade("spread", 3.5, 23, 20, 0.6) == "loss"      # gana por 3, no cubre


def test_spread_underdog_positivo_cubre():
    # linea -2.5 = visitante favorito; local pierde por 1 -> local +2.5 cubre
    assert _grade("spread", -2.5, 20, 21, 0.6) == "win"
    assert _grade("spread", -2.5, 17, 21, 0.4) == "win"      # se eligio visitante -2.5, gana por 4


def test_pick_em_y_push():
    assert _grade("spread", 0.0, 20, 20, 0.55) == "push"
    assert _grade("spread", 0.0, 21, 20, 0.55) == "win"


def test_etiqueta_spread_con_signo_correcto():
    row = SimpleNamespace(home_team="KC", away_team="BUF")
    # linea 2.5 (KC favorito por 2.5): lado local -> "KC -2.5"; lado visitante -> "BUF +2.5"
    line = 2.5
    assert f"{row.home_team} {-line:+g}" == "KC -2.5"
    assert f"{row.away_team} {line:+g}" == "BUF +2.5"


def test_linea_de_referencia_termina_en_medio():
    assert P._ref_line(47.9) == 47.5 and P._ref_line(47.1) == 47.5 and P._ref_line(-3.2) == -3.5


# ---------------------------------------------------------------- no-vig y cuotas
def test_abreviaturas_de_the_odds_api():
    assert abbr_from_name("Kansas City Chiefs") == "KC"
    assert abbr_from_name("Washington Commanders") == "WAS"
    assert abbr_from_name("Equipo Inexistente") is None


def test_consenso_sin_vig_y_sin_cuotas_posteriores(tmp_path):
    init_db()
    gid = "TEST_9999_A_B"
    now = datetime(2030, 1, 1, 12, 0, 0)
    with session_scope() as s:
        s.query(NflOdds).filter(NflOdds.game_id == gid).delete()
        for sel, price, when in (("over", -110, now - timedelta(hours=2)),
                                 ("under", -110, now - timedelta(hours=2)),
                                 ("over", -150, now + timedelta(hours=1)),   # posterior: no cuenta
                                 ("under", +130, now + timedelta(hours=1))):
            from shared.odds import american_to_prob
            s.add(NflOdds(game_id=gid, bookmaker="bk", market="total", selection=sel, line=47.5,
                          price_american=price, implied_prob=american_to_prob(price), available_at=when))
        s.flush()
        c = consensus(s, gid, "total", now)
        assert c is not None and abs(c["p_home"] - 0.5) < 1e-9 and c["line"] == 47.5
        assert consensus(s, gid, "total", now - timedelta(hours=3)) is None   # nada antes
        s.query(NflOdds).filter(NflOdds.game_id == gid).delete()


# ---------------------------------------------------------------- leakage
@needs_nfl
def test_ninguna_feature_es_de_resultado_ni_de_mercado():
    X = load()
    for name, cols in families(X).items():
        assert not (set(cols) & TARGET_COLS), name
        assert not (set(cols) & MARKET_COLS), name
        assert not any(c.endswith("_ts") or c.startswith("market_") for c in cols), name


@needs_nfl
def test_los_timestamps_de_las_fuentes_son_anteriores_al_kickoff():
    X = load()
    fin = X[X.home_score.notna()]
    for c in P.SRC_TS:
        t = pd.to_datetime(fin[c])
        ok = t.isna() | (t < fin.kickoff_utc)
        assert ok.all(), f"{c}: {(~ok).sum()} filas con fuente no anterior al kickoff"


@needs_nfl
def test_las_lineas_de_cierre_no_estan_en_las_features():
    from NFL.markets.gating import OUT
    r = json.loads((OUT / "research.json").read_text())
    for mk in ("total", "spread"):
        f = set(r["targets"][mk]["features"])
        assert not (f & MARKET_COLS) and not (f & TARGET_COLS)


@needs_nfl
def test_walk_forward_no_entrena_con_la_temporada_de_prueba():
    """Los archivos de predicciones walk-forward solo contienen temporadas >= 2020 y
    cada fila de prueba de la temporada T se genero sin filas de T (se recalcula
    un caso y debe coincidir)."""
    from NFL.markets.data import numeric
    from NFL.markets.research import algos
    d = pd.read_parquet(OUT / "preds_total.parquet")
    assert d.season.min() >= 2020
    X = load(); r = json.loads((OUT / "research.json").read_text())["targets"]["total"]
    T = 2023
    tr = X[(X.season < T) & X.total_pts.notna()]
    m = algos()[r["algoritmo_elegido"]](); m.fit(numeric(tr, r["features"]), tr.total_pts.values)
    te = X[X.season == T].head(5)
    got = m.predict(numeric(te, r["features"]))
    ref = d[d.season == T].set_index("game_id").loc[te.game_id.values, "pred"].values
    assert np.allclose(got, ref, atol=1e-6)


# ---------------------------------------------------------------- gating y calibracion
@needs_research
def test_totales_y_spread_no_son_pick_sin_evidencia():
    d = decide()
    for m in ("total", "spread"):
        g = d[m]
        if g["publish_pick"]:
            assert g["picks"]["z"] >= 1.96 and g["picks"]["n"] >= 100
        else:
            assert g["mode"] in ("proyeccion", "bloqueado")


@needs_research
def test_calibracion_por_buckets_esta_medida():
    d = decide()
    for m in ("total", "spread"):
        b = d[m]["calibration"]
        assert set(b) == {"50-55", "55-60", "60-65", "65-70", "70-100"}
        assert sum(x["n"] for x in b.values()) > 1000


@needs_research
def test_baselines_presentes_por_temporada():
    d = decide()
    for T, v in d["spread"]["por_temporada"].items():
        assert {"siempre_favorito", "siempre_underdog"} <= set(v["vs_linea"]["baselines"])
    for T, v in d["total"]["por_temporada"].items():
        assert {"siempre_over", "siempre_under"} <= set(v["vs_linea"]["baselines"])


@needs_model
def test_el_calibrador_no_cambia_de_lado():
    import joblib
    for m in ("total", "spread"):
        art = joblib.load(NFL_MARKETS_MODELS_DIR / f"{m}_v1.joblib")
        assert art["calibrator_intercept"] == 0.0
        for p in (0.3, 0.45, 0.55, 0.7):
            assert (P._calibrate(p, art) >= 0.5) == (p >= 0.5)
            assert abs(P._calibrate(p, art) - 0.5) <= abs(p - 0.5) + 1e-9   # solo encoge


# ---------------------------------------------------------------- predicciones guardadas
@needs_model
def test_predicciones_versionadas_con_todos_los_campos():
    init_db()
    with session_scope() as s:
        rows = s.execute(select(NflMarketPrediction).where(
            NflMarketPrediction.status != "superseded")).scalars().all()
    for p in rows:
        assert p.market_type in ("total", "spread") and p.model_version in ("NFL_TOTAL_v1", "NFL_SPREAD_v1")
        assert p.prediction_timestamp is not None and p.line is not None
        assert p.selection and p.pick in ("pick", "proyeccion", "no_pick")
        assert 0.5 <= p.pick_probability <= 1.0
        assert p.confidence in ("ALTA", "MEDIA", "BAJA")
        if p.cutoff_timestamp is not None:
            assert p.cutoff_timestamp < p.kickoff_utc
        if p.pick != "pick":
            assert p.confidence == "BAJA"


@needs_model
def test_una_activa_por_partido_y_mercado():
    init_db()
    with session_scope() as s:
        rows = s.execute(select(NflMarketPrediction).where(
            NflMarketPrediction.status != "superseded")).scalars().all()
    seen = set()
    for p in rows:
        assert (p.game_id, p.market_type) not in seen
        seen.add((p.game_id, p.market_type))


@needs_model
def test_cambio_de_linea_genera_nueva_version(monkeypatch):
    """Si la linea cambia, la prediccion anterior queda superseded y nace v+1."""
    init_db()
    with session_scope() as s:
        p = s.execute(select(NflMarketPrediction).where(NflMarketPrediction.market_type == "total",
                                                        NflMarketPrediction.status != "superseded")).scalars().first()
        if p is None:
            pytest.skip("sin predicciones")
        gid, v0, line0 = p.game_id, p.version, p.line
        row = SimpleNamespace(game_id=gid, season=p.season, week=p.week, kickoff_utc=p.kickoff_utc,
                              home_team=p.home_team, away_team=p.away_team)
        art = {"name": p.model_version, "res_std": p.expected_std}
        stored, rev = P._store(s, row, "total", art, p.expected_value, p.expected_median, line0 + 1.0,
                               "test", None, p.p_over_raw, p.model_probability, None, p.final_probability,
                               p.selection, p.pick, p.pick_probability, p.confidence, {}, [], None,
                               datetime.utcnow(), None)
        assert stored == 1 and rev == 1
        act = s.execute(select(NflMarketPrediction).where(NflMarketPrediction.game_id == gid,
                                                          NflMarketPrediction.market_type == "total",
                                                          NflMarketPrediction.status != "superseded")).scalars().all()
        assert len(act) == 1 and act[0].version == v0 + 1 and act[0].parent_id == p.id
        # limpieza: se deshace el cambio de prueba
        s.delete(act[0]); p.status = "published"


# ---------------------------------------------------------------- aislamiento
@needs_nfl
def test_el_motor_nfl_original_no_cambia():
    """Base y features del motor NFL: mismo hash antes y despues de un ciclo de prediccion."""
    files = [NFL_HOME / "data" / "processed" / "features.parquet"]
    before = {f: hashlib.md5(f.read_bytes()).hexdigest() for f in files}
    from NFL.markets.pipeline import run
    run(days_ahead=3, with_odds=False)
    after = {f: hashlib.md5(f.read_bytes()).hexdigest() for f in files}
    assert before == after


def test_moneyline_nfl_sigue_viniendo_del_motor_original():
    """El adaptador de moneyline no importa nada de NFL.markets."""
    src = (NFL_HOME.parent / "Sports-Prediction-Center" / "NFL" / "adapter.py")
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "adapter.py"
    assert "NFL.markets" not in src.read_text()


def test_los_mercados_nfl_no_tocan_mlb():
    from pathlib import Path
    import re
    for f in (Path(__file__).resolve().parents[1] / "markets").glob("*.py"):
        code = "\n".join(l for l in f.read_text().splitlines() if not l.strip().startswith(("#", '"""')))
        assert not re.search(r"\bfrom MLB\b|\bimport MLB\b|MLB_", code), f.name


# ---------------------------------------------------------------- dashboard
@needs_model
def test_dashboard_muestra_tres_mercados_nfl():
    from dashboard.backend.main import games
    g = games("NFL", "custom", "2026-09-01", "2026-12-31")
    mk = {}
    for d in g["days"]:
        for c in d["games"]:
            mk.setdefault(c["game_id"], set()).add((c.get("extra") or {}).get("market_key") or c["market"].lower())
    assert any({"total", "spread"} <= v for v in mk.values())


@needs_model
def test_top_picks_no_mezcla_proyecciones_con_picks():
    from dashboard.backend.main import top_picks
    t = top_picks("NFL", "custom", "2026-09-01", "2026-12-31", limit=30)
    seen_proj = False
    for p in t["picks"]:
        if not p["validated_pick"]:
            seen_proj = True
        else:
            assert not seen_proj, "un pick validado aparece despues de una proyeccion"
    for p in t["picks"]:
        if (p.get("extra") or {}).get("market_key") in ("total", "spread"):
            assert p["validated_pick"] == (p["extra"]["mode"] == "pick")


@needs_model
def test_markets_endpoint_declara_modo_de_cada_mercado_nfl():
    from dashboard.backend.main import markets
    n = markets()["NFL"]
    assert n["moneyline"]["mode"] == "pick"
    for m in ("total", "spread"):
        assert n[m]["mode"] in ("pick", "proyeccion", "bloqueado") and n[m]["reason"]
