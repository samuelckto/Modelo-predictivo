"""Motor MLB: mercados habilitados, incertidumbre, lado del pick y puntuacion."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from MLB.database.models import ModelVersion, Prediction
from MLB.database.session import session_scope
from MLB.engine import uncertainty as unc
from MLB.engine.elo import EloConfig, run as run_elo
from MLB.engine.gating import decide, enabled_markets
from MLB.engine.models import EXCLUDE_PREFIX, feature_columns
from shared.paths import MLB_PROCESSED_DIR

FEAT = MLB_PROCESSED_DIR / "features.parquet"
needs_data = pytest.mark.skipif(not FEAT.exists(), reason="sin features construidas")


# ---------------------------------------------------------------- gating
@needs_data
def test_los_totales_nunca_son_pick():
    """Over/Under se publica solo como probabilidad informativa: el backtest no
    demostro que el modelo proyecte carreras mejor que la media historica."""
    d = decide()
    assert d["total"]["enabled"] is True
    assert d["total"]["publish_pick"] is False
    assert "informativa" in d["total"]["reason"]


@needs_data
def test_run_line_no_se_publica_como_pick():
    d = decide()["run_line"]
    assert d["enabled"] is True and d["publish_pick"] is False
    assert d["significant_95"] is False
    assert "NO es significativa" in d["reason"]


@needs_data
def test_el_gating_declara_cuanto_inflaba_el_leakage():
    """Cada mercado debe decir cuantos pp le anadia el backtest contaminado."""
    d = decide()
    for m in ("moneyline", "f5_moneyline", "run_line"):
        assert "leakage_inflation_pp" in d[m], m


@needs_data
def test_solo_moneyline_es_pick_validado():
    """ANTES de la auditoria este test exigia que moneyline Y F5 fueran picks.

    La auditoria (§3-§5) demostro que la ventaja de F5 dependia de las features
    del abridor, que en el historico usan informacion del propio partido. Con la
    cota limpia F5 baja a z=1.29 y deja de ser significativo. El test se actualiza
    a la evidencia nueva; el hallazgo queda escrito aqui para que no se olvide.
    """
    d = decide()
    assert d["moneyline"]["enabled"] and d["moneyline"]["publish_pick"]
    assert d["moneyline"]["z"] > 1.645
    assert d["moneyline"]["edge_vs_majority_pp"] > 0
    assert d["f5_moneyline"]["enabled"] is True
    assert d["f5_moneyline"]["publish_pick"] is False


# --------------------------------------------------------- incertidumbre
def test_el_riesgo_es_la_probabilidad_de_fallar():
    assert unc.uncertainty(0.62) == pytest.approx(38.0)
    assert unc.uncertainty(0.50) == pytest.approx(50.0)
    assert unc.band(0.70) == "LOW RISK"
    assert unc.band(0.56) == "MEDIUM RISK"
    assert unc.band(0.51) == "HIGH RISK"


def test_el_indice_compuesto_quedo_descartado_y_documentado():
    v = unc.VALIDATION
    assert v["composite_index"]["validated"] is False
    assert v["probability"]["validated"] is True
    # el compuesto no separaba: los terciles quedan casi planos
    e = v["composite_index"]["error_by_tercile"]
    assert max(e) - min(e) < 0.02
    # la probabilidad si separa
    p = v["probability"]["error_by_tercile"]
    assert p[0] - p[-1] > 0.04


def test_los_avisos_no_afirman_quien_gana():
    fl = unc.flags(dispersion=0.06, p_elo=0.35, p_selection=0.52, missing=["sin cuotas"],
                   lineup_confirmed=False, starter_known=False, p_market=0.62)
    codes = {f["code"] for f in fl}
    assert {"COIN_FLIP", "MODEL_DISAGREEMENT", "ELO_GAP", "LINEUP_UNCONFIRMED",
            "PITCHER_UNKNOWN"} <= codes
    for f in fl:
        t = f["text"].lower()
        assert not any(x in t for x in ("va a ganar", "seguro", "garantiz"))


def test_sin_incertidumbre_no_se_inventan_avisos():
    assert unc.flags(dispersion=0.01, p_elo=0.66, p_selection=0.68) == []


# --------------------------------------------------------------- features
@needs_data
def test_las_features_no_incluyen_el_resultado():
    X = pd.read_parquet(FEAT)
    cols = feature_columns(X)
    for c in cols:
        assert not c.startswith(EXCLUDE_PREFIX), c
    assert len(cols) > 80
    assert all(c.startswith("d_") or c in
               ("park_runs_factor", "is_doubleheader", "month", "h_rest_days", "a_rest_days")
               for c in cols)


# -------------------------------------------------------------------- elo
@needs_data
def test_elo_solo_usa_partidos_anteriores():
    X = pd.read_parquet(FEAT)
    g = X[["game_id", "season", "start_utc", "home_team_id", "away_team_id",
           "home_score", "away_score"]]
    p = run_elo(g, EloConfig())
    first = g.sort_values("start_utc").iloc[0]
    row = p[p.game_id == first.game_id].iloc[0]
    # primer partido de la historia: ambos equipos en la base, prob = ventaja de local
    assert row.elo_home_pre == pytest.approx(1500.0)
    assert row.elo_away_pre == pytest.approx(1500.0)
    assert 0.5 < row.p_home_elo < 0.56


# ------------------------------------------------------- predicciones vivas
def test_la_probabilidad_guardada_es_la_del_pick():
    """Si el pick es el visitante, la probabilidad NO puede ser la del local."""
    with session_scope() as s:
        rows = s.execute(select(Prediction).where(Prediction.market == "moneyline")
                         .limit(300)).scalars().all()
    if not rows:
        pytest.skip("todavia no hay predicciones generadas")
    for p in rows:
        assert p.ensemble_probability >= 0.5 - 1e-9, (
            f"{p.selection}: se guardo {p.ensemble_probability}, que es la del rival")
        if p.selection == p.away_abbr:
            assert p.ensemble_probability >= 0.5


def test_totales_solo_probabilidad_y_con_linea():
    """Toda prediccion de total lleva linea, seleccion Over/Under y la nota de
    que no es pick; la probabilidad guardada es la de la seleccion."""
    with session_scope() as s:
        tot = s.execute(select(Prediction).where(
            Prediction.market == "total", Prediction.status != "superseded")).scalars().all()
    for p in tot:
        assert p.line is not None
        assert p.selection.split()[0] in ("Over", "Under")
        assert p.ensemble_probability >= 0.5
        assert p.elo_probability is None
    from MLB.engine.provider import games
    for c in games("2000-01-01", "2100-01-01"):
        if c["extra"]["market_key"] == "total":
            assert c["extra"]["publish_pick"] is False


def test_f5_retirado_de_la_vista():
    from MLB.engine.markets import HIDDEN
    from MLB.engine.provider import games
    assert "f5_moneyline" in HIDDEN
    assert all(c["extra"]["market_key"] != "f5_moneyline"
               for c in games("2000-01-01", "2100-01-01"))


def test_las_predicciones_no_se_sobrescriben():
    """Al revisar, la anterior queda como 'superseded', nunca se borra."""
    with session_scope() as s:
        rev = s.execute(select(Prediction).where(Prediction.version > 1)).scalars().all()
        if not rev:
            pytest.skip("todavia no hay revisiones")
        for p in rev:
            assert p.parent_prediction_id is not None
            prev = s.get(Prediction, p.parent_prediction_id)
            assert prev is not None and prev.status == "superseded"
            assert prev.prediction_timestamp <= p.prediction_timestamp


def test_toda_prediccion_tiene_timestamp_y_trazabilidad():
    with session_scope() as s:
        rows = s.execute(select(Prediction).limit(200)).scalars().all()
    if not rows:
        pytest.skip("sin predicciones")
    for p in rows:
        assert p.prediction_timestamp is not None
        assert p.model_version and p.feature_version
        assert p.sources and "mlb_stats_api" in p.sources
        assert p.data_completeness is not None
        if p.market_probability is None:
            assert "sin cuotas de mercado" in (p.data_completeness or {}).get("missing", [])


# --------------------------------------------- ficha detallada e historico
@needs_data
def test_la_ficha_detallada_no_inventa_lo_que_no_tiene():
    import pandas as pd

    from MLB.engine.detail import game_detail
    X = pd.read_parquet(FEAT)
    fut = X[X.game_date >= "2026-09-07"]
    if fut.empty:
        pytest.skip("sin partidos recientes")
    d = game_detail(int(fut.iloc[0].game_id))
    assert "error" not in d
    # clima y umpire no se ingieren: deben decirlo, no inventarlo
    assert d["contexto"]["clima"]["disponible"] is False
    assert d["contexto"]["umpire"]["disponible"] is False
    for lado in ("home", "away"):
        a = d["abridores"][lado]
        # si no hay abridor, no puede haber estadisticas suyas con valor
        if a["name"] is None:
            assert all(s["value"] is None for s in a["stats"])
        assert d["alineaciones"][lado]["estado"] in (
            "confirmed", "probable", "projected", "no confirmada")


def test_las_reasignaciones_a_menores_no_cuentan_como_lesion():
    from MLB.ingest.statsapi import IL_CODES
    assert "RM" not in IL_CODES and "MIN" not in IL_CODES
    with session_scope() as s:
        from MLB.database.models import Injury
        malos = s.execute(select(Injury).where(
            Injury.status.in_(("RM", "MIN"))).limit(1)).scalars().first()
    assert malos is None


def test_el_historico_avisa_cuando_la_muestra_es_pequena():
    from MLB.engine.history import MIN_N_UTIL, history
    h = history()
    for m, v in h["por_mercado"].items():
        if v["n"] < MIN_N_UTIL:
            assert v["muestra_util"] is False and v["aviso"], m
        assert v["ic95"] is None or v["ic95"][0] <= (v["accuracy"] or 0) <= v["ic95"][1]
    if h["total"]["n"]:
        assert h["total"]["aciertos"] + h["total"]["fallos"] == h["total"]["n"]


def test_el_historico_solo_cuenta_predicciones_ya_evaluadas():
    from MLB.engine.history import history
    h = history()
    for p in h["predicciones"]:
        assert p["resultado"] in ("win", "loss")
        assert isinstance(p["acierto"], bool)
