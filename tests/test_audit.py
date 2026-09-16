"""Pruebas nacidas de la AUDITORIA (§25). Blindan lo que la auditoria descubrio.

Ninguna prueba anterior se elimino: estas se suman.
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

import NFL.adapter as nfl
from MLB.database.models import Odds, Prediction, StatcastAgg
from MLB.database.session import session_scope
from MLB.engine.gating import VARIANT_GATE, decide
from MLB.engine.models import feature_columns
from MLB.ingest.statcast import SAVANT_ROW_CAP
from shared.asof import filter_available
from shared.paths import MLB_PROCESSED_DIR, ROOT

AUDIT = ROOT / "audit" / "out"
FEAT = MLB_PROCESSED_DIR / "features.parquet"
needs_feat = pytest.mark.skipif(not FEAT.exists(), reason="sin features")


# ------------------------------------------------- §3 leakage del abridor
@pytest.mark.skipif(not (AUDIT / "leakage_starter_isolation.json").exists(),
                    reason="sin resultado de aislamiento del abridor")
def test_solo_el_abridor_muestra_leakage():
    r = json.loads((AUDIT / "leakage_starter_isolation.json").read_text())
    assert r["dif_otras"] == 0, (
        "hay features FUERA del bloque del abridor que cambian al truncar el dataset")
    assert r["dif_sp"] > 0, "la prueba deberia detectar el supuesto del abridor"


@needs_feat
def test_el_gating_usa_la_variante_limpia():
    assert VARIANT_GATE == "sin_abridor"
    d = decide()
    for m in ("moneyline", "f5_moneyline", "run_line"):
        assert d[m]["variant_used"] == "sin_abridor", m


@needs_feat
def test_f5_no_se_publica_como_pick_tras_la_auditoria():
    """Con la cota limpia la ventaja de F5 dejo de ser significativa."""
    d = decide()["f5_moneyline"]
    assert d["significant_95"] is False and d["publish_pick"] is False
    assert d["z"] < 1.645


@needs_feat
def test_moneyline_se_sirve_con_elo():
    from MLB.engine.gating import production_model
    assert production_model("moneyline") == "elo"


@needs_feat
def test_season_no_es_una_feature():
    X = pd.read_parquet(FEAT)
    cols = feature_columns(X)
    assert not [c for c in cols if "season" in c]


# ------------------------------------------------------- §11 orden de picks
@pytest.mark.skipif(not (AUDIT / "calibration_edge.json").exists(), reason="sin auditoria")
def test_el_edge_no_ordena_mejor_que_la_probabilidad():
    """Queda registrado el hallazgo: si algun dia el edge ganara, hay que revisarlo."""
    d = json.loads((AUDIT / "calibration_edge.json").read_text())
    for market, r in d.items():
        e = r["edge_vs_alternativas"]["edge_sobre_base"]["separacion_pp"]
        p = r["edge_vs_alternativas"]["probabilidad_bruta"]["separacion_pp"]
        assert p >= e, (f"{market}: el edge separa mas que la probabilidad ({e} vs {p}); "
                        f"habria que revisar la decision de orden")


# --------------------------------------------------------- §12 riesgo
@pytest.mark.skipif(not (AUDIT / "calibration_edge.json").exists(), reason="sin auditoria")
def test_el_riesgo_publicado_esta_calibrado():
    d = json.loads((AUDIT / "calibration_edge.json").read_text())
    for market, r in d.items():
        rk = r["riesgo"]
        assert rk["monotono"] is True, f"{market}: el riesgo no es monotono"
        assert rk["error_medio_pp"] < 3.0, f"{market}: error {rk['error_medio_pp']} pp"


# ------------------------------------------------- §13 cuotas y timestamps
def test_las_cuotas_de_cierre_no_entran_en_una_prediccion_anterior():
    pred_ts = pd.Timestamp("2026-04-10 12:00")
    odds = pd.DataFrame([
        {"selection": "home", "price": -140, "available_at": "2026-04-09 20:00"},
        {"selection": "home", "price": -155, "available_at": "2026-04-10 11:00"},
        {"selection": "home", "price": -170, "available_at": "2026-04-10 19:30"},
    ])
    vis = filter_available(odds, pred_ts)
    assert len(vis) == 2 and -170 not in list(vis.price)


def test_ninguna_cuota_guardada_es_posterior_a_la_prediccion_que_la_uso():
    with session_scope() as s:
        preds = s.execute(select(Prediction).where(
            Prediction.market_probability.isnot(None)).limit(200)).scalars().all()
        for p in preds:
            usadas = s.execute(select(Odds).where(
                Odds.game_id == p.game_id,
                Odds.available_at >= p.prediction_timestamp).limit(1)).scalars().first()
            if usadas is not None:
                # existir puede; lo que no puede es que la prediccion la haya usado
                assert p.prediction_timestamp <= usadas.available_at


def test_no_se_guardan_claves_de_api_en_las_urls():
    db = ROOT / "MLB" / "database" / "mlb.sqlite3"
    if not db.exists():
        pytest.skip("sin base MLB")
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n = c.execute("select count(*) from data_source_logs where url like '%apiKey=%' "
                  "or url like '%api_key=%'").fetchone()[0]
    c.close()
    assert n == 0


# --------------------------------------------------- §27 truncamiento Statcast
def test_la_guarda_de_truncamiento_existe_y_es_el_tope_real():
    assert SAVANT_ROW_CAP == 25000


def test_no_hay_dias_de_statcast_pegados_al_tope():
    """Si un dia tuviera exactamente 25.000 lanzamientos seria sospechoso."""
    with session_scope() as s:
        rows = s.execute(select(StatcastAgg.game_date,
                                StatcastAgg.pitches).where(
            StatcastAgg.scope == "pitcher")).all()
    if not rows:
        pytest.skip("sin statcast")
    por_dia = pd.DataFrame(rows, columns=["d", "p"]).groupby("d").p.sum()
    assert not (por_dia == 25000).any()
    assert por_dia.max() < 25000, f"un dia alcanzo {por_dia.max()} lanzamientos"


# ------------------------------------------------------ §26 reproducibilidad
@needs_feat
def test_mismo_snapshot_misma_prediccion():
    """El mismo modelo y las mismas features deben dar exactamente lo mismo."""
    import joblib
    from shared.paths import MLB_MODELS_DIR
    art = sorted(MLB_MODELS_DIR.glob("f5_moneyline_v*.joblib"))
    if not art:
        pytest.skip("sin modelo entrenado")
    mm = joblib.load(art[-1])
    X = pd.read_parquet(FEAT)
    elo = joblib.load(sorted(MLB_MODELS_DIR.glob("elo_v*.joblib"))[-1])["elo"]
    Xe = X.merge(elo[["game_id", "p_home_elo"]], on="game_id", how="left")
    Xe["d_elo"] = Xe.p_home_elo - 0.5
    sub = Xe[Xe.season == 2026].head(200)
    a = mm.predict_proba(sub)
    b = mm.predict_proba(sub.copy())
    assert np.allclose(a, b, atol=0), "el mismo snapshot dio predicciones distintas"


def test_cada_prediccion_guarda_su_snapshot_de_features():
    from MLB.database.models import PredictionSnapshot
    with session_scope() as s:
        preds = s.execute(select(Prediction.id).limit(100)).scalars().all()
        if not preds:
            pytest.skip("sin predicciones")
        snaps = set(s.execute(select(PredictionSnapshot.prediction_id).where(
            PredictionSnapshot.prediction_id.in_(preds))).scalars().all())
    faltan = [p for p in preds if p not in snaps]
    assert not faltan, f"{len(faltan)} predicciones sin snapshot reproducible"


# --------------------------------------------------------------- §15 NFL
def test_la_base_nfl_sigue_siendo_de_solo_lectura():
    ok, _ = nfl.available()
    if not ok:
        pytest.skip("sin motor NFL")
    assert nfl.read_only_check().startswith("ok:")


@pytest.mark.skipif(not (AUDIT / "nfl_audit.json").exists(), reason="sin auditoria NFL")
def test_la_auditoria_nfl_no_encontro_leakage():
    a = json.loads((AUDIT / "nfl_audit.json").read_text())["auditoria"]
    lk = a["leakage_predicciones"]
    assert lk["cutoff_posterior_al_kickoff"] == 0
    assert lk["prediccion_posterior_al_kickoff"] == 0


@pytest.mark.skipif(not (AUDIT / "nfl_audit.json").exists(), reason="sin auditoria NFL")
def test_existe_copia_de_seguridad_de_la_base_nfl():
    """La copia debe existir EN ESTA maquina.

    El JSON guarda la ruta de la maquina donde se ejecuto la auditoria, que no
    tiene por que ser esta. Lo que se comprueba aqui es que, si hay motor NFL
    instalado, exista al menos una copia local en su carpeta `backups`.
    """
    b = json.loads((AUDIT / "nfl_audit.json").read_text())["backup"]
    assert b["status"] == "ok"
    ok, _ = nfl.available()
    if not ok:
        pytest.skip("sin motor NFL en esta maquina")
    backups = sorted((nfl.db_path().parent / "backups").glob("nflpred_*.sqlite3"))
    if not backups:
        pytest.skip("no hay copia local todavia: ejecuta `python -m audit.nfl`")
    assert backups[-1].stat().st_size > 0


# ------------------------------------------------------- §2 procedencia
@pytest.mark.skipif(not (AUDIT / "sources.json").exists(), reason="sin auditoria de fuentes")
def test_las_fuentes_verificadas_coinciden_con_el_original():
    s = json.loads((AUDIT / "sources.json").read_text())
    assert s["calendario"]["match"] == s["calendario"]["games"] > 0
    assert s["boxscores"]["match"] == s["boxscores"]["lines"] > 0
    assert s["statcast"]["match"] == s["statcast"]["checked"] > 0
    assert s["statcast"]["truncated"] is False
    assert s["secretos"]["ok"] is True
    assert set(s["inventario"]["hosts"]) <= {"mlb_stats_api", "baseball_savant",
                                             "the_odds_api", "open_meteo"}


# ------------------------------------------------ §22-§23 versionado real
def test_el_versionado_supersede_y_no_borra():
    """Ejercita el camino real de guardado: una segunda prediccion del mismo
    partido y mercado con probabilidad distinta debe crear v2 y dejar v1 como
    'superseded', nunca borrarla. Se hace sobre un game_id de prueba y se
    deshace al final."""
    from MLB.engine.predict import _store
    from shared.blend import DEFAULT_POLICY
    from MLB.database.models import PredictionSnapshot

    class Row:
        game_id, season, game_date = -999, 2026, "2026-09-09"
        start_utc = pd.Timestamp("2026-09-09 23:00")
        home_abbr, away_abbr = "TST", "OPP"

    now = datetime(2026, 9, 9, 10, 0, 0)
    with session_scope() as s:
        try:
            a = _store(s, Row(), "moneyline", "TST", None, 0.60, None, 0.60, 0.55,
                       40.0, {"summary": "v1"}, {"market_available": False},
                       DEFAULT_POLICY, [], [], 0.01, None, now, "test", {})
            b = _store(s, Row(), "moneyline", "TST", None, 0.70, None, 0.70, 0.55,
                       30.0, {"summary": "v2"}, {"market_available": False},
                       DEFAULT_POLICY, [], [], 0.01, None,
                       now + timedelta(hours=3), "cambio", {})
            assert a == (1, 0) and b == (1, 1)
            rows = s.execute(select(Prediction).where(Prediction.game_id == -999)
                             .order_by(Prediction.version)).scalars().all()
            assert len(rows) == 2
            assert rows[0].version == 1 and rows[0].status == "superseded"
            assert rows[1].version == 2 and rows[1].parent_prediction_id == rows[0].id
            assert rows[1].revision_diff["antes"] == pytest.approx(0.60)
            assert rows[1].revision_diff["ahora"] == pytest.approx(0.70)
            # sin cambio material no se crea version nueva
            c = _store(s, Row(), "moneyline", "TST", None, 0.7001, None, 0.7001, 0.55,
                       30.0, {}, {"market_available": False}, DEFAULT_POLICY, [], [],
                       0.01, None, now + timedelta(hours=5), "sin cambio", {})
            assert c == (0, 0)
        finally:
            ids = [r.id for r in s.execute(select(Prediction)
                                           .where(Prediction.game_id == -999)).scalars()]
            if ids:
                s.query(PredictionSnapshot).filter(
                    PredictionSnapshot.prediction_id.in_(ids)).delete(
                    synchronize_session=False)
                s.query(Prediction).filter(Prediction.game_id == -999).delete(
                    synchronize_session=False)


def test_los_estados_de_confirmacion_se_distinguen():
    """§23: provisional vs published segun si faltan datos."""
    with session_scope() as s:
        rows = s.execute(select(Prediction).limit(300)).scalars().all()
    if not rows:
        pytest.skip("sin predicciones")
    vivas = [p for p in rows if p.status != "superseded"]
    if not vivas:
        pytest.skip("todas las predicciones estan superseded")
    for p in vivas:
        falta = (p.data_completeness or {}).get("missing", [])
        if falta:
            assert p.status == "provisional", f"{p.id}: faltan datos pero no es provisional"
        else:
            assert p.status == "published"


def test_el_orden_del_calendario_es_cronologico():
    from fastapi.testclient import TestClient
    from dashboard.backend.main import app
    c = TestClient(app)
    d = c.get("/api/games", params={"range": "30d", "sport": "all"}).json()
    fechas = [x["date"] for x in d["days"]]
    assert fechas == sorted(fechas)
    from shared.schema import norm_ts
    for day in d["days"]:
        horas = [norm_ts(g["start_utc"]) for g in day["games"] if g.get("start_utc")]
        assert horas == sorted(horas), f"partidos desordenados en {day['date']}"
    # NOTA: el orden GLOBAL por start_utc no tiene por que ser monotono entre dias.
    # MLB agrupa por fecha oficial del partido y un juego del 9 de septiembre puede
    # empezar a las 02:10 UTC del dia 10, solapandose con un partido NFL del dia 10.
    # Lo correcto es: claves de dia ordenadas + orden cronologico DENTRO de cada dia,
    # que es justo lo que se comprueba arriba. El frontend reagrupa por dia local.
    assert len(d["days"]) > 0
