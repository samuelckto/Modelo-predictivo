"""Aislamiento NFL <-> MLB. Estas pruebas deben fallar si alguien cruza los motores."""
import sqlite3
from pathlib import Path

import pytest

import NFL.adapter as nfl
from MLB.database.session import MLB_DATABASE_URL, init_db, table_names
from shared.paths import (MLB_DIR, NFL_DIR, NFL_HOME, ROOT, IsolationError, assert_writable, sport_root)


def test_bases_de_datos_distintas():
    mlb_db = Path(MLB_DATABASE_URL.replace("sqlite:///", ""))
    nfl_db = nfl.db_path()
    assert mlb_db != nfl_db
    assert MLB_DIR in mlb_db.parents
    assert MLB_DIR not in nfl_db.parents


def test_mlb_no_puede_escribir_en_el_arbol_nfl():
    with pytest.raises(IsolationError):
        assert_writable(NFL_HOME / "database" / "nflpred.sqlite3", "MLB")
    with pytest.raises(IsolationError):
        assert_writable(NFL_HOME / "models" / "x.joblib", "MLB")


def test_nfl_no_puede_escribir_en_el_arbol_mlb():
    with pytest.raises(IsolationError):
        assert_writable(MLB_DIR / "database" / "mlb.sqlite3", "NFL")
    with pytest.raises(IsolationError):
        assert_writable(MLB_DIR / "models" / "x.joblib", "NFL")


def test_cada_deporte_escribe_en_lo_suyo():
    assert assert_writable(MLB_DIR / "data" / "raw" / "a.parquet", "MLB")
    assert assert_writable(NFL_DIR / "database" / "nfl_markets.sqlite3", "NFL")


def test_nadie_escribe_en_el_motor_nfl_original():
    for sport in ("NFL", "MLB"):
        with pytest.raises(IsolationError):
            assert_writable(NFL_HOME / "data" / "raw" / "a.parquet", sport)


def test_ningun_deporte_escribe_fuera_del_proyecto():
    with pytest.raises(IsolationError):
        assert_writable(Path.home() / "algo.txt", "MLB")
    with pytest.raises(IsolationError):
        assert_writable(ROOT / "config" / "x.yml", "MLB")   # config es compartida: solo lectura


@pytest.mark.skipif(not nfl.available()[0], reason="sistema NFL no instalado en esta maquina")
def test_la_conexion_nfl_es_de_solo_lectura():
    assert nfl.read_only_check().startswith("ok:")
    with pytest.raises(sqlite3.OperationalError):
        with nfl._conn() as c:
            c.execute("delete from predictions")


def test_el_esquema_mlb_no_contiene_tablas_de_nfl():
    init_db()
    t = set(table_names())
    for prohibida in ("team_game_stats", "depth_charts", "weekly_reports", "odds_snapshots"):
        assert prohibida not in t
    assert {"games", "predictions", "data_source_logs"} <= t


def test_no_hay_rutas_cruzadas_hardcodeadas_en_el_motor_mlb():
    """Ningun archivo de MLB/ puede mencionar rutas o modulos del sistema NFL."""
    ofensas = []
    for f in (MLB_DIR).rglob("*.py"):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        for pat in ("nflpred", "nfl-prediction-app", "NFL_HOME"):
            if pat in txt:
                ofensas.append(f"{f.relative_to(ROOT)}: {pat}")
    assert not ofensas, ofensas


def test_sport_root_rechaza_deportes_desconocidos():
    with pytest.raises(ValueError):
        sport_root("NHL")
