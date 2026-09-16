"""Tests de SOCCER: aislamiento, anti-leakage, distribuciones, mercados y gating.

Los tests de leakage no se limitan a comprobar que hoy no hay fuga: INYECTAN
datos del futuro y exigen que la auditoria los cace. Una auditoria que solo dice
que si a todo no sirve de nada.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SOCCER.evaluation import leakage
from SOCCER.features.build import FEATURES_PARQUET, ID_COLS, TARGET_COLS, VENTANAS, EstadoEquipo
from SOCCER.ingestion.teams import Resolver, clave, normaliza
from SOCCER.markets.gating import MERCADOS, evaluar_mercado
from SOCCER.models import corners as C
from SOCCER.models import goals as G
from SOCCER.leagues import CODES, CON_HISTORICO, LIGAS, liga
from shared.paths import (MLB_DIR, NBA_DIR, NFL_DIR, NFL_HOME, SOCCER_DIR, TENIS_DIR,
                          IsolationError, assert_writable)

needs_data = pytest.mark.skipif(not FEATURES_PARQUET.exists(), reason="sin features de futbol")


@pytest.fixture(scope="module")
def X():
    return pd.read_parquet(FEATURES_PARQUET)


# --------------------------------------------------------------------------- #
# 1. Aislamiento
# --------------------------------------------------------------------------- #
def test_soccer_solo_escribe_en_su_arbol():
    assert assert_writable(SOCCER_DIR / "database" / "x.sqlite3", "SOCCER")
    for otro in (MLB_DIR, NFL_DIR, NBA_DIR, TENIS_DIR):
        with pytest.raises(IsolationError):
            assert_writable(otro / "database" / "x.sqlite3", "SOCCER")


def test_soccer_no_puede_escribir_en_el_motor_nfl_original():
    with pytest.raises(IsolationError):
        assert_writable(NFL_HOME / "cualquier.db", "SOCCER")


def test_los_demas_deportes_no_pueden_escribir_en_soccer():
    for deporte in ("MLB", "NFL", "NBA", "TENIS"):
        with pytest.raises(IsolationError):
            assert_writable(SOCCER_DIR / "database" / "soccer_markets.sqlite3", deporte)


def test_la_base_de_soccer_es_propia():
    from SOCCER.db import SOCCER_DATABASE_URL
    assert "soccer_markets.sqlite3" in SOCCER_DATABASE_URL
    for otra in ("mlb.sqlite3", "nba_markets", "tenis_markets", "nfl_markets"):
        assert otra not in SOCCER_DATABASE_URL


# --------------------------------------------------------------------------- #
# 2. Ligas
# --------------------------------------------------------------------------- #
def test_las_competiciones_soportadas_son_las_pedidas():
    assert set(CODES) == {"EPL", "LALIGA", "SERIEA", "BUNDES", "LIGUE1", "LIGAMX", "SAUDI", "UCL"}


def test_saudi_no_tiene_fuente_historica_declarada():
    assert liga("SAUDI").historico is None
    assert "SAUDI" not in CON_HISTORICO


def test_la_champions_es_copa_y_no_liga():
    from SOCCER.leagues import COPAS
    assert "UCL" in COPAS
    assert liga("UCL").historico is None      # su historico viene de otro repositorio


def test_un_club_es_el_mismo_en_su_liga_y_en_la_champions():
    """Sin esto el Arsenal llegaria a la Champions sin historia."""
    r = Resolver()
    a, _ = r.resolver("EPL", "Arsenal")
    b, _ = r.resolver("UCL", "Arsenal FC (ENG)")
    assert a == b
    c, _ = r.resolver("BUNDES", "Bayern Munich")
    d, _ = r.resolver("UCL", "FC Bayern München (GER)")
    assert c == d


def test_el_codigo_de_pais_no_forma_parte_del_nombre():
    from SOCCER.ingestion.teams import pais_de
    assert pais_de("Arsenal FC (ENG)") == "ENG"
    assert "eng" not in clave("Arsenal FC (ENG)", "UCL")


def test_el_pool_de_clubes_es_global_no_por_liga():
    r = Resolver()
    r.resolver("EPL", "Arsenal")
    assert isinstance(r.canonicos, dict)
    assert all(isinstance(v, str) for v in r.canonicos.values())


def test_todas_las_ligas_tienen_clave_de_cuotas():
    for lg in LIGAS:
        assert lg.odds_key.startswith("soccer_")


# --------------------------------------------------------------------------- #
# 3. Anti-leakage: la auditoria tiene que CAZAR fugas inyectadas
# --------------------------------------------------------------------------- #
@needs_data
def test_auditoria_limpia_en_los_datos_reales(X):
    r = leakage.audit(X, corte="2022-01-01")
    assert r["ok"], r["pruebas"]


@needs_data
def test_detecta_una_feature_que_es_el_resultado(X):
    envenenado = X.head(3000).copy()
    envenenado["total_goals_futuro"] = envenenado["total_goals"]
    r = leakage.correlacion_sospechosa(envenenado)
    assert not r["ok"]
    assert any(s["feature"] == "total_goals_futuro" for s in r["sospechosas"])


@needs_data
def test_detecta_una_columna_prohibida_por_nombre(X):
    envenenado = X.head(500).copy()
    envenenado["closing_odds"] = 1.85
    r = leakage.columnas_prohibidas(envenenado)
    assert not r["ok"] and "closing_odds" in r["prohibidas_encontradas"]


@needs_data
def test_detecta_un_cutoff_posterior_al_saque(X):
    envenenado = X[X["kickoff_utc"].notna()].head(300).copy()
    envenenado["available_at"] = pd.to_datetime(envenenado["kickoff_utc"]) + pd.Timedelta(hours=3)
    r = leakage.cutoff_ordenado(envenenado)
    assert not r["ok"] and r["available_at_posterior_al_saque"] > 0


@needs_data
def test_detecta_una_ventana_que_incluye_el_partido_actual(X):
    envenenado = X.head(6000).copy()
    # la forma del local pasa a incluir sus goles de HOY: eso es leakage
    envenenado["home_gf_5"] = envenenado["home_goals"].astype(float)
    r = leakage.ventanas_excluyen(envenenado, muestra=120)
    assert not r["ok"] and r["n_fallos"] > 0


@needs_data
def test_ninguna_feature_es_un_objetivo(X):
    feats = [c for c in X.columns if c not in ID_COLS + TARGET_COLS]
    for prohibida in ("home_goals", "away_goals", "total_goals", "resultado", "btts",
                      "home_corners", "away_corners", "total_corners"):
        assert prohibida not in feats


@needs_data
def test_available_at_nunca_pasa_del_dia_del_partido(X):
    a = pd.to_datetime(X["available_at"])
    d = pd.to_datetime(X["match_date"])
    assert (a <= d + pd.Timedelta(days=1)).all()


def test_el_estado_de_un_equipo_solo_mira_hacia_atras():
    e = EstadoEquipo()
    antes = e.medias()
    assert antes["n_partidos"] == 0
    e.registrar(3, 1, True, date(2024, 1, 1))
    despues = e.medias()
    assert despues["n_partidos"] == 1 and despues["gf_media"] == 3.0
    # la lectura previa no cambia retroactivamente
    assert antes["gf_media"] != 3.0


def test_la_ventana_de_forma_no_guarda_mas_de_n_partidos():
    e = EstadoEquipo()
    for i in range(12):
        e.registrar(i % 3, 1, True, date(2024, 1, 1) + timedelta(days=i))
    for v in VENTANAS:
        assert len(e.ventanas[v]) == v


def test_los_corners_solo_cuentan_partidos_que_los_traen():
    e = EstadoEquipo()
    e.registrar(1, 1, True, date(2024, 1, 1))          # partido sin corners
    assert e.medias_corners()["c_n_partidos"] == 0
    e.registrar_corners(7, 3, 14, 9, True)
    assert e.medias_corners()["c_n_partidos"] == 1
    assert e.medias_corners()["cf_media"] == 7.0


# --------------------------------------------------------------------------- #
# 4. Nombres de equipo
# --------------------------------------------------------------------------- #
def test_no_fusiona_manchester_united_con_manchester_city():
    r = Resolver()
    a, _ = r.resolver("EPL", "Man City")
    b, _ = r.resolver("EPL", "Man United")
    assert a != b


def test_une_la_misma_grafia_entre_fuentes():
    r = Resolver()
    a, _ = r.resolver("EPL", "Arsenal")
    b, _ = r.resolver("EPL", "Arsenal FC")
    assert a == b


def test_apostrofo_no_parte_el_nombre():
    assert clave("Nott'm Forest", "EPL") == clave("Nottingham Forest", "EPL")


def test_normaliza_quita_acentos():
    assert normaliza("Atlético Madrid") == "atletico madrid"


# --------------------------------------------------------------------------- #
# 5. Distribucion de goles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fam", ["poisson", "dixon_coles", "negative_binomial",
                                 "bivariate_poisson"])
def test_1x2_suma_uno(fam):
    r = G.mercados(1.6, 1.1, fam)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < 1e-9


@pytest.mark.parametrize("fam", ["poisson", "dixon_coles", "negative_binomial",
                                 "bivariate_poisson"])
def test_btts_si_mas_no_suman_uno(fam):
    r = G.mercados(1.6, 1.1, fam)
    assert abs(r["btts_yes"] + r["btts_no"] - 1.0) < 1e-9


@pytest.mark.parametrize("linea", [0.5, 1.5, 2.5, 3.5, 4.5])
def test_over_under_medio_punto_no_tiene_push(linea):
    r = G.mercados(1.5, 1.2, "dixon_coles", lineas=(linea,))
    t = r["totales"][f"{linea:g}"]
    assert t["push"] == 0.0
    assert abs(t["over"] + t["under"] - 1.0) < 1e-9


@pytest.mark.parametrize("linea", [2.0, 3.0, 4.0])
def test_linea_entera_produce_push(linea):
    m = G.joint(1.5, 1.2, "poisson")
    o, u, p = G.prob_over(m, linea)
    assert p > 0
    assert abs(o + u + p - 1.0) < 1e-9


def test_dixon_coles_sube_el_empate_frente_a_poisson():
    p = G.mercados(1.4, 1.2, "poisson")
    d = G.mercados(1.4, 1.2, "dixon_coles", rho=-0.05)
    assert d["p_draw"] > p["p_draw"]


def test_btts_es_coherente_con_over_05():
    r = G.mercados(1.7, 1.0, "dixon_coles", lineas=(0.5,))
    assert r["btts_yes"] <= r["totales"]["0.5"]["over"] + 1e-12


def test_goles_esperados_coinciden_con_las_lambdas():
    r = G.mercados(2.0, 1.0, "poisson")
    assert abs(r["expected_goals"]["home"] - 2.0) < 0.02
    assert abs(r["expected_goals"]["away"] - 1.0) < 0.02


@pytest.mark.parametrize("fam", ["poisson", "dixon_coles", "negative_binomial",
                                 "bivariate_poisson"])
def test_version_vectorizada_da_lo_mismo(fam):
    lh, la = np.array([1.8, 0.9]), np.array([1.0, 1.7])
    b = G.mercados_batch(lh, la, fam, lineas=(2.5,))
    for i in range(2):
        s = G.mercados(lh[i], la[i], fam, lineas=(2.5,))
        assert abs(b["p_home"][i] - s["p_home"]) < 1e-9
        assert abs(b["btts_yes"][i] - s["btts_yes"]) < 1e-9


# --------------------------------------------------------------------------- #
# 6. Corners
# --------------------------------------------------------------------------- #
def test_corners_over_under_suman_uno():
    r = C.mercados([5.4], [4.6], "nbinom", phi=8.0)
    for ln, v in r["totales"].items():
        assert abs(v["over"][0] + v["under"][0] + v["push"][0] - 1.0) < 1e-9


def test_corners_linea_entera_tiene_push():
    D = C.dist_total(np.array([5.5]), np.array([4.5]), "poisson")
    o, u, p = C.over_under(D, 10.0)
    assert p[0] > 0 and abs(o[0] + u[0] + p[0] - 1.0) < 1e-9


def test_corners_esperados_coinciden_con_las_lambdas():
    D = C.dist_total(np.array([6.0]), np.array([4.0]), "poisson")
    assert abs(C.esperado(D)[0] - 10.0) < 0.05


def test_mas_corners_esperados_sube_el_over():
    bajo = C.over_under(C.dist_total(np.array([4.0]), np.array([4.0]), "nbinom", 8.0), 9.5)[0][0]
    alto = C.over_under(C.dist_total(np.array([7.0]), np.array([7.0]), "nbinom", 8.0), 9.5)[0][0]
    assert alto > bajo


def test_binomial_negativa_tiene_mas_varianza_que_poisson():
    lam = np.array([5.0])
    dp = C.dist_total(lam, lam, "poisson")
    dn = C.dist_total(lam, lam, "nbinom", phi=4.0)
    k = np.arange(dp.shape[1])
    vp = float((dp[0] * k ** 2).sum() - (dp[0] * k).sum() ** 2)
    vn = float((dn[0] * k ** 2).sum() - (dn[0] * k).sum() ** 2)
    assert vn > vp


@needs_data
def test_corners_solo_existen_en_las_ligas_con_fuente(X):
    con = X[X["total_corners"].notna()]
    assert set(con["league_code"].unique()) <= {"EPL", "LALIGA", "SERIEA", "BUNDES", "LIGUE1"}
    for code in ("LIGAMX", "SAUDI", "UCL"):
        assert X[(X.league_code == code) & X["total_corners"].notna()].empty


@needs_data
def test_la_champions_tiene_historico_propio(X):
    u = X[X.league_code == "UCL"]
    assert len(u) > 1200
    assert u.home_team.nunique() > 80


@needs_data
def test_los_clubes_grandes_llegan_a_la_champions_con_historia(X):
    """La prueba de que el estado va por club: en la Champions reciente los
    equipos de las 6 ligas traen cientos de partidos, no cero."""
    u = X[(X.league_code == "UCL") & (pd.to_datetime(X.match_date) >= "2024-01-01")]
    assert not u.empty
    grandes = u[u.home_team.isin(["arsenal", "real madrid", "bayern munchen",
                                  "barcelona", "liverpool", "internazionale"])]
    assert not grandes.empty
    assert grandes.home_n_partidos.min() > 200


# --------------------------------------------------------------------------- #
# 7. Gating
# --------------------------------------------------------------------------- #
def test_un_modelo_que_no_supera_al_baseline_queda_bloqueado():
    r = evaluar_mercado("corners", 0.6950, 0.6940, 0.51, 0.51, 5000)
    assert r["estado"] == "blocked"


def test_sin_cuotas_historicas_nunca_pasa_de_proyeccion():
    r = evaluar_mercado("corners", 0.6500, 0.6900, 0.60, 0.50, 5000,
                        hay_cuotas_historicas=False)
    assert r["estado"] == "projection" and r["permite_pick"] is False


def test_con_muestra_pequena_es_insufficient_data():
    r = evaluar_mercado("corners", 0.6500, 0.6900, 0.60, 0.50, 100)
    assert r["estado"] == "insufficient_data"


def test_los_tres_mercados_publicados_son_los_pedidos():
    assert set(MERCADOS) == {"double_chance", "btts", "total_goals"}
    assert "1x2" not in MERCADOS and "moneyline" not in MERCADOS
    assert "corners" not in MERCADOS      # sustituido por doble oportunidad


def test_doble_oportunidad_sale_de_la_distribucion_de_goles():
    """1X = P(gana local) + P(empate). No es un modelo aparte."""
    r = G.mercados(1.6, 1.1, "dixon_coles")
    p_1x = r["p_home"] + r["p_draw"]
    p_x2 = r["p_draw"] + r["p_away"]
    assert 0 < p_1x < 1 and 0 < p_x2 < 1
    # 1X y X2 se solapan en el empate, asi que suman mas de 1 justo por P(empate)
    assert abs((p_1x + p_x2) - (1 + r["p_draw"])) < 1e-9


def test_doble_oportunidad_es_mas_probable_que_ganar_seco():
    r = G.mercados(1.9, 0.9, "dixon_coles")
    assert r["p_home"] + r["p_draw"] > r["p_home"]


def test_poca_historia_no_es_lo_mismo_que_falta_de_fuente():
    """Una liga con poco historico publica probabilidad como NO PICK; un mercado
    sin fuente no publica nada. Confundirlos borro los datos de la liga Saudi."""
    from SOCCER.markets.gating import estado_por_liga
    g = {"mercados": {m: {"estado": "projection", "publica_probabilidad": True,
                          "permite_pick": False} for m in MERCADOS}}
    cob = {"SAUDI": {"con_corners": 0}}
    est = {"SAUDI": {"partidos": 267, "estado": "insufficient_data"}}
    r = estado_por_liga(g, cob, est)["SAUDI"]
    for m in MERCADOS:
        assert r[m]["estado"] == "no_pick", m
        assert r[m]["publica_probabilidad"] is True


# --------------------------------------------------------------------------- #
# 8. Datos reales cargados
# --------------------------------------------------------------------------- #
@needs_data
def test_hay_historico_real_en_las_seis_ligas(X):
    for code in ("EPL", "LALIGA", "SERIEA", "BUNDES", "LIGUE1", "LIGAMX"):
        assert len(X[X.league_code == code]) > 3000, code


@needs_data
def test_saudi_tiene_pocos_partidos_y_por_eso_se_bloquea(X):
    assert len(X[X.league_code == "SAUDI"]) < 1200


@needs_data
def test_no_hay_partidos_duplicados(X):
    g = X.groupby(["league_code", "match_date", "home_team", "away_team"]).size()
    assert int((g > 1).sum()) == 0


@needs_data
def test_las_features_llegan_hasta_fechas_recientes(X):
    ult = pd.to_datetime(X["match_date"]).max()
    assert ult >= pd.Timestamp("2026-05-01")
