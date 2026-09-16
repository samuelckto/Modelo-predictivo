"""Entrenamiento final. Se entrena con TODO el historico, una sola vez.

Los modelos que se guardan aqui son los que se eligieron en el walk-forward, no
otros: corners con el estimador ganador y goles con Dixon-Coles sobre lambdas de
gradient boosting. Cambiar de modelo exige volver a pasar por la validacion.

Cada artefacto guarda su periodo de entrenamiento y sus features, para que una
prediccion pueda auditarse despues.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from SOCCER.db import SoccerModelVersion, init_db, session_scope
from SOCCER.features.build import FEATURES_PARQUET
from SOCCER.models.cards import EstimadorTarjetas
from SOCCER.models.corners import EstimadorCorners
from SOCCER.models.lambdas import ESTIMADORES, ajustar_rho, columnas
from shared.paths import SOCCER_MODELS_DIR, assert_writable
from shared.timeutil import utcnow

VERSION_CORNERS = "SOCCER_CORNERS_v1"
VERSION_BTTS = "SOCCER_BTTS_v1"
VERSION_TOTAL = "SOCCER_TOTAL_v1"
VERSION_DC = "SOCCER_DC_v1"
VERSION_CARDS = "SOCCER_CARDS_v1"

# Elegidos por el walk-forward, no por preferencia.
CONFIG_CORNERS = {"algoritmo": "hgb", "objetivo": "equipos", "familia": "nbinom"}
CONFIG_GOLES = {"estimador": "hgb_poisson", "familia": "dixon_coles"}
# TARJETAS. El bloque de seleccion (14 432 partidos, 2016-17 a 2023-24) separa a
# las cinco configuraciones por el CUARTO decimal de log loss: 0.63854 la mejor y
# 0.63893 la peor. Esa diferencia no distingue nada, asi que elegir "la mejor"
# habria sido elegir ruido. La decision se toma por tres razones que no dependen
# de esa cuarta cifra:
#   1. el camino 'equipos' es el unico que produce los mercados por equipo
#      (¿recibe tarjeta este equipo?, ¿los dos?), que el camino 'total' no puede dar;
#   2. a nivel de equipo los datos estan INFRAdispersos (media 2.03, varianza 1.89),
#      asi que la binomial negativa colapsa a Poisson. El walk-forward lo confirma:
#      hgb_equipos_poisson y hgb_equipos_nbinom dan numeros IDENTICOS (0.63649);
#   3. el GLM se descarta por no converger en 800 iteraciones, no por su puntuacion.
# Se prefiere la familia mas simple que los datos soportan.
CONFIG_TARJETAS = {"algoritmo": "hgb", "objetivo": "equipos", "familia": "poisson"}


def entrenar(progress=None) -> dict:
    init_db()
    SOCCER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    assert_writable(SOCCER_MODELS_DIR, "SOCCER")
    X = pd.read_parquet(FEATURES_PARQUET).dropna(subset=["home_goals", "away_goals"])
    ahora = utcnow()
    periodo = f"{X.match_date.min()} .. {X.match_date.max()}"
    cols = columnas(X)
    out = {}

    # --- goles: lambdas + Dixon-Coles -------------------------------------
    if progress:
        progress("entrenando modelo de goles")
    est = ESTIMADORES[CONFIG_GOLES["estimador"]]().fit(X, X["home_goals"], X["away_goals"])
    lh, la = est.predict(X)
    rho = ajustar_rho(X, lh, la)
    g = np.concatenate([X["home_goals"].to_numpy(float), X["away_goals"].to_numpy(float)])
    media, var = float(g.mean()), float(g.var())
    phi = max(media ** 2 / max(var - media, 1e-6), 0.5) if var > media else 50.0
    art_goles = {"estimador": est, "features": cols, "rho": rho, "phi": phi,
                 "familia": CONFIG_GOLES["familia"], "entrenado_en": ahora.isoformat(),
                 "periodo": periodo, "n": int(len(X))}
    joblib.dump(art_goles, SOCCER_MODELS_DIR / "goles_v1.joblib")
    out["goles"] = {"rho": round(rho, 4), "phi": round(phi, 3), "n": int(len(X))}

    # --- corners: solo partidos que los tienen -----------------------------
    C = X[X["total_corners"].notna()]
    if progress:
        progress(f"entrenando modelo de corners con {len(C)} partidos")
    ec = EstimadorCorners(CONFIG_CORNERS["algoritmo"], CONFIG_CORNERS["objetivo"]).fit(
        C, C["home_corners"], C["away_corners"], C["total_corners"])
    art_corners = {"estimador": ec, "features": ec.cols, "phi": ec.phi,
                   "familia": CONFIG_CORNERS["familia"], "camino": CONFIG_CORNERS["objetivo"],
                   "entrenado_en": ahora.isoformat(),
                   "periodo": f"{C.match_date.min()} .. {C.match_date.max()}", "n": int(len(C))}
    joblib.dump(art_corners, SOCCER_MODELS_DIR / "corners_v1.joblib")
    out["corners"] = {"phi": round(float(ec.phi), 3), "n": int(len(C)),
                      "ligas": sorted(C.league_code.unique().tolist())}

    # --- tarjetas: solo partidos que las tienen ----------------------------
    T = X[X["total_yellow"].notna()]
    if progress:
        progress(f"entrenando modelo de tarjetas con {len(T)} partidos")
    et = EstimadorTarjetas(CONFIG_TARJETAS["algoritmo"], CONFIG_TARJETAS["objetivo"]).fit(
        T, T["home_yellow"], T["away_yellow"], T["total_yellow"])
    art_cards = {"estimador": et, "features": et.cols, "phi": et.phi,
                 "familia": CONFIG_TARJETAS["familia"], "camino": CONFIG_TARJETAS["objetivo"],
                 "entrenado_en": ahora.isoformat(),
                 "periodo": f"{T.match_date.min()} .. {T.match_date.max()}",
                 "n": int(len(T)),
                 "sin_arbitro": True}
    joblib.dump(art_cards, SOCCER_MODELS_DIR / "cards_v1.joblib")
    out["tarjetas"] = {"phi": round(float(et.phi), 3), "n": int(len(T)),
                       "ligas": sorted(T.league_code.unique().tolist()),
                       "holdout_log_loss": 0.63649, "holdout_baseline": 0.65761,
                       "holdout_accuracy": 0.6372, "baseline_accuracy": 0.6360,
                       "limitacion": ("no hay arbitro en la base; es el factor conocido "
                                      "mas fuerte de este mercado y falta")}

    with session_scope() as s:
        for nombre, mercado, alg, extra in (
                (VERSION_CORNERS, "corners", "hgb_poisson + binomial negativa", out["corners"]),
                (VERSION_TOTAL, "total_goals", "hgb_poisson + Dixon-Coles", out["goles"]),
                (VERSION_BTTS, "btts", "hgb_poisson + Dixon-Coles", out["goles"]),
                (VERSION_DC, "double_chance", "hgb_poisson + Dixon-Coles", out["goles"]),
                (VERSION_CARDS, "cards", "hgb_poisson por equipo", out["tarjetas"])):
            s.add(SoccerModelVersion(
                name=nombre, market=mercado, scope="global", league_code=None,
                algorithm=alg, features=json.dumps(cols),
                training_period=periodo, validation_period="walk-forward 2016-17..2023-24",
                calibration=None, metrics=json.dumps(extra), created_at=ahora))
    return out
