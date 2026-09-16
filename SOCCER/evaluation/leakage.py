"""Auditoria anti-leakage. Tiene que ser capaz de ENCONTRAR fugas, no de jurar que no hay.

Cinco comprobaciones independientes, cada una ataca un modo distinto de fallo:

1. `columnas_prohibidas`  ninguna feature es un objetivo disfrazado.
2. `correlacion_sospechosa` ninguna feature correlaciona casi perfecto con el
   resultado del propio partido (asi se detecta una fuga que no sabiamos que
   existia, sin tener que adivinar su nombre).
3. `truncamiento`         reconstruir las features usando SOLO los partidos
   anteriores a una fecha debe dar exactamente lo mismo que el dataset completo.
   Es la prueba dura: si una feature mirara al futuro, cambiaria.
4. `cutoff_ordenado`      available_at nunca es posterior al saque.
5. `ventanas_excluyen`    la ventana de forma de un equipo no contiene su propio
   partido (se verifica recalculando a mano una muestra).
"""
from __future__ import annotations

import math
import random
from datetime import datetime

import pandas as pd

from SOCCER.features.build import ID_COLS, TARGET_COLS, VENTANAS

# Nombres que jamas pueden ser features: son el resultado del propio partido.
PROHIBIDAS = {"home_goals", "away_goals", "total_goals", "goal_diff", "resultado", "btts",
              "ht_home_goals", "ht_away_goals", "score", "ft", "winner",
              "closing_odds", "closing_line", "odds_close"}
UMBRAL_CORR = 0.90


def columnas_prohibidas(X: pd.DataFrame) -> dict:
    feats = [c for c in X.columns if c not in ID_COLS + TARGET_COLS]
    malas = [c for c in feats if c.lower() in PROHIBIDAS]
    return {"ok": not malas, "features": len(feats), "prohibidas_encontradas": malas}


def correlacion_sospechosa(X: pd.DataFrame, umbral: float = UMBRAL_CORR) -> dict:
    """Una feature legitima predice; una feature con fuga casi copia el resultado."""
    feats = [c for c in X.columns if c not in ID_COLS + TARGET_COLS]
    num = X[feats].apply(pd.to_numeric, errors="coerce")
    sospechosas = []
    for objetivo in ("total_goals", "goal_diff", "btts"):
        y = pd.to_numeric(X[objetivo], errors="coerce")
        for c in feats:
            v = num[c]
            if v.nunique(dropna=True) < 2:
                continue
            r = v.corr(y)
            if r is not None and not math.isnan(r) and abs(r) >= umbral:
                sospechosas.append({"feature": c, "objetivo": objetivo, "r": round(float(r), 4)})
    return {"ok": not sospechosas, "umbral": umbral, "sospechosas": sospechosas}


def cutoff_ordenado(X: pd.DataFrame) -> dict:
    """available_at <= kickoff, y nunca despues del dia del partido."""
    x = X.copy()
    x["available_at"] = pd.to_datetime(x["available_at"])
    x["kickoff_utc"] = pd.to_datetime(x["kickoff_utc"])
    x["dia"] = pd.to_datetime(x["match_date"])
    con_ko = x[x["kickoff_utc"].notna()]
    tras_saque = int((con_ko["available_at"] > con_ko["kickoff_utc"]).sum())
    tras_dia = int((x["available_at"] > x["dia"] + pd.Timedelta(days=1)).sum())
    return {"ok": tras_saque == 0 and tras_dia == 0,
            "con_hora_de_saque": int(len(con_ko)),
            "available_at_posterior_al_saque": tras_saque,
            "available_at_posterior_al_dia": tras_dia}


def truncamiento(X: pd.DataFrame, corte: str = "2023-01-01", muestra: int = 400) -> dict:
    """Prueba dura: reconstruir con datos truncados debe dar features identicas.

    Se vuelve a construir el dataset usando UNICAMENTE los partidos anteriores al
    corte y se comparan las features de esos mismos partidos. Si alguna feature
    usara informacion posterior, aqui cambiaria.
    """
    from SOCCER.features.build import EstadoEquipo, _elo_esperado   # noqa: PLC0415
    from SOCCER.leagues import COPAS                                # noqa: PLC0415

    x = X.copy()
    x["match_date"] = pd.to_datetime(x["match_date"])
    corte_ts = pd.Timestamp(corte)
    antes = x[x["match_date"] < corte_ts].sort_values(["match_date", "match_id"])
    if antes.empty:
        return {"ok": True, "nota": "sin partidos antes del corte", "comparadas": 0}

    # Recalculo independiente de dos features representativas (Elo y forma a 5).
    # El estado va por CLUB, igual que en build(): un club es el mismo en su liga
    # y en la Champions. Si esta prueba usara otra clave no estaria comprobando
    # el mismo calculo y su veredicto no valdria nada.
    estados: dict[str, EstadoEquipo] = {}
    recalculado: dict[str, dict] = {}
    for r in antes.itertuples(index=False):
        eh = estados.setdefault(r.home_team, EstadoEquipo())
        ea = estados.setdefault(r.away_team, EstadoEquipo())
        if r.league_code not in COPAS:          # una copa no abre temporada
            eh.nueva_temporada(r.season)
            ea.nueva_temporada(r.season)
        mh, ma = eh.medias(), ea.medias()
        p = _elo_esperado(eh.elo, ea.elo, r.ventaja_local_liga)
        recalculado[r.match_id] = {"elo_home": eh.elo, "elo_away": ea.elo,
                                   "home_gf_5": mh["gf_5"], "away_gc_5": ma["gc_5"],
                                   "home_n_partidos": mh["n_partidos"]}
        hg, ag = int(r.home_goals), int(r.away_goals)
        real = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margen = abs(hg - ag)
        mult = 1.0 if margen <= 1 else (1.5 if margen == 2 else (1.75 + (margen - 3) / 8))
        cambio = 20.0 * mult * (real - p)
        eh.elo += cambio
        ea.elo -= cambio
        eh.registrar(hg, ag, True, r.match_date.date())
        ea.registrar(ag, hg, False, r.match_date.date())

    ids = list(recalculado)
    random.Random(7).shuffle(ids)
    ids = ids[:muestra]
    orig = antes.set_index("match_id")
    difs = []
    for mid in ids:
        for k, v in recalculado[mid].items():
            o = float(orig.at[mid, k])
            if abs(o - float(v)) > 1e-6:
                difs.append({"match_id": mid, "feature": k, "completo": o, "truncado": float(v)})
    return {"ok": not difs, "comparadas": len(ids), "features_por_partido": 5,
            "diferencias": difs[:10], "n_diferencias": len(difs),
            "nota": ("las features reconstruidas solo con el pasado coinciden exactamente "
                     "con las del dataset completo" if not difs else "HAY DIFERENCIAS")}


def ventanas_excluyen(X: pd.DataFrame, muestra: int = 300) -> dict:
    """La forma a N partidos de un equipo no puede incluir el partido actual."""
    x = X.copy()
    x["match_date"] = pd.to_datetime(x["match_date"])
    x = x.sort_values(["match_date", "match_id"])
    fallos = []
    equipos = x["home_team"].dropna().unique().tolist()
    random.Random(11).shuffle(equipos)
    revisados = 0
    for eq in equipos:
        if revisados >= muestra:
            break
        propios = x[(x["home_team"] == eq) | (x["away_team"] == eq)]
        if len(propios) < 8:
            continue
        filas = propios.iloc[5:12]
        for r in filas.itertuples(index=False):
            en_casa = r.home_team == eq
            previos = propios[propios["match_date"] < r.match_date]
            if len(previos) < 5:
                continue
            ult5 = previos.tail(5)
            gf = [(p.home_goals if p.home_team == eq else p.away_goals) for p in
                  ult5.itertuples(index=False)]
            esperado = sum(gf) / len(gf)
            observado = float(getattr(r, "home_gf_5" if en_casa else "away_gf_5"))
            if abs(esperado - observado) > 1e-6:
                fallos.append({"equipo": eq, "match_id": r.match_id,
                               "esperado_sin_partido_actual": round(esperado, 6),
                               "observado": round(observado, 6)})
            revisados += 1
            if revisados >= muestra:
                break
    return {"ok": not fallos, "revisados": revisados, "fallos": fallos[:10],
            "n_fallos": len(fallos)}


def audit(X: pd.DataFrame | None = None, corte: str = "2023-01-01") -> dict:
    """Ejecuta las cinco comprobaciones y devuelve un veredicto unico."""
    if X is None:
        from SOCCER.features.build import FEATURES_PARQUET      # noqa: PLC0415
        X = pd.read_parquet(FEATURES_PARQUET)
    pruebas = {
        "columnas_prohibidas": columnas_prohibidas(X),
        "correlacion_sospechosa": correlacion_sospechosa(X),
        "cutoff_ordenado": cutoff_ordenado(X),
        "ventanas_excluyen_partido_actual": ventanas_excluyen(X),
        "truncamiento": truncamiento(X, corte),
    }
    ok = all(p["ok"] for p in pruebas.values())
    return {"ok": ok, "partidos": int(len(X)), "ejecutado_en": datetime.utcnow().isoformat(),
            "pruebas": pruebas}
