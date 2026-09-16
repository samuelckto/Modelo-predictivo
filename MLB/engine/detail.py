"""Ficha detallada de un partido MLB para el dashboard.

Todo lo que se muestra sale de datos ya ingeridos y de las features as-of, que
respetan la regla `available_at < inicio del partido`. Lo que no existe se marca
como no disponible en vez de rellenarse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from MLB.database.models import (Game, Injury, Lineup, Odds, Player, Prediction,
                                 ProbablePitcher, Team)
from MLB.database.session import session_scope
from MLB.ingest.statsapi import IL_LABEL
from shared.paths import MLB_PROCESSED_DIR

_FEAT: pd.DataFrame | None = None
_ELO: pd.DataFrame | None = None


def invalidate() -> None:
    """Se llama tras reconstruir features para que la ficha no sirva datos viejos."""
    global _FEAT, _ELO
    _FEAT, _ELO = None, None


def _feat() -> pd.DataFrame:
    global _FEAT
    if _FEAT is None:
        f = MLB_PROCESSED_DIR / "features.parquet"
        _FEAT = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    return _FEAT


def _elo() -> pd.DataFrame:
    """Calificaciones Elo previas al partido. Viven en el artefacto del modelo,
    no en las features, porque se calculan en orden cronologico."""
    global _ELO
    if _ELO is None:
        try:
            import joblib

            from MLB.engine.elo import run as run_elo
            from shared.paths import MLB_MODELS_DIR
            arts = sorted(MLB_MODELS_DIR.glob("elo_v*.joblib"))
            X = _feat()
            # recalculado con los resultados actuales, no la tabla congelada
            _ELO = run_elo(X, joblib.load(arts[-1])["cfg"]) if (arts and not X.empty) \
                else pd.DataFrame()
        except Exception:                                    # noqa: BLE001
            _ELO = pd.DataFrame()
    return _ELO


def _v(row, col):
    """Valor de una feature, o None si no existe / es NaN."""
    if row is None or col not in row.index:
        return None
    x = row[col]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x) if isinstance(x, (int, float, np.number)) else x


SP_FIELDS = [
    ("sp_era_s", "ERA", "{:.2f}", "temporada"),
    ("sp_fip_s", "FIP aprox.", "{:.2f}", "temporada"),
    ("sp_whip_s", "WHIP", "{:.2f}", "temporada"),
    ("sp_k_s", "K%", "{:.1%}", "temporada"),
    ("sp_bb_s", "BB%", "{:.1%}", "temporada"),
    ("sp_kbb_s", "K-BB%", "{:.1%}", "temporada"),
    ("sp_hr9_s", "HR/9", "{:.2f}", "temporada"),
    ("sp_ip_start_s", "entradas por apertura", "{:.1f}", "temporada"),
    ("sp_pitches_start_s", "lanzamientos por apertura", "{:.0f}", "temporada"),
    ("sp_xwoba_s", "xwOBA permitido", "{:.3f}", "Statcast"),
    ("sp_hard_s", "Hard hit%", "{:.1%}", "Statcast"),
    ("sp_barrel_s", "Barrel%", "{:.1%}", "Statcast"),
    ("sp_csw_s", "CSW%", "{:.1%}", "Statcast"),
    ("sp_swstr_s", "SwStr%", "{:.1%}", "Statcast"),
    ("sp_velo_s", "velocidad media", "{:.1f} mph", "Statcast"),
    ("sp_spin_s", "spin medio", "{:.0f} rpm", "Statcast"),
    ("sp_era_d45", "ERA últimos 45 días", "{:.2f}", "forma reciente"),
    ("sp_starts_s", "aperturas en la temporada", "{:.0f}", "muestra"),
    ("sp_rest_days", "días de descanso", "{:.0f}", "contexto"),
]

TEAM_FIELDS = [
    ("off_rpg_d30", "carreras por partido (30 d)", "{:.2f}"),
    ("off_obp_d30", "OBP (30 d)", "{:.3f}"),
    ("off_slg_d30", "SLG (30 d)", "{:.3f}"),
    ("off_iso_d30", "ISO (30 d)", "{:.3f}"),
    ("off_k_d30", "K% (30 d)", "{:.1%}"),
    ("off_bb_d30", "BB% (30 d)", "{:.1%}"),
    ("off_rpg_d7", "carreras por partido (7 d)", "{:.2f}"),
    ("def_rpg_d30", "carreras recibidas (30 d)", "{:.2f}"),
    ("stf_era_d30", "ERA del staff (30 d)", "{:.2f}"),
    ("stf_k_d30", "K% del staff (30 d)", "{:.1%}"),
    ("off_games_s", "partidos jugados", "{:.0f}"),
]

BULLPEN_FIELDS = [
    ("bp_pitches_d1", "lanzamientos ayer", "{:.0f}"),
    ("bp_pitches_d2", "lanzamientos 2 días", "{:.0f}"),
    ("bp_pitches_d3", "lanzamientos 3 días", "{:.0f}"),
    ("bp_pitches_d5", "lanzamientos 5 días", "{:.0f}"),
    ("bp_relievers_d3", "relevistas usados (3 d)", "{:.0f}"),
    ("bp_b2b_d3", "relevistas en días seguidos (3 d)", "{:.0f}"),
]


def _fmt(v, f):
    return None if v is None else f.format(v)


def _side_block(row, prefix, fields):
    out = []
    for col, label, f, *rest in fields:
        v = _v(row, f"{prefix}_{col}")
        out.append({"label": label, "value": _fmt(v, f), "raw": v,
                    "group": rest[0] if rest else None})
    return out


def _elo_block(game_id: int) -> dict:
    e = _elo()
    if e.empty or "game_id" not in e.columns:
        return {"elo_home": None, "elo_away": None, "elo_prob_home": None}
    m = e[e.game_id == game_id]
    if m.empty:
        return {"elo_home": None, "elo_away": None, "elo_prob_home": None}
    r = m.iloc[0]
    f = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    return {"elo_home": f(r.get("elo_home_pre")), "elo_away": f(r.get("elo_away_pre")),
            "elo_prob_home": f(r.get("p_home_elo"))}


def game_detail(game_id: int) -> dict:
    X = _feat()
    row = None
    if not X.empty:
        m = X[X.game_id == game_id]
        if len(m):
            row = m.iloc[0]

    with session_scope() as s:
        g = s.get(Game, int(game_id))
        if g is None:
            return {"error": f"partido {game_id} no encontrado"}
        teams = {t.id: t for t in s.execute(select(Team)).scalars()}
        names = {p.id: p for p in s.execute(select(Player)).scalars()}

        # --- abridores -------------------------------------------------
        starters = {}
        for side, tid in (("home", g.home_team_id), ("away", g.away_team_id)):
            pid = _v(row, f"{'h' if side == 'home' else 'a'}_sp_player_id")
            src = None
            if row is not None:
                src = row.get(f"{'h' if side == 'home' else 'a'}_sp_source")
            pp = s.execute(
                select(ProbablePitcher).where(ProbablePitcher.game_id == game_id,
                                              ProbablePitcher.team_id == tid)
                .order_by(ProbablePitcher.available_at.desc()).limit(1)).scalars().first()
            pid = int(pid) if pid else (pp.player_id if pp else None)
            pl = names.get(pid)
            starters[side] = {
                "player_id": pid,
                "name": (pl.full_name if pl else (pp.player_name if pp else None)),
                "throws": (pl.throws if pl else None),
                "estado": ("anunciado por la MLB" if src == "probable_announced" else
                           "supuesto (abridor real del partido)" if src == "actual_starter_assumed"
                           else ("anunciado" if pp else "sin anunciar")),
                "anunciado_at": (pp.available_at if pp else None),
                "stats": _side_block(row, "h" if side == "home" else "a", SP_FIELDS),
            }

        # --- alineaciones ----------------------------------------------
        lineups = {}
        for side, tid in (("home", g.home_team_id), ("away", g.away_team_id)):
            lu = s.execute(
                select(Lineup).where(Lineup.game_id == game_id, Lineup.team_id == tid)
                .order_by(Lineup.available_at.desc()).limit(1)).scalars().first()
            lineups[side] = {
                "estado": (lu.state if lu else "no confirmada"),
                "available_at": (lu.available_at if lu else None),
                "orden": ([{"pos": i + 1, "player_id": pid,
                            "name": (names[pid].full_name if pid in names else str(pid)),
                            "bats": (names[pid].bats if pid in names else None)}
                           for i, pid in enumerate(lu.batting_order or [])]
                          if lu else []),
            }

        # --- lesiones --------------------------------------------------
        inj = {}
        for side, tid in (("home", g.home_team_id), ("away", g.away_team_id)):
            rows = s.execute(select(Injury).where(Injury.team_id == tid)
                             .order_by(Injury.available_at.desc())).scalars().all()
            vistos, lista = set(), []
            for r in rows:
                if r.player_id in vistos:
                    continue
                vistos.add(r.player_id)
                pl = names.get(r.player_id)
                lista.append({"player_id": r.player_id,
                              "name": r.player_name or (pl.full_name if pl else None),
                              "posicion": (pl.primary_position if pl else None),
                              "estado": IL_LABEL.get(r.status, r.status),
                              "codigo": r.status,
                              "actualizado": r.available_at})
            inj[side] = lista

        # --- cuotas por casa -------------------------------------------
        odds_rows = s.execute(
            select(Odds).where(Odds.game_id == game_id)
            .order_by(Odds.available_at.desc())).scalars().all()
        books: dict = {}
        for o in odds_rows:
            k = (o.bookmaker, o.market, o.selection)
            if k in books:
                continue
            books[k] = o
        odds = {}
        for (bk, mkt, sel), o in books.items():
            odds.setdefault(mkt, {}).setdefault(bk, []).append(
                {"seleccion": sel, "linea": o.line, "precio": o.price_american,
                 "prob_implicita": o.implied_prob, "cierre": bool(o.is_closing),
                 "capturado": o.available_at})

        # --- predicciones y resultado ----------------------------------
        preds = s.execute(
            select(Prediction).where(Prediction.game_id == game_id,
                                     Prediction.status != "superseded")
            .order_by(Prediction.market)).scalars().all()
        predicciones = [{
            "market": p.market, "selection": p.selection, "line": p.line,
            "probabilidad": p.ensemble_probability, "modelo": p.model_probability,
            "mercado": p.market_probability, "elo": p.elo_probability,
            "riesgo": p.upset_risk, "banda": p.upset_label,
            "version": p.version, "generada": p.prediction_timestamp,
            "modelo_version": p.model_version,
            "resultado": p.result, "acierto": p.correct,
            "explicacion": p.upset_explanation,
            "completitud": p.data_completeness,
        } for p in preds]

        info = {
            "game_id": g.id, "fecha": g.game_date, "inicio_utc": g.start_utc,
            "estado": g.status, "sede": g.venue_name,
            "home": {"abbr": g.home_abbr,
                     "nombre": (teams[g.home_team_id].name if g.home_team_id in teams else None),
                     "carreras": g.home_score, "carreras_f5": g.home_score_f5},
            "away": {"abbr": g.away_abbr,
                     "nombre": (teams[g.away_team_id].name if g.away_team_id in teams else None),
                     "carreras": g.away_score, "carreras_f5": g.away_score_f5},
            "doubleheader": g.doubleheader not in (None, "N"),
        }

    return {
        "partido": info,
        "abridores": starters,
        "equipos": {"home": _side_block(row, "h", TEAM_FIELDS),
                    "away": _side_block(row, "a", TEAM_FIELDS)},
        "bullpen": {"home": _side_block(row, "h", BULLPEN_FIELDS),
                    "away": _side_block(row, "a", BULLPEN_FIELDS)},
        "alineaciones": lineups,
        "lesiones": inj,
        "contexto": {
            "park_runs_factor": _v(row, "park_runs_factor"),
            **_elo_block(game_id),
            "descanso_home": _v(row, "h_rest_days"),
            "descanso_away": _v(row, "a_rest_days"),
            "clima": {"disponible": False,
                      "motivo": "el sistema todavia no ingiere datos meteorologicos"},
            "umpire": {"disponible": False,
                       "motivo": "la fuente no publica el umpire de home antes del partido"},
        },
        "cuotas": odds or {"disponible": False,
                           "motivo": "sin cuotas guardadas para este partido"},
        "predicciones": predicciones,
    }
