"""Features as-of de tenis, en formato SIMETRICO (p1 / p2).

Clave metodologica: los datos historicos vienen como ganador/perdedor. Si se
usaran asi, el modelo aprenderia el resultado. Por eso cada partido se convierte
en una fila donde p1/p2 se asignan de forma DETERMINISTA por id de jugador
(no por resultado), y el objetivo es `p1_win`.

Corte as-of: 00:00 del dia del partido. Cada jugador aporta solo partidos de
dias ANTERIORES (el calendario de tenis no da hora fiable en el historico).

Features por jugador (ventana de 12 meses y ventana de superficie de 24 meses,
mas ultimos 20 partidos):
  * spw  = puntos ganados al saque / puntos al saque
  * rpw  = puntos ganados al resto (1 - spw del rival en ese partido)
  * ajustados por rival: spw_adj = spw - media de rpw de los rivales enfrentados
  * ace%, doble falta%, break points salvados/enfrentados, juegos ganados al saque
  * forma: % de partidos ganados, juegos por partido
  * Elo global y Elo por superficie (K con ajuste por nivel de torneo)
  * descanso (dias desde el ultimo partido), partidos en 14 dias, ranking y puntos
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from TENIS.markets.db import TenisMatch, init_db, session_scope
from shared.paths import TENIS_OUT_DIR

FEATURES_PARQUET = TENIS_OUT_DIR / "features.parquet"
FEATURE_VERSION = "tenis_features_v1"
WINDOW_DAYS = 365
SURFACE_DAYS = 730
LAST_N = 20
MIN_SVPT = 200                      # puntos al saque minimos para fiarse de spw
ELO_K = {"G": 32.0, "M": 28.0, "A": 24.0, "F": 28.0, "D": 20.0, "C": 18.0, "S": 18.0}
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def _elo_expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def load_matches() -> pd.DataFrame:
    init_db()
    with session_scope() as s:
        rows = s.execute(select(TenisMatch)).scalars().all()
        d = pd.DataFrame([{
            "match_id": m.match_id, "tour": m.tour, "season": m.season, "date": pd.Timestamp(m.match_date),
            "tourney_id": m.tourney_id, "tourney_name": m.tourney_name, "level": m.tourney_level,
            "surface": m.surface or "Unknown", "round": m.round, "best_of": m.best_of or 3,
            "minutes": m.minutes, "score": m.score, "retirement": bool(m.retirement),
            "w_id": m.winner_id, "l_id": m.loser_id, "w_name": m.winner_name, "l_name": m.loser_name,
            "w_rank": m.winner_rank, "l_rank": m.loser_rank, "w_pts": m.winner_rank_points,
            "l_pts": m.loser_rank_points, "w_age": m.winner_age, "l_age": m.loser_age,
            "w_hand": m.winner_hand, "l_hand": m.loser_hand, "w_ht": m.winner_ht, "l_ht": m.loser_ht,
            "g_w": m.games_winner, "g_l": m.games_loser, "games_total": m.games_total,
            "s_w": m.sets_winner, "s_l": m.sets_loser, **(m.serve or {})} for m in rows])
    return d.sort_values(["date", "match_id"]).reset_index(drop=True)


class Window:
    """Ventana con sumas incrementales (O(1) amortizado). Solo mira el pasado."""
    KEYS = ("svpt", "spw_n", "rpt", "rpw_n", "ace", "df", "bp_saved", "bp_faced",
            "win", "g_for", "g_against", "opp_rpw_s", "opp_rpw_n", "opp_spw_s", "opp_spw_n")

    def __init__(self, days=None, maxlen=None):
        self.days, self.maxlen = days, maxlen
        self.dq = deque()
        self.s = dict.fromkeys(self.KEYS, 0.0)
        self.n = 0

    def _apply(self, rec, sign):
        s = self.s
        for k in ("svpt", "spw_n", "rpt", "rpw_n", "ace", "df", "bp_saved", "bp_faced",
                  "win", "g_for", "g_against"):
            v = rec[k]
            s[k] += sign * (0.0 if v is None or v != v else float(v))
        if rec["opp_rpw"] is not None and rec["opp_rpw"] == rec["opp_rpw"]:
            s["opp_rpw_s"] += sign * rec["opp_rpw"]; s["opp_rpw_n"] += sign
        if rec["opp_spw"] is not None and rec["opp_spw"] == rec["opp_spw"]:
            s["opp_spw_s"] += sign * rec["opp_spw"]; s["opp_spw_n"] += sign
        self.n += sign

    def add(self, dt, rec):
        self.dq.append((dt, rec)); self._apply(rec, 1)
        if self.maxlen:
            while len(self.dq) > self.maxlen:
                _, old = self.dq.popleft(); self._apply(old, -1)

    def trim(self, cutoff):
        if not self.days:
            return
        lim = cutoff - timedelta(days=self.days)
        while self.dq and self.dq[0][0] < lim:
            _, old = self.dq.popleft(); self._apply(old, -1)

    def agg(self, cutoff):
        self.trim(cutoff)
        if self.n == 0:
            return None
        s = self.s
        return {
            "n": self.n, "svpt": s["svpt"],
            "spw": (s["spw_n"] / s["svpt"]) if s["svpt"] else None,
            "rpw": (s["rpw_n"] / s["rpt"]) if s["rpt"] else None,
            "opp_rpw": (s["opp_rpw_s"] / s["opp_rpw_n"]) if s["opp_rpw_n"] else None,
            "opp_spw": (s["opp_spw_s"] / s["opp_spw_n"]) if s["opp_spw_n"] else None,
            "ace": (s["ace"] / s["svpt"]) if s["svpt"] else None,
            "df": (s["df"] / s["svpt"]) if s["svpt"] else None,
            "bp_saved": (s["bp_saved"] / s["bp_faced"]) if s["bp_faced"] else None,
            "win": s["win"] / self.n,
            "games_won_pct": (s["g_for"] / (s["g_for"] + s["g_against"])) if (s["g_for"] + s["g_against"]) else None,
            "games_pm": (s["g_for"] + s["g_against"]) / self.n,
        }


class PlayerState:
    """Historial as-of de un jugador: se consulta ANTES de anadir el partido."""

    def __init__(self):
        self.y1 = Window(days=WINDOW_DAYS)
        self.l20 = Window(maxlen=LAST_N)
        self.srf = {s: Window(days=SURFACE_DAYS) for s in list(SURFACES) + ["Unknown"]}
        self.elo = 1500.0
        self.elo_surface = defaultdict(lambda: 1500.0)
        self.last_date = None
        self.n_matches = 0
        self.recent_dates = deque()

    def features(self, cutoff, surface):
        f = {}
        for tag, w in (("y1", self.y1), ("l20", self.l20),
                       ("srf", self.srf.get(surface) or self.srf["Unknown"])):
            a = w.agg(cutoff)
            if a:
                for k, v in a.items():
                    f[f"{tag}_{k}"] = v
        f["elo"] = self.elo
        f["elo_surface"] = self.elo_surface[surface]
        f["rest_days"] = (cutoff - self.last_date).days if self.last_date is not None else None
        while self.recent_dates and self.recent_dates[0] < cutoff - timedelta(days=14):
            self.recent_dates.popleft()
        f["matches_14d"] = len(self.recent_dates)
        f["career_matches"] = self.n_matches
        return f

    def add(self, dt, surface, rec):
        self.y1.add(dt, rec); self.l20.add(dt, rec)
        (self.srf.get(surface) or self.srf["Unknown"]).add(dt, rec)
        self.last_date = dt
        self.recent_dates.append(dt)
        self.n_matches += 1


def _num(v):
    """NaN de pandas -> 0. Sin esto un solo NaN contamina las sumas acumuladas
    para siempre (bug detectado al validar la cobertura de las features)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def _serve_rec(pref, opref, r, win, g_for, g_against):
    svpt = _num(r.get(f"{pref}_svpt")); osvpt = _num(r.get(f"{opref}_svpt"))
    won = _num(r.get(f"{pref}_1stWon")) + _num(r.get(f"{pref}_2ndWon")) if svpt else 0.0
    owon = _num(r.get(f"{opref}_1stWon")) + _num(r.get(f"{opref}_2ndWon")) if osvpt else 0.0
    return {
        "svpt": svpt, "spw_n": won,
        "rpt": osvpt, "rpw_n": (osvpt - owon) if osvpt else 0.0,
        "opp_spw": (owon / osvpt) if osvpt else None,
        "opp_rpw": ((svpt - won) / svpt) if svpt else None,
        "ace": _num(r.get(f"{pref}_ace")), "df": _num(r.get(f"{pref}_df")),
        "bp_saved": _num(r.get(f"{pref}_bpSaved")), "bp_faced": _num(r.get(f"{pref}_bpFaced")),
        "win": 1.0 if win else 0.0, "g_for": _num(g_for), "g_against": _num(g_against),
    }


def build(save: bool = True, progress=None) -> pd.DataFrame:
    d = load_matches()
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    rows = []
    for r in d.itertuples():
        dt = r.date
        surface = r.surface
        # p1/p2 deterministas por id (NO por resultado)
        p1_is_winner = str(r.w_id) < str(r.l_id)
        p1, p2 = (r.w_id, r.l_id) if p1_is_winner else (r.l_id, r.w_id)
        s1, s2 = states[p1], states[p2]
        f1, f2 = s1.features(dt, surface), s2.features(dt, surface)
        base = {
            "match_id": r.match_id, "tour": r.tour, "season": r.season, "date": dt, "surface": surface,
            "level": r.level, "round": r.round, "best_of": r.best_of, "tourney_name": r.tourney_name,
            "p1_id": p1, "p2_id": p2,
            "p1_name": (r.w_name if p1_is_winner else r.l_name), "p2_name": (r.l_name if p1_is_winner else r.w_name),
            "p1_rank": (r.w_rank if p1_is_winner else r.l_rank), "p2_rank": (r.l_rank if p1_is_winner else r.w_rank),
            "p1_pts": (r.w_pts if p1_is_winner else r.l_pts), "p2_pts": (r.l_pts if p1_is_winner else r.w_pts),
            "p1_age": (r.w_age if p1_is_winner else r.l_age), "p2_age": (r.l_age if p1_is_winner else r.w_age),
            "p1_ht": (r.w_ht if p1_is_winner else r.l_ht), "p2_ht": (r.l_ht if p1_is_winner else r.w_ht),
            "p1_hand": (r.w_hand if p1_is_winner else r.l_hand), "p2_hand": (r.l_hand if p1_is_winner else r.w_hand),
            "p1_win": 1.0 if p1_is_winner else 0.0,
            "games_total": r.games_total, "retirement": r.retirement,
            "p1_games": (r.g_w if p1_is_winner else r.g_l), "p2_games": (r.g_l if p1_is_winner else r.g_w),
            "minutes": r.minutes,
        }
        base["games_margin"] = (None if base["p1_games"] is None or base["p2_games"] is None
                                else base["p1_games"] - base["p2_games"])
        for tag, f in (("p1", f1), ("p2", f2)):
            for k, v in f.items():
                base[f"{tag}_{k}"] = v
        rows.append(base)
        # --- actualizar estado DESPUES de generar la fila -----------------------
        gw, gl = r.g_w, r.g_l
        rec_w = _serve_rec("w", "l", r._asdict(), True, gw, gl)
        rec_l = _serve_rec("l", "w", r._asdict(), False, gl, gw)
        sw, sl = states[r.w_id], states[r.l_id]
        k = ELO_K.get(r.level, 22.0)
        exp = _elo_expected(sw.elo, sl.elo)
        delta = k * (1.0 - exp)
        sw.elo += delta; sl.elo -= delta
        exps = _elo_expected(sw.elo_surface[surface], sl.elo_surface[surface])
        ds = k * (1.0 - exps)
        sw.elo_surface[surface] += ds; sl.elo_surface[surface] -= ds
        sw.add(dt, surface, rec_w); sl.add(dt, surface, rec_l)
    X = pd.DataFrame(rows)
    # diferencias simetricas
    new = {}
    for c in [c for c in X.columns if c.startswith("p1_") and c not in ("p1_id", "p1_name", "p1_hand", "p1_win")]:
        c2 = "p2_" + c[3:]
        if c2 in X.columns and pd.api.types.is_numeric_dtype(X[c]) and pd.api.types.is_numeric_dtype(X[c2]):
            new["d_" + c[3:]] = X[c] - X[c2]
            new["sum_" + c[3:]] = X[c] + X[c2]
    new["log_rank_diff"] = np.log(X.p1_rank.clip(lower=1)) - np.log(X.p2_rank.clip(lower=1))
    new["is_slam"] = (X.level == "G").astype(float)
    new["best_of_5"] = (X.best_of == 5).astype(float)
    new["surface_hard"] = (X.surface == "Hard").astype(float)
    new["surface_clay"] = (X.surface == "Clay").astype(float)
    new["surface_grass"] = (X.surface == "Grass").astype(float)
    new["elo_prob_p1"] = 1.0 / (1.0 + 10 ** ((X.p2_elo - X.p1_elo) / 400.0))
    new["elo_surface_prob_p1"] = 1.0 / (1.0 + 10 ** ((X.p2_elo_surface - X.p1_elo_surface) / 400.0))
    new["cutoff_utc"] = X.date
    new["max_source_available_at"] = X.date - pd.Timedelta(seconds=1)
    X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    # spw ajustado por rival (proxy de fuerza real del saque)
    for tag in ("p1", "p2"):
        for win in ("y1", "srf"):
            spw, opp = f"{tag}_{win}_spw", f"{tag}_{win}_opp_rpw"
            if spw in X.columns and opp in X.columns:
                X[f"{tag}_{win}_spw_adj"] = X[spw] - (X[opp] - X[opp].mean())
    for win in ("y1", "srf"):
        if f"p1_{win}_spw_adj" in X.columns:
            X[f"d_{win}_spw_adj"] = X[f"p1_{win}_spw_adj"] - X[f"p2_{win}_spw_adj"]
    if save:
        TENIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(FEATURES_PARQUET, index=False)
    if progress:
        progress(f"features tenis: {X.shape}")
    return X


TARGET_COLS = {"p1_win", "games_total", "games_margin", "p1_games", "p2_games", "minutes", "retirement"}
ID_COLS = {"match_id", "tour", "season", "date", "surface", "level", "round", "tourney_name",
           "p1_id", "p2_id", "p1_name", "p2_name", "p1_hand", "p2_hand", "cutoff_utc",
           "max_source_available_at", "best_of"}


def families(X: pd.DataFrame) -> dict[str, list[str]]:
    cols = set(X.columns) - TARGET_COLS - ID_COLS
    num = {c for c in cols if pd.api.types.is_numeric_dtype(X[c])}
    def pick(*subs, pre=("d_", "sum_")):
        return sorted(c for c in num if c.startswith(pre) and any(s in c for s in subs))
    fam = {
        "elo": [c for c in ("p1_elo", "p2_elo", "d_elo", "elo_prob_p1") if c in num],
        "elo_surface": [c for c in ("p1_elo_surface", "p2_elo_surface", "d_elo_surface",
                                    "elo_surface_prob_p1") if c in num],
        "ranking": [c for c in ("d_rank", "d_pts", "log_rank_diff", "p1_rank", "p2_rank") if c in num],
        "serve": pick("spw", "ace", "df"),
        "return": pick("rpw", "bp_saved"),
        "form": pick("_win", "games_won_pct", "games_pm", "_n"),
        "surface_form": [c for c in num if c.startswith(("d_srf", "sum_srf"))],
        "physical": [c for c in ("d_age", "d_ht", "sum_age", "sum_ht", "d_career_matches") if c in num],
        "schedule": [c for c in num if any(k in c for k in ("rest_days", "matches_14d"))],
        "context": [c for c in ("is_slam", "best_of_5", "surface_hard", "surface_clay", "surface_grass") if c in num],
    }
    for k, v in fam.items():
        assert not (set(v) & TARGET_COLS), k
    return fam


def numeric(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X[cols].apply(pd.to_numeric, errors="coerce").astype(float)
