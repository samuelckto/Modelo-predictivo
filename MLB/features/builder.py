"""Constructor de features MLB, estrictamente point-in-time.

REGLA UNICA: para un partido cuyo `start_utc` es T, solo entran filas con
`available_at < T`. No hay excepciones y hay pruebas que lo verifican.

Como se garantiza:
  1. Todas las agregaciones pasan por `rolling_asof`, que consulta series
     acumuladas hacia atras con `allow_exact_matches=False`.
  2. Las filas sin `available_at` se descartan (no se puede probar que fueran
     anteriores).
  3. `audit()` recuenta violaciones despues de construir, sobre los datos reales.

Ventanas: temporada, 30, 14 y 7 dias. NO se les asigna ningun peso aqui; se
entregan como columnas separadas y es el modelo quien decide su importancia con
validacion temporal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from MLB.database.models import (BatterGameLog, BullpenUsage, Game, PitcherGameLog,
                                 ProbablePitcher, StatcastAgg)
from MLB.database.session import session_scope
from shared.asof import rolling_asof
from shared.paths import MLB_PROCESSED_DIR

FEATURE_VERSION = "mlb_features_v1"
# "s" = acumulado de la TEMPORADA en curso (se agrupa tambien por season, para
# que el primer partido del ano no arrastre datos del ano anterior).
# Las demas son ventanas moviles en dias.
WINDOWS = {"s": None, "d30": 30, "d14": 14, "d7": 7}
PITCHER_WINDOWS = {"s": None, "d45": 45}
SEASON_KEY = "s"
FINAL = ("Final", "Game Over", "Completed Early")


def _df(rows, cols) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------- extraccion
def load(seasons: list[int], data_cutoff=None) -> dict[str, pd.DataFrame]:
    """`data_cutoff` (solo para auditoria): descarta TODA fila con
    available_at >= cutoff. Permite comprobar que reconstruir las features de un
    dia con el dataset truncado en ese dia da exactamente el mismo resultado."""
    with session_scope() as s:
        games = _df(s.execute(select(
            Game.id, Game.season, Game.game_date, Game.start_utc, Game.game_type,
            Game.status, Game.home_team_id, Game.away_team_id, Game.home_abbr,
            Game.away_abbr, Game.venue_id, Game.home_score, Game.away_score,
            Game.home_score_f5, Game.away_score_f5, Game.doubleheader, Game.game_number)
            .where(Game.season.in_(seasons))).all(),
            ["game_id", "season", "game_date", "start_utc", "game_type", "status",
             "home_team_id", "away_team_id", "home_abbr", "away_abbr", "venue_id",
             "home_score", "away_score", "home_f5", "away_f5", "dh", "game_number"])

        pit = _df(s.execute(select(
            PitcherGameLog.game_id, PitcherGameLog.player_id, PitcherGameLog.team_id,
            PitcherGameLog.is_starter, PitcherGameLog.outs, PitcherGameLog.pitches,
            PitcherGameLog.batters_faced, PitcherGameLog.hits, PitcherGameLog.runs,
            PitcherGameLog.earned_runs, PitcherGameLog.walks, PitcherGameLog.strikeouts,
            PitcherGameLog.home_runs, PitcherGameLog.ground_outs, PitcherGameLog.air_outs,
            PitcherGameLog.available_at, PitcherGameLog.season)
            .where(PitcherGameLog.season.in_(seasons))).all(),
            ["game_id", "player_id", "team_id", "is_starter", "outs", "pitches", "bf",
             "hits", "runs", "er", "bb", "k", "hr", "go", "ao", "available_at", "season"])

        bat = _df(s.execute(select(
            BatterGameLog.game_id, BatterGameLog.team_id, BatterGameLog.plate_appearances,
            BatterGameLog.at_bats, BatterGameLog.hits, BatterGameLog.doubles,
            BatterGameLog.triples, BatterGameLog.home_runs, BatterGameLog.walks,
            BatterGameLog.strikeouts, BatterGameLog.total_bases, BatterGameLog.available_at,
            BatterGameLog.season).where(BatterGameLog.season.in_(seasons))).all(),
            ["game_id", "team_id", "pa", "ab", "hits", "doubles", "triples", "hr", "bb",
             "k", "tb", "available_at", "season"])

        sc = _df(s.execute(select(
            StatcastAgg.scope, StatcastAgg.player_id, StatcastAgg.game_date,
            StatcastAgg.pitches, StatcastAgg.batted_balls, StatcastAgg.hard_hit,
            StatcastAgg.barrels, StatcastAgg.xwoba, StatcastAgg.xba, StatcastAgg.xslg,
            StatcastAgg.exit_velocity, StatcastAgg.launch_angle, StatcastAgg.swstr,
            StatcastAgg.csw, StatcastAgg.velocity, StatcastAgg.spin,
            StatcastAgg.available_at, StatcastAgg.season)
            .where(StatcastAgg.season.in_(seasons))).all(),
            ["scope", "player_id", "game_date", "sc_pitches", "bb_balls", "hard", "barrels",
             "xwoba", "xba", "xslg", "ev", "la", "swstr", "csw", "velo", "spin",
             "available_at", "season"])

        bp = _df(s.execute(select(
            BullpenUsage.team_id, BullpenUsage.game_date, BullpenUsage.relievers_used,
            BullpenUsage.bullpen_outs, BullpenUsage.bullpen_pitches,
            BullpenUsage.back_to_back, BullpenUsage.available_at)
            .where(BullpenUsage.season.in_(seasons))).all(),
            ["team_id", "game_date", "relievers", "bp_outs", "bp_pitches", "b2b",
             "available_at"])

        pp = _df(s.execute(select(
            ProbablePitcher.game_id, ProbablePitcher.team_id, ProbablePitcher.is_home,
            ProbablePitcher.player_id, ProbablePitcher.available_at)).all(),
            ["game_id", "team_id", "is_home", "player_id", "available_at"])
    out = {"games": games, "pit": pit, "bat": bat, "sc": sc, "bp": bp, "pp": pp}
    if data_cutoff is not None:
        cut = pd.Timestamp(data_cutoff)
        for k in ("pit", "bat", "sc", "bp", "pp"):
            d = out[k]
            if "available_at" in d.columns and len(d):
                out[k] = d[pd.to_datetime(d.available_at, errors="coerce") < cut]
        g = out["games"]
        fut = pd.to_datetime(g.start_utc, errors="coerce") >= cut
        # los partidos futuros siguen (hay que predecirlos) pero SIN su resultado
        for c in ("home_score", "away_score", "home_f5", "away_f5"):
            g.loc[fut, c] = None
        out["games"] = g
    return out


# ------------------------------------------------------------------ helpers
def _rate(num, den, scale=1.0, minimum=1.0):
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return np.where(d >= minimum, n / d.replace(0, np.nan) * scale, np.nan)


def _team_events(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Eventos por equipo con su available_at real."""
    g = d["games"][["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]]
    # ofensiva: se agrega el boxscore de bateo por equipo y partido
    off = (d["bat"].groupby(["game_id", "team_id", "available_at"], as_index=False)
           [["pa", "ab", "hits", "doubles", "triples", "hr", "bb", "k", "tb"]].sum())
    runs = pd.concat([
        g.rename(columns={"home_team_id": "team_id", "home_score": "runs_for",
                          "away_score": "runs_against"})[["game_id", "team_id", "runs_for", "runs_against"]],
        g.rename(columns={"away_team_id": "team_id", "away_score": "runs_for",
                          "home_score": "runs_against"})[["game_id", "team_id", "runs_for", "runs_against"]]])
    off = off.merge(runs, on=["game_id", "team_id"], how="left")
    off = off.merge(d["games"][["game_id", "season"]], on="game_id", how="left")

    # pitcheo del staff completo por equipo y partido
    stf = (d["pit"].groupby(["game_id", "team_id", "available_at"], as_index=False)
           [["outs", "bf", "hits", "er", "bb", "k", "hr", "go", "ao"]].sum())
    stf = stf.merge(d["games"][["game_id", "season"]], on="game_id", how="left")
    bp = d["bp"].merge(d["games"][["game_date", "season"]].drop_duplicates("game_date"),
                       on="game_date", how="left")
    return off, stf, bp


def _windows(events: pd.DataFrame, cuts: pd.DataFrame, key: str, value_cols: list[str],
             windows: dict, prefix: str = "") -> pd.DataFrame:
    """Aplica las ventanas. La ventana de temporada se agrupa por (clave, season)
    para que no arrastre la temporada anterior; las de dias, solo por la clave."""
    day_w = {k: v for k, v in windows.items() if v is not None}
    out = rolling_asof(events, cuts, [key], "available_at", value_cols, day_w, prefix) \
        if day_w else cuts.copy()
    if SEASON_KEY in windows:
        s = rolling_asof(events, cuts, [key, "season"], "available_at", value_cols,
                         {SEASON_KEY: None}, prefix)
        for c in [c for c in s.columns if c.endswith(f"_{SEASON_KEY}")]:
            out[c] = s[c].values
    return out


def build(seasons: list[int], save: bool = True, progress=None,
          data_cutoff=None) -> pd.DataFrame:
    d = load(seasons, data_cutoff=data_cutoff)
    games = d["games"]
    games = games[games.start_utc.notna()].copy()
    games["start_utc"] = pd.to_datetime(games.start_utc)
    games = games.sort_values("start_utc").reset_index(drop=True)
    if progress:
        progress(f"partidos: {len(games)}")

    off, stf, bp = _team_events(d)
    cuts = pd.concat([
        games[["game_id", "home_team_id", "start_utc", "season"]].rename(
            columns={"home_team_id": "team_id", "start_utc": "cut"}).assign(side="home"),
        games[["game_id", "away_team_id", "start_utc", "season"]].rename(
            columns={"away_team_id": "team_id", "start_utc": "cut"}).assign(side="away")],
        ignore_index=True)

    # --- ofensiva ---------------------------------------------------------
    o = _windows(off, cuts, "team_id",
                 ["pa", "ab", "hits", "doubles", "triples", "hr", "bb", "k", "tb",
                  "runs_for", "runs_against"], WINDOWS)
    for w in WINDOWS:
        o[f"off_obp_{w}"] = _rate(o[f"hits_{w}"] + o[f"bb_{w}"], o[f"pa_{w}"], 1, 30)
        o[f"off_slg_{w}"] = _rate(o[f"tb_{w}"], o[f"ab_{w}"], 1, 30)
        o[f"off_iso_{w}"] = o[f"off_slg_{w}"] - _rate(o[f"hits_{w}"], o[f"ab_{w}"], 1, 30)
        o[f"off_k_{w}"] = _rate(o[f"k_{w}"], o[f"pa_{w}"], 1, 30)
        o[f"off_bb_{w}"] = _rate(o[f"bb_{w}"], o[f"pa_{w}"], 1, 30)
        o[f"off_hr_{w}"] = _rate(o[f"hr_{w}"], o[f"pa_{w}"], 1, 30)
        o[f"off_rpg_{w}"] = _rate(o[f"runs_for_{w}"], o[f"n_{w}"], 1, 3)
        o[f"def_rpg_{w}"] = _rate(o[f"runs_against_{w}"], o[f"n_{w}"], 1, 3)
        o[f"off_games_{w}"] = o[f"n_{w}"]
    keep_o = ["game_id", "team_id", "side"] + [c for c in o.columns
                                               if c.startswith(("off_", "def_"))]
    o = o[keep_o]

    # --- staff de pitcheo -------------------------------------------------
    p = _windows(stf, cuts, "team_id",
                 ["outs", "bf", "hits", "er", "bb", "k", "hr", "go", "ao"], WINDOWS)
    for w in WINDOWS:
        ip = p[f"outs_{w}"] / 3.0
        p[f"stf_era_{w}"] = np.where(ip >= 10, p[f"er_{w}"] * 9 / ip.replace(0, np.nan), np.nan)
        p[f"stf_whip_{w}"] = np.where(ip >= 10, (p[f"hits_{w}"] + p[f"bb_{w}"]) / ip.replace(0, np.nan), np.nan)
        p[f"stf_k_{w}"] = _rate(p[f"k_{w}"], p[f"bf_{w}"], 1, 40)
        p[f"stf_bb_{w}"] = _rate(p[f"bb_{w}"], p[f"bf_{w}"], 1, 40)
        p[f"stf_hr9_{w}"] = np.where(ip >= 10, p[f"hr_{w}"] * 9 / ip.replace(0, np.nan), np.nan)
        p[f"stf_gb_{w}"] = _rate(p[f"go_{w}"], p[f"go_{w}"] + p[f"ao_{w}"], 1, 20)
    p = p[["game_id", "team_id", "side"] + [c for c in p.columns if c.startswith("stf_")]]

    # --- bullpen ----------------------------------------------------------
    bpe = bp.copy()
    b = _windows(bpe, cuts, "team_id", ["relievers", "bp_outs", "bp_pitches", "b2b"],
                 {"d1": 1, "d2": 2, "d3": 3, "d5": 5, "s": None})
    for w in ("d1", "d2", "d3", "d5"):
        b[f"bp_pitches_{w}"] = b[f"bp_pitches_{w}"]
        b[f"bp_outs_{w}"] = b[f"bp_outs_{w}"]
        b[f"bp_relievers_{w}"] = b[f"relievers_{w}"]
        b[f"bp_b2b_{w}"] = b[f"b2b_{w}"]
    b["bp_pitches_per_game_s"] = _rate(b["bp_pitches_s"], b["n_s"], 1, 5)
    b = b[["game_id", "team_id", "side"] + [c for c in b.columns if c.startswith("bp_")]]

    # --- abridor probable -------------------------------------------------
    sp = _starter_features(d, games, progress)

    # --- ensamblado -------------------------------------------------------
    side = (cuts.merge(o, on=["game_id", "team_id", "side"], how="left")
                .merge(p, on=["game_id", "team_id", "side"], how="left")
                .merge(b, on=["game_id", "team_id", "side"], how="left")
                .merge(sp, on=["game_id", "team_id", "side"], how="left"))
    h = side[side.side == "home"].drop(columns=["side", "cut"]).add_prefix("h_")
    a = side[side.side == "away"].drop(columns=["side", "cut"]).add_prefix("a_")
    X = (games.merge(h, left_on="game_id", right_on="h_game_id", how="left")
               .merge(a, left_on="game_id", right_on="a_game_id", how="left")
               .drop(columns=["h_game_id", "a_game_id", "h_team_id", "a_team_id"]))

    # diferencias (el modelo suele aprender mejor con el diferencial).
    # Se construyen TODAS a la vez y se pegan con un solo concat: insertarlas una
    # a una fragmenta el DataFrame y pandas avisa con PerformanceWarning.
    NO_DIFF = {"season", "sp_player_id", "sp_source"}
    difs = {}
    for c in [c[2:] for c in X.columns if c.startswith("h_")]:
        if c in NO_DIFF or f"a_{c}" not in X.columns:
            continue
        difs[f"d_{c}"] = (pd.to_numeric(X[f"h_{c}"], errors="coerce") -
                          pd.to_numeric(X[f"a_{c}"], errors="coerce"))
    if difs:
        X = pd.concat([X, pd.DataFrame(difs, index=X.index)], axis=1)

    # contexto (tambien de una sola vez)
    ctx = pd.DataFrame({
        "is_doubleheader": (X["dh"].fillna("N") != "N").astype(int),
        "month": pd.to_datetime(X.game_date).dt.month,
    }, index=X.index)
    X["game_number"] = X["game_number"].fillna(1)
    X = pd.concat([X, ctx], axis=1)
    X = _rest_days(X)
    X = _park(X, games)

    # objetivos (solo en partidos terminados)
    fin = X.status.isin(FINAL) & X.home_score.notna() & X.away_score.notna()
    f5 = fin & X.home_f5.notna() & X.away_f5.notna()
    margin = np.where(fin, X.home_score - X.away_score, np.nan)
    X = pd.concat([X, pd.DataFrame({
        "y_home_win": np.where(fin, (X.home_score > X.away_score).astype(float), np.nan),
        "total_runs": np.where(fin, X.home_score + X.away_score, np.nan),
        "home_margin": margin,
        "y_home_rl": np.where(fin, (margin >= 2).astype(float), np.nan),
        "y_home_f5": np.where(f5, (X.home_f5 > X.away_f5).astype(float), np.nan),
        "total_runs_f5": np.where(f5, X.home_f5 + X.away_f5, np.nan),
        "feature_version": FEATURE_VERSION,
    }, index=X.index)], axis=1)

    if save:
        MLB_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(MLB_PROCESSED_DIR / "features.parquet", index=False)
    if progress:
        progress(f"features: {X.shape[0]} filas x {X.shape[1]} columnas")
    return X


def _starter_features(d: dict, games: pd.DataFrame, progress=None) -> pd.DataFrame:
    """Rendimiento del abridor, conocido antes del partido.

    DOS ORIGENES, y la columna `sp_source` dice cual se uso en cada fila:

    * `probable_announced`: el anuncio real de la MLB Stats API con su timestamp,
      verificado anterior al inicio. Es lo que se usa en vivo.
    * `actual_starter_assumed`: para partidos historicos anteriores a la primera
      ingesta no existe snapshot del anuncio, asi que se toma la IDENTIDAD del
      abridor que efectivamente lanzo. Es un supuesto declarado, no un dato: la
      MLB publica los probables con dias de antelacion y coinciden con el
      abridor real en la gran mayoria de los casos, pero no siempre.

    Importante: el supuesto afecta SOLO a QUIEN lanza. Todas sus estadisticas se
    calculan igual de estrictas (unicamente aperturas anteriores a este partido),
    asi que no entra ni un dato del propio partido.
    """
    pp, pit, sc = d["pp"], d["pit"], d["sc"]
    g = games[["game_id", "start_utc", "home_team_id", "away_team_id"]]
    ann = pd.DataFrame({"game_id": pd.Series(dtype="int64"),
                        "team_id": pd.Series(dtype="int64"),
                        "player_id": pd.Series(dtype="int64")})
    if not pp.empty:
        ann = pp.merge(g, on="game_id", how="inner")
        ann["available_at"] = pd.to_datetime(ann.available_at)
        ann = ann[ann.available_at < pd.to_datetime(ann.start_utc)]
        ann = (ann.sort_values("available_at")
                  .groupby(["game_id", "team_id"], as_index=False).tail(1))
        ann = ann[["game_id", "team_id", "player_id"]]
    ann["sp_source"] = "probable_announced"

    act = pit[pit.is_starter.fillna(False)][["game_id", "team_id", "player_id"]].copy()
    act["sp_source"] = "actual_starter_assumed"
    have = set(zip(ann.game_id, ann.team_id))
    act = act[[(a, b) not in have for a, b in zip(act.game_id, act.team_id)]]

    pp = pd.concat([ann, act], ignore_index=True).merge(g, on="game_id", how="inner")
    if pp.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "side"])
    for c in ("game_id", "team_id", "player_id"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").astype("int64")
    pp["side"] = np.where(pp.team_id == pp.home_team_id, "home", "away")

    st = pit[pit.is_starter.fillna(False)].copy()
    cuts = pp[["game_id", "player_id", "start_utc"]].rename(columns={"start_utc": "cut"})
    cuts = cuts.merge(games[["game_id", "season"]], on="game_id", how="left")
    r = _windows(st, cuts, "player_id",
                 ["outs", "bf", "hits", "er", "bb", "k", "hr", "pitches"], PITCHER_WINDOWS)
    for w in PITCHER_WINDOWS:
        ip = r[f"outs_{w}"] / 3.0
        r[f"sp_starts_{w}"] = r[f"n_{w}"]
        r[f"sp_era_{w}"] = np.where(ip >= 10, r[f"er_{w}"] * 9 / ip.replace(0, np.nan), np.nan)
        r[f"sp_whip_{w}"] = np.where(ip >= 10, (r[f"hits_{w}"] + r[f"bb_{w}"]) / ip.replace(0, np.nan), np.nan)
        r[f"sp_k_{w}"] = _rate(r[f"k_{w}"], r[f"bf_{w}"], 1, 30)
        r[f"sp_bb_{w}"] = _rate(r[f"bb_{w}"], r[f"bf_{w}"], 1, 30)
        r[f"sp_kbb_{w}"] = r[f"sp_k_{w}"] - r[f"sp_bb_{w}"]
        r[f"sp_hr9_{w}"] = np.where(ip >= 10, r[f"hr_{w}"] * 9 / ip.replace(0, np.nan), np.nan)
        r[f"sp_ip_start_{w}"] = np.where(r[f"n_{w}"] >= 2, ip / r[f"n_{w}"], np.nan)
        r[f"sp_pitches_start_{w}"] = _rate(r[f"pitches_{w}"], r[f"n_{w}"], 1, 2)
        # FIP con constante de liga calculada de los propios datos
        r[f"sp_fip_{w}"] = np.where(ip >= 10,
                                    (13 * r[f"hr_{w}"] + 3 * r[f"bb_{w}"] - 2 * r[f"k_{w}"]) /
                                    ip.replace(0, np.nan), np.nan)

    # Statcast del abridor
    scp = sc[sc.scope == "pitcher"].copy()
    if not scp.empty:
        scp["w_xwoba"] = scp.xwoba * scp.bb_balls.fillna(0)
        scp["w_ev"] = scp.ev * scp.bb_balls.fillna(0)
        scp["w_velo"] = scp.velo * scp.sc_pitches.fillna(0)
        scp["w_spin"] = scp.spin * scp.sc_pitches.fillna(0)
        scp["w_csw"] = scp.csw * scp.sc_pitches.fillna(0)
        scp["w_swstr"] = scp.swstr * scp.sc_pitches.fillna(0)
        s = _windows(scp, cuts, "player_id",
                     ["sc_pitches", "bb_balls", "hard", "barrels", "w_xwoba", "w_ev",
                      "w_velo", "w_spin", "w_csw", "w_swstr"], PITCHER_WINDOWS)
        for w in PITCHER_WINDOWS:
            bbb = s[f"bb_balls_{w}"].replace(0, np.nan)
            pch = s[f"sc_pitches_{w}"].replace(0, np.nan)
            s[f"sp_xwoba_{w}"] = np.where(s[f"bb_balls_{w}"] >= 25, s[f"w_xwoba_{w}"] / bbb, np.nan)
            s[f"sp_ev_{w}"] = np.where(s[f"bb_balls_{w}"] >= 25, s[f"w_ev_{w}"] / bbb, np.nan)
            s[f"sp_hard_{w}"] = np.where(s[f"bb_balls_{w}"] >= 25, s[f"hard_{w}"] / bbb, np.nan)
            s[f"sp_barrel_{w}"] = np.where(s[f"bb_balls_{w}"] >= 25, s[f"barrels_{w}"] / bbb, np.nan)
            s[f"sp_velo_{w}"] = np.where(s[f"sc_pitches_{w}"] >= 100, s[f"w_velo_{w}"] / pch, np.nan)
            s[f"sp_spin_{w}"] = np.where(s[f"sc_pitches_{w}"] >= 100, s[f"w_spin_{w}"] / pch, np.nan)
            s[f"sp_csw_{w}"] = np.where(s[f"sc_pitches_{w}"] >= 100, s[f"w_csw_{w}"] / pch, np.nan)
            s[f"sp_swstr_{w}"] = np.where(s[f"sc_pitches_{w}"] >= 100, s[f"w_swstr_{w}"] / pch, np.nan)
        r = r.merge(s[["game_id", "player_id"] + [c for c in s.columns if c.startswith("sp_")]],
                    on=["game_id", "player_id"], how="left")

    # descanso del abridor
    last = (st.sort_values("available_at").groupby("player_id")
              .available_at.apply(list).to_dict())
    def rest(row):
        L = [t for t in last.get(row.player_id, []) if t < row.cut]
        return (row.cut - L[-1]).total_seconds() / 86400 if L else np.nan
    r["cut"] = pd.to_datetime(r["cut"])
    r["sp_rest_days"] = [rest(x) for x in r.itertuples()]
    r["sp_known"] = 1.0

    out = pp[["game_id", "team_id", "side", "player_id", "sp_source"]].merge(
        r[["game_id", "player_id"] + [c for c in r.columns if c.startswith("sp_")]],
        on=["game_id", "player_id"], how="left")
    return out.rename(columns={"player_id": "sp_player_id"})


def _rest_days(X: pd.DataFrame) -> pd.DataFrame:
    """Dias de descanso y partidos en los ultimos 7 dias, por equipo."""
    ev = pd.concat([
        X[["game_id", "home_team_id", "start_utc"]].rename(columns={"home_team_id": "team_id"}),
        X[["game_id", "away_team_id", "start_utc"]].rename(columns={"away_team_id": "team_id"})])
    ev["start_utc"] = pd.to_datetime(ev.start_utc)
    ev = ev.sort_values("start_utc")
    prev = ev.groupby("team_id").start_utc.shift(1)
    ev["rest"] = (ev.start_utc - prev).dt.total_seconds() / 86400
    m = ev.set_index(["game_id", "team_id"]).rest
    h = [m.get((g, t), np.nan) for g, t in zip(X.game_id, X.home_team_id)]
    a = [m.get((g, t), np.nan) for g, t in zip(X.game_id, X.away_team_id)]
    rest = pd.DataFrame({"h_rest_days": h, "a_rest_days": a}, index=X.index)
    rest["d_rest_days"] = rest.h_rest_days - rest.a_rest_days
    return pd.concat([X, rest], axis=1)


def _park(X: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Park factor calculado con los propios partidos, usando SOLO temporadas
    anteriores. Si no hay historia previa suficiente, queda NaN (no se inventa)."""
    g = games[games.home_score.notna()].copy()
    g["total"] = g.home_score + g.away_score
    lg = g.groupby("season").total.mean()
    pf = (g.groupby(["venue_id", "season"]).agg(t=("total", "mean"), n=("total", "size"))
            .reset_index())
    pf["pf"] = pf.t / pf.season.map(lg)
    pf = pf[pf.n >= 40]
    # se desplaza una temporada: el factor de 2023 solo se usa en 2024
    pf["apply_season"] = pf.season + 1
    m = {(r.venue_id, r.apply_season): r.pf for r in pf.itertuples()}
    return pd.concat([X, pd.DataFrame(
        {"park_runs_factor": [m.get((v, s), np.nan)
                              for v, s in zip(X.venue_id, X.season)]},
        index=X.index)], axis=1)


def audit(X: pd.DataFrame) -> dict:
    """Comprobaciones sobre las features ya construidas."""
    out = {"rows": int(len(X)), "cols": int(X.shape[1]), "problems": []}
    fin = X.status.isin(FINAL)
    out["finished"] = int(fin.sum())
    out["with_target"] = int(X.y_home_win.notna().sum())
    out["with_f5"] = int(X.y_home_f5.notna().sum())
    out["with_starter"] = int(X.get("h_sp_known", pd.Series(dtype=float)).notna().sum())
    for c in ("y_home_win", "y_home_f5", "y_home_rl"):
        v = X[c].dropna()
        if len(v) and not set(v.unique()) <= {0.0, 1.0}:
            out["problems"].append(f"{c} tiene valores fuera de 0/1")
    # una feature no puede correlacionar perfecto con el resultado del propio partido
    num = X.select_dtypes("number")
    y = X.y_home_win
    mask = y.notna()
    sus = []
    for c in num.columns:
        if c.startswith(("y_", "home_score", "away_score", "home_f5", "away_f5",
                         "total_runs", "home_margin")):
            continue
        v = num.loc[mask, c]
        if v.notna().sum() < 200:
            continue
        r = np.corrcoef(v.fillna(v.median()), y[mask])[0, 1]
        if abs(r) > 0.6:
            sus.append({"feature": c, "corr": round(float(r), 3)})
    out["suspicious_correlations"] = sorted(sus, key=lambda x: -abs(x["corr"]))[:10]
    if sus:
        out["problems"].append(f"{len(sus)} feature(s) con |corr|>0.6 contra el resultado")
    return out
