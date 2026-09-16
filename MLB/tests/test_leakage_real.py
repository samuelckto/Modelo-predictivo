"""Anti-leakage verificado contra los datos REALES ya ingeridos.

No basta con probar el helper con datos sinteticos: aqui se recalculan features
a mano, de forma independiente del constructor, y se comparan. Si el constructor
usara informacion posterior al inicio del partido, estas pruebas fallan.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from MLB.database.models import BatterGameLog, Game, PitcherGameLog, StatcastAgg
from MLB.database.session import session_scope
from shared.paths import MLB_PROCESSED_DIR

FEAT = MLB_PROCESSED_DIR / "features.parquet"
pytestmark = pytest.mark.skipif(not FEAT.exists(),
                                reason="todavia no se han construido las features MLB")


@pytest.fixture(scope="module")
def X():
    return pd.read_parquet(FEAT)


@pytest.fixture(scope="module")
def raw():
    with session_scope() as s:
        bat = pd.DataFrame(s.execute(select(
            BatterGameLog.game_id, BatterGameLog.team_id, BatterGameLog.pa if False else
            BatterGameLog.plate_appearances, BatterGameLog.hits, BatterGameLog.walks,
            BatterGameLog.available_at)).all(),
            columns=["game_id", "team_id", "pa", "hits", "bb", "available_at"])
        pit = pd.DataFrame(s.execute(select(
            PitcherGameLog.game_id, PitcherGameLog.player_id, PitcherGameLog.is_starter,
            PitcherGameLog.outs, PitcherGameLog.earned_runs, PitcherGameLog.available_at)).all(),
            columns=["game_id", "player_id", "is_starter", "outs", "er", "available_at"])
        gm = pd.DataFrame(s.execute(select(
            Game.id, Game.season, Game.start_utc, Game.home_team_id, Game.away_team_id,
            Game.home_score, Game.away_score)).all(),
            columns=["game_id", "season", "start_utc", "home_team_id", "away_team_id",
                     "hs", "as_"])
        pit_season = pd.DataFrame(s.execute(select(
            PitcherGameLog.game_id, PitcherGameLog.season)).all(),
            columns=["game_id", "season"])
    # drop_duplicates: pit_season trae una fila por lanzador, no por partido
    pit = pit.merge(pit_season.drop_duplicates("game_id"), on="game_id", how="left")
    for d in (bat, pit):
        d["available_at"] = pd.to_datetime(d.available_at)
    gm["start_utc"] = pd.to_datetime(gm.start_utc)
    return {"bat": bat, "pit": pit, "gm": gm}


def test_ninguna_fila_usada_es_posterior_al_inicio(raw):
    """Comprobacion directa sobre la base: los eventos de un partido nunca son
    anteriores a su propio inicio, por lo que jamas pueden entrar en sus features."""
    m = raw["pit"].merge(raw["gm"][["game_id", "start_utc"]], on="game_id")
    violaciones = m[m.available_at < m.start_utc]
    assert len(violaciones) == 0, (
        f"{len(violaciones)} lineas de pitcheo con available_at ANTERIOR al inicio de su "
        f"propio partido: el supuesto de disponibilidad estaria mal")


def test_obp_ofensivo_coincide_con_el_calculo_manual(X, raw):
    """Se recalcula off_obp_s para 150 partidos con un filtro estricto e
    independiente y debe coincidir con lo que produjo el constructor."""
    bat = (raw["bat"].groupby(["game_id", "team_id", "available_at"], as_index=False)
           [["pa", "hits", "bb"]].sum())
    bat = bat.merge(raw["gm"][["game_id"]].assign(_k=1)[["game_id"]], on="game_id")
    seas = raw["gm"].set_index("game_id")
    bat["season"] = [pd.Timestamp(seas.start_utc.get(g)).year if g in seas.index else None
                     for g in bat.game_id]
    # Los partidos de los ultimos 10 dias se excluyen: entre un `mlb-daily` y el
    # siguiente llegan boxscores nuevos y el parquet de features queda un paso
    # atras (no es fuga de datos, es desfase de refresco). El resto debe cuadrar
    # exactamente.
    lim = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=10)
    sample = X[X.h_off_obp_s.notna() & (pd.to_datetime(X.start_utc) < lim)].sample(150, random_state=7)
    difs = []
    for r in sample.itertuples():
        cut = pd.Timestamp(r.start_utc)
        prev = bat[(bat.team_id == r.home_team_id) & (bat.available_at < cut) &
                   (bat.season == r.season)]
        if prev.pa.sum() < 30:
            continue
        manual = (prev.hits.sum() + prev.bb.sum()) / prev.pa.sum()
        if not np.isclose(manual, r.h_off_obp_s, atol=1e-9):
            difs.append((r.game_id, manual, r.h_off_obp_s))
    assert not difs, f"{len(difs)} discrepancias, p.ej. {difs[:3]}"


def test_era_del_abridor_no_incluye_su_propia_apertura(X, raw):
    """La ERA del abridor debe calcularse solo con aperturas ANTERIORES."""
    st = raw["pit"][raw["pit"].is_starter.fillna(False)]
    sample = X[(X.h_sp_era_s.notna()) & (X.h_sp_player_id.notna())].sample(120, random_state=11)
    difs = []
    for r in sample.itertuples():
        cut = pd.Timestamp(r.start_utc)
        prev = st[(st.player_id == r.h_sp_player_id) & (st.available_at < cut) &
                  (st.season == r.season)]
        ip = prev.outs.sum() / 3.0
        if ip < 10:
            continue
        manual = prev.er.sum() * 9 / ip
        if not np.isclose(manual, r.h_sp_era_s, atol=1e-6):
            difs.append((r.game_id, round(manual, 4), round(r.h_sp_era_s, 4)))
        # y la apertura del propio partido NO puede estar dentro
        assert r.game_id not in set(prev.game_id), \
            f"la apertura del partido {r.game_id} entro en su propia ERA"
    assert not difs, f"{len(difs)} discrepancias, p.ej. {difs[:3]}"


def test_el_primer_partido_de_cada_equipo_no_tiene_historia(X):
    """El PRIMER partido de la temporada de un equipo (como local o visitante) no
    puede tener medias acumuladas: no existe historia previa de esa temporada."""
    long = pd.concat([
        X[["game_id", "season", "start_utc", "home_team_id"]]
            .rename(columns={"home_team_id": "team"}).assign(side="h"),
        X[["game_id", "season", "start_utc", "away_team_id"]]
            .rename(columns={"away_team_id": "team"}).assign(side="a")])
    first = long.sort_values("start_utc").groupby(["season", "team"]).head(1)
    idx = X.set_index("game_id")
    malos = []
    for r in first.itertuples():
        col = "h_off_obp_s" if r.side == "h" else "a_off_obp_s"
        v = idx.at[r.game_id, col]
        if pd.notna(v):
            malos.append((r.game_id, r.season, r.team, col, v))
    assert not malos, f"{len(malos)} primeros partidos con historia, p.ej. {malos[:3]}"


def test_statcast_disponible_solo_al_dia_siguiente():
    with session_scope() as s:
        rows = s.execute(select(StatcastAgg.game_date, StatcastAgg.available_at).limit(2000)).all()
    bad = [(d, a) for d, a in rows if a is None or
           a <= pd.Timestamp(d) + pd.Timedelta(hours=6)]
    assert not bad, f"{len(bad)} filas de statcast disponibles demasiado pronto: {bad[:3]}"


def test_los_objetivos_no_aparecen_entre_las_features(X):
    prohibidas = {"home_score", "away_score", "home_f5", "away_f5", "total_runs",
                  "home_margin", "total_runs_f5"}
    feats = [c for c in X.columns if c.startswith(("h_", "a_", "d_")) or
             c in ("park_runs_factor", "is_doubleheader", "month", "game_number")]
    assert not (set(feats) & prohibidas)
    for c in feats:
        assert not c.startswith("y_"), c
