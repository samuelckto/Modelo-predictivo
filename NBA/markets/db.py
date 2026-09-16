"""Base de datos PROPIA de NBA. Nadie mas escribe aqui; NBA no escribe fuera."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float, Integer, String, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.orm import declarative_base, sessionmaker

from shared.paths import NBA_DB_DIR

NBA_DB = NBA_DB_DIR / "nba_markets.sqlite3"
Base = declarative_base()


class NbaGame(Base):
    __tablename__ = "nba_games"
    game_id = Column(String, primary_key=True)
    season = Column(Integer, index=True)          # 2016 = temporada 2015-16
    season_type = Column(String)                  # Regular Season | Playoffs
    game_date = Column(Date, index=True)
    start_utc = Column(DateTime)                  # tipoff si se conoce; si no, None
    home_team_id = Column(String)
    away_team_id = Column(String)
    home_abbr = Column(String)
    away_abbr = Column(String)
    home_points = Column(Integer)
    away_points = Column(Integer)
    home_poss = Column(Integer)
    away_poss = Column(Integer)
    status = Column(String, default="final")
    source = Column(String, default="pbpstats")
    ingested_at = Column(DateTime)


class NbaTeamGameLog(Base):
    """Estadisticas de UN equipo en UN partido (derivadas del play-by-play).
    available_at = fin del dia del partido: nunca se usan para partidos del mismo dia."""
    __tablename__ = "nba_team_game_logs"
    id = Column(Integer, primary_key=True)
    game_id = Column(String, index=True)
    team_id = Column(String, index=True)
    opp_id = Column(String)
    season = Column(Integer, index=True)
    game_date = Column(Date)
    is_home = Column(Boolean)
    stats = Column(JSON)                          # dict con todas las columnas de pbpstats
    available_at = Column(DateTime)
    __table_args__ = (UniqueConstraint("game_id", "team_id", name="uq_gamelog"),)


class NbaTeam(Base):
    __tablename__ = "nba_teams"
    team_id = Column(String, primary_key=True)
    abbr = Column(String)
    name = Column(String)


class NbaModelVersion(Base):
    __tablename__ = "nba_model_versions"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)          # NBA_ML_v1 | NBA_SPREAD_v1 | NBA_TOTAL_v1
    market = Column(String, nullable=False)        # moneyline | spread | total
    version = Column(String, nullable=False)
    algorithm = Column(String)
    training_period = Column(String)
    validation_period = Column(String)
    features = Column(JSON)
    calibration = Column(JSON)                     # metodo y parametros
    metrics = Column(JSON)
    artifact_path = Column(String)
    is_production = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class NbaPrediction(Base):
    __tablename__ = "nba_predictions"
    id = Column(Integer, primary_key=True)
    game_id = Column(String, index=True, nullable=False)
    season = Column(Integer)
    game_date = Column(Date)
    start_utc = Column(DateTime)
    home_abbr = Column(String)
    away_abbr = Column(String)
    market_type = Column(String, nullable=False)   # moneyline | spread | total
    model_version = Column(String)
    calibration_version = Column(String)
    prediction_timestamp = Column(DateTime, nullable=False)
    cutoff_timestamp = Column(DateTime)
    expected_value = Column(Float)                 # margen esperado (local-visit) | total esperado | None
    expected_std = Column(Float)
    line = Column(Float)                           # spread del local (negativo = local favorito) | total
    line_source = Column(String)
    line_timestamp = Column(DateTime)
    odds_home = Column(Float)                      # precio americano del lado local/over
    odds_away = Column(Float)
    p_raw = Column(Float)                          # P(local gana | local cubre | over) sin calibrar
    model_probability = Column(Float)              # calibrada, lado local/over
    market_probability = Column(Float)             # sin vig, lado local/over
    final_probability = Column(Float)
    selection = Column(String)
    pick_probability = Column(Float)               # del lado mostrado
    status_market = Column(String)                 # pick | projection | no_pick | blocked
    confidence = Column(String)
    gap_pp = Column(Float)
    explanation = Column(JSON)
    data_completeness = Column(JSON)
    status = Column(String, default="published")   # published | superseded
    version = Column(Integer, default=1)
    parent_id = Column(Integer)
    result = Column(String)                        # win | loss | push
    correct = Column(Boolean)
    actual_value = Column(Float)


class NbaOdds(Base):
    __tablename__ = "nba_odds_snapshots"
    id = Column(Integer, primary_key=True)
    game_id = Column(String, index=True, nullable=False)
    bookmaker = Column(String)
    market = Column(String)                        # moneyline | spread | total
    selection = Column(String)                     # home | away | over | under
    line = Column(Float)
    price_american = Column(Float)
    implied_prob = Column(Float)
    available_at = Column(DateTime, index=True)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)
    source = Column(String, default="the_odds_api")


class NbaAvailability(Base):
    """Disponibilidad de jugadores anotada EN VIVO con timestamp propio (no hay historico)."""
    __tablename__ = "nba_availability"
    id = Column(Integer, primary_key=True)
    game_id = Column(String, index=True)
    team_id = Column(String)
    player = Column(String)
    status = Column(String)                        # out | questionable | probable | available
    available_at = Column(DateTime)
    source = Column(String)


class NbaPipelineRun(Base):
    __tablename__ = "nba_pipeline_runs"
    id = Column(Integer, primary_key=True)
    kind = Column(String)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    status = Column(String)
    summary = Column(JSON)


class NbaSourceLog(Base):
    __tablename__ = "nba_source_log"
    id = Column(Integer, primary_key=True)
    source = Column(String)
    domain = Column(String)
    url = Column(String)
    status = Column(String)
    records = Column(Integer)
    error = Column(Text)
    retrieved_at = Column(DateTime)


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        NBA_DB.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{NBA_DB}", future=True,
                                connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi, _):
            c = dbapi.cursor(); c.execute("PRAGMA journal_mode=DELETE"); c.close()
        _Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
    return _engine


def init_db():
    Base.metadata.create_all(engine())
    return NBA_DB


@contextmanager
def session_scope():
    engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
