"""Base de datos PROPIA de TENIS."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float, Integer, String, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.orm import declarative_base, sessionmaker

from shared.paths import TENIS_DB_DIR

TENIS_DB = TENIS_DB_DIR / "tenis_markets.sqlite3"
Base = declarative_base()


class TenisMatch(Base):
    """Un partido jugado. `p1` es el GANADOR en los historicos (asi vienen los datos);
    las features se construyen despues de forma simetrica para no filtrar el resultado."""
    __tablename__ = "tenis_matches"
    match_id = Column(String, primary_key=True)      # tour-tourney_id-match_num
    tour = Column(String, index=True)                # ATP | WTA
    season = Column(Integer, index=True)
    tourney_id = Column(String)
    tourney_name = Column(String)
    tourney_level = Column(String)                   # G(slam) M A D F C S
    surface = Column(String, index=True)             # Hard | Clay | Grass | Carpet
    draw_size = Column(Integer)
    match_date = Column(Date, index=True)
    round = Column(String)
    best_of = Column(Integer)
    minutes = Column(Integer)
    score = Column(String)
    retirement = Column(Boolean, default=False)
    winner_id = Column(String, index=True)
    loser_id = Column(String, index=True)
    winner_name = Column(String)
    loser_name = Column(String)
    winner_rank = Column(Integer)
    loser_rank = Column(Integer)
    winner_rank_points = Column(Integer)
    loser_rank_points = Column(Integer)
    winner_age = Column(Float)
    loser_age = Column(Float)
    winner_hand = Column(String)
    loser_hand = Column(String)
    winner_ht = Column(Integer)
    loser_ht = Column(Integer)
    games_winner = Column(Integer)                   # juegos ganados por el ganador
    games_loser = Column(Integer)
    games_total = Column(Integer)
    sets_winner = Column(Integer)
    sets_loser = Column(Integer)
    serve = Column(JSON)                             # w_ace, w_svpt, ... l_bpFaced
    source = Column(String, default="sackmann-archive")
    ingested_at = Column(DateTime)


class TenisPlayer(Base):
    __tablename__ = "tenis_players"
    player_id = Column(String, primary_key=True)
    tour = Column(String)
    name = Column(String)
    hand = Column(String)
    height = Column(Integer)
    country = Column(String)
    birth = Column(String)


class TenisSchedule(Base):
    """Partido futuro (de The Odds API): no hay resultado todavia."""
    __tablename__ = "tenis_schedule"
    event_id = Column(String, primary_key=True)
    tour = Column(String)
    sport_key = Column(String)
    tourney_name = Column(String)
    surface = Column(String)
    start_utc = Column(DateTime, index=True)
    p1_name = Column(String)
    p2_name = Column(String)
    p1_id = Column(String)
    p2_id = Column(String)
    best_of = Column(Integer)
    resolved = Column(Boolean, default=False)
    ingested_at = Column(DateTime)


class TenisResult(Base):
    """Resultado real de un partido del calendario, traido de una fuente en vivo.
    El archivo historico esta congelado, asi que sin esto los partidos nuevos
    nunca se podrian calificar."""
    __tablename__ = "tenis_results"
    event_id = Column(String, primary_key=True)
    tour = Column(String)
    match_date = Column(Date, index=True)
    p1_name = Column(String)
    p2_name = Column(String)
    winner_name = Column(String)
    p1_games = Column(Integer)
    p2_games = Column(Integer)
    games_total = Column(Integer)
    games_margin = Column(Integer)
    sets_p1 = Column(Integer)
    sets_p2 = Column(Integer)
    status = Column(String)              # final | retired | walkover
    detail = Column(String)
    source = Column(String)
    fetched_at = Column(DateTime)


class TenisModelVersion(Base):
    __tablename__ = "tenis_model_versions"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    market = Column(String, nullable=False)          # winner | total_games | handicap_games
    tour = Column(String)                            # ATP | WTA | both
    version = Column(String, nullable=False)
    algorithm = Column(String)
    training_period = Column(String)
    validation_period = Column(String)
    features = Column(JSON)
    calibration = Column(JSON)
    metrics = Column(JSON)
    artifact_path = Column(String)
    is_production = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class TenisPrediction(Base):
    __tablename__ = "tenis_predictions"
    id = Column(Integer, primary_key=True)
    event_id = Column(String, index=True, nullable=False)
    match_id = Column(String)                        # cuando se resuelve contra el historico
    tour = Column(String)
    season = Column(Integer)
    match_date = Column(Date)
    start_utc = Column(DateTime)
    tourney_name = Column(String)
    surface = Column(String)
    best_of = Column(Integer)
    p1_name = Column(String)
    p2_name = Column(String)
    market_type = Column(String, nullable=False)     # winner | total_games | handicap_games
    model_version = Column(String)
    engine = Column(String)                          # punto_a_punto | elo
    calibration_version = Column(String)
    prediction_timestamp = Column(DateTime, nullable=False)
    cutoff_timestamp = Column(DateTime)
    expected_value = Column(Float)                   # juegos totales esperados | margen de juegos
    expected_std = Column(Float)
    p_serve_1 = Column(Float)                        # prob. de ganar punto al saque proyectada
    p_serve_2 = Column(Float)
    line = Column(Float)
    line_source = Column(String)
    line_timestamp = Column(DateTime)
    odds_1 = Column(Float)
    odds_2 = Column(Float)
    p_raw = Column(Float)
    model_probability = Column(Float)
    market_probability = Column(Float)
    final_probability = Column(Float)
    selection = Column(String)
    pick_probability = Column(Float)
    status_market = Column(String)                   # pick | projection | no_pick | blocked
    confidence = Column(String)
    gap_pp = Column(Float)
    explanation = Column(JSON)
    data_completeness = Column(JSON)
    status = Column(String, default="published")
    version = Column(Integer, default=1)
    parent_id = Column(Integer)
    result = Column(String)
    correct = Column(Boolean)
    actual_value = Column(Float)


class TenisOdds(Base):
    __tablename__ = "tenis_odds_snapshots"
    id = Column(Integer, primary_key=True)
    event_id = Column(String, index=True, nullable=False)
    bookmaker = Column(String)
    market = Column(String)                          # winner | total_games | handicap_games
    selection = Column(String)                       # p1 | p2 | over | under
    line = Column(Float)
    price_american = Column(Float)
    implied_prob = Column(Float)
    available_at = Column(DateTime, index=True)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)
    source = Column(String, default="the_odds_api")


class TenisPipelineRun(Base):
    __tablename__ = "tenis_pipeline_runs"
    id = Column(Integer, primary_key=True)
    kind = Column(String)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    status = Column(String)
    summary = Column(JSON)


class TenisSourceLog(Base):
    __tablename__ = "tenis_source_log"
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
        TENIS_DB.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{TENIS_DB}", future=True,
                                connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi, _):
            c = dbapi.cursor(); c.execute("PRAGMA journal_mode=DELETE"); c.close()
        _Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
    return _engine


def init_db():
    Base.metadata.create_all(engine())
    return TENIS_DB


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
