"""Base de datos PROPIA del Sports Prediction Center para los mercados NFL.

Separada del motor NFL original (que no se toca) y de la base MLB.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer, String, Text,
                        create_engine, event)
from sqlalchemy.orm import declarative_base, sessionmaker

from shared.paths import ROOT

NFL_MARKETS_DB = ROOT / "NFL" / "database" / "nfl_markets.sqlite3"
Base = declarative_base()


class NflModelVersion(Base):
    __tablename__ = "nfl_model_versions"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)          # NFL_TOTAL_v1, NFL_SPREAD_v1
    market = Column(String, nullable=False)        # total | spread
    version = Column(String, nullable=False)
    algorithm = Column(String)
    train_seasons = Column(String)
    features = Column(JSON)
    metrics = Column(JSON)                         # walk-forward, calibracion, gating
    artifact_path = Column(String)
    is_production = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class NflMarketPrediction(Base):
    __tablename__ = "nfl_market_predictions"
    id = Column(Integer, primary_key=True)
    game_id = Column(String, index=True, nullable=False)
    season = Column(Integer)
    week = Column(Integer)
    kickoff_utc = Column(DateTime)
    home_team = Column(String)
    away_team = Column(String)
    market_type = Column(String, nullable=False)   # total | spread
    model_version = Column(String)
    prediction_timestamp = Column(DateTime, nullable=False)
    cutoff_timestamp = Column(DateTime)            # ultima fuente usada en las features
    expected_value = Column(Float)                 # total esperado | margen esperado (local - visitante)
    expected_median = Column(Float)
    expected_std = Column(Float)
    line = Column(Float)                           # linea evaluada (total | spread del local)
    line_source = Column(String)                   # the_odds_api | nflverse | referencia
    line_timestamp = Column(DateTime)
    p_over_raw = Column(Float)                     # P(over) / P(local cubre) sin calibrar
    model_probability = Column(Float)              # calibrada, lado "over"/"local cubre"
    market_probability = Column(Float)             # sin vig, mismo lado
    final_probability = Column(Float)
    selection = Column(String)                     # "OVER 47.5" | "KC -2.5" | None
    pick = Column(String)                          # pick | proyeccion | no_pick
    pick_probability = Column(Float)               # probabilidad del lado elegido/mostrado
    confidence = Column(String)                    # ALTA | MEDIA | BAJA
    gap_pp = Column(Float)                         # modelo - mercado (puntos porcentuales)
    explanation = Column(JSON)
    data_completeness = Column(JSON)
    status = Column(String, default="published")   # published | superseded
    version = Column(Integer, default=1)
    parent_id = Column(Integer)
    result = Column(String)                        # win | loss | push
    correct = Column(Boolean)
    actual_value = Column(Float)


class NflOdds(Base):
    __tablename__ = "nfl_odds"
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


class NflPipelineRun(Base):
    __tablename__ = "nfl_pipeline_runs"
    id = Column(Integer, primary_key=True)
    kind = Column(String)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    status = Column(String)
    summary = Column(JSON)


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        NFL_MARKETS_DB.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{NFL_MARKETS_DB}", future=True,
                                connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi, _):
            c = dbapi.cursor(); c.execute("PRAGMA journal_mode=DELETE"); c.close()
        _Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
    return _engine


def init_db():
    Base.metadata.create_all(engine())
    return NFL_MARKETS_DB


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
