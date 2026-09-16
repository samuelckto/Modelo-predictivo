"""Base PROPIA de las combinadas. No se escribe en ningun deporte."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float, Integer, String, Text,
                        create_engine, event)
from sqlalchemy.orm import declarative_base, sessionmaker

from shared.paths import PARLAY_DB_DIR

PARLAY_DB = PARLAY_DB_DIR / "parlays.sqlite3"
Base = declarative_base()


class Parlay(Base):
    __tablename__ = "parlays"
    id = Column(Integer, primary_key=True)
    parlay_key = Column(String, index=True, unique=True)   # huella de las patas: no se duplica
    size = Column(Integer)                                 # 2 o 3
    mixed = Column(Boolean)                                # patas de mas de un deporte
    sports = Column(String)
    created_at = Column(DateTime, nullable=False)
    first_start_utc = Column(DateTime)
    last_start_utc = Column(DateTime)
    match_date = Column(Date, index=True)
    probability = Column(Float)                            # producto de las patas (independientes)
    fair_odds = Column(Float)                              # 1 / probabilidad
    market_probability = Column(Float)                     # producto del mercado sin vig (si hay)
    market_odds = Column(Float)
    min_leg_probability = Column(Float)
    rank = Column(Integer)                                 # posicion en la generacion del dia
    status = Column(String, default="open")                # open | graded
    result = Column(String)                                # win | loss | push_partial
    legs_won = Column(Integer)
    legs_total = Column(Integer)
    graded_at = Column(DateTime)
    notes = Column(Text)


class ParlayLeg(Base):
    __tablename__ = "parlay_legs"
    id = Column(Integer, primary_key=True)
    parlay_id = Column(Integer, index=True)
    sport = Column(String)
    game_id = Column(String)
    prediction_id = Column(Integer)
    market = Column(String)
    market_key = Column(String)
    selection = Column(String)
    probability = Column(Float)
    market_probability = Column(Float)
    start_utc = Column(DateTime)
    teams = Column(String)
    result = Column(String)                                # win | loss | push | pendiente
    correct = Column(Boolean)


class ParlayRun(Base):
    __tablename__ = "parlay_runs"
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    summary = Column(JSON)


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        PARLAY_DB.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{PARLAY_DB}", future=True,
                                connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi, _):
            c = dbapi.cursor(); c.execute("PRAGMA journal_mode=DELETE"); c.close()
        _Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
    return _engine


def init_db():
    Base.metadata.create_all(engine())
    return PARLAY_DB


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
