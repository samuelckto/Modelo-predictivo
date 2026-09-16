"""Esquema de SOCCER: base propia, aislada de los demas deportes.

Separacion clave del modulo, y la razon de que haya dos tablas de partidos:

- `SoccerMatch`     partido HISTORICO ya jugado -> entrenar, validar, calibrar.
- `SoccerFixture`   partido PROXIMO del calendario en vivo -> predecir.

Cuando un fixture termina y se conoce el marcador, pasa tambien a `SoccerMatch`
con `source='live'`: asi el historico crece solo y una liga sin datos (Saudi)
puede activarse con el tiempo sin tocar la arquitectura.

Todo dato guardado lleva el momento REAL en que se recibio (`fetched_at`). Nunca
se inventa un timestamp: si la fuente no lo da, queda en NULL y la feature que
dependa de el no se usa en produccion.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from shared.paths import SOCCER_DATABASE_URL, SOCCER_DB_DIR, assert_writable

SOCCER_DB_DIR.mkdir(parents=True, exist_ok=True)
assert_writable(SOCCER_DB_DIR, "SOCCER")

_engine = create_engine(SOCCER_DATABASE_URL, future=True)
_Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class SoccerLeague(Base):
    """Catalogo de ligas y su estado de datos (se recalcula, no se escribe a mano)."""
    __tablename__ = "leagues"
    code: Mapped[str] = mapped_column(String(16), primary_key=True)
    nombre: Mapped[str] = mapped_column(String(64))
    pais: Mapped[str] = mapped_column(String(64))
    odds_key: Mapped[str] = mapped_column(String(64))
    historico_file: Mapped[str | None] = mapped_column(String(32), nullable=True)
    partidos_historicos: Mapped[int] = mapped_column(Integer, default=0)
    primera_fecha: Mapped[date | None] = mapped_column(Date, nullable=True)
    ultima_fecha: Mapped[date | None] = mapped_column(Date, nullable=True)
    estado: Mapped[str] = mapped_column(String(24), default="unknown")   # ok / insufficient_data
    actualizado_en: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SoccerTeam(Base):
    """Nombres de equipo por liga. `alias` guarda como lo llama The Odds API."""
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("league_code", "nombre", name="uq_team_liga"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_code: Mapped[str] = mapped_column(String(16), ForeignKey("leagues.code"), index=True)
    nombre: Mapped[str] = mapped_column(String(96), index=True)
    nombre_norm: Mapped[str] = mapped_column(String(96), index=True)
    alias_odds: Mapped[str | None] = mapped_column(String(96), nullable=True)
    visto_por_ultima_vez: Mapped[date | None] = mapped_column(Date, nullable=True)


class SoccerMatch(Base):
    """Partido JUGADO. Es la unica fuente para entrenar."""
    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("league_code", "match_date", "home_team", "away_team",
                                       name="uq_match"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    league_code: Mapped[str] = mapped_column(String(16), index=True)
    season: Mapped[str] = mapped_column(String(12), index=True)
    stage: Mapped[str | None] = mapped_column(String(48), nullable=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    # Hora real de saque SOLO si la fuente la da. Si es NULL, el corte anti-leakage
    # usa las 00:00 del dia del partido, que es mas conservador.
    kickoff_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    home_team: Mapped[str] = mapped_column(String(96), index=True)
    away_team: Mapped[str] = mapped_column(String(96), index=True)
    home_goals: Mapped[int] = mapped_column(Integer)
    away_goals: Mapped[int] = mapped_column(Integer)
    ht_home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Estadisticas de partido. NULL cuando la fuente no las da: nunca se rellenan
    # con ceros, porque un 0 significa "cero corners", no "no lo se".
    home_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_sot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_sot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_yellow: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_yellow: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stats_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32))          # footballcsv / openfootball / live
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class SoccerFixture(Base):
    """Partido PROXIMO del calendario en vivo (The Odds API)."""
    __tablename__ = "fixtures"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    league_code: Mapped[str] = mapped_column(String(16), index=True)
    odds_key: Mapped[str] = mapped_column(String(64))
    commence_utc: Mapped[datetime] = mapped_column(DateTime, index=True)
    home_team: Mapped[str] = mapped_column(String(96))
    away_team: Mapped[str] = mapped_column(String(96))
    home_match: Mapped[str | None] = mapped_column(String(96), nullable=True)  # nombre historico
    away_match: Mapped[str | None] = mapped_column(String(96), nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    primera_vez: Mapped[datetime] = mapped_column(DateTime)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class SoccerOdds(Base):
    """Snapshot de cuotas con el momento REAL en que se recibio."""
    __tablename__ = "odds_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    league_code: Mapped[str] = mapped_column(String(16), index=True)
    market: Mapped[str] = mapped_column(String(24), index=True)   # h2h / totals / btts
    bookmaker: Mapped[str] = mapped_column(String(48))
    selection: Mapped[str] = mapped_column(String(96))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float] = mapped_column(Float)                   # decimal
    implied: Mapped[float | None] = mapped_column(Float, nullable=True)
    novig: Mapped[float | None] = mapped_column(Float, nullable=True)
    bookmaker_update: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    snapshot_kind: Mapped[str] = mapped_column(String(16), default="current")  # opening/current/closing


class SoccerPrediction(Base):
    """Prediccion versionada. Nunca se sobrescribe: la vieja pasa a superseded."""
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    league_code: Mapped[str] = mapped_column(String(16), index=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    kickoff_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    home_team: Mapped[str] = mapped_column(String(96))
    away_team: Mapped[str] = mapped_column(String(96))
    market: Mapped[str] = mapped_column(String(24), index=True)   # 1x2 / btts / total
    selection: Mapped[str] = mapped_column(String(64))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 1X2 guarda las tres; btts/total guardan p_yes / p_over en `probability`.
    p_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability: Mapped[float] = mapped_column(Float)             # prob del lado mostrado
    lambda_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    lambda_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    gap_pp: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    bookmaker: Mapped[str | None] = mapped_column(String(48), nullable=True)
    odds_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)   # pick/projection/no_pick/
    #                                                               insufficient_data/blocked
    model_version: Mapped[str] = mapped_column(String(48))
    calibration_version: Mapped[str | None] = mapped_column(String(48), nullable=True)
    feature_cutoff: Mapped[datetime] = mapped_column(DateTime)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    record_status: Mapped[str] = mapped_column(String(16), default="active")  # active/superseded
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)      # JSON
    features_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_completeness: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str | None] = mapped_column(String(12), nullable=True)     # win/loss/push
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    actual_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SoccerGoalsDist(Base):
    """Distribucion de goles publicada para un partido (auditable despues)."""
    __tablename__ = "goals_distribution"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(48))
    lambda_home: Mapped[float] = mapped_column(Float)
    lambda_away: Mapped[float] = mapped_column(Float)
    rho: Mapped[float | None] = mapped_column(Float, nullable=True)   # Dixon-Coles
    matrix_json: Mapped[str] = mapped_column(Text)                    # P(i,j) truncada
    created_at: Mapped[datetime] = mapped_column(DateTime)


class SoccerModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(48), index=True)
    market: Mapped[str] = mapped_column(String(24))
    scope: Mapped[str] = mapped_column(String(24))          # global / por_liga / global_liga
    league_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    algorithm: Mapped[str] = mapped_column(String(48))
    features: Mapped[str] = mapped_column(Text)
    training_period: Mapped[str] = mapped_column(String(48))
    validation_period: Mapped[str] = mapped_column(String(48))
    calibration: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class SoccerSourceLog(Base):
    """Cada consulta a una fuente: cuando, a que, y como salio."""
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(48), index=True)
    scope: Mapped[str | None] = mapped_column(String(64), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    records: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime, index=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope():
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
