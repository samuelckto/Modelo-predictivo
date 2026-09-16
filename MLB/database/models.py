"""Esquema SQLite del motor MLB. Independiente del de NFL.

Regla anti-leakage: toda tabla con informacion que cambia dentro del dia lleva
`available_at` = momento en que ESE dato quedo disponible. El constructor de
features solo puede usar filas con available_at <= prediction_timestamp.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


SPORT = "MLB"          # constante: este esquema jamas guarda otro deporte


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)            # mlbam team id
    abbr = Column(String(5), index=True)
    name = Column(String(80))
    league = Column(String(4))
    division = Column(String(20))
    venue_id = Column(Integer)
    venue_name = Column(String(120))


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)            # mlbam player id
    full_name = Column(String(120), index=True)
    primary_position = Column(String(8))
    bats = Column(String(2))
    throws = Column(String(2))
    team_id = Column(Integer, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Game(Base):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True)            # mlbam gamePk
    season = Column(Integer, index=True)
    game_date = Column(String(10), index=True)        # YYYY-MM-DD (fecha oficial)
    start_utc = Column(DateTime, index=True)
    game_type = Column(String(4))                     # R, F, D, L, W, S
    status = Column(String(24))                       # Scheduled / Final / Postponed...
    doubleheader = Column(String(2))
    game_number = Column(Integer, default=1)
    home_team_id = Column(Integer, index=True)
    away_team_id = Column(Integer, index=True)
    home_abbr = Column(String(5))
    away_abbr = Column(String(5))
    venue_id = Column(Integer)
    venue_name = Column(String(120))
    roof = Column(String(20))
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_score_f5 = Column(Integer)                   # primeras 5 entradas
    away_score_f5 = Column(Integer)
    innings = Column(Integer)
    source = Column(String(40))
    retrieved_at = Column(DateTime)
    __table_args__ = (Index("ix_games_season_date", "season", "game_date"),)


class ProbablePitcher(Base):
    """Pitcher abridor. `state` = probable | confirmed. Se guarda historia completa."""
    __tablename__ = "probable_pitchers"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), index=True)
    team_id = Column(Integer, index=True)
    is_home = Column(Boolean)
    player_id = Column(Integer, index=True)
    player_name = Column(String(120))
    state = Column(String(12), default="probable")
    available_at = Column(DateTime, index=True)       # cuando se supo
    source = Column(String(40))
    __table_args__ = (Index("ix_pp_game_team_at", "game_id", "team_id", "available_at"),)


class Lineup(Base):
    """Alineacion. `state` = projected | probable | confirmed. Nunca se pisa una fila."""
    __tablename__ = "lineups"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), index=True)
    team_id = Column(Integer, index=True)
    is_home = Column(Boolean)
    state = Column(String(12), default="projected")
    batting_order = Column(JSON)                      # [player_id, ...] 1..9
    details = Column(JSON)
    available_at = Column(DateTime, index=True)
    source = Column(String(40))
    __table_args__ = (Index("ix_lineup_game_at", "game_id", "available_at"),)


class PitcherGameLog(Base):
    """Linea real de un abridor/relevista en un partido terminado (para features as-of)."""
    __tablename__ = "pitcher_game_logs"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, index=True)
    game_date = Column(String(10), index=True)
    season = Column(Integer, index=True)
    player_id = Column(Integer, index=True)
    team_id = Column(Integer, index=True)
    is_starter = Column(Boolean)
    outs = Column(Integer)
    pitches = Column(Integer)
    batters_faced = Column(Integer)
    hits = Column(Integer)
    runs = Column(Integer)
    earned_runs = Column(Integer)
    walks = Column(Integer)
    strikeouts = Column(Integer)
    home_runs = Column(Integer)
    ground_outs = Column(Integer)
    air_outs = Column(Integer)
    available_at = Column(DateTime, index=True)       # fin del partido
    source = Column(String(40))
    __table_args__ = (UniqueConstraint("game_id", "player_id", name="uq_pgl"),)


class BatterGameLog(Base):
    __tablename__ = "batter_game_logs"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, index=True)
    game_date = Column(String(10), index=True)
    season = Column(Integer, index=True)
    player_id = Column(Integer, index=True)
    team_id = Column(Integer, index=True)
    plate_appearances = Column(Integer)
    at_bats = Column(Integer)
    hits = Column(Integer)
    doubles = Column(Integer)
    triples = Column(Integer)
    home_runs = Column(Integer)
    walks = Column(Integer)
    strikeouts = Column(Integer)
    rbi = Column(Integer)
    total_bases = Column(Integer)
    available_at = Column(DateTime, index=True)
    source = Column(String(40))
    __table_args__ = (UniqueConstraint("game_id", "player_id", name="uq_bgl"),)


class StatcastAgg(Base):
    """Agregados Statcast por jugador y fecha (calidad de contacto)."""
    __tablename__ = "statcast_agg"
    id = Column(Integer, primary_key=True)
    scope = Column(String(10))                        # pitcher | batter
    player_id = Column(Integer, index=True)
    game_date = Column(String(10), index=True)
    season = Column(Integer, index=True)
    pitches = Column(Integer)
    batted_balls = Column(Integer)
    hard_hit = Column(Integer)
    barrels = Column(Integer)
    xba = Column(Float)
    xslg = Column(Float)
    xwoba = Column(Float)
    exit_velocity = Column(Float)
    launch_angle = Column(Float)
    swstr = Column(Float)
    csw = Column(Float)
    velocity = Column(Float)
    spin = Column(Float)
    available_at = Column(DateTime, index=True)
    source = Column(String(40))
    __table_args__ = (UniqueConstraint("scope", "player_id", "game_date", name="uq_sc"),)


class BullpenUsage(Base):
    """Uso del bullpen por equipo y fecha; se deriva de pitcher_game_logs."""
    __tablename__ = "bullpen_usage"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, index=True)
    game_date = Column(String(10), index=True)
    season = Column(Integer, index=True)
    relievers_used = Column(Integer)
    bullpen_outs = Column(Integer)
    bullpen_pitches = Column(Integer)
    back_to_back = Column(Integer)
    available_at = Column(DateTime, index=True)
    source = Column(String(40))
    __table_args__ = (UniqueConstraint("team_id", "game_date", name="uq_bp"),)


class Injury(Base):
    __tablename__ = "injuries"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, index=True)
    player_id = Column(Integer, index=True)
    player_name = Column(String(120))
    status = Column(String(40))                       # IL-10, IL-60, Day-To-Day...
    description = Column(Text)
    available_at = Column(DateTime, index=True)
    source = Column(String(40))


class Odds(Base):
    """Cuotas. Cada snapshot es una fila nueva; nunca se sobrescribe."""
    __tablename__ = "odds"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, index=True)
    bookmaker = Column(String(60), index=True)
    market = Column(String(40), index=True)           # moneyline|run_line|total|f5_moneyline|f5_total|team_total
    selection = Column(String(60))                    # home|away|over|under|<abbr>
    line = Column(Float)                              # -1.5, 8.5 ... null en moneyline
    price_american = Column(Float)
    price_decimal = Column(Float)
    implied_prob = Column(Float)                      # con vig
    is_closing = Column(Boolean, default=False)
    available_at = Column(DateTime, index=True)       # timestamp de la cuota
    source = Column(String(40))
    __table_args__ = (Index("ix_odds_game_mkt_at", "game_id", "market", "available_at"),)


class Weather(Base):
    __tablename__ = "weather"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, index=True)
    temperature_c = Column(Float)
    humidity = Column(Float)
    wind_speed_kmh = Column(Float)
    wind_direction_deg = Column(Float)
    precipitation_mm = Column(Float)
    is_forecast = Column(Boolean, default=True)
    available_at = Column(DateTime, index=True)
    source = Column(String(40))


class ParkFactor(Base):
    """Park factors calculados con datos propios; nunca copiados sin fuente."""
    __tablename__ = "park_factors"
    id = Column(Integer, primary_key=True)
    venue_id = Column(Integer, index=True)
    season = Column(Integer, index=True)
    runs_factor = Column(Float)
    hr_factor = Column(Float)
    n_games = Column(Integer)
    method = Column(String(60))
    available_at = Column(DateTime, index=True)
    source = Column(String(40))
    __table_args__ = (UniqueConstraint("venue_id", "season", name="uq_pf"),)


class Prediction(Base):
    """Una prediccion por partido y mercado. Nunca se borra: se crean versiones."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT, index=True)
    game_id = Column(Integer, index=True)
    season = Column(Integer, index=True)
    game_date = Column(String(10), index=True)
    game_start_utc = Column(DateTime, index=True)
    home_abbr = Column(String(5))
    away_abbr = Column(String(5))
    market = Column(String(40), index=True)
    selection = Column(String(60))
    line = Column(Float)
    model_probability = Column(Float)
    market_probability = Column(Float)
    ensemble_probability = Column(Float)
    elo_probability = Column(Float)
    confidence = Column(Float)
    upset_risk = Column(Float)
    upset_label = Column(String(20))
    upset_explanation = Column(JSON)
    model_market_gap = Column(Float)
    directional_disagreement = Column(Boolean)
    agreement_bucket = Column(String(30))
    model_dispersion = Column(Float)
    market_signal = Column(String(200))
    blend_policy = Column(JSON)
    data_completeness = Column(JSON)                  # que falta y desde cuando
    status = Column(String(20), default="published")  # published|provisional|blocked|superseded
    status_note = Column(Text)
    version = Column(Integer, default=1)
    parent_prediction_id = Column(Integer, index=True)
    revision_diff = Column(JSON)
    model_version = Column(String(60))
    feature_version = Column(String(60))
    data_snapshot = Column(String(200))
    sources = Column(JSON)
    prediction_timestamp = Column(DateTime, index=True)
    result = Column(String(20))                       # win|loss|push|void|pending
    correct = Column(Boolean)
    __table_args__ = (Index("ix_pred_game_mkt_ver", "game_id", "market", "version"),)


class PredictionSnapshot(Base):
    """Copia inmutable de las features usadas: permite reproducir la prediccion."""
    __tablename__ = "prediction_snapshots"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), index=True)
    features = Column(JSON)
    inputs = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT)
    name = Column(String(60))
    version = Column(String(30))
    market = Column(String(40))
    training_period = Column(String(40))
    validation_period = Column(String(40))
    test_period = Column(String(40))
    features = Column(JSON)
    hyperparameters = Column(JSON)
    metrics = Column(JSON)
    blend_policy = Column(JSON)
    artifact_path = Column(String(300))
    code_version = Column(String(60))
    is_production = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestRun(Base):
    __tablename__ = "backtests"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT)
    label = Column(String(80))
    config = Column(JSON)
    summary = Column(JSON)
    model_version_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataSource(Base):
    __tablename__ = "data_sources"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT)
    name = Column(String(60), unique=True)
    base_url = Column(String(300))
    docs_url = Column(String(300))
    domains = Column(JSON)
    notes = Column(Text)


class DataSourceLog(Base):
    __tablename__ = "data_source_logs"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT, index=True)
    source = Column(String(60), index=True)
    domain = Column(String(40), index=True)
    url = Column(String(500))
    status = Column(String(16), index=True)
    records = Column(Integer, default=0)
    retrieved_at = Column(DateTime, index=True)
    data_timestamp = Column(DateTime)
    checksum = Column(String(64))
    error = Column(Text)
    notes = Column(Text)


class SourceConflict(Base):
    """Discrepancias entre fuentes. Nunca se resuelven en silencio."""
    __tablename__ = "source_conflicts"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT)
    domain = Column(String(40))
    key = Column(String(120))
    values = Column(JSON)
    chosen = Column(String(60))
    policy = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id = Column(Integer, primary_key=True)
    sport = Column(String(4), default=SPORT)
    kind = Column(String(40), index=True)
    trigger = Column(String(40))
    status = Column(String(20), default="running")
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    detail = Column(JSON)
    error = Column(Text)
