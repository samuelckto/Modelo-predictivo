"""Utilidades de tiempo. Todo se guarda en UTC naive para poder comparar timestamps."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def to_utc(dt: datetime, tz: str) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def iso(dt) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


def json_safe(o):
    """Convierte escalares de numpy/pandas y fechas en valores serializables."""
    import numpy as np
    import pandas as pd
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [json_safe(v) for v in o.tolist()]
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o
