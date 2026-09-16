"""Registro de fuentes y trazabilidad.

Cada dato importante que entra al sistema debe poder responder:
  de donde salio (source, url), cuando se descargo (retrieved_at),
  a que momento corresponde el dato (data_timestamp), y si la descarga fallo.

Politica de conflicto entre fuentes: NUNCA se decide en silencio. Se guarda el
valor de cada fuente y se aplica la prioridad declarada aqui, dejando registro.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from shared.timeutil import utcnow

# Prioridad declarada por deporte y dominio. Menor numero = mas prioritaria.
PRIORITY = {
    "MLB": {
        "schedule":   ["mlb_stats_api"],
        "boxscore":   ["mlb_stats_api"],
        "statcast":   ["baseball_savant", "mlb_stats_api"],
        "pitching":   ["mlb_stats_api", "baseball_savant"],
        "batting":    ["mlb_stats_api", "baseball_savant"],
        "lineups":    ["mlb_stats_api"],
        "injuries":   ["mlb_stats_api"],
        "odds":       ["the_odds_api"],
        "weather":    ["open_meteo"],
        "park":       ["derived_from_boxscores"],
    },
    "NFL": {
        "schedule":  ["nflverse"],
        "pbp":       ["nflverse"],
        "rosters":   ["nflverse"],
        "injuries":  ["nflverse"],
        "odds":      ["the_odds_api", "nflverse"],
    },
}

DOCS = {
    "mlb_stats_api": "https://statsapi.mlb.com/api/v1/",
    "baseball_savant": "https://baseballsavant.mlb.com/statcast_search",
    "the_odds_api": "https://api.the-odds-api.com/v4",
    "open_meteo": "https://api.open-meteo.com/v1/forecast",
    "nflverse": "https://github.com/nflverse/nflverse-data",
}


@dataclass
class Fetch:
    """Resultado de una descarga. `status` es ok | empty | error | skipped."""
    sport: str
    source: str
    domain: str
    url: str
    status: str = "ok"
    records: int = 0
    retrieved_at: object = field(default_factory=utcnow)
    data_timestamp: object | None = None
    error: str | None = None
    checksum: str | None = None
    notes: str | None = None

    def to_row(self) -> dict:
        return {"sport": self.sport, "source": self.source, "domain": self.domain,
                "url": self.url, "status": self.status, "records": int(self.records),
                "retrieved_at": self.retrieved_at, "data_timestamp": self.data_timestamp,
                "error": (self.error or "")[:800], "checksum": self.checksum,
                "notes": self.notes}


def checksum(payload) -> str:
    b = payload if isinstance(payload, (bytes, bytearray)) else str(payload).encode("utf-8", "replace")
    return hashlib.sha256(b).hexdigest()[:32]


def resolve_conflict(sport: str, domain: str, values: dict):
    """values = {source: valor}. Devuelve (valor_elegido, informe_de_conflicto|None).

    Si dos fuentes discrepan se elige por PRIORITY y se DEVUELVE el conflicto para
    que quede registrado. Nunca se promedia ni se ignora en silencio.
    """
    vals = {s: v for s, v in values.items() if v is not None}
    if not vals:
        return None, None
    order = PRIORITY.get(sport.upper(), {}).get(domain, [])
    rank = {s: i for i, s in enumerate(order)}
    chosen_src = sorted(vals, key=lambda s: (rank.get(s, 99), s))[0]
    distinct = {repr(v) for v in vals.values()}
    conflict = None
    if len(distinct) > 1:
        conflict = {"sport": sport, "domain": domain, "values": {s: v for s, v in vals.items()},
                    "chosen": chosen_src, "policy": order or "sin prioridad declarada"}
    return vals[chosen_src], conflict
