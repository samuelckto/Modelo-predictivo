"""Ligas soportadas en v1 y de donde sale cada cosa.

Una liga tiene DOS fuentes independientes, y esa separacion es deliberada:

- `historico`: archivo del espejo de GitHub, para ENTRENAR y VALIDAR.
- `odds_key` : clave de The Odds API, para el CALENDARIO y las CUOTAS de AHORA.

Una liga puede tener la segunda sin la primera (caso Saudi): se muestran sus
partidos y sus cuotas, pero no se entrena ni se publica pick. `min_partidos` es
el umbral a partir del cual el pipeline la activa solo, sin tocar codigo.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_PARTIDOS_ENTRENAR = 1500          # ~4 temporadas completas de una liga grande


@dataclass(frozen=True)
class Liga:
    code: str                          # identificador interno estable
    nombre: str
    pais: str
    historico: str | None              # archivo en el espejo footballcsv, None = sin historico
    odds_key: str                      # clave de The Odds API
    equipos: int                       # equipos por temporada (referencia, no se usa para modelar)


LIGAS: tuple[Liga, ...] = (
    Liga("EPL", "Premier League", "Inglaterra", "eng.1", "soccer_epl", 20),
    Liga("LALIGA", "LaLiga", "Espana", "es.1", "soccer_spain_la_liga", 20),
    Liga("SERIEA", "Serie A", "Italia", "it.1", "soccer_italy_serie_a", 20),
    Liga("BUNDES", "Bundesliga", "Alemania", "de.1", "soccer_germany_bundesliga", 18),
    Liga("LIGUE1", "Ligue 1", "Francia", "fr.1", "soccer_france_ligue_one", 18),
    Liga("LIGAMX", "Liga MX", "Mexico", "mx.1", "soccer_mexico_ligamx", 18),
    # Champions League: no es una liga, es una COPA con equipos de muchos paises.
    # Su historico vive en otro repositorio (openfootball/champions-league) y por
    # eso `historico` va en None: lo carga `ingestion/cups.py`, no el espejo.
    Liga("UCL", "Champions League", "Europa", None, "soccer_uefa_champs_league", 36),
    # Saudi: The Odds API si la cubre, pero no hay historico gratuito suficiente.
    # No se inventa: entra como INSUFFICIENT DATA hasta acumular partidos propios.
    Liga("SAUDI", "Saudi Pro League", "Arabia Saudita", None,
         "soccer_saudi_arabia_pro_league", 18),
)

# Competiciones donde los equipos vienen de otras ligas. Su fuerza NO se calcula
# aparte: se hereda del club, que es el mismo en su liga nacional y aqui.
COPAS = ("UCL",)

POR_CODE = {x.code: x for x in LIGAS}
POR_ODDS_KEY = {x.odds_key: x for x in LIGAS}
CODES = tuple(x.code for x in LIGAS)
CON_HISTORICO = tuple(x.code for x in LIGAS if x.historico)


def liga(code: str) -> Liga:
    try:
        return POR_CODE[code.upper()]
    except KeyError:
        raise ValueError(f"liga desconocida: {code}") from None
