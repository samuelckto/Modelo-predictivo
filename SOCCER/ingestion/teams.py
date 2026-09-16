"""Nombre canonico de equipo. Sin esto el modulo no sirve para predecir hoy.

El historico viene de dos fuentes que escriben distinto el mismo club:

    footballcsv   'Arsenal'      'Man United'    'Club America'
    openfootball  'Arsenal FC'   'Manchester United FC'  'CF America'
    The Odds API  'Arsenal'      'Manchester United'     'Club America'

Si no se unifican, un equipo pierde toda su historia al cambiar de fuente y sus
features vuelven al prior. Eso se detecto en el walk-forward: el log loss de
2024-25 se disparo de 1.06 a 1.15 justo donde cambia la fuente.

La resolucion es en dos pasos y siempre DENTRO de la misma liga:
  1. normalizar (sin acentos, sin sufijos de club, sin ruido)
  2. si no hay coincidencia exacta, similitud de tokens con umbral alto

Nunca se fuerza una coincidencia: por debajo del umbral se devuelve el nombre
normalizado tal cual y queda registrado como equipo distinto, que es preferible a
fusionar dos clubes por error.
"""
from __future__ import annotations

import re
import unicodedata


# Palabras que no distinguen a un club de otro.
# OJO: 'united' y 'city' NO van aqui. Quitarlas fusionaba Manchester United con
# Manchester City en un solo equipo, que es peor que no unificar nada.
RUIDO = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "rc", "cd", "ud", "sd", "cs",
    "club", "futbol", "football", "calcio", "sv", "vfl", "vfb", "tsg",
    "fsv", "sge", "bsc", "kv", "rcd", "ca", "ssc", "ogc", "rcd", "cfc",
    "de", "del", "y", "and", "the",
    # sufijos que aparecen sobre todo en los nombres largos de competicion europea
    "bc", "sfp", "pae", "sk", "afc", "fk", "sc", "cp", "ac", "as", "aek",
}
# Codigos de pais que openfootball anade en competiciones europeas:
# 'Arsenal FC (ENG)'. Se quitan del nombre, pero se guardan por si algun dia hay
# que desempatar dos clubes con el mismo nombre en paises distintos.
PAIS = re.compile(r"\(([A-Z]{3})\)\s*$")
# Casos que la normalizacion no resuelve sola (nombres realmente distintos).
ALIAS = {
    "EPL": {"man united": "manchester united", "man utd": "manchester united",
            "man city": "manchester city", "spurs": "tottenham hotspur",
            "wolves": "wolverhampton wanderers", "nottm forest": "nottingham forest",
            "sheffield united": "sheffield utd", "west brom": "west bromwich albion",
            "newcastle": "newcastle united", "leeds": "leeds united",
            "west ham": "west ham united", "brighton": "brighton hove albion",
            "bournemouth": "bournemouth", "leicester": "leicester city",
            "norwich": "norwich city", "hull": "hull city", "stoke": "stoke city",
            "swansea": "swansea city", "cardiff": "cardiff city", "luton": "luton town",
            "ipswich": "ipswich town", "birmingham": "birmingham city",
            "huddersfield": "huddersfield town", "blackburn": "blackburn rovers",
            "qpr": "queens park rangers"},
    "LALIGA": {"ath bilbao": "athletic bilbao", "ath madrid": "atletico madrid",
               "espanol": "espanyol", "sociedad": "real sociedad",
               "vallecano": "rayo vallecano", "betis": "real betis",
               "celta": "celta vigo", "la coruna": "deportivo la coruna",
               "sp gijon": "sporting gijon", "alaves": "alaves",
               "espanyol barcelona": "espanyol", "rcd espanyol": "espanyol",
               "athletic": "athletic bilbao", "athletic club": "athletic bilbao",
               "real betis balompie": "real betis", "villarreal": "villarreal",
               "real valladolid": "valladolid", "cadiz": "cadiz",
               "real sociedad": "real sociedad"},
    "SERIEA": {"inter": "internazionale", "milan": "ac milan",
               "verona": "hellas verona", "spal": "spal"},
    "BUNDES": {"bayern munich": "bayern munchen", "ein frankfurt": "eintracht frankfurt",
               "dortmund": "borussia dortmund", "mgladbach": "borussia monchengladbach",
               "m gladbach": "borussia monchengladbach", "leverkusen": "bayer leverkusen",
               "hertha": "hertha berlin", "schalke": "schalke 04",
               "hoffenheim": "1899 hoffenheim", "stuttgart": "vfb stuttgart",
               "wolfsburg": "vfl wolfsburg", "werder bremen": "werder bremen",
               "fc koln": "koln", "cologne": "koln", "hamburg": "hamburger sv",
               "mainz": "mainz 05", "union berlin": "union berlin",
               "rb leipzig": "rb leipzig", "leipzig": "rb leipzig"},
    "LIGUE1": {"paris sg": "paris saint germain", "psg": "paris saint germain",
               "st etienne": "saint etienne", "marseille": "olympique marseille",
               "lyon": "olympique lyonnais", "nimes": "nimes olympique",
               "clermont": "clermont foot",
               "stade rennais": "rennes", "stade rennais fc": "rennes",
               "stade brestois": "brest", "stade brestois 29": "brest",
               "stade de reims": "reims", "stade reims": "reims",
               "losc lille": "lille", "fco dijon": "dijon"},
    "LIGAMX": {"america": "club america", "u n a m pumas": "pumas unam",
               "unam pumas": "pumas unam", "u a n l tigres": "tigres uanl",
               "uanl tigres": "tigres uanl", "guadalajara chivas": "guadalajara",
               "deportivo guadalajara": "guadalajara", "chivas": "guadalajara",
               "atlas guadalajara": "atlas", "cruz azul": "cruz azul",
               "monterrey": "monterrey", "cf monterrey": "monterrey",
               "santos laguna": "santos laguna", "leon": "leon", "club leon": "leon",
               "pachuca": "pachuca", "cf pachuca": "pachuca", "toluca": "toluca",
               "deportivo toluca": "toluca", "necaxa": "necaxa", "club necaxa": "necaxa",
               "tijuana": "tijuana", "club tijuana": "tijuana",
               "queretaro": "queretaro", "gallos blancos": "queretaro",
               "juarez": "juarez", "fc juarez": "juarez",
               "mazatlan": "mazatlan", "mazatlan fc": "mazatlan",
               "puebla": "puebla", "puebla fc": "puebla",
               "san luis": "atletico san luis", "atl san luis": "atletico san luis",
               "atletico san luis": "atletico san luis"},
    "SAUDI": {},
}
# Tabla combinada. La usan las COPAS (Champions), donde el mismo club llega con
# el nombre largo europeo y hay que reconocerlo contra el alias de su liga.
_GLOBAL = {}
for _tabla in ALIAS.values():
    _GLOBAL.update(_tabla)

UMBRAL = 0.80
MARGEN = 0.08          # ventaja minima sobre el segundo candidato para aceptar la union


def pais_de(nombre: str) -> str | None:
    m = PAIS.search(str(nombre or "").strip())
    return m.group(1) if m else None


def normaliza(nombre: str) -> str:
    s = PAIS.sub("", str(nombre or "").strip())           # 'Arsenal FC (ENG)' -> 'Arsenal FC'
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = s.replace("'", "")             # Nott'm -> nottm, no 'nott m'
    s = re.sub(r"[\.\-_/,&]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(nombre: str) -> list[str]:
    return [t for t in normaliza(nombre).split() if t not in RUIDO and not t.isdigit()]


def clave(nombre: str, liga: str) -> str:
    """Forma comparable: alias, tokens sin ruido, y alias otra vez.

    La segunda pasada importa: 'RCD Espanyol Barcelona' no esta en la tabla, pero
    al quitar 'rcd' queda 'espanyol barcelona', que si lo esta.
    """
    propia = ALIAS.get(liga.upper())
    # En una copa el equipo NO es de esa competicion: viene de su liga. Por eso
    # se consulta la tabla combinada y no una tabla vacia.
    tabla = propia if propia else _GLOBAL
    n = normaliza(nombre)
    n = tabla.get(n, n)
    toks = [t for t in n.split() if t not in RUIDO and not t.isdigit()]
    if not toks:                       # el nombre era todo ruido: se conserva entero
        toks = n.split()
    k = " ".join(toks)
    k = tabla.get(k, k)
    return " ".join(t for t in k.split() if t not in RUIDO) or k


def _similitud(a: str, b: str) -> float:
    """Comparacion POR TOKENS, nunca por cadena.

    La similitud de cadena daba 0.81 entre 'manchester united' y 'manchester city'
    y llegaba a fusionarlos. Por tokens da 0.5 y quedan separados, que es lo
    correcto. La contencion ('espanyol' dentro de 'espanyol barcelona') si vale,
    pero solo la acepta el resolver cuando ningun otro candidato compite: de eso
    se encarga la regla de margen.
    """
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    contencion = len(ta & tb) / min(len(ta), len(tb))
    return max(jaccard, contencion)


class Resolver:
    """Mapea cualquier grafia al club canonico. El pool es GLOBAL, no por liga.

    Tiene que serlo: el Arsenal de la Champions es el mismo Arsenal de la Premier.
    Con pools separados el club llegaba a la copa sin historia y sus features
    volvian al prior. Ademas, los partidos de Champions son los unicos que
    conectan los Elo de ligas distintas y los hacen comparables entre si.

    El riesgo de un pool global es fusionar dos clubes homonimos de paises
    distintos. Se comprobo sobre los datos reales: 221 clubes, 0 nombres
    repetidos entre ligas. Aun asi, la regla de margen sigue protegiendo.
    """

    def __init__(self) -> None:
        self.canonicos: dict[str, str] = {}              # clave -> canonico (global)
        self.cache: dict[tuple[str, str], str] = {}
        self.paises: dict[str, str] = {}                 # canonico -> pais si se supo

    def registrar(self, liga: str, nombre: str) -> str:
        k = clave(nombre, liga)
        self.canonicos.setdefault(k, k)
        return self.canonicos[k]

    def resolver(self, liga: str, nombre: str) -> tuple[str, float]:
        """Devuelve (canonico, similitud). 1.0 = coincidencia exacta tras normalizar."""
        liga = liga.upper()
        ck = (liga, nombre)
        if ck in self.cache:
            return self.cache[ck], 1.0
        k = clave(nombre, liga)
        pais = pais_de(nombre)
        conocidos = self.canonicos
        if k in conocidos:
            self.cache[ck] = conocidos[k]
            if pais:
                self.paises.setdefault(conocidos[k], pais)
            return conocidos[k], 1.0
        puntuadas = sorted(((_similitud(k, c), c) for c in conocidos), reverse=True)
        mejor_s, mejor = puntuadas[0] if puntuadas else (0.0, None)
        segunda = puntuadas[1][0] if len(puntuadas) > 1 else 0.0
        # Se exige margen sobre el segundo candidato: sin el, 'manchester' encajaria
        # igual de bien con 'manchester united' que con 'manchester city' y se
        # fusionarian dos clubes distintos.
        if mejor is not None and mejor_s >= UMBRAL and (mejor_s - segunda) >= MARGEN:
            self.cache[ck] = conocidos[mejor]
            if pais:
                self.paises.setdefault(conocidos[mejor], pais)
            return conocidos[mejor], mejor_s
        # sin coincidencia fiable: se registra como equipo nuevo, no se fuerza
        conocidos[k] = k
        self.cache[ck] = k
        if pais:
            self.paises.setdefault(k, pais)
        return k, 0.0
