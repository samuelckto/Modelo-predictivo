"""Features as-of. El orden de las operaciones ES la garantia anti-leakage.

Se recorre el historico en orden cronologico estricto y, para cada partido:

    1. se LEE el estado acumulado de ambos equipos  -> fila de features
    2. solo DESPUES se actualiza ese estado con el marcador del partido

Por construccion, ninguna feature de un partido puede contener informacion de ese
partido ni de ninguno posterior. No hay ventanas deslizantes calculadas sobre el
dataset completo, que es donde suele colarse el leakage.

`available_at` de cada fila es el corte: la hora de saque si la fuente la da, o
las 00:00 del dia del partido si no. Ese valor se guarda en la fila y es lo que
audita `SOCCER/evaluation/leakage.py`.

Sobre el empate de fechas: dos partidos del mismo equipo el mismo dia no existen,
pero partidos distintos del mismo dia se ordenan por (fecha, match_id) para que el
resultado sea reproducible.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import date, datetime

import pandas as pd
from sqlalchemy import select

from SOCCER.db import SoccerMatch, init_db, session_scope
from SOCCER.ingestion.teams import Resolver
from SOCCER.leagues import COPAS
from shared.paths import SOCCER_FEATURES_DIR, assert_writable

FEATURES_PARQUET = SOCCER_FEATURES_DIR / "soccer_features.parquet"

VENTANAS = (3, 5, 10)              # se comparan por validacion OOS, no se elige a dedo
ELO_INICIAL = 1500.0
ELO_K = 20.0
ELO_REGRESION = 0.25               # regresion a la media entre temporadas
MEDIA_GOLES_PREVIA = 1.35          # prior neutro por equipo mientras no hay historia
CORNERS_PREVIOS = 5.2              # prior de corners por equipo y partido
# Amarillas por equipo y partido. NO es un numero elegido: es la media real de
# los 25 237 partidos con tarjetas en la base (50 475 observaciones de equipo,
# media 2.0314, varianza 1.8938). Solo se usa mientras un equipo no tiene
# historia propia.
TARJETAS_PREVIAS = 2.03

ID_COLS = ["match_id", "league_code", "season", "match_date", "kickoff_utc", "available_at",
           "home_team", "away_team", "home_team_fuente", "away_team_fuente"]
TARGET_COLS = ["home_goals", "away_goals", "total_goals", "resultado", "btts", "goal_diff",
               "home_corners", "away_corners", "total_corners",
               "home_yellow", "away_yellow", "total_yellow"]


class EstadoEquipo:
    """Acumulados incrementales de un equipo. Solo se actualiza con partidos pasados."""

    def __init__(self) -> None:
        self.elo = ELO_INICIAL
        self.partidos = 0
        self.gf = 0.0
        self.gc = 0.0
        self.gf_casa = 0.0
        self.gc_casa = 0.0
        self.partidos_casa = 0
        self.gf_fuera = 0.0
        self.gc_fuera = 0.0
        self.partidos_fuera = 0
        self.btts = 0
        self.over25 = 0
        self.porteria_cero = 0
        self.sin_marcar = 0
        self.puntos = 0
        self.ult_fecha: date | None = None
        self.ventanas = {v: deque(maxlen=v) for v in VENTANAS}   # (gf, gc, pts)
        self.season = None
        # --- corners: contador aparte, porque no todos los partidos los traen ---
        self.c_partidos = 0
        self.cf = 0.0                 # corners a favor
        self.cc = 0.0                 # corners en contra
        self.cf_casa = 0.0
        self.cc_casa = 0.0
        self.c_partidos_casa = 0
        self.cf_fuera = 0.0
        self.cc_fuera = 0.0
        self.c_partidos_fuera = 0
        self.c_ventanas = {v: deque(maxlen=v) for v in VENTANAS}  # (cf, cc)
        self.tiros = 0.0
        self.tiros_contra = 0.0
        # --- tarjetas: mismo tratamiento que corners y por la misma razon.
        # Solo 5 de las 8 competiciones traen amarillas, asi que su contador va
        # aparte: si se mezclara con `partidos`, un equipo de Champions con 40
        # partidos y 0 con tarjetas parecería tener una media de 0 amarillas.
        self.t_partidos = 0
        self.tf = 0.0                 # tarjetas propias
        self.tc = 0.0                 # tarjetas del rival
        self.tf_casa = 0.0
        self.t_partidos_casa = 0
        self.tf_fuera = 0.0
        self.t_partidos_fuera = 0
        self.faltas = 0.0
        self.t_ventanas = {v: deque(maxlen=v) for v in VENTANAS}   # (tf, tc)

    # -- lectura (antes del partido) ----------------------------------------
    def medias(self) -> dict:
        n = max(self.partidos, 1)
        out = {
            "n_partidos": self.partidos,
            "gf_media": self.gf / n if self.partidos else MEDIA_GOLES_PREVIA,
            "gc_media": self.gc / n if self.partidos else MEDIA_GOLES_PREVIA,
            "btts_rate": self.btts / n if self.partidos else 0.5,
            "over25_rate": self.over25 / n if self.partidos else 0.5,
            "cs_rate": self.porteria_cero / n if self.partidos else 0.25,
            "fts_rate": self.sin_marcar / n if self.partidos else 0.25,
            "ppp": self.puntos / n if self.partidos else 1.0,
        }
        nc, nf = max(self.partidos_casa, 1), max(self.partidos_fuera, 1)
        out["gf_casa"] = self.gf_casa / nc if self.partidos_casa else MEDIA_GOLES_PREVIA
        out["gc_casa"] = self.gc_casa / nc if self.partidos_casa else MEDIA_GOLES_PREVIA
        out["gf_fuera"] = self.gf_fuera / nf if self.partidos_fuera else MEDIA_GOLES_PREVIA
        out["gc_fuera"] = self.gc_fuera / nf if self.partidos_fuera else MEDIA_GOLES_PREVIA
        for v, dq in self.ventanas.items():
            if dq:
                out[f"gf_{v}"] = sum(x[0] for x in dq) / len(dq)
                out[f"gc_{v}"] = sum(x[1] for x in dq) / len(dq)
                out[f"pts_{v}"] = sum(x[2] for x in dq) / len(dq)
                out[f"n_{v}"] = len(dq)
            else:
                out[f"gf_{v}"] = MEDIA_GOLES_PREVIA
                out[f"gc_{v}"] = MEDIA_GOLES_PREVIA
                out[f"pts_{v}"] = 1.0
                out[f"n_{v}"] = 0
        return out

    def medias_corners(self) -> dict:
        """Solo cuenta partidos que TRAIAN corners. Sin ellos devuelve el prior."""
        n = max(self.c_partidos, 1)
        out = {"c_n_partidos": self.c_partidos,
               "cf_media": self.cf / n if self.c_partidos else CORNERS_PREVIOS,
               "cc_media": self.cc / n if self.c_partidos else CORNERS_PREVIOS,
               "tiros_media": self.tiros / n if self.c_partidos else 12.0,
               "tiros_contra_media": self.tiros_contra / n if self.c_partidos else 12.0}
        nc, nf = max(self.c_partidos_casa, 1), max(self.c_partidos_fuera, 1)
        out["cf_casa"] = self.cf_casa / nc if self.c_partidos_casa else CORNERS_PREVIOS
        out["cc_casa"] = self.cc_casa / nc if self.c_partidos_casa else CORNERS_PREVIOS
        out["cf_fuera"] = self.cf_fuera / nf if self.c_partidos_fuera else CORNERS_PREVIOS
        out["cc_fuera"] = self.cc_fuera / nf if self.c_partidos_fuera else CORNERS_PREVIOS
        for v, dq in self.c_ventanas.items():
            if dq:
                out[f"cf_{v}"] = sum(x[0] for x in dq) / len(dq)
                out[f"cc_{v}"] = sum(x[1] for x in dq) / len(dq)
                out[f"cn_{v}"] = len(dq)
            else:
                out[f"cf_{v}"] = CORNERS_PREVIOS
                out[f"cc_{v}"] = CORNERS_PREVIOS
                out[f"cn_{v}"] = 0
        return out

    def medias_tarjetas(self) -> dict:
        """Solo cuenta partidos que TRAIAN tarjetas. Sin ellos devuelve el prior."""
        n = max(self.t_partidos, 1)
        out = {"t_n_partidos": self.t_partidos,
               "tf_media": self.tf / n if self.t_partidos else TARJETAS_PREVIAS,
               "tc_media": self.tc / n if self.t_partidos else TARJETAS_PREVIAS,
               "faltas_media": self.faltas / n if self.t_partidos else 13.0}
        nc, nf = max(self.t_partidos_casa, 1), max(self.t_partidos_fuera, 1)
        out["tf_casa"] = self.tf_casa / nc if self.t_partidos_casa else TARJETAS_PREVIAS
        out["tf_fuera"] = self.tf_fuera / nf if self.t_partidos_fuera else TARJETAS_PREVIAS
        for v, dq in self.t_ventanas.items():
            if dq:
                out[f"tf_{v}"] = sum(x[0] for x in dq) / len(dq)
                out[f"tc_{v}"] = sum(x[1] for x in dq) / len(dq)
                out[f"tn_{v}"] = len(dq)
            else:
                out[f"tf_{v}"] = TARJETAS_PREVIAS
                out[f"tc_{v}"] = TARJETAS_PREVIAS
                out[f"tn_{v}"] = 0
        return out

    def registrar_tarjetas(self, tf: int, tc: int, faltas, en_casa: bool) -> None:
        self.t_partidos += 1
        self.tf += tf
        self.tc += tc
        if faltas is not None:
            self.faltas += faltas
        if en_casa:
            self.t_partidos_casa += 1
            self.tf_casa += tf
        else:
            self.t_partidos_fuera += 1
            self.tf_fuera += tf
        for dq in self.t_ventanas.values():
            dq.append((tf, tc))

    def registrar_corners(self, cf: int, cc: int, tiros, tiros_contra, en_casa: bool) -> None:
        self.c_partidos += 1
        self.cf += cf
        self.cc += cc
        if tiros is not None:
            self.tiros += tiros
        if tiros_contra is not None:
            self.tiros_contra += tiros_contra
        if en_casa:
            self.c_partidos_casa += 1
            self.cf_casa += cf
            self.cc_casa += cc
        else:
            self.c_partidos_fuera += 1
            self.cf_fuera += cf
            self.cc_fuera += cc
        for dq in self.c_ventanas.values():
            dq.append((cf, cc))

    def descanso(self, f: date) -> int | None:
        return (f - self.ult_fecha).days if self.ult_fecha else None

    # -- actualizacion (despues de emitir la fila) --------------------------
    def nueva_temporada(self, season) -> None:
        if self.season is not None and season != self.season:
            self.elo = self.elo + ELO_REGRESION * (ELO_INICIAL - self.elo)
        self.season = season

    def registrar(self, gf: int, gc: int, en_casa: bool, f: date) -> None:
        self.partidos += 1
        self.gf += gf
        self.gc += gc
        pts = 3 if gf > gc else 1 if gf == gc else 0
        self.puntos += pts
        if en_casa:
            self.partidos_casa += 1
            self.gf_casa += gf
            self.gc_casa += gc
        else:
            self.partidos_fuera += 1
            self.gf_fuera += gf
            self.gc_fuera += gc
        if gf > 0 and gc > 0:
            self.btts += 1
        if gf + gc > 2.5:
            self.over25 += 1
        if gc == 0:
            self.porteria_cero += 1
        if gf == 0:
            self.sin_marcar += 1
        for dq in self.ventanas.values():
            dq.append((gf, gc, pts))
        self.ult_fecha = f


def _elo_esperado(a: float, b: float, ventaja_local: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a + ventaja_local - b) / 400.0))


def _cutoff(m: SoccerMatch) -> datetime:
    """Corte anti-leakage: saque real si existe, 00:00 del dia si no (conservador)."""
    if m.kickoff_utc is not None:
        return m.kickoff_utc
    return datetime(m.match_date.year, m.match_date.month, m.match_date.day)


def build(progress=None) -> dict:
    """Recorre el historico en orden y escribe el parquet de features."""
    init_db()
    SOCCER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    assert_writable(SOCCER_FEATURES_DIR, "SOCCER")

    with session_scope() as s:
        partidos = s.execute(select(SoccerMatch).order_by(
            SoccerMatch.match_date, SoccerMatch.match_id)).scalars().all()
        # Nombre canonico por liga. Se resuelve en orden cronologico, asi que la
        # grafia mas antigua fija el canonico y las fuentes nuevas se enganchan a
        # ella en vez de crear un equipo duplicado sin historia.
        res = Resolver()
        datos = []
        for m in partidos:
            h, sh = res.resolver(m.league_code, m.home_team)
            a, sa = res.resolver(m.league_code, m.away_team)
            datos.append({
                "match_id": m.match_id, "league_code": m.league_code, "season": m.season,
                "match_date": m.match_date, "kickoff_utc": m.kickoff_utc,
                "available_at": _cutoff(m),
                "home_team": h, "away_team": a,
                "home_team_fuente": m.home_team, "away_team_fuente": m.away_team,
                "home_goals": m.home_goals, "away_goals": m.away_goals,
                "home_corners": m.home_corners, "away_corners": m.away_corners,
                "home_shots": m.home_shots, "away_shots": m.away_shots,
                "home_yellow": m.home_yellow, "away_yellow": m.away_yellow,
                "home_fouls": m.home_fouls, "away_fouls": m.away_fouls,
            })

    # Ventaja local por liga: se estima con lo YA visto, nunca con el futuro.
    ventaja: dict[str, float] = {}
    vistos_liga: dict[str, list[float]] = {}

    # El estado se guarda por CLUB, no por (liga, club). Es lo que permite que el
    # Arsenal llegue a la Champions con la historia de su Premier, y lo que hace
    # que los Elo de ligas distintas se vuelvan comparables al cruzarse en copa.
    estados: dict[str, EstadoEquipo] = {}
    filas = []
    for i, d in enumerate(datos):
        code, season = d["league_code"], d["season"]
        kh, ka = d["home_team"], d["away_team"]
        eh = estados.setdefault(kh, EstadoEquipo())
        ea = estados.setdefault(ka, EstadoEquipo())
        if code not in COPAS:          # una copa no abre temporada para el club
            eh.nueva_temporada(season)
            ea.nueva_temporada(season)

        va = ventaja.get(code, 60.0)          # 60 pts Elo: prior estandar hasta tener datos
        mh, ma = eh.medias(), ea.medias()
        ch, ca = eh.medias_corners(), ea.medias_corners()
        th, ta = eh.medias_tarjetas(), ea.medias_tarjetas()
        p_elo = _elo_esperado(eh.elo, ea.elo, va)

        fila = {
            **{k: d[k] for k in ID_COLS},
            "elo_home": eh.elo, "elo_away": ea.elo, "elo_diff": eh.elo - ea.elo,
            "elo_p_home": p_elo, "ventaja_local_liga": va,
            "descanso_home": eh.descanso(d["match_date"]),
            "descanso_away": ea.descanso(d["match_date"]),
        }
        for tag, m in (("home", mh), ("away", ma)):
            for k, v in m.items():
                fila[f"{tag}_{k}"] = v
        for tag, m in (("home", ch), ("away", ca)):
            for k, v in m.items():
                fila[f"{tag}_{k}"] = v
        for tag, m in (("home", th), ("away", ta)):
            for k, v in m.items():
                fila[f"{tag}_{k}"] = v
        # Cruce de tarjetas: a un equipo lo amonestan por lo que hace EL, pero
        # tambien por como se juega el partido. Por eso se cruzan las propias
        # con las que provoca el rival, igual que con los corners.
        fila["tarjetas_esperadas_previas"] = (th["tf_casa"] + ta["tc_media"]
                                              + ta["tf_fuera"] + th["tc_media"]) / 2
        fila["tarjetas_medio_5"] = (th["tf_5"] + th["tc_5"] + ta["tf_5"] + ta["tc_5"]) / 2
        fila["tf_diff_5"] = th["tf_5"] - ta["tf_5"]
        fila["faltas_medio"] = (th["faltas_media"] + ta["faltas_media"]) / 2
        # cuantos partidos con tarjetas respaldan la fila: el gating lo usa
        fila["tarjetas_historial"] = min(th["t_n_partidos"], ta["t_n_partidos"])
        # cruce de corners: los saca quien ataca contra quien los concede
        fila["corners_esperados_previos"] = (ch["cf_casa"] + ca["cc_fuera"]
                                             + ca["cf_fuera"] + ch["cc_casa"]) / 2
        fila["cf_diff_5"] = ch["cf_5"] - ca["cf_5"]
        fila["corners_medio_5"] = (ch["cf_5"] + ch["cc_5"] + ca["cf_5"] + ca["cc_5"]) / 2
        fila["tiros_medio"] = (ch["tiros_media"] + ca["tiros_media"]) / 2
        # cuantos partidos con corners respaldan la fila: el gating lo usa
        fila["corners_historial"] = min(ch["c_n_partidos"], ca["c_n_partidos"])
        # cruces: la fuerza relativa es lo que mueve los goles
        fila["ataque_vs_defensa_home"] = mh["gf_casa"] * ma["gc_fuera"]
        fila["ataque_vs_defensa_away"] = ma["gf_fuera"] * mh["gc_casa"]
        fila["gf_diff_5"] = mh["gf_5"] - ma["gf_5"]
        fila["gc_diff_5"] = mh["gc_5"] - ma["gc_5"]
        fila["ppp_diff"] = mh["ppp"] - ma["ppp"]
        fila["btts_medio"] = (mh["btts_rate"] + ma["btts_rate"]) / 2
        fila["over25_medio"] = (mh["over25_rate"] + ma["over25_rate"]) / 2
        fila["goles_esperados_previos"] = (mh["gf_casa"] + ma["gc_fuera"]
                                           + ma["gf_fuera"] + mh["gc_casa"]) / 2
        # contexto de calendario (todo conocido antes del saque)
        for tag, e in (("home", eh), ("away", ea)):
            dr = e.descanso(d["match_date"])
            fila[f"{tag}_congestion"] = 1 if (dr is not None and dr <= 3) else 0
        fila["entre_semana"] = 1 if d["match_date"].weekday() in (1, 2, 3) else 0
        fila["mes"] = d["match_date"].month
        fila["tramo_temporada"] = min(max(eh.partidos, ea.partidos), 38) / 38.0

        # objetivos (NO son features: se separan en el entrenamiento)
        hg, ag = d["home_goals"], d["away_goals"]
        fila["home_goals"] = hg
        fila["away_goals"] = ag
        fila["total_goals"] = hg + ag
        fila["goal_diff"] = hg - ag
        fila["resultado"] = 0 if hg > ag else 1 if hg == ag else 2      # 0 local, 1 empate, 2 vis
        fila["btts"] = 1 if (hg > 0 and ag > 0) else 0
        hc, ac = d["home_corners"], d["away_corners"]
        fila["home_corners"] = hc
        fila["away_corners"] = ac
        fila["total_corners"] = (hc + ac) if (hc is not None and ac is not None) else None
        hy, ay = d["home_yellow"], d["away_yellow"]
        fila["home_yellow"] = hy
        fila["away_yellow"] = ay
        fila["total_yellow"] = (hy + ay) if (hy is not None and ay is not None) else None
        filas.append(fila)

        # ---- a partir de aqui SI se puede usar el resultado ----
        real = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margen = abs(hg - ag)
        mult = 1.0 if margen <= 1 else (1.5 if margen == 2 else (1.75 + (margen - 3) / 8))
        cambio = ELO_K * mult * (real - p_elo)
        eh.elo += cambio
        ea.elo -= cambio
        eh.registrar(hg, ag, True, d["match_date"])
        ea.registrar(ag, hg, False, d["match_date"])
        if hc is not None and ac is not None:
            eh.registrar_corners(hc, ac, d["home_shots"], d["away_shots"], True)
            ea.registrar_corners(ac, hc, d["away_shots"], d["home_shots"], False)
        if hy is not None and ay is not None:
            eh.registrar_tarjetas(hy, ay, d["home_fouls"], True)
            ea.registrar_tarjetas(ay, hy, d["away_fouls"], False)
        vistos_liga.setdefault(code, []).append(real)
        if len(vistos_liga[code]) >= 200:
            cuota = sum(vistos_liga[code]) / len(vistos_liga[code])
            cuota = min(max(cuota, 0.35), 0.75)
            ventaja[code] = -400 * math.log10(1 / cuota - 1)
        if progress and i % 5000 == 0 and i:
            progress(f"{i}/{len(datos)} partidos procesados")

    X = pd.DataFrame(filas)
    X.to_parquet(FEATURES_PARQUET, index=False)
    return {"partidos": len(X), "columnas": len(X.columns),
            "features": len([c for c in X.columns if c not in ID_COLS + TARGET_COLS]),
            "desde": str(X.match_date.min()), "hasta": str(X.match_date.max()),
            "archivo": str(FEATURES_PARQUET)}


def estado_actual():
    """Estado de cada equipo DESPUES de todo el historico. Base para predecir hoy.

    Devuelve (estados, ventaja_por_liga, resolver, ultima_fecha). Es el mismo
    recorrido cronologico que `build`, asi que un fixture futuro recibe
    exactamente las mismas features que habria recibido un partido historico.
    """
    init_db()
    with session_scope() as s:
        partidos = s.execute(select(SoccerMatch).order_by(
            SoccerMatch.match_date, SoccerMatch.match_id)).scalars().all()
        crudo = [(m.league_code, m.season, m.match_date, m.home_team, m.away_team,
                  m.home_goals, m.away_goals, m.home_corners, m.away_corners,
                  m.home_shots, m.away_shots,
                  m.home_yellow, m.away_yellow, m.home_fouls, m.away_fouls)
                 for m in partidos]
    res = Resolver()
    estados: dict[str, EstadoEquipo] = {}
    ventaja: dict[str, float] = {}
    vistos: dict[str, list[float]] = {}
    ultima: dict[str, date] = {}
    for (code, season, f, home, away, hg, ag, hc, ac, hs, as_, hy, ay, hf, af) in crudo:
        h, _ = res.resolver(code, home)
        a, _ = res.resolver(code, away)
        eh = estados.setdefault(h, EstadoEquipo())
        ea = estados.setdefault(a, EstadoEquipo())
        if code not in COPAS:
            eh.nueva_temporada(season)
            ea.nueva_temporada(season)
        p = _elo_esperado(eh.elo, ea.elo, ventaja.get(code, 60.0))
        real = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margen = abs(hg - ag)
        mult = 1.0 if margen <= 1 else (1.5 if margen == 2 else (1.75 + (margen - 3) / 8))
        cambio = ELO_K * mult * (real - p)
        eh.elo += cambio
        ea.elo -= cambio
        eh.registrar(hg, ag, True, f)
        ea.registrar(ag, hg, False, f)
        if hc is not None and ac is not None:
            eh.registrar_corners(hc, ac, hs, as_, True)
            ea.registrar_corners(ac, hc, as_, hs, False)
        if hy is not None and ay is not None:
            eh.registrar_tarjetas(hy, ay, hf, True)
            ea.registrar_tarjetas(ay, hy, af, False)
        vistos.setdefault(code, []).append(real)
        if len(vistos[code]) >= 200:
            c = min(max(sum(vistos[code]) / len(vistos[code]), 0.35), 0.75)
            ventaja[code] = -400 * math.log10(1 / c - 1)
        ultima[code] = f if code not in ultima or f > ultima[code] else ultima[code]
    return estados, ventaja, res, ultima


def fila_fixture(estados, ventaja, resolver, code: str, home: str, away: str,
                 fecha: date, kickoff=None) -> dict:
    """Construye la fila de features de un partido AUN NO JUGADO."""
    h, sim_h = resolver.resolver(code, home)
    a, sim_a = resolver.resolver(code, away)
    eh = estados.get(h) or EstadoEquipo()
    ea = estados.get(a) or EstadoEquipo()
    va = ventaja.get(code, 60.0)
    mh, ma = eh.medias(), ea.medias()
    ch, ca = eh.medias_corners(), ea.medias_corners()
    th, ta = eh.medias_tarjetas(), ea.medias_tarjetas()
    fila = {
        "match_id": None, "league_code": code, "season": None,
        "match_date": fecha, "kickoff_utc": kickoff,
        "available_at": kickoff or datetime(fecha.year, fecha.month, fecha.day),
        "home_team": h, "away_team": a,
        "home_team_fuente": home, "away_team_fuente": away,
        "elo_home": eh.elo, "elo_away": ea.elo, "elo_diff": eh.elo - ea.elo,
        "elo_p_home": _elo_esperado(eh.elo, ea.elo, va), "ventaja_local_liga": va,
        "descanso_home": eh.descanso(fecha), "descanso_away": ea.descanso(fecha),
    }
    for tag, m in (("home", mh), ("away", ma), ("home", ch), ("away", ca),
                   ("home", th), ("away", ta)):
        for k, v in m.items():
            fila[f"{tag}_{k}"] = v
    fila["ataque_vs_defensa_home"] = mh["gf_casa"] * ma["gc_fuera"]
    fila["ataque_vs_defensa_away"] = ma["gf_fuera"] * mh["gc_casa"]
    fila["gf_diff_5"] = mh["gf_5"] - ma["gf_5"]
    fila["gc_diff_5"] = mh["gc_5"] - ma["gc_5"]
    fila["ppp_diff"] = mh["ppp"] - ma["ppp"]
    fila["btts_medio"] = (mh["btts_rate"] + ma["btts_rate"]) / 2
    fila["over25_medio"] = (mh["over25_rate"] + ma["over25_rate"]) / 2
    fila["goles_esperados_previos"] = (mh["gf_casa"] + ma["gc_fuera"]
                                       + ma["gf_fuera"] + mh["gc_casa"]) / 2
    fila["corners_esperados_previos"] = (ch["cf_casa"] + ca["cc_fuera"]
                                         + ca["cf_fuera"] + ch["cc_casa"]) / 2
    fila["cf_diff_5"] = ch["cf_5"] - ca["cf_5"]
    fila["corners_medio_5"] = (ch["cf_5"] + ch["cc_5"] + ca["cf_5"] + ca["cc_5"]) / 2
    fila["tiros_medio"] = (ch["tiros_media"] + ca["tiros_media"]) / 2
    fila["corners_historial"] = min(ch["c_n_partidos"], ca["c_n_partidos"])
    fila["tarjetas_esperadas_previas"] = (th["tf_casa"] + ta["tc_media"]
                                          + ta["tf_fuera"] + th["tc_media"]) / 2
    fila["tarjetas_medio_5"] = (th["tf_5"] + th["tc_5"] + ta["tf_5"] + ta["tc_5"]) / 2
    fila["tf_diff_5"] = th["tf_5"] - ta["tf_5"]
    fila["faltas_medio"] = (th["faltas_media"] + ta["faltas_media"]) / 2
    fila["tarjetas_historial"] = min(th["t_n_partidos"], ta["t_n_partidos"])
    for tag, e in (("home", eh), ("away", ea)):
        dr = e.descanso(fecha)
        fila[f"{tag}_congestion"] = 1 if (dr is not None and dr <= 3) else 0
    fila["entre_semana"] = 1 if fecha.weekday() in (1, 2, 3) else 0
    fila["mes"] = fecha.month
    fila["tramo_temporada"] = min(max(eh.partidos, ea.partidos), 38) / 38.0
    fila["_similitud_home"] = sim_h
    fila["_similitud_away"] = sim_a
    fila["_partidos_home"] = eh.partidos
    fila["_partidos_away"] = ea.partidos
    return fila


def familias() -> dict[str, list[str]]:
    """Grupos de features para la ablacion por familias."""
    return {
        "elo": ["elo_home", "elo_away", "elo_diff", "elo_p_home", "ventaja_local_liga"],
        "ataque_defensa": ["home_gf_media", "home_gc_media", "away_gf_media", "away_gc_media",
                           "home_gf_casa", "home_gc_casa", "away_gf_fuera", "away_gc_fuera",
                           "ataque_vs_defensa_home", "ataque_vs_defensa_away",
                           "goles_esperados_previos"],
        "forma": [f"{t}_{k}_{v}" for t in ("home", "away") for k in ("gf", "gc", "pts")
                  for v in VENTANAS] + ["gf_diff_5", "gc_diff_5", "ppp_diff",
                                        "home_ppp", "away_ppp"],
        "corners": ["home_cf_media", "home_cc_media", "away_cf_media", "away_cc_media",
                    "home_cf_casa", "home_cc_casa", "away_cf_fuera", "away_cc_fuera",
                    "corners_esperados_previos", "cf_diff_5", "corners_medio_5",
                    "tiros_medio", "home_tiros_media", "away_tiros_media"]
                   + [f"{t}_{k}_{v}" for t in ("home", "away") for k in ("cf", "cc")
                      for v in VENTANAS],
        "tarjetas": ["home_tf_media", "home_tc_media", "away_tf_media", "away_tc_media",
                     "home_tf_casa", "away_tf_fuera", "tarjetas_esperadas_previas",
                     "tf_diff_5", "tarjetas_medio_5", "faltas_medio",
                     "home_faltas_media", "away_faltas_media"]
                    + [f"{t}_{k}_{v}" for t in ("home", "away") for k in ("tf", "tc")
                       for v in VENTANAS],
        "mercado_historico": ["home_btts_rate", "away_btts_rate", "btts_medio",
                              "home_over25_rate", "away_over25_rate", "over25_medio",
                              "home_cs_rate", "away_cs_rate", "home_fts_rate", "away_fts_rate"],
        "contexto": ["descanso_home", "descanso_away", "home_congestion", "away_congestion",
                     "entre_semana", "mes", "tramo_temporada",
                     "home_n_partidos", "away_n_partidos"],
    }


def numeric(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
