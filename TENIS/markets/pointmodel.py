"""Modelo punto a punto de tenis (cerrado, sin simulacion Monte Carlo).

Dadas p1 = prob. de que el jugador 1 gane un punto CON SU SAQUE y p2 la del
jugador 2, se calculan de forma exacta:
  * P(ganar un juego al saque)            -> formula binomial con deuce
  * P(ganar el tie-break)                 -> recursion sobre estados
  * P(ganar el set) y P(ganar el partido)  -> recursion sobre juegos y sets
  * distribucion del numero de juegos del partido (para totales y handicap)

Todo es analitico salvo la distribucion de juegos del set, que se obtiene por
recursion completa sobre marcadores. Nada de esto es un ajuste estadistico: es
la consecuencia matematica de p1 y p2. Lo que SI se estima con datos es p1/p2.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=100_000)
def game_prob(p: float) -> float:
    """P(ganar el juego al saque) ganando cada punto con prob. p."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = 1 - p
    # marcadores directos: 40-0, 40-15, 40-30 y deuce
    p40_0 = p ** 4
    p40_15 = 4 * p ** 4 * q
    p40_30 = 10 * p ** 4 * q ** 2
    p_deuce = 20 * p ** 3 * q ** 3
    win_deuce = p ** 2 / (p ** 2 + q ** 2)
    return p40_0 + p40_15 + p40_30 + p_deuce * win_deuce


@lru_cache(maxsize=200_000)
def tiebreak_prob(p: float, q: float, target: int = 7) -> float:
    """P(el jugador 1 gane el tie-break) sacando el primer punto.

    Saques del tie-break: el 1 saca 1 punto, luego se alternan de dos en dos.
    Iterativo (no recursivo) sobre estados; desde 6-6 se usa la formula cerrada
    de la ventaja de dos puntos, porque el par de saques se repite.
    """
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = min(max(q, 1e-6), 1 - 1e-6)

    def server(points_played: int) -> int:
        return 1 if ((points_played + 1) // 2) % 2 == 0 else 2

    # dos puntos consecutivos desde 6-6: el 1 saca uno y resta otro (y viceversa)
    a2 = p * (1 - q)                 # gana los dos
    b2 = (1 - p) * q                 # pierde los dos
    win_from_deuce = a2 / (a2 + b2) if (a2 + b2) > 0 else 0.5

    states = {(0, 0): 1.0}
    win = 0.0
    for n in range(0, 4 * target + 8):
        nxt = {}
        for (a, b), pr in states.items():
            if a >= target and a - b >= 2:
                win += pr; continue
            if b >= target and b - a >= 2:
                continue
            if a >= target - 1 and b >= target - 1:      # 6-6 o mas con diferencia < 2
                win += pr * win_from_deuce
                continue
            pp = p if server(a + b) == 1 else 1 - q
            nxt[(a + 1, b)] = nxt.get((a + 1, b), 0.0) + pr * pp
            nxt[(a, b + 1)] = nxt.get((a, b + 1), 0.0) + pr * (1 - pp)
        states = nxt
        if not states:
            break
    return win


def set_game_distribution(p: float, q: float) -> dict[tuple[int, int], float]:
    """Distribucion de marcadores de juegos de UN set: {(juegos1, juegos2): prob}.
    p = P(el jugador 1 gane un punto con su saque); q = la del jugador 2.
    Iterativo sobre marcadores; el saque alterna cada juego."""
    gp1, gp2 = game_prob(p), game_prob(q)
    dist: dict[tuple[int, int], float] = {}
    states = {(0, 0, 1): 1.0}                      # (juegos1, juegos2, quien saca)
    for _ in range(16):
        nxt: dict[tuple[int, int, int], float] = {}
        for (a, b, srv), pr in states.items():
            if pr < 1e-14:
                continue
            if a == 6 and b <= 4:
                dist[(6, b)] = dist.get((6, b), 0.0) + pr; continue
            if b == 6 and a <= 4:
                dist[(a, 6)] = dist.get((a, 6), 0.0) + pr; continue
            if (a, b) == (7, 5):
                dist[(7, 5)] = dist.get((7, 5), 0.0) + pr; continue
            if (a, b) == (5, 7):
                dist[(5, 7)] = dist.get((5, 7), 0.0) + pr; continue
            if (a, b) == (6, 6):
                tb = tiebreak_prob(p, q)
                dist[(7, 6)] = dist.get((7, 6), 0.0) + pr * tb
                dist[(6, 7)] = dist.get((6, 7), 0.0) + pr * (1 - tb)
                continue
            w = gp1 if srv == 1 else 1 - gp2
            s2 = 2 if srv == 1 else 1
            nxt[(a + 1, b, s2)] = nxt.get((a + 1, b, s2), 0.0) + pr * w
            nxt[(a, b + 1, s2)] = nxt.get((a, b + 1, s2), 0.0) + pr * (1 - w)
        states = nxt
        if not states:
            break
    return dist


def match_distribution(p1_serve: float, p2_serve: float, best_of: int = 3) -> dict:
    """P(gana el jugador 1), distribucion de juegos totales y de margen de juegos.
    Iterativo sobre sets ganados (no recursivo)."""
    sd = set_game_distribution(p1_serve, p2_serve)
    set_win = sum(pr for (a, b), pr in sd.items() if a > b)
    need = 2 if best_of == 3 else 3
    games_total: dict[int, float] = {}
    games_margin: dict[int, float] = {}
    p_match = 0.0
    states = {(0, 0, 0, 0): 1.0}                   # (sets1, sets2, juegos_tot, margen)
    for _ in range(2 * need):
        nxt: dict[tuple[int, int, int, int], float] = {}
        for (w1, w2, tot, marg), pr in states.items():
            if pr < 1e-13:
                continue
            for (a, b), sp in sd.items():
                n1, n2 = w1 + (1 if a > b else 0), w2 + (1 if b > a else 0)
                t2, m2, p2 = tot + a + b, marg + a - b, pr * sp
                if n1 == need or n2 == need:
                    games_total[t2] = games_total.get(t2, 0.0) + p2
                    games_margin[m2] = games_margin.get(m2, 0.0) + p2
                    if n1 == need:
                        p_match += p2
                else:
                    k = (n1, n2, t2, m2)
                    nxt[k] = nxt.get(k, 0.0) + p2
        states = nxt
        if not states:
            break
    return {"p_win": p_match, "p_set": set_win, "games_total": games_total, "games_margin": games_margin,
            "exp_games": sum(k * v for k, v in games_total.items()),
            "exp_margin": sum(k * v for k, v in games_margin.items())}


def p_over_games(dist: dict, line: float) -> float:
    """P(juegos totales > linea). Con linea entera, el empate exacto es push y se
    reparte fuera (el llamador lo trata)."""
    return sum(v for k, v in dist["games_total"].items() if k > line)


def p_cover_games(dist: dict, line: float) -> float:
    """P(margen del jugador 1 + linea > 0). linea negativa = el 1 da juegos."""
    return sum(v for k, v in dist["games_margin"].items() if k + line > 0)


def push_prob(dist: dict, line: float, kind: str) -> float:
    if float(line) != int(line):
        return 0.0
    key = int(line)
    d = dist["games_total"] if kind == "total" else dist["games_margin"]
    return d.get(key if kind == "total" else -key, 0.0)


def solve_serve_probs(p_win_target: float, avg_serve: float, best_of: int = 3,
                      lo: float = -0.25, hi: float = 0.25, iters: int = 24) -> tuple[float, float]:
    """Busca (p1, p2) con media `avg_serve` cuya P(gana 1) sea `p_win_target`.
    Se usa para el motor Elo: Elo da la probabilidad de partido y de ahi se
    derivan juegos y handicap de forma coherente."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        p1, p2 = avg_serve + mid, avg_serve - mid
        pw = match_distribution(min(max(p1, .3), .95), min(max(p2, .3), .95), best_of)["p_win"]
        if pw < p_win_target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return min(max(avg_serve + d, .3), .95), min(max(avg_serve - d, .3), .95)
