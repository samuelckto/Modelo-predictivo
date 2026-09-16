"""Agrega las partes de audit/parts en tablas comparables."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

PARTS = Path(__file__).resolve().parent / "parts"
ALG_ORDER = ["logistic", "forest", "xgboost", "lightgbm", "ensemble", "elo",
             "siempre_local", "siempre_visitante", "aleatorio"]


def load(market: str, variant: str) -> list[dict]:
    out = []
    for p in sorted(PARTS.glob(f"[0-9][0-9][0-9][0-9]_{market}_{variant}.json")):
        out.append(json.loads(p.read_text()))
    return sorted(out, key=lambda r: r["season"])


def z_vs_majority(acc: float, base: float, n: int) -> float:
    major = max(base, 1 - base)
    se = math.sqrt(major * (1 - major) / n)
    return (acc - major) / se if se else 0.0


def z_paired(a_acc, b_acc, n):
    """Aproximacion normal para dos estrategias sobre la misma muestra."""
    d = a_acc - b_acc
    se = math.sqrt((a_acc * (1 - a_acc) + b_acc * (1 - b_acc)) / n)
    return d / se if se else 0.0


def table(market: str, variant: str) -> dict:
    rows = load(market, variant)
    if not rows:
        return {}
    per_season, agg = {}, defaultdict(lambda: {"n": 0, "hits": 0.0, "ll": 0.0,
                                               "br": 0.0, "ece": 0.0})
    for r in rows:
        per_season[r["season"]] = {
            k: {"accuracy": v["accuracy"], "log_loss": v["log_loss"],
                "brier": v["brier"], "ece": v["ece"], "n": v["n"]}
            for k, v in r["strategies"].items()}
        for k, v in r["strategies"].items():
            a = agg[k]
            a["n"] += r["n"]
            a["hits"] += v["accuracy"] * r["n"]
            a["ll"] += v["log_loss"] * r["n"]
            a["br"] += v["brier"] * r["n"]
            a["ece"] += v["ece"] * r["n"]
    total = {k: {"n": v["n"], "accuracy": v["hits"] / v["n"], "log_loss": v["ll"] / v["n"],
                 "brier": v["br"] / v["n"], "ece": v["ece"] / v["n"]}
             for k, v in agg.items()}
    base = float(np.average([r["base_rate"] for r in rows],
                            weights=[r["n"] for r in rows]))
    n_tot = sum(r["n"] for r in rows)
    for k, v in total.items():
        v["z_vs_mayoritaria"] = round(z_vs_majority(v["accuracy"], base, n_tot), 2)
    if "elo" in total and "ensemble" in total:
        total["ensemble"]["z_vs_elo"] = round(
            z_paired(total["ensemble"]["accuracy"], total["elo"]["accuracy"], n_tot), 2)
    return {"market": market, "variant": variant, "base_rate": base, "n": n_tot,
            "seasons": [r["season"] for r in rows], "per_season": per_season,
            "total": total, "n_features": rows[0]["n_features"],
            "elo_cfg_por_temporada": {r["season"]: r["elo_cfg"] for r in rows}}


def fmt(t: dict) -> str:
    if not t:
        return "(sin datos)"
    L = [f"\n=== {t['market'].upper()}  [{t['variant']}]  n={t['n']}  "
         f"tasa base={t['base_rate']*100:.2f}%  features={t['n_features']}"]
    L.append(f"{'algoritmo':<20}{'acc':>8}{'logloss':>10}{'brier':>9}{'ECE':>8}{'z vs may.':>11}")
    for k in ALG_ORDER:
        if k not in t["total"]:
            continue
        v = t["total"][k]
        L.append(f"{k:<20}{v['accuracy']*100:>7.2f}%{v['log_loss']:>10.4f}"
                 f"{v['brier']:>9.4f}{v['ece']:>8.4f}{v['z_vs_mayoritaria']:>11.2f}")
    L.append("\npor temporada (accuracy):")
    algs = [a for a in ALG_ORDER if a in t["total"]]
    L.append(f"{'temporada':<12}" + "".join(f"{a[:9]:>11}" for a in algs))
    for s, d in sorted(t["per_season"].items()):
        L.append(f"{s:<12}" + "".join(f"{d[a]['accuracy']*100:>10.2f}%" for a in algs))
    return "\n".join(L)


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "moneyline"
    for v in ("con_abridor", "sin_abridor"):
        print(fmt(table(mk, v)))
    a, b = table(mk, "con_abridor"), table(mk, "sin_abridor")
    if a and b:
        print(f"\n=== IMPACTO DEL LEAKAGE DEL ABRIDOR ({mk}) ===")
        print(f"{'algoritmo':<20}{'con':>9}{'sin':>9}{'dif':>9}")
        for k in ALG_ORDER:
            if k in a["total"] and k in b["total"]:
                x, y = a["total"][k]["accuracy"], b["total"][k]["accuracy"]
                print(f"{k:<20}{x*100:>8.2f}%{y*100:>8.2f}%{(x-y)*100:>+8.2f}pp")
