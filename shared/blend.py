"""Politica de combinacion modelo <-> mercado, comun a NFL y MLB.

Restricciones de diseno, declaradas explicitamente:

1. NINGUN peso se escribe a mano. `select_policy` elige entre candidatos usando
   SOLO datos anteriores a la temporada/periodo que se va a evaluar.
2. El mercado nunca puede ser la unica fuente (ALLOW_MARKET_ONLY = False):
   (a) es una regla del usuario;
   (b) en backtest el mercado disponible suele ser la linea de cierre, que NO
       existe en el momento real de predecir, asi que un `market_only` medido en
       backtest sobreestima lo que se podria lograr en vivo.
3. El pick lo decide SIEMPRE la probabilidad final combinada. No hay ninguna
   regla que compare con el mercado y cambie el pick por separado.
4. Si no hay historial suficiente NO se optimiza nada: se usa una politica a
   priori 50/50 y se dice que es a priori.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from shared.calibration import brier, log_loss
from shared.odds import logit, sigmoid

WEIGHT_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
ALLOW_MARKET_ONLY = False
MIN_MODEL_WEIGHT = 0.10
MIN_HISTORY = 400          # observaciones minimas para intentar optimizar
MIN_VALID_SEASONS = 2      # periodos minimos en el historial


@dataclass
class BlendPolicy:
    kind: str = "fixed_weight"        # model_only | market_only | fixed_weight | stacking
    market_weight: float = 0.5
    coef: list | None = None          # [w_model, w_market, w_elo] en espacio logit
    intercept: float = 0.0
    use_elo: bool = False
    fitted_on: dict = field(default_factory=dict)
    selection: list = field(default_factory=list)
    rationale: str = ""

    # -- aplicacion ---------------------------------------------------------
    def blend(self, p_model, p_market=None, p_elo=None) -> float:
        if p_model is None:
            raise ValueError("p_model es obligatorio")
        if p_market is None:               # sin cuotas -> solo modelo, nunca inventar
            return float(p_model)
        if self.kind == "model_only":
            return float(p_model)
        if self.kind == "market_only":
            return float(p_market)
        if self.kind == "stacking" and self.coef:
            z = self.intercept + self.coef[0] * logit(p_model) + self.coef[1] * logit(p_market)
            if self.use_elo and len(self.coef) > 2 and p_elo is not None:
                z += self.coef[2] * logit(p_elo)
            return sigmoid(z)
        w = float(self.market_weight)
        return sigmoid((1 - w) * logit(p_model) + w * logit(p_market))

    def blend_array(self, pm, pk=None, pe=None):
        pm = np.asarray(pm, dtype=float)
        if pk is None:
            return pm
        pk = np.asarray(pk, dtype=float)
        out = np.empty_like(pm)
        for i in range(len(pm)):
            k = None if np.isnan(pk[i]) else pk[i]
            e = None if pe is None or np.isnan(np.asarray(pe, float)[i]) else float(np.asarray(pe, float)[i])
            out[i] = self.blend(pm[i], k, e)
        return out

    def describe(self) -> str:
        if self.kind == "model_only":
            return "solo modelo (el mercado no mejoro nada en la validacion temporal)"
        if self.kind == "market_only":
            return "solo mercado"
        if self.kind == "stacking":
            c = self.coef or []
            return ("regresion logistica sobre logits: " +
                    ", ".join(f"{n}={v:+.3f}" for n, v in zip(("modelo", "mercado", "elo"), c)))
        return f"peso fijo: {(1-self.market_weight)*100:.0f}% modelo / {self.market_weight*100:.0f}% mercado"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "market_weight": self.market_weight, "coef": self.coef,
                "intercept": self.intercept, "use_elo": self.use_elo,
                "fitted_on": self.fitted_on, "selection": self.selection,
                "rationale": self.rationale, "description": self.describe()}

    @staticmethod
    def from_dict(d) -> "BlendPolicy":
        if isinstance(d, str):
            d = json.loads(d)
        if not d:
            return DEFAULT_POLICY
        return BlendPolicy(kind=d.get("kind", "fixed_weight"),
                           market_weight=float(d.get("market_weight", 0.5)),
                           coef=d.get("coef"), intercept=float(d.get("intercept", 0.0)),
                           use_elo=bool(d.get("use_elo", False)),
                           fitted_on=d.get("fitted_on", {}), selection=d.get("selection", []),
                           rationale=d.get("rationale", ""))


DEFAULT_POLICY = BlendPolicy(
    kind="fixed_weight", market_weight=0.5,
    rationale="por defecto a priori (50/50): sin historial suficiente no se optimiza nada")


# --------------------------------------------------------------- seleccion
def select_policy(history, metric: str = "log_loss", allow_market_only: bool = ALLOW_MARKET_ONLY,
                  period_col: str = "season") -> BlendPolicy:
    """Elige la politica con datos PASADOS unicamente.

    `history` debe traer columnas: <period_col>, p_model, p_market, y (0/1) y
    opcionalmente p_elo. Se ajustan los candidatos con todos los periodos menos
    el ultimo y se validan en el ultimo. Nunca ve el periodo que se evaluara.
    """
    import pandas as pd
    if history is None or len(history) == 0:
        return DEFAULT_POLICY
    h = pd.DataFrame(history).dropna(subset=["p_model", "p_market", "y"])
    if len(h) < MIN_HISTORY or h[period_col].nunique() < MIN_VALID_SEASONS:
        p = BlendPolicy(**{**DEFAULT_POLICY.__dict__})
        p.fitted_on = {"seasons": sorted(map(_num, h[period_col].unique())), "n": int(len(h))}
        p.rationale = (f"historial insuficiente ({len(h)} obs, "
                       f"{h[period_col].nunique()} periodos): se usa la politica a priori 50/50")
        return p

    periods = sorted(h[period_col].unique())
    valid_p = periods[-1]
    fit = h[h[period_col] < valid_p] if np.isscalar(valid_p) else h[h[period_col] != valid_p]
    val = h[h[period_col] == valid_p]
    if len(fit) < 100 or len(val) < 50:
        fit, val = h, h        # ultimo recurso; se declara en rationale

    score = (lambda y, p: log_loss(y, p)) if metric == "log_loss" else (lambda y, p: brier(y, p))
    cands: list[tuple[str, BlendPolicy, float]] = []

    cands.append(("model_only", BlendPolicy(kind="model_only"), score(val.y, val.p_model)))
    if allow_market_only:
        cands.append(("market_only", BlendPolicy(kind="market_only"), score(val.y, val.p_market)))
    for w in WEIGHT_GRID:
        if not allow_market_only and w > 1 - MIN_MODEL_WEIGHT:
            continue
        pol = BlendPolicy(kind="fixed_weight", market_weight=float(w))
        cands.append((f"w={w:.1f}", pol, score(val.y, pol.blend_array(val.p_model, val.p_market))))

    for use_elo in ((False, True) if "p_elo" in h.columns else (False,)):
        pol = _fit_stacking(fit, use_elo)
        if pol is not None:
            pe = val.p_elo if (use_elo and "p_elo" in val) else None
            cands.append((("stacking+elo" if use_elo else "stacking"), pol,
                          score(val.y, pol.blend_array(val.p_model, val.p_market, pe))))

    cands = [(n, p, s) for n, p, s in cands if s is not None and np.isfinite(s)]
    if not cands:
        return DEFAULT_POLICY
    cands.sort(key=lambda t: t[2])
    name, best, best_s = cands[0]
    best.selection = [{"candidate": n, metric: round(float(s), 6)} for n, _, s in cands]
    best.fitted_on = {"seasons": sorted(map(_num, fit[period_col].unique())),
                      "validation_season": _num(valid_p), "n": int(len(h)),
                      "n_fit": int(len(fit)), "n_validation": int(len(val)), "metric": metric}
    base = [s for n, _, s in cands if n == "model_only"][0]
    best.rationale = (f"elegido por validacion temporal en {valid_p}: '{name}' con "
                      f"{metric}={best_s:.4f} frente a {base:.4f} de solo-modelo. "
                      f"El mercado nunca es la unica fuente (ALLOW_MARKET_ONLY={allow_market_only}).")
    return best


def _fit_stacking(fit, use_elo: bool):
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    cols = ["p_model", "p_market"] + (["p_elo"] if use_elo and "p_elo" in fit.columns else [])
    d = fit.dropna(subset=cols + ["y"])
    if len(d) < 200:
        return None
    X = np.column_stack([[logit(v) for v in d[c]] for c in cols])
    try:
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, d.y.astype(int))
    except Exception:
        return None
    coef = list(map(float, lr.coef_[0]))
    if len(coef) == 2:
        coef.append(0.0)
    if not allow_model_weight_ok(coef):
        return None
    return BlendPolicy(kind="stacking", coef=coef, intercept=float(lr.intercept_[0]),
                       use_elo=bool(use_elo and len(cols) == 3))


def allow_model_weight_ok(coef) -> bool:
    """Rechaza un stacking que en la practica anule al modelo (equivaldria a market_only)."""
    if ALLOW_MARKET_ONLY:
        return True
    tot = abs(coef[0]) + abs(coef[1])
    return tot > 0 and abs(coef[0]) / tot >= 0.02


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------- desacuerdo
def disagreement(p_model, p_market, p_final, ensemble_std: float = 0.0, p_elo=None) -> dict:
    """Metricas de desacuerdo. NO afirman quien tiene razon."""
    d = {
        "market_available": p_market is not None,
        "model_confidence": abs(float(p_model) - 0.5) * 2,
        "market_confidence": (abs(float(p_market) - 0.5) * 2) if p_market is not None else None,
        "ensemble_confidence": abs(float(p_final) - 0.5) * 2,
        "model_dispersion": float(ensemble_std),
        "market_model_gap": None,
        "directional_disagreement": None,
        "agreement_bucket": "sin_mercado",
    }
    if p_market is None:
        return d
    gap = abs(float(p_model) - float(p_market))
    d["market_model_gap"] = round(gap, 6)
    d["directional_disagreement"] = bool((float(p_model) >= 0.5) != (float(p_market) >= 0.5))
    if d["directional_disagreement"]:
        d["agreement_bucket"] = "desacuerdo_direccional"
    elif gap < 0.05:
        d["agreement_bucket"] = "0-5pp"
    elif gap < 0.10:
        d["agreement_bucket"] = "5-10pp"
    elif gap < 0.20:
        d["agreement_bucket"] = "10-20pp"
    else:
        d["agreement_bucket"] = "20pp+"
    return d


def signal_label(d) -> str | None:
    if not d or not d.get("market_available"):
        return None
    if d.get("directional_disagreement"):
        return "DESACUERDO DE DIRECCION: modelo y mercado favorecen lados distintos"
    g = d.get("market_model_gap") or 0
    if g >= 0.10:
        return f"Discrepancia de confianza: {g*100:.0f} puntos porcentuales entre modelo y mercado"
    return None
