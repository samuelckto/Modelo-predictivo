"""Guardas anti-leakage genericas (sirven a NFL y a MLB).

Principio: para una prediccion hecha en T solo puede usarse informacion cuyo
`available_at` sea ESTRICTAMENTE anterior a T. Nada de "<=", porque un dato con
el mismo timestamp exacto que la prediccion no estaba disponible al calcularla.
"""
from __future__ import annotations

import pandas as pd


class LeakageError(AssertionError):
    """Se intento usar informacion posterior al momento de la prediccion."""


def to_ts(v):
    if v is None:
        return None
    t = pd.to_datetime(v, errors="coerce", utc=False)
    return None if pd.isna(t) else (t.tz_localize(None) if getattr(t, "tzinfo", None) else t)


def filter_available(df: pd.DataFrame, cutoff, col: str = "available_at") -> pd.DataFrame:
    """Deja solo las filas conocidas ANTES del cutoff. Filas sin timestamp se descartan:
    si no sabemos cuando se supo un dato, no podemos garantizar que fuera anterior."""
    if df is None or len(df) == 0:
        return df
    c = to_ts(cutoff)
    if c is None:
        raise LeakageError("cutoff invalido: no se puede filtrar sin momento de prediccion")
    ts = pd.to_datetime(df[col], errors="coerce")
    return df[ts.notna() & (ts < c)]


def latest_before(df: pd.DataFrame, cutoff, keys, col: str = "available_at") -> pd.DataFrame:
    """Ultima version de cada clave conocida antes del cutoff (lineups, cuotas, lesiones)."""
    d = filter_available(df, cutoff, col)
    if d is None or len(d) == 0:
        return d
    return (d.sort_values(col).groupby(list(keys), as_index=False).tail(1))


def asof_join(left: pd.DataFrame, right: pd.DataFrame, on: str, by, suffix: str = "") -> pd.DataFrame:
    """merge_asof con allow_exact_matches=False: nunca usa la fila del propio partido."""
    l = left.sort_values(on).copy()
    r = right.sort_values(on).copy()
    l[on] = pd.to_datetime(l[on])
    r[on] = pd.to_datetime(r[on])
    return pd.merge_asof(l, r, on=on, by=by, direction="backward",
                         allow_exact_matches=False, suffixes=("", suffix or "_hist"))


def assert_no_leakage(used: pd.DataFrame, cutoff, col: str = "available_at", label: str = "") -> None:
    """Lanza LeakageError si alguna fila usada es posterior o igual al cutoff."""
    if used is None or len(used) == 0:
        return
    c = to_ts(cutoff)
    ts = pd.to_datetime(used[col], errors="coerce")
    bad = used[ts.isna() | (ts >= c)]
    if len(bad):
        raise LeakageError(
            f"{label or 'dataset'}: {len(bad)} fila(s) con {col} >= prediction_timestamp ({c}). "
            f"Ejemplos: {bad[col].head(3).tolist()}")


def audit_frame(df: pd.DataFrame, cutoff_col: str, ts_cols: list[str]) -> dict:
    """Auditoria posterior: cuenta violaciones por columna de timestamp."""
    out = {"rows": int(len(df)), "violations": {}}
    if len(df) == 0:
        return out
    cut = pd.to_datetime(df[cutoff_col], errors="coerce")
    for c in ts_cols:
        if c not in df.columns:
            continue
        t = pd.to_datetime(df[c], errors="coerce")
        out["violations"][c] = int((t.notna() & (t >= cut)).sum())
    out["total_violations"] = int(sum(out["violations"].values()))
    return out


# ------------------------------------------------------- ventanas acumuladas
def rolling_asof(events: pd.DataFrame, cutoffs: pd.DataFrame, keys: list[str],
                 ts: str, value_cols: list[str], windows: dict[str, int | None],
                 prefix: str = "") -> pd.DataFrame:
    """Sumas por ventana temporal SIN mirar el futuro.

    Para cada fila de `cutoffs` (con columnas `keys` + `cut`) devuelve, por cada
    ventana en dias, la suma de `value_cols` de los eventos cuyo `ts` es
    ESTRICTAMENTE anterior a `cut` y posterior a `cut - ventana`.

    Se implementa con sumas acumuladas y `merge_asof(allow_exact_matches=False)`:
    dos lecturas por ventana (en `cut` y en `cut - ventana`) y una resta. Es
    exacto y no puede filtrarse informacion posterior, porque la serie acumulada
    solo se consulta hacia atras.

    windows: {"s": None} = desde el inicio de la serie; {"d30": 30} = 30 dias.
    """
    ev = events.dropna(subset=[ts]).copy()
    ev[ts] = pd.to_datetime(ev[ts])
    ev = ev.sort_values(ts)
    ev["_n"] = 1.0
    cols = list(value_cols) + ["_n"]
    for c in cols:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0.0)
    cum = ev.groupby(keys, sort=False)[cols].cumsum()
    cum[keys] = ev[keys].values
    cum[ts] = ev[ts].values
    cum = cum.sort_values(ts)

    out = cutoffs.copy()
    out["cut"] = pd.to_datetime(out["cut"])
    out = out.sort_values("cut")

    def read_at(when: pd.Series, tag: str) -> pd.DataFrame:
        left = out[keys].copy()
        left["_t"] = when.values
        left = left.sort_values("_t")
        r = pd.merge_asof(left, cum.rename(columns={ts: "_t"}), on="_t", by=keys,
                          direction="backward", allow_exact_matches=False)
        r = r[[*keys, "_t", *cols]].rename(columns={c: f"{c}{tag}" for c in cols})
        return r.set_index(left.index)

    at_cut = read_at(out["cut"], "_hi")
    res = out.copy()
    for name, days in windows.items():
        if days is None:
            for c in cols:
                res[f"{prefix}{c}_{name}"] = at_cut[f"{c}_hi"].reindex(res.index).fillna(0.0)
        else:
            lo = read_at(out["cut"] - pd.Timedelta(days=days), "_lo")
            for c in cols:
                hi = at_cut[f"{c}_hi"].reindex(res.index).fillna(0.0)
                low = lo[f"{c}_lo"].reindex(res.index).fillna(0.0)
                res[f"{prefix}{c}_{name}"] = hi - low
    return res.rename(columns={f"{prefix}_n_{k}": f"{prefix}n_{k}" for k in windows})
