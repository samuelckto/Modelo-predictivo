"""Cliente HTTP con reintentos, registro de fuente y CERO invencion.

Si una descarga falla se devuelve `Fetch(status="error")` y el llamador debe
marcar el dato como unavailable. Nunca se rellena con un valor por defecto.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from shared.sources import Fetch, checksum
from shared.timeutil import utcnow

UA = "Sports-Prediction-Center/0.2 (uso personal, sin scraping masivo)"
TIMEOUT = 120


def _open(url: str, timeout: int = TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    return urllib.request.urlopen(req, timeout=timeout)


NET_RETRIES = 6          # fallos de red (DNS, timeout, conexion): se insiste mas
NET_BACKOFF = (2, 4, 8, 12, 20)


def _is_network_error(e: Exception) -> bool:
    """DNS caido, resolvedor saturado, timeout, conexion rechazada: transitorios."""
    s = f"{type(e).__name__}: {e}".lower()
    return any(k in s for k in ("getaddrinfo", "name or service", "nodename",
                                "timed out", "timeout", "connection reset",
                                "connection refused", "remote end closed",
                                "temporary failure", "network is unreachable"))


def get_json(url: str, sport: str, source: str, domain: str, retries: int = 3,
             timeout: int = TIMEOUT) -> tuple[dict | None, Fetch]:
    """Descarga JSON. Los errores del cliente (400/401/403/404) no se reintentan;
    los errores de RED se reintentan hasta NET_RETRIES veces con esperas
    crecientes, porque en Windows el resolvedor DNS se satura con muchas
    descargas en paralelo y falla de forma transitoria (visto en produccion:
    `URLError: getaddrinfo failed` en 59 descargas de una misma ejecucion)."""
    last = None
    max_tries = max(retries, NET_RETRIES)
    for i in range(max_tries):
        try:
            t0 = time.time()
            with _open(url, timeout) as r:
                raw = r.read()
            data = json.loads(raw)
            return data, Fetch(sport, source, domain, url, "ok",
                               records=_count(data), retrieved_at=utcnow(),
                               checksum=checksum(raw),
                               notes=f"{time.time()-t0:.1f}s" +
                                     (f" tras {i} reintento(s)" if i else ""))
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404):         # no reintentar errores del cliente
                break
            if not _is_network_error(e) and i + 1 >= retries:
                break
            time.sleep(NET_BACKOFF[min(i, len(NET_BACKOFF) - 1)])
    return None, Fetch(sport, source, domain, url, "error", retrieved_at=utcnow(), error=last)


def get_csv(url: str, sport: str, source: str, domain: str, retries: int = 3,
            timeout: int = TIMEOUT) -> tuple[pd.DataFrame | None, Fetch]:
    last = None
    max_tries = max(retries, NET_RETRIES)
    for i in range(max_tries):
        try:
            t0 = time.time()
            with _open(url, timeout) as r:
                raw = r.read()
            if not raw.strip():
                return None, Fetch(sport, source, domain, url, "empty", retrieved_at=utcnow())
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            return df, Fetch(sport, source, domain, url, "ok" if len(df) else "empty",
                             records=len(df), retrieved_at=utcnow(), checksum=checksum(raw),
                             notes=f"{time.time()-t0:.1f}s {len(raw)/1e6:.1f}MB")
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            if getattr(e, "code", None) in (400, 401, 403, 404):
                break
            if not _is_network_error(e) and i + 1 >= retries:
                break
            time.sleep(NET_BACKOFF[min(i, len(NET_BACKOFF) - 1)])
    return None, Fetch(sport, source, domain, url, "error", retrieved_at=utcnow(), error=last)


def _count(d) -> int:
    if isinstance(d, list):
        return len(d)
    if isinstance(d, dict):
        for k in ("dates", "roster", "teams", "people", "records"):
            if isinstance(d.get(k), list):
                return len(d[k])
    return 1


def log_fetch(session, fetch: Fetch) -> None:
    """Escribe la fila en data_source_logs. Se llama SIEMPRE, tambien en error."""
    from MLB.database.models import DataSourceLog
    session.add(DataSourceLog(**fetch.to_row()))
    session.flush()
