"""El dashboard solo lee, respeta el filtro por deporte y nunca inventa partidos."""
import pytest
from fastapi.testclient import TestClient

from dashboard.backend.main import app
from shared.schema import empty_card, validate_card

c = TestClient(app)


def test_status_reporta_los_motores_con_honestidad():
    s = c.get("/api/status").json()["sports"]
    assert {"NFL", "MLB"} <= set(s)
    assert isinstance(s["MLB"]["missing"], list)
    if not s["MLB"]["ready"]:
        assert s["MLB"]["missing"], "si no esta listo debe decir que le falta"


def test_filtro_por_deporte():
    """La suma de cada deporte por separado debe dar el total: ningun deporte se
    queda fuera del calendario unificado ni se cuenta dos veces."""
    from shared.paths import SPORTS
    todos = c.get("/api/games", params={"range": "30d", "sport": "all"}).json()["count"]
    partes = {s: c.get("/api/games", params={"range": "30d", "sport": s}).json()["count"] for s in SPORTS}
    assert todos == sum(partes.values()), partes


def test_las_tarjetas_cumplen_el_esquema_unico():
    for d in c.get("/api/games", params={"range": "30d"}).json()["days"]:
        for g in d["games"]:
            assert validate_card(g) == []


def test_los_dias_vienen_ordenados():
    days = [d["date"] for d in c.get("/api/games", params={"range": "30d"}).json()["days"]]
    assert days == sorted(days)


def test_top_picks_ordena_por_probabilidad_calibrada():
    """La auditoria §11 demostro que ordenar por 'edge sobre la tasa base' da
    PEORES picks que ordenar por probabilidad. Este test blinda el orden correcto:
    primero los mercados con ventaja demostrada, y dentro de cada grupo por
    probabilidad descendente."""
    t = c.get("/api/top-picks", params={"range": "30d"}).json()
    picks = t["picks"]
    claves = [(not p.get("validated_pick", True), -float(p["ensemble_probability"]))
              for p in picks]
    assert claves == sorted(claves), "el orden de Top Picks no es el auditado"
    assert any("probabilidad calibrada" in x for x in t["caveats"])


def test_top_picks_avisa_de_los_mercados_no_validados():
    t = c.get("/api/top-picks", params={"range": "30d", "sport": "MLB"}).json()
    sin_pick = [p for p in t["picks"] if (p.get("extra") or {}).get("publish_pick") is False]
    if sin_pick:
        assert any("solo como probabilidad" in x for x in t["caveats"])


def test_sin_oportunidades_dice_no_pick_y_no_inventa():
    t = c.get("/api/top-picks", params={"range": "custom", "start": "1990-01-01",
                                        "end": "1990-01-02"}).json()
    assert t["count"] == 0 and t["picks"] == []
    assert any("NO PICK" in x for x in t["caveats"])


def test_el_dashboard_no_expone_endpoints_de_escritura():
    metodos = {r.methods and tuple(sorted(r.methods)) for r in app.routes if hasattr(r, "methods")}
    for m in metodos:
        assert not ({"POST", "PUT", "DELETE", "PATCH"} & set(m or ()))


def test_validate_card_rechaza_basura():
    assert validate_card(empty_card(sport="NBA")) != []
    assert "probabilidad" not in ""  # marcador
    bad = empty_card(sport="MLB", game_id=1, start_utc="x", home="LAD", away="SD",
                     model_probability=1.4)
    assert any("fuera de [0,1]" in p for p in validate_card(bad))
