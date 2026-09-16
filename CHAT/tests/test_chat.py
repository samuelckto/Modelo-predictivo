"""Tests del chat: intencion, registro, API interna y la regla de no inventar.

El test mas importante de este archivo es `test_nunca_devuelve_una_probabilidad
_sin_modelo`. Todo lo demas es comodidad; ese es el contrato.
"""
from __future__ import annotations

import json

import pytest

from CHAT import answer, api, intent, registry
from PARLAY.engine import calculator as calc


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------

def test_el_registro_es_coherente():
    for mid, d in registry.MERCADOS.items():
        assert d["market_id"] == mid
        assert d["visibilidad"] in registry.VISIBILIDAD
        assert d["estado"] in registry.ESTADOS
        assert d["distribution_type"] in registry.DISTRIBUCIONES
        assert mid.startswith(d["sport"].lower() + ".")


def test_los_secundarios_no_son_primarios():
    sec = {d["market_id"] for d in registry.secundarios()}
    pri = {d["market_id"] for d in registry.listar(visibilidad="primary")}
    assert sec and pri
    assert not (sec & pri)


def test_tarjetas_y_f5_son_secundarios():
    """Requisito explicito: no pueden salir en Top Picks."""
    assert registry.get("soccer.cards")["visibilidad"] == "secondary"
    assert registry.get("mlb.f5_total")["visibilidad"] == "secondary"
    assert registry.get("mlb.runs_dist")["visibilidad"] == "secondary"


def test_los_secundarios_declaran_sus_limitaciones():
    for d in registry.secundarios():
        assert isinstance(d["limitaciones"], list)
    # el de tarjetas debe decir que no hay arbitro
    lim = " ".join(registry.get("soccer.cards")["limitaciones"]).lower()
    assert "arbitro" in lim


def test_buscar_no_confunde_mercados_parecidos():
    """Por tokens declarados, no por parecido de cadenas."""
    assert registry.buscar("cuantas tarjetas se esperan")[0]["market_id"] == "soccer.cards"
    assert registry.buscar("cuantos corners")[0]["market_id"] == "soccer.corners"
    assert registry.buscar("probabilidad de over 2.5 goles")[0]["market_id"] == \
        "soccer.total_goals"


# --------------------------------------------------------------------------
# Intencion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("texto,mid", [
    ("¿cuantas carreras se esperan?", "mlb.total"),
    ("¿cuantas tarjetas se esperan?", "soccer.cards"),
    ("¿ambos reciben tarjeta?", "soccer.cards"),
    ("¿habra carreras en las primeras 5 entradas?", "mlb.f5_total"),
    ("¿que probabilidad hay de over 2.5 goles?", "soccer.total_goals"),
    ("¿ambos marcan?", "soccer.btts"),
    ("¿cuantos corners?", "soccer.corners"),
])
def test_detecta_el_mercado(texto, mid):
    assert intent.detectar(texto)["market_id"] == mid


def test_detecta_la_linea_y_el_lado():
    d = intent.detectar("¿que probabilidad hay de over 2.5 goles?")
    assert d["linea"] == 2.5
    assert d["lado"] == "over"
    d2 = intent.detectar("¿menos de 8.5 carreras?")
    assert d2["linea"] == 8.5
    assert d2["lado"] == "under"


def test_detecta_la_cantidad_de_una_pregunta_de_equipo():
    d = intent.detectar("¿que probabilidad hay de que Boston anote 4 carreras?")
    assert d["cantidad"] == 4


def test_distingue_explicacion_de_probabilidad():
    assert intent.detectar("¿por que 63%?")["intencion"] == "explicacion"
    assert intent.detectar("¿en que te basas?")["intencion"] == "explicacion"


def test_distingue_comparacion_con_el_mercado():
    d = intent.detectar("¿por que el modelo esta tan diferente de la casa?")
    # 'por que' es explicacion y va primero: la comparacion sale en la respuesta
    assert d["intencion"] in ("explicacion", "comparacion")
    assert intent.detectar("¿que dice el mercado de este partido?")["intencion"] == \
        "comparacion"


def test_una_pregunta_sin_mercado_no_se_fuerza():
    d = intent.detectar("¿va a llover manana?")
    assert d["intencion"] == "desconocida"
    assert d["market_id"] is None
    assert d["motivo"]


def test_pregunta_vacia():
    assert intent.detectar("")["intencion"] == "desconocida"


# --------------------------------------------------------------------------
# La regla que no se puede romper
# --------------------------------------------------------------------------

def test_nunca_devuelve_una_probabilidad_sin_modelo():
    """Un mercado inexistente NO puede producir un numero, pase lo que pase."""
    r = api.get_prediction("inventado", "mlb.no_existe")
    assert r["disponible"] is False
    assert "no existe el mercado" in r["motivo"]
    assert not any(isinstance(v, float) for k, v in r.items()
                   if k.lower().startswith(("prob", "model_prob")))


def test_partido_inexistente_no_produce_numero():
    r = api.get_prediction("999999999", "soccer.total_goals")
    assert r["disponible"] is False
    assert r.get("MODEL_PROBABILITY_CALIBRATED") is None


def test_mercado_binario_no_tiene_distribucion():
    r = api.get_distribution("x", "nfl.spread")
    assert r["disponible"] is False
    assert "binario" in r["motivo"]


def test_respuesta_a_pregunta_sin_modelo_lo_dice():
    """Regla 6: mejor 'no tengo modelo' que un numero plausible."""
    r = answer.responder("¿cuantos home runs pega Judge?", game_id="123")
    assert r["ok"] is False
    assert r["tipo"] in ("sin_modelo", "falta_mercado")
    assert r["status"] in ("BLOCKED", "NO MARKET")
    assert "inventado" in r["respuesta"] or "no de que mercado" in r["respuesta"]
    # y aun asi ofrece lo que SI puede contestar
    assert r["primarios"]


def test_pregunta_ininteligible_es_sin_modelo_no_falta_mercado():
    r = answer.responder("asdfgh qwerty")
    assert r["tipo"] == "sin_modelo"
    assert r["status"] == "BLOCKED"


# --------------------------------------------------------------------------
# Conversacion. La regla de no inventar aplica a las PROBABILIDADES, no a
# saludar. Contestar "estado BLOCKED" a un "hola" no es rigor, es mala interfaz.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("saludo", [
    "hola", "Hola!", "buenas", "buenos dias", "hey", "que tal", "qué onda",
    "gracias", "ok", "perfecto", "adios",
])
def test_los_saludos_se_reconocen(saludo):
    assert intent.detectar(saludo)["intencion"] == "saludo"


def test_un_saludo_no_se_bloquea():
    r = answer.responder("hola")
    assert r["ok"] is True
    assert r["tipo"] == "saludo"
    assert r["status"] is None
    assert "BLOCKED" not in str(r.get("status"))
    assert "Caballo Chat" in r["respuesta"]


def test_el_saludo_no_se_confunde_con_un_mercado():
    """'hola' no puede caer por casualidad en ningun patron de mercado."""
    d = intent.detectar("hola")
    assert d["market_id"] is None
    assert d["mercado_desconocido"] is False


@pytest.mark.parametrize("q", [
    "¿que tienes de este partido?", "dime todo", "¿como ves este partido?",
    "resumen", "analiza el partido", "¿que sabes de esto?",
])
def test_las_preguntas_abiertas_piden_resumen(q):
    assert intent.detectar(q)["intencion"] == "resumen"


def test_el_resumen_recorre_todos_los_mercados_del_deporte(monkeypatch):
    """No espera a que el usuario acierte el nombre del mercado."""
    vistos = []

    def falso_pred(gid, mid):
        vistos.append(mid)
        return {"disponible": False, "motivo": "sin datos"}

    monkeypatch.setattr(api, "get_prediction", falso_pred)
    monkeypatch.setattr(api, "get_distribution",
                        lambda gid, mid: {"disponible": False, "motivo": "sin datos"})
    partido = {"sport": "SOCCER", "game_id": "9", "nombre": "A vs B",
               "start_utc": None, "liga": "EPL"}
    r = answer._resumen("dime todo", {"intencion": "resumen"}, partido, "9", "SOCCER")
    assert set(vistos) >= {"soccer.total_goals", "soccer.btts", "soccer.double_chance"}
    # sin ningun mercado disponible NO se inventa nada
    assert r["ok"] is False
    assert r["status"] == "INSUFFICIENT DATA"


def test_los_mercados_publicados_no_van_al_motor_de_distribucion():
    """Bug real: 'cuantos goles' se enrutaba al motor de conteo, no encontraba
    ninguno y contestaba 'no disponible' teniendo la prediccion publicada."""
    for mid in ("soccer.total_goals", "soccer.double_chance", "mlb.total"):
        assert mid not in api.DISTRIBUCIONES
    for mid in ("soccer.cards", "mlb.f5_total", "mlb.runs_dist"):
        assert mid in api.DISTRIBUCIONES


def test_el_catalogo_no_se_marca_como_no_entendido():
    d = intent.detectar("¿que puedes contestar?")
    assert d["intencion"] == "catalogo"
    assert d["mercado_desconocido"] is False
    assert d["motivo"] is None


def test_el_catalogo_separa_primarios_de_secundarios():
    r = answer.responder("¿que puedes contestar?")
    assert r["tipo"] == "catalogo"
    assert r["primarios"] and r["secundarios"]
    ids = {d["market_id"] for d in r["secundarios"]}
    assert "soccer.cards" in ids
    assert "solo se consultan aqui" in r["respuesta"] or "Top Picks" in r["respuesta"]


def test_sin_partido_pide_el_partido_en_vez_de_responder():
    r = answer.responder("¿cuantas tarjetas se esperan?")
    assert r["ok"] is False
    assert r["tipo"] == "falta_partido"


# --------------------------------------------------------------------------
# Calculadora de parlay
# --------------------------------------------------------------------------

def test_cuota_combinada_y_pago():
    r = calc.calcular([{"odds_decimal": 2.0, "selection": "A", "event_id": "1"},
                       {"odds_decimal": 2.0, "selection": "B", "event_id": "2"}],
                      stake=10)
    assert r["cuota_decimal"] == pytest.approx(4.0)
    assert r["payout"] == pytest.approx(40.0)
    assert r["profit"] == pytest.approx(30.0)
    assert r["probabilidad_implicita_combinada"] == pytest.approx(0.25)


def test_acepta_americana_y_decimal_mezcladas():
    r = calc.calcular([{"odds_american": -110, "selection": "A", "event_id": "1"},
                       {"odds_decimal": 2.5, "selection": "B", "event_id": "2"}])
    assert r["n_validas"] == 2
    assert r["cuota_decimal"] == pytest.approx(1.909091 * 2.5, abs=1e-4)


def test_cuota_invalida_se_descarta_y_se_explica():
    r = calc.calcular([{"odds_decimal": 2.0, "selection": "A", "event_id": "1"},
                       {"odds_american": -9, "selection": "B", "event_id": "2"}])
    assert r["n_validas"] == 1
    assert r["descartadas"][0]["motivo"]


def test_avisa_de_patas_del_mismo_partido():
    r = calc.calcular([{"odds_decimal": 2.0, "selection": "gana A", "event_id": "77"},
                       {"odds_decimal": 1.8, "selection": "over 2.5", "event_id": "77"}])
    assert r["nivel_correlacion"] == "ALTA"
    assert r["avisos_correlacion"][0]["tipo"] == "mismo_evento"
    assert "NO es correcta" in r["advertencia"]


def test_avisa_de_participante_repetido():
    r = calc.calcular([
        {"odds_decimal": 2.0, "selection": "gana Sabalenka", "event_id": "1",
         "participants": ["Sabalenka"]},
        {"odds_decimal": 1.8, "selection": "Sabalenka -2.5", "event_id": "2",
         "participants": ["Sabalenka"]}])
    assert r["nivel_correlacion"] == "ALTA"
    assert r["avisos_correlacion"][0]["comun"] == ["Sabalenka"]


def test_patas_independientes_no_disparan_aviso():
    r = calc.calcular([{"odds_decimal": 2.0, "event_id": "1", "sport": "NBA",
                        "participants": ["LAL"]},
                       {"odds_decimal": 2.0, "event_id": "2", "sport": "MLB",
                        "participants": ["BOS"]}])
    assert r["nivel_correlacion"] == "NINGUNA"
    assert r["advertencia"] is None


def test_siempre_declara_el_supuesto_de_independencia():
    r = calc.calcular([{"odds_decimal": 2.0, "event_id": "1"},
                       {"odds_decimal": 2.0, "event_id": "2"}])
    assert "INDEPENDENCIA" in r["supuesto"]


def test_probabilidad_del_modelo_solo_si_la_tienen_todas():
    parcial = calc.calcular([
        {"odds_decimal": 2.0, "event_id": "1", "model_probability": 0.55},
        {"odds_decimal": 2.0, "event_id": "2"}])
    assert parcial["probabilidad_modelo_combinada"] is None
    assert "no es de ninguno de los dos" in parcial["nota_modelo"]

    completa = calc.calcular([
        {"odds_decimal": 2.0, "event_id": "1", "model_probability": 0.6},
        {"odds_decimal": 2.0, "event_id": "2", "model_probability": 0.5}])
    assert completa["probabilidad_modelo_combinada"] == pytest.approx(0.30)
    assert completa["gap_pp"] == pytest.approx(5.0, abs=1e-6)


def test_sin_ninguna_cuota_valida_no_calcula():
    r = calc.calcular([{"selection": "A"}, {"selection": "B"}])
    assert r["cuota_decimal"] is None
    assert "ninguna seleccion tiene cuota valida" in r["advertencia"]


def test_conversor_suelto():
    assert calc.convertir(american=-200)["decimal"] == pytest.approx(1.5)
    assert calc.convertir(american=-50)["valida"] is False
    assert calc.convertir(decimal=3.0)["implicita"] == pytest.approx(1 / 3)


def test_el_registro_se_puede_serializar():
    """El dashboard lo envia por JSON."""
    json.dumps(registry.listar())
