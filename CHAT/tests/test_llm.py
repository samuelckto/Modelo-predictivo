"""Tests de la capa de lenguaje.

La regla que estos tests protegen: el LLM NO calcula. Recibe datos ya
calculados y los redacta. Un LLM inventando un 62 % que suena verosímil es el
peor fallo posible aquí, porque no deja rastro.
"""
from __future__ import annotations

import json

import pytest

from CHAT import answer, llm


def test_sin_clave_no_esta_disponible(monkeypatch):
    for v in llm.CLAVES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("SPC_CHAT_BASE_URL", raising=False)
    d = llm.disponible()
    assert d["ok"] is False
    assert "no hay ningún motor" in d["motivo"]


def test_sin_clave_devuelve_none_y_no_inventa(monkeypatch):
    """Sin clave NO se degrada a 'contesto con lo que sepa'."""
    for v in llm.CLAVES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("SPC_CHAT_BASE_URL", raising=False)
    assert llm.responder("¿quién va a ganar el mundial?") is None


def test_una_clave_corta_no_cuenta_como_clave(monkeypatch):
    for v in llm.CLAVES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("SPC_CHAT_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc")
    assert llm.disponible()["ok"] is False


@pytest.fixture
def limpio(monkeypatch):
    """Sin ninguna variable de LLM en el entorno."""
    for v in llm.CLAVES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("SPC_CHAT_BASE_URL", raising=False)
    return monkeypatch


def test_detecta_la_clave_y_el_proveedor(limpio):
    limpio.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 40)
    d = llm.disponible()
    assert d["ok"] is True and d["proveedor"] == "anthropic"

    limpio.delenv("ANTHROPIC_API_KEY")
    limpio.setenv("OPENAI_API_KEY", "sk-" + "y" * 40)
    assert llm.disponible()["proveedor"] == "openai"


@pytest.mark.parametrize("var,base", [
    ("GROQ_API_KEY", "groq.com"),
    ("DEEPSEEK_API_KEY", "deepseek.com"),
    ("OPENROUTER_API_KEY", "openrouter.ai"),
    ("TOGETHER_API_KEY", "together.xyz"),
])
def test_otros_proveedores_compatibles(limpio, var, base):
    """No solo Anthropic: todo lo que habla el protocolo de OpenAI sirve."""
    limpio.setenv(var, "k" * 40)
    d = llm.disponible()
    assert d["ok"] is True
    assert d["proveedor"] == "openai"
    assert base in d["base_url"]


def test_un_servidor_local_no_necesita_clave(limpio):
    """Ollama y LM Studio corren en tu PC: no hay a quién autenticarse."""
    limpio.setenv("SPC_CHAT_BASE_URL", "http://localhost:11434/v1")
    d = llm.disponible()
    assert d["ok"] is True
    assert d["proveedor"] == "local"
    assert d["coste"] == "gratis"
    assert d["privado"] is True
    assert d["variable"] is None


def test_una_url_remota_si_necesita_clave(limpio):
    limpio.setenv("SPC_CHAT_BASE_URL", "https://mi-servidor.com/v1")
    d = llm.disponible()
    assert d["ok"] is False
    assert "no es local" in d["motivo"]


def test_sin_nada_ofrece_las_opciones(limpio):
    d = llm.disponible()
    assert d["ok"] is False
    nombres = " ".join(o["nombre"] for o in d["opciones"])
    assert "Ollama" in nombres and "Groq" in nombres
    gratis = [o for o in d["opciones"] if o["coste"] == "gratis"]
    assert gratis and all(o["clave"] is False for o in gratis)


def test_el_error_local_explica_como_arreglarlo(limpio):
    limpio.setenv("SPC_CHAT_BASE_URL", "http://localhost:11434/v1")

    def revienta(*a, **k):
        raise ConnectionRefusedError("nada escuchando")
    limpio.setattr(llm, "_llamar_openai", revienta)
    r = llm.responder("hola")
    assert r["ok"] is False
    assert "localhost:11434" in r["respuesta"]
    assert "Start Server" in r["respuesta"]


def test_el_prompt_prohibe_calcular():
    s = llm.SISTEMA
    assert "no calculas nada" in s.lower()
    assert "prohibido estimar" in s.lower()
    assert "PROJECTION" in s
    # y prohibe explicitamente llamar valor a un gap
    assert "no es \"valor\"" in s or 'NO es "valor"' in s


def test_el_prompt_si_permite_explicar_conceptos():
    """La regla es sobre NÚMEROS, no sobre ideas. Con el prompt anterior el
    modelo contestaba 'no hay números en el contexto que permitan explicar qué
    es el log loss', que es absurdo: eso no necesita ningún dato."""
    s = llm.SISTEMA.lower()
    assert "concepto" in s
    assert "log loss" in s
    assert "no necesita ningún dato" in s or "no necesita ningun dato" in s


def test_el_contexto_lleva_datos_reales_y_es_serializable():
    p = llm.contexto_del_sistema()
    assert "catalogo" in p and "calendario" in p and "estado_del_sistema" in p
    json.dumps(p, default=str)          # tiene que poder viajar al LLM


def test_el_contexto_incluye_el_veredicto_honesto():
    p = llm.contexto_del_sistema()
    e = p.get("estado_del_sistema") or {}
    if e:
        assert "NO se ha demostrado ventaja rentable" in e["veredicto"]


def test_un_fallo_de_red_no_tumba_el_chat(limpio):
    limpio.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 40)

    def revienta(*a, **k):
        raise ConnectionError("sin red")
    limpio.setattr(llm, "_llamar_anthropic", revienta)
    r = llm.responder("hola")
    assert r["ok"] is False
    assert r["status"] == "ERROR"
    assert "No pude consultar" in r["respuesta"]


# --------------------------------------------------------------------------
# Los tres bugs que hacian que el LLM pareciera no funcionar
# --------------------------------------------------------------------------

def test_un_error_del_llm_se_muestra_en_vez_de_tragarse(limpio):
    """El bug peor: si el LLM fallaba, el chat decia 'no reconozco la pregunta'.
    Mentira: la pregunta se entendio, lo que fallo fue el modelo. El usuario se
    quedaba creyendo que el chat no sabia.

    OJO con el fixture `limpio`: sin el, este test lee el .env real, ve el
    Ollama del usuario y se queda esperando la respuesta de verdad.
    """
    limpio.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 40)
    limpio.setattr(llm, "_llamar_anthropic",
                   lambda *a, **k: (_ for _ in ()).throw(TimeoutError("tarde")))
    r = answer.responder("cuentame algo raro que no este en mis reglas")
    assert r["tipo"] == "llm_error"
    assert "no reconozco" not in (r.get("motivo") or "")


def test_el_env_se_lee_en_cada_llamada_no_al_importar(monkeypatch):
    """Otro bug real: leer SPC_CHAT_BASE_URL a nivel de modulo. Si algo
    importaba CHAT.llm antes de load_dotenv, el .env no servia para nada."""
    monkeypatch.setenv("SPC_CHAT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("SPC_CHAT_MODEL", "modelo-de-prueba")
    d = llm.disponible()
    assert d["base_url"] == "http://127.0.0.1:9999/v1"
    assert d["modelo"] == "modelo-de-prueba"


def test_un_modelo_local_tiene_mas_margen_de_tiempo(monkeypatch):
    """60 s no alcanzan: medido, llama3.1 tarda 186 s con 6000 tokens."""
    monkeypatch.delenv("SPC_CHAT_TIMEOUT", raising=False)
    assert llm._timeout(local=True) > llm._timeout(local=False)
    assert llm._timeout(local=True) >= 180


def test_el_contexto_compacto_es_mucho_mas_pequeno():
    """En un modelo local el tiempo crece brutalmente con el contexto."""
    grande = json.dumps(llm.contexto_del_sistema(compacto=False), default=str)
    chico = json.dumps(llm.contexto_del_sistema(compacto=True), default=str)
    assert len(chico) < len(grande) * 0.6


def test_un_timeout_sugiere_un_modelo_mas_pequeno(monkeypatch, limpio):
    limpio.setenv("SPC_CHAT_BASE_URL", "http://localhost:11434/v1")
    limpio.setenv("SPC_CHAT_TIMEOUT", "1")

    def lento(*a, **k):
        import time as t
        t.sleep(1.2)
        raise TimeoutError("timed out")
    limpio.setattr(llm, "_llamar_openai", lento)
    r = llm.responder("hola")
    assert r["ok"] is False
    assert "qwen2.5:3b" in r["respuesta"]
    assert r["segundos"] >= 1


def test_sin_llm_el_chat_lo_dice_en_vez_de_fingir(monkeypatch):
    for v in llm.CLAVES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("SPC_CHAT_BASE_URL", raising=False)
    r = answer.responder("cuentame un chiste de beisbol")
    assert r["ok"] is False
    assert r["llm"]["ok"] is False
    assert r["llm"]["opciones"]


def test_el_llm_solo_entra_cuando_las_reglas_no_saben(monkeypatch):
    """Una pregunta que las reglas SÍ entienden no debe ir al LLM."""
    llamadas = []
    monkeypatch.setattr(llm, "responder",
                        lambda *a, **k: llamadas.append(a) or None)
    answer.responder("¿que partidos hay de beisbol hoy?")
    answer.responder("hola")
    assert llamadas == []
