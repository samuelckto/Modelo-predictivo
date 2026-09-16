"""Configuracion comun de los tests del chat.

Un test no puede depender del .env de quien lo ejecuta. Sin esto, en la maquina
del usuario (que tiene Ollama configurado) los tests de las REGLAS se iban al
modelo de lenguaje real: la suite pasaba de 4 segundos a 6 minutos y dos tests
fallaban por motivos que no tenian nada que ver con lo que comprobaban.

Asi que por defecto NO hay LLM. Los tests que quieren uno lo configuran ellos.
"""
from __future__ import annotations

import pytest

VARIABLES = ("SPC_CHAT_BASE_URL", "SPC_CHAT_MODEL", "SPC_CHAT_API_KEY",
             "SPC_CHAT_TIMEOUT", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY",
             "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
             "OPENROUTER_API_KEY", "TOGETHER_API_KEY")


@pytest.fixture(autouse=True)
def sin_llm_por_defecto(monkeypatch):
    for v in VARIABLES:
        monkeypatch.delenv(v, raising=False)
