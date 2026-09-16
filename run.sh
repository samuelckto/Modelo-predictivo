#!/usr/bin/env bash
# Arrancar manualmente en primer plano (para depuracion o pruebas)
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -d ".venv" ]; then
  echo "Entorno virtual no encontrado. Ejecuta primero bash setup_server.sh"
  exit 1
fi
source .venv/bin/activate
python spc.py serve --host 0.0.0.0 --port 8100
