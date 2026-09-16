#!/usr/bin/env bash
# ==============================================================================
# Sports Prediction Center - Script de Instalacion y Servicio 24/7 para DietPi
# ==============================================================================
set -e

echo "==================================================================="
echo "  Configurando Sports Prediction Center en DietPi / Linux"
echo "==================================================================="

# 1. Verificar si se ejecuta como root (estandar en DietPi)
if [ "$EUID" -ne 0 ]; then
  echo "[-] Este script necesita privilegios de superusuario."
  echo "    Por favor ejecutalo con: sudo bash setup_server.sh"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Instalando dependencias del sistema operativo (Debian/DietPi)..."
apt-get update -y
apt-get install -y python3 python3-pip python3-venv build-essential libgomp1 curl git

echo "[2/6] Desempaquetando bases de datos e historico..."
python3 unpack_data.py

echo "[3/6] Preparando entorno virtual de Python (.venv)..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

echo "[4/6] Instalando requerimientos de Python..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "[5/6] Configurando archivo de entorno (.env)..."
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "    Archivo .env creado a partir de .env.example."
else
  echo "    Archivo .env existente preservado."
fi

# Asegurar host 0.0.0.0 en .env si no esta configurado
if ! grep -q "SPC_API_HOST=" .env; then
  echo "SPC_API_HOST=0.0.0.0" >> .env
fi
if ! grep -q "SPC_API_PORT=" .env; then
  echo "SPC_API_PORT=8100" >> .env
fi

# Asegurar esquemas actualizados
.venv/bin/python spc.py init

echo "[6/6] Creando y activando servicio systemd (spc.service) para ejecucion 24/7..."
SERVICE_FILE="/etc/systemd/system/spc.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Sports Prediction Center Dashboard & Engine
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/spc.py serve --host 0.0.0.0 --port 8100
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=SPC_API_HOST=0.0.0.0
Environment=SPC_API_PORT=8100

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable spc.service
systemctl restart spc.service

echo ""
echo "==================================================================="
echo "  ¡INSTALACION Y CONFIGURACION COMPLETADA CON EXITO!"
echo "==================================================================="
echo ""
echo "El servidor esta corriendo en segundo plano como un servicio del sistema."
echo "Se iniciara automaticamente cada vez que enciendas tu DietPi."
echo ""
echo "Puedes acceder al Dashboard en tu navegador web desde:"
IP_LOCAL=$(hostname -I | awk '{print $1}')
echo "  -> Red Local (LAN): http://${IP_LOCAL}:8100"
if command -v tailscale >/dev/null 2>&1; then
  TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || true)
  if [ -n "$TAILSCALE_IP" ]; then
    echo "  -> Tailscale:       http://${TAILSCALE_IP}:8100"
  fi
fi
echo ""
echo "Comandos utiles para gestionar el servicio:"
echo "  - Ver estado:     systemctl status spc.service"
echo "  - Ver logs:       journalctl -u spc.service -f"
echo "  - Reiniciar:      systemctl restart spc.service"
echo "  - Detener:        systemctl stop spc.service"
echo "  - O usa el script rapido: ./service.sh {status|restart|stop|logs}"
echo "==================================================================="
