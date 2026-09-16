#!/usr/bin/env bash
# ==============================================================================
# Helper rapido para gestionar el servicio de Sports Prediction Center
# ==============================================================================
case "$1" in
  status)
    systemctl status spc.service
    ;;
  restart)
    systemctl restart spc.service
    echo "[✓] Servicio reiniciado."
    ;;
  stop)
    systemctl stop spc.service
    echo "[✓] Servicio detenido."
    ;;
  start)
    systemctl start spc.service
    echo "[✓] Servicio iniciado."
    ;;
  logs)
    journalctl -u spc.service -f -n 50
    ;;
  *)
    echo "Uso: ./service.sh {status|restart|stop|start|logs}"
    ;;
esac
