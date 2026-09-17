"""Empaqueta y comprime las bases de datos y archivos >100MB para Git estandar.

Divide cualquier archivo que supere 75 MB en partes de 40 MB para que GitHub
los acepte SIEMPRE sin necesidad de Git LFS ni extensiones adicionales.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "data_bundle"
CHUNK_SIZE = 45 * 1024 * 1024  # 45 MB por bloque (muy por debajo de los 100MB de GitHub)

TARGETS = [
    ("MLB/database/mlb.sqlite3", "mlb.sqlite3"),
    ("NBA/database/nba_markets.sqlite3", "nba_markets.sqlite3"),
    ("TENIS/database/tenis_markets.sqlite3", "tenis_markets.sqlite3"),
    ("SOCCER/database/soccer_markets.sqlite3", "soccer_markets.sqlite3"),
    ("NFL/database/nfl_markets.sqlite3", "nfl_markets.sqlite3"),
    ("NFL/database/nflpred.sqlite3", "nflpred.sqlite3"),
    ("PARLAY/database/parlays.sqlite3", "parlays.sqlite3"),
    ("TENIS/markets/out/features.parquet", "tenis_features.parquet"),
]


def pack():
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    # Limpiar bundle previo si existia
    for f in BUNDLE_DIR.glob("*"):
        if f.is_file():
            f.unlink()

    manifest = []
    print("Comprimiendo y preparando archivos para Git estandar (sin LFS)...")

    for rel_path, bundle_name in TARGETS:
        src = ROOT / rel_path
        if not src.exists():
            print(f"[-] Omitiendo {rel_path} (no existe en este entorno)")
            continue

        raw_size = src.stat().st_size / (1024 * 1024)
        print(f"[>] Procesando {rel_path} ({raw_size:.1f} MB)...")

        gz_target = BUNDLE_DIR / f"{bundle_name}.gz"
        with open(src, "rb") as f_in, gzip.open(gz_target, "wb", compresslevel=6) as f_out:
            while chunk := f_in.read(1024 * 1024 * 8):
                f_out.write(chunk)

        gz_size = gz_target.stat().st_size
        gz_mb = gz_size / (1024 * 1024)
        print(f"    Comprimido a {gz_mb:.1f} MB")

        # Si el comprimido supera 75 MB, lo dividimos en partes de 45 MB
        if gz_size > 75 * 1024 * 1024:
            print(f"    Dividiendo {bundle_name}.gz en partes de 45 MB para GitHub...")
            with open(gz_target, "rb") as f_gz:
                part_idx = 0
                while pchunk := f_gz.read(CHUNK_SIZE):
                    part_file = BUNDLE_DIR / f"{bundle_name}.gz.part{part_idx:02d}"
                    part_file.write_bytes(pchunk)
                    part_idx += 1
            gz_target.unlink()  # Eliminar el .gz grande, quedan solo los .partXX
            manifest.append((rel_path, bundle_name, "split", part_idx))
            print(f"    Dividido en {part_idx} partes.")
        else:
            manifest.append((rel_path, bundle_name, "single", 1))

    # Guardar manifiesto
    with open(BUNDLE_DIR / "manifest.txt", "w", encoding="utf-8") as f_m:
        for rel_path, bundle_name, kind, count in manifest:
            f_m.write(f"{rel_path}|{bundle_name}|{kind}|{count}\n")

    print("\n[OK] Empaquetado completado en data_bundle/")
    print("     Ningun archivo supera los 75 MB. GitHub aceptara todo limpiamente.")


if __name__ == "__main__":
    pack()
