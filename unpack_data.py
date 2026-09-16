"""Restaura las bases de datos y datasets en el servidor DietPi."""
from __future__ import annotations

import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "data_bundle"
MANIFEST = BUNDLE_DIR / "manifest.txt"


def unpack():
    if not MANIFEST.exists():
        print("[!] No hay bundle de datos comprimido en data_bundle/. Omitiendo descompresion.")
        return

    print("Restaurando bases de datos y datasets desde data_bundle/...")
    with open(MANIFEST, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        rel_path, bundle_name, kind, count = line.split("|")
        dest = ROOT / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"[>] Restaurando {rel_path}...")
        gz_path = BUNDLE_DIR / f"{bundle_name}.gz"

        if kind == "split":
            # Reensamblar las partes
            total_parts = int(count)
            with open(gz_path, "wb") as f_out:
                for idx in range(total_parts):
                    part_file = BUNDLE_DIR / f"{bundle_name}.gz.part{idx:02d}"
                    if not part_file.exists():
                        raise FileNotFoundError(f"Falta parte {part_file}")
                    f_out.write(part_file.read_bytes())

        # Descomprimir a destino
        with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
            while chunk := f_in.read(1024 * 1024 * 8):
                f_out.write(chunk)

        # Si era split, limpiar el .gz ensamblado temporal
        if kind == "split" and gz_path.exists():
            gz_path.unlink()

        final_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    [OK] {rel_path} restaurado ({final_mb:.1f} MB)")

    print("\n[OK] Todas las bases de datos y features han sido restauradas.")


if __name__ == "__main__":
    unpack()
