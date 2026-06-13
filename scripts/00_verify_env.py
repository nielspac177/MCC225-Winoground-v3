#!/usr/bin/env python3
"""Verifica el entorno e imprime la hoja de trazabilidad (§10 del examen)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_logging import print_snapshot  # noqa: E402

if __name__ == "__main__":
    snap = print_snapshot(ROOT)
    missing = [k for k, v in snap.items() if str(v).startswith("missing")]
    if missing:
        print(f"\n[ADVERTENCIA] dependencias ausentes: {missing}")
        sys.exit(1)
    print("\nEntorno OK.")
