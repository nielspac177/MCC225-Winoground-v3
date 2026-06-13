"""Registro de entorno para trazabilidad y reproducibilidad (patrón del Cuaderno 10).

Imprime/serializa versiones de librerías, revisión de git, dispositivo y semilla.
Se usa en la celda inicial del notebook como 'hoja de trazabilidad' embebida.
"""
from __future__ import annotations

import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict


def _safe_version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "version_not_exposed"))
    except Exception as exc:  # noqa: BLE001
        return f"missing ({type(exc).__name__})"


def git_revision(root: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "no_git"
    except Exception:  # noqa: BLE001
        return "no_git"


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        if torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        return "cpu"
    except Exception:  # noqa: BLE001
        return "cpu (torch no disponible)"


def environment_snapshot(root: str | Path = ".") -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": git_revision(root),
        "device": detect_device(),
        "torch": _safe_version("torch"),
        "open_clip_torch": _safe_version("open_clip"),
        "transformers": _safe_version("transformers"),
        "datasets": _safe_version("datasets"),
        "numpy": _safe_version("numpy"),
        "faiss": _safe_version("faiss"),
    }


def print_snapshot(root: str | Path = ".") -> Dict[str, str]:
    snap = environment_snapshot(root)
    width = max(len(k) for k in snap)
    print("=" * 60)
    print("HOJA DE TRAZABILIDAD — ENTORNO DE EJECUCIÓN (MCC225)")
    print("=" * 60)
    for k, v in snap.items():
        print(f"  {k.rjust(width)} : {v}")
    print("=" * 60)
    return snap


if __name__ == "__main__":
    snap = print_snapshot(Path(__file__).resolve().parents[1])
    print(json.dumps(snap, indent=2, ensure_ascii=False))
