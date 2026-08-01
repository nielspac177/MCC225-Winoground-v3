"""Genera los entregables auditables que exige §3.2 del Examen Final.

Todo se deriva de `outputs/metrics/`, de modo que los CSV de `results/` nunca se
editan a mano y siempre corresponden a la última ejecución.

Nota sobre versiones: se leen con `importlib.metadata`, que consulta los
metadatos del paquete sin importarlo. En este entorno `import torch` cuesta ~120 s
y `import huggingface_hub` 163 s; importar cinco librerías solo para registrar su
número de versión costaría más que todo el análisis.

Salidas:
  results/configuracion_experimental.json   semilla, modelos, versiones, hardware, commit
  results/metricas.csv                      métrica, resultado, interpretación
  data/MANIFIESTO.md                        procedencia y restricciones del dataset

Uso:
    python scripts/14_build_deliverables.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MET = ROOT / "outputs" / "metrics"
RES = ROOT / "results"

PAQUETES = [
    "torch", "open_clip_torch", "transformers", "datasets",
    "huggingface_hub", "numpy", "pandas", "scipy", "faiss-cpu", "Pillow",
]


def _version(nombre: str) -> str:
    try:
        return version(nombre)
    except PackageNotFoundError:
        return "no instalado"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "desconocido"


def configuracion_experimental() -> dict:
    """Registro completo para reproducir la corrida (§3.2, §4.10)."""
    modelos = []
    ruta = MET / "audit_modelos.csv"
    if ruta.exists():
        df = pd.read_csv(ruta)
        modelos = sorted(df["modelo"].unique().tolist())
    return {
        "proyecto": "Evaluación del razonamiento composicional en Winoground",
        "estudiante": "Niels Victor Pacheco Barrios",
        "curso": "MCC225 — IA Generativa y Aprendizaje Multimodal",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "commit_corto": _git("rev-parse", "--short", "HEAD"),
            "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "arbol_limpio": _git("status", "--porcelain") == "",
        },
        "semilla": 42,
        "bootstrap_rondas": 2000,
        "dataset": {
            "nombre": "facebook/winoground",
            "n_ejemplos": 400,
            "acceso": "gated (requiere aceptar la licencia en HuggingFace)",
            "revision": "b400e173549071916ad1b3d449293bc8d8b4b763",
        },
        "modelos_evaluados": modelos,
        "metricas": {
            "text_score": {"azar": 0.25},
            "image_score": {"azar": 0.25},
            "group_score": {"azar": 1 / 6},
            "group_match_score": {"azar": 0.5},
        },
        "hardware": {
            "plataforma": platform.platform(),
            "procesador": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "versiones": {p: _version(p) for p in PAQUETES},
        "limitaciones_conocidas": [
            "scripts/02_run_winoground.py se interbloquea en su bloque FAISS: carga el "
            "libomp de FAISS y el de PyTorch en el mismo proceso. El análisis se ejecuta "
            "con scripts/11_metric_audit.py, que no importa ninguno de los dos.",
            "Las descargas de HuggingFace desde Python se cuelgan en este entorno; los "
            "pesos de BLIP se obtuvieron con curl a data/models/ (ver data/MANIFIESTO.md).",
        ],
    }


def metricas_csv() -> pd.DataFrame:
    """Tabla plana y auditable de todos los resultados principales."""
    filas = []
    modelos = MET / "audit_modelos.csv"
    if modelos.exists():
        for _, r in pd.read_csv(modelos).iterrows():
            supera = "sí" if r["supera_azar"] else "no"
            filas.append({
                "experimento": "auditoria-metrica",
                "modelo": r["modelo"],
                "datos": "400 pares Winoground oficial",
                "metrica": r["metrica"],
                "resultado": round(float(r["score"]), 4),
                "ic95_lo": round(float(r["ic_lo"]), 4),
                "ic95_hi": round(float(r["ic_hi"]), 4),
                "azar": round(float(r["azar"]), 4),
                "interpretacion": (
                    f"IC95 [{r['ic_lo']:.3f}, {r['ic_hi']:.3f}]; "
                    f"¿el límite inferior supera el azar ({r['azar']:.3f})? {supera}."
                ),
            })
    potencia = MET / "audit_potencia.csv"
    if potencia.exists():
        for _, r in pd.read_csv(potencia).iterrows():
            filas.append({
                "experimento": "potencia-estadistica",
                "modelo": "(todos)",
                "datos": "n=400, alfa=0.05, potencia=0.80",
                "metrica": f"dif_minima_detectable_{r['metrica']}",
                "resultado": round(float(r["dif_minima_detectable"]), 4),
                "ic95_lo": "", "ic95_hi": "", "azar": "",
                "interpretacion": (
                    f"Dos modelos que difieran menos de "
                    f"{r['dif_minima_detectable'] * 100:.1f} puntos en {r['metrica']} "
                    "no son distinguibles con 400 ejemplos."
                ),
            })
    margenes = MET / "audit_margenes.csv"
    if margenes.exists():
        for _, r in pd.read_csv(margenes).iterrows():
            filas.append({
                "experimento": "margenes-de-decision",
                "modelo": r["modelo"],
                "datos": "1600 comparaciones (4 por ejemplo)",
                "metrica": "margen_mediano",
                "resultado": round(float(r["p50"]), 5),
                "ic95_lo": round(float(r["p5"]), 5),
                "ic95_hi": round(float(r["p90"]), 5),
                "azar": "",
                "interpretacion": (
                    f"La mitad de las decisiones se resuelve por un margen menor a "
                    f"{r['p50']:.4f}, sobre similitudes en [{r['sim_min']:.3f}, "
                    f"{r['sim_max']:.3f}]. El scorer usa `>` estricto: no hay empates "
                    "exactos, pero las decisiones no son holgadas."
                ),
            })
    return pd.DataFrame(filas)


MANIFIESTO = """# Manifiesto del dataset

## Winoground (fuente principal)

| Campo | Valor |
|---|---|
| Identificador | `facebook/winoground` en HuggingFace |
| Referencia | Thrush et al., *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*, CVPR 2022, arXiv:2204.03162 |
| Revisión usada | `b400e173549071916ad1b3d449293bc8d8b4b763` |
| Tamaño | 400 ejemplos = 800 imágenes + 800 captions |
| Formato | Parquet con las imágenes embebidas (367 MB) |
| Acceso | **Gated.** Hay que iniciar sesión en HuggingFace y aceptar el acuerdo de licencia. La aprobación es automática, pero sin ella la descarga devuelve 401. |

### Restricciones de uso

El acuerdo de licencia del dataset restringe su uso a **investigación no comercial**. Las
imágenes proceden de Getty Images y **no se redistribuyen** en este repositorio: solo se
versionan los *embeddings* derivados (`data/embeddings/winoground_real_vitb32.npz`) y las
matrices de similitud, que no permiten reconstruir las imágenes originales.

`.gitignore` excluye `data/winoground_cache/` justamente por esto.

### Estructura relevante

| Columna | Uso en este proyecto |
|---|---|
| `id` | 0–399, clave del ejemplo |
| `image_0`, `image_1` | las dos imágenes del par mínimo |
| `caption_0`, `caption_1` | los dos captions; comparten exactamente el mismo conjunto de palabras |
| `collapsed_tag` | `Object` (141), `Relation` (233), `Both` (26); base del análisis por tag |

Convención del scorer: `caption_0` corresponde a `image_0` y `caption_1` a `image_1`.

## Pesos de modelos

| Modelo | Origen | Ubicación local |
|---|---|---|
| CLIP ViT-B/32, ViT-B/16, ViT-L/14 | OpenCLIP (`open_clip.create_model_and_transforms`) | caché de HuggingFace |
| BLIP ITM base COCO | `Salesforce/blip-itm-base-coco` | `data/models/blip-itm-base-coco/` |

**Por qué BLIP está en `data/models/` y no en la caché de HuggingFace:** en este entorno las
descargas iniciadas desde Python se quedan colgadas indefinidamente — el proceso queda a 0 %
de CPU sin abrir socket — mientras que `curl` funciona con normalidad. Los pesos se
descargaron con:

```bash
curl -L -H "Authorization: Bearer $HF_TOKEN" \\
  -o data/models/blip-itm-base-coco/pytorch_model.bin \\
  https://huggingface.co/Salesforce/blip-itm-base-coco/resolve/main/pytorch_model.bin
```

`src/blip_utils.py::resolve_model_id` prefiere esa copia local si existe, de modo que la
ejecución es reproducible y completamente offline. El directorio está en `.gitignore`
(895 MB).

## Datos derivados que sí se versionan

| Ruta | Contenido | Regenerable con |
|---|---|---|
| `data/winoground_meta.csv` | ids, captions y tags; sin imágenes | `scripts/10_export_sims.py` |
| `data/embeddings/*.npz` | embeddings de CLIP ViT-B/32 | `scripts/02_run_winoground.py` |
| `outputs/sims/*.npz` | matrices 2×2 por ejemplo y modelo | `scripts/10_export_sims.py` |
"""


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)

    cfg = configuracion_experimental()
    (RES / "configuracion_experimental.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[ok] results/configuracion_experimental.json  (commit {cfg['git']['commit_corto']})")

    df = metricas_csv()
    if df.empty:
        print("[aviso] no hay métricas en outputs/metrics/; ejecuta antes 11_metric_audit.py")
    else:
        df.to_csv(RES / "metricas.csv", index=False)
        print(f"[ok] results/metricas.csv  ({len(df)} filas)")

    (ROOT / "data" / "MANIFIESTO.md").write_text(MANIFIESTO, encoding="utf-8")
    print("[ok] data/MANIFIESTO.md")


if __name__ == "__main__":
    main()
