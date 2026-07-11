# Imagen CPU reproducible para la evaluación de Winoground (MCC225).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/data/winoground_cache/hf

WORKDIR /workspace

# Dependencias del sistema mínimas (faiss/torch CPU, PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
# Incluye extra 'notebook' (jupyter/nbconvert/ipykernel) para ejecutar el Cuaderno14 headless.
RUN pip install -e ".[dev,notebook]"

COPY . .

# Por defecto: genera datos, corre el pipeline y las figuras (pipeline Winoground).
# -----------------------------------------------------------------------------
# Comandos alternativos (Exposición 2 / Cuaderno14) — ejecútalos con:
#   docker compose run --rm avance      (servicio dedicado en docker-compose.yml)
# o sobreescribiendo el CMD, p.ej.:
#   docker run --rm mcc225-winoground:latest bash -lc "\
#       python scripts/build_local_manifest.py 120 && \
#       jupyter nbconvert --to notebook --execute --inplace \
#           notebooks/Cuaderno14_MCC225_resuelto.ipynb && \
#       python scripts/02_run_winoground.py && \
#       python scripts/03_make_figures.py"
#   1) build_local_manifest.py 120 -> data/raw/manifest_local.csv (120 pares reales)
#   2) nbconvert --execute --inplace -> ejecuta Cuaderno14 en sitio
#   3) 02_run_winoground.py / 03_make_figures.py -> métricas + figuras reproducibles
# -----------------------------------------------------------------------------
CMD ["bash", "-lc", "python scripts/01_prepare_data.py && python scripts/02_run_winoground.py && python scripts/03_make_figures.py"]
