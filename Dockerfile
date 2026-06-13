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
RUN pip install -e ".[dev]"

COPY . .

# Por defecto: genera datos, corre el pipeline y las figuras.
CMD ["bash", "-lc", "python scripts/01_prepare_data.py && python scripts/02_run_winoground.py && python scripts/03_make_figures.py"]
