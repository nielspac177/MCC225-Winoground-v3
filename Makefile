.PHONY: setup data run figures test validate models manifest notebook annotate figura avance demo slides all clean help docker-avance
PY := .venv/bin/python

help:
	@echo "make setup       - crea venv (uv, python 3.12) e instala dependencias"
	@echo "make data        - genera set curado + intenta cachear Winoground real"
	@echo "make run         - corre el pipeline de evaluación -> outputs/metrics"
	@echo "make figures     - genera figuras -> outputs/figures"
	@echo "make test        - pytest"
	@echo "make validate    - valida el scorer contra clip.jsonl oficial"
	@echo "make models      - convierte CLIP/BLIP a safetensors local (models_local/)"
	@echo "make manifest    - construye data/raw/manifest_local.csv (120 pares reales)"
	@echo "make notebook    - ejecuta el Cuaderno14 headless (in-proc, in place)"
	@echo "make annotate    - anota la hoja de análisis de errores"
	@echo "make figura      - genera figures/ejemplos_evaluados.png (Actividad5)"
	@echo "make avance      - reproducción Exposición 2 (models+manifest+notebook+annotate)"
	@echo "make demo        - DEMO EN VIVO: CLIP sobre pares mínimos de Winoground"
	@echo "make slides      - compila Beamer defensa + exposición2 y genera el PPTX"
	@echo "make docker-avance - corre la reproducción Exposición 2 en Docker"
	@echo "make all         - setup data run figures test"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,notebook]"

data:
	$(PY) scripts/01_prepare_data.py

run:
	$(PY) scripts/02_run_winoground.py

figures:
	$(PY) scripts/03_make_figures.py

test:
	$(PY) -m pytest -q

validate:   ## valida el scorer contra los scores oficiales de CLIP (clip.jsonl)
	$(PY) scripts/validate_against_official.py

models:   ## convierte los pesos oficiales CLIP/BLIP a safetensors local
	$(PY) scripts/convert_models_to_safetensors.py

manifest:   ## construye el manifest local del Cuaderno14 (120 pares reales)
	$(PY) scripts/build_local_manifest.py 120

notebook:   ## ejecuta el Cuaderno14 in-proc (robusto, in place)
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PY) scripts/run_notebook_inproc.py notebooks/Cuaderno14_MCC225_resuelto.ipynb

annotate:   ## anota la hoja de análisis de errores del Cuaderno14
	$(PY) scripts/annotate_error_sheet.py

figura:   ## genera la figura de ejemplos evaluados (Actividad5)
	$(PY) scripts/generar_figura_ejemplos.py

avance: models manifest notebook annotate   ## reproducción completa (Exposición 2)
	@echo "Reproducción C14 completa. Revisa el notebook ejecutado, outputs/ y reports/."

demo:   ## demo en vivo (terminal): CLIP sobre pares mínimos de Winoground
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PY) scripts/demo_winoground.py --n 5 --retrieval

demo-visual:   ## demo VISUAL: paneles con imágenes + heatmap 2x2 + GIF animado
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PY) scripts/demo_visual.py --n 6

slides:
	cd slides/latex && latexmk -pdf defensa_winoground.tex || pdflatex defensa_winoground.tex
	cd slides/latex && latexmk -pdf exposicion2_winoground.tex || pdflatex exposicion2_winoground.tex
	$(PY) slides/pptx/build_pptx.py

docker-avance:   ## reproducción Exposición 2 en Docker
	docker compose run --rm avance

all: data run figures test
	@echo "Pipeline completo. Revisa outputs/ y notebooks/."

clean:
	rm -rf outputs/metrics/* outputs/figures/* .pytest_cache
