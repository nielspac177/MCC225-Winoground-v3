.PHONY: setup data run figures test slides all clean help
PY := .venv/bin/python

help:
	@echo "make setup    - crea venv (uv, python 3.12) e instala dependencias"
	@echo "make data     - genera set curado + intenta cachear Winoground real"
	@echo "make run      - corre el pipeline de evaluación -> outputs/metrics"
	@echo "make figures  - genera figuras -> outputs/figures"
	@echo "make test     - pytest"
	@echo "make slides   - compila Beamer (PDF) y genera el PPTX"
	@echo "make all      - setup data run figures test"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

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

slides:
	cd slides/latex && latexmk -pdf defensa_winoground.tex || pdflatex defensa_winoground.tex
	$(PY) slides/pptx/build_pptx.py

all: data run figures test
	@echo "Pipeline completo. Revisa outputs/ y notebooks/."

clean:
	rm -rf outputs/metrics/* outputs/figures/* .pytest_cache
