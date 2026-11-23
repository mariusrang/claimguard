# Short commands for everything. `make help` lists them.
CONFIG ?= configs/default.yaml
TRAIN_CONFIG ?= configs/train_verifier.yaml
PY ?= python

.PHONY: help install data sample train train-seeds sweep eval extras tables report demo demo-offline mcp test lint docker docker-run clean

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ :/'

install: ## install the dependencies and the package
	pip install -r requirements-dev.txt && pip install -e .

data: ## download SciFact + Climate-FEVER, preprocess, build the contrast set
	$(PY) data/scripts/download.py
	$(PY) data/scripts/preprocess.py --raw data/raw/scifact --out data/processed/scifact
	$(PY) data/scripts/build_contrast_set.py --data data/processed/scifact

sample: ## prepare the toy sample (offline mode, tests)
	$(PY) data/scripts/preprocess.py --raw data/sample --out data/processed/sample
	$(PY) data/scripts/build_contrast_set.py --data data/processed/sample

train: ## fine-tune the verifier (1 seed)
	$(PY) scripts/train_verifier.py --config $(TRAIN_CONFIG)

train-seeds: ## fine-tune with the 3 seeds of the config
	$(PY) scripts/train_verifier.py --config $(TRAIN_CONFIG) --all-seeds

sweep: ## hyperparameter grid search
	$(PY) scripts/train_verifier.py --config $(TRAIN_CONFIG) --sweep

eval: ## main experiments: retrieval, baselines, pipeline + ablations, calibration, errors
	$(PY) scripts/evaluate.py retrieval --config $(CONFIG)
	$(PY) scripts/evaluate.py baselines --config $(CONFIG)
	$(PY) scripts/evaluate.py calibration --config $(CONFIG)
	$(PY) scripts/evaluate.py pipeline --config $(CONFIG)
	$(PY) scripts/evaluate.py errors --config $(CONFIG)

extras: ## stress tests: contrast set, French bank set, Climate-FEVER, efficiency
	$(PY) scripts/evaluate.py given --dataset contrast --config $(CONFIG)
	$(PY) scripts/evaluate.py given --dataset french --config $(CONFIG)
	$(PY) scripts/evaluate.py given --dataset climate --config $(CONFIG)
	$(PY) scripts/evaluate.py efficiency --config $(CONFIG)

tables: ## markdown tables + figures from results/
	$(PY) scripts/make_tables.py
	$(PY) scripts/make_figures.py

report: ## report/report_draft.md -> report.pdf (needs pandoc + xelatex)
	pandoc report/report_draft.md -o report.pdf --pdf-engine=xelatex -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono"

demo: ## Gradio demo on http://localhost:7860
	$(PY) app.py

demo-offline: sample ## demo without any model download (rule-based baseline)
	CLAIMGUARD_CONFIG=configs/offline.yaml $(PY) app.py

mcp: ## start the MCP server (stdio)
	$(PY) -m claimguard.mcp_server

test: sample ## unit tests with coverage
	$(PY) -m pytest --cov=claimguard --cov-report=term-missing

lint: ## code style
	ruff check .

docker: ## build the image
	docker build -t claimguard .

docker-run: ## run the demo in Docker
	docker run --rm -p 7860:7860 claimguard

clean:
	rm -rf results/*.json results/*.jsonl results/figures logs/*.jsonl .pytest_cache
