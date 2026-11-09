# Replication pipeline. Run `make all` after placing the WADA PDFs (see README).
PY := .venv/bin/python -W ignore

.PHONY: all setup index runs score analyse benchmark

all: index runs score analyse

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# clause-level retrieval index for each List (needs the PDFs in data/raw/)
index:
	for y in 2021 2023 2026 2027; do $(PY) src/parse_list.py $$y; done

# model calls; released outputs in results/runs/ are reused, so only missing items are called.
# Delete results/runs/ first to re-run every call from scratch.
runs:
	sh scripts/run_all.sh

score:
	$(PY) src/score.py

# metrics, statistics, figures and tables (written to results/ and paper/)
analyse:
	$(PY) src/analyse.py
	$(PY) src/summary.py

# optional: regenerate the benchmark from the Lists (rule tables with a deterministic and an LLM pass,
# then the questions). The released benchmark/ files are the ones used in the study.
benchmark:
	for y in 2026 2027; do $(PY) src/parse_list.py $$y && $(PY) src/rules_pass_a.py $$y && \
	  $(PY) src/rules_pass_b.py $$y && $(PY) src/reconcile_rules.py $$y && $(PY) src/apply_overrides.py $$y; done
	$(PY) src/merge_agreement.py
	$(PY) src/list_delta.py
	$(PY) src/build_benchmark.py
	$(PY) src/make_subsets.py
	$(PY) src/build_status_change.py
