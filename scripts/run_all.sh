#!/bin/sh
# All model runs, one model at a time. Each run resumes from its JSONL file.
set -e
PY=".venv/bin/python -W ignore"
MODELS="${MODELS:-qwen3:30b-a3b qwen3:14b gemma3:12b llama3.1:8b}"
# embed every retrieval query first, then unload the embedding model, so it never competes for GPU memory
$PY src/retriever.py
curl -s http://localhost:11434/api/generate -d '{"model": "bge-m3", "keep_alive": 0}' > /dev/null || true
for m in $MODELS; do
  for c in C0 C1 C2; do $PY src/run_experiments.py --model "$m" --condition $c --tag main; done
done
for m in $MODELS; do
  for c in C0 C1; do $PY src/run_experiments.py --model "$m" --condition $c --temperature 0.7 --repeats 3 --subset consistency --tag consistency; done
done
if [ "${ABLATION:-1}" = "1" ]; then
  for m in ${ABLATION_MODELS:-qwen3:30b-a3b qwen3:14b}; do
    for k in 3 10; do $PY src/run_experiments.py --model "$m" --condition C1 --topk $k --tag k$k; done
  done
fi
# status-change arm (pre-specified in DECISIONS.md): closed-book, current index, historical index
if [ "${STATUSCHANGE:-1}" = "1" ]; then
  for m in $MODELS; do
    for c in C0 C1 CH; do
      $PY src/run_experiments.py --model "$m" --condition $c --tag statuschange --items benchmark/items_status_change.jsonl
    done
  done
fi
