"""Run one model under one condition over the benchmark. Resumable: one JSONL line per call.

Conditions: C0 closed-book, C1 RAG over the 2027 index, C2 RAG over the 2026 (stale) index,
CH RAG over each item's historical index (stale_year; status-change arm, --items benchmark/items_status_change.jsonl).
The question always carries the 2027 date frame.

Usage:
  python src/run_experiments.py --model qwen3:14b --condition C1 [--temperature 0] [--repeats 1]
         [--subset consistency] [--topk 6] [--tag main]
Writes results/runs/<model>__<condition>__<tag>.jsonl
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat, model_digest  # noqa: E402
from retriever import Index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "v1"
SEED = 42


def load_prompts():
    p = ROOT / "prompts" / PROMPT_VERSION
    return {k: (p / f"{k}.txt").read_text() for k in ("system_rag", "system_closed", "user_rag", "user_closed")}, \
        json.loads((p / "schema.json").read_text())


def items(subset, path="benchmark/items.jsonl"):
    its = [json.loads(l) for l in open(ROOT / path)]
    if subset:
        ids = set(json.load(open(ROOT / f"benchmark/subset_{subset}.json")))
        its = [i for i in its if i["item_id"] in ids]
    return its


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["C0", "C1", "C2", "CH"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--subset", default="")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--items", default="benchmark/items.jsonl")
    a = ap.parse_args()
    prompts, schema = load_prompts()
    idx = None if a.condition in ("C0", "CH") else Index(2027 if a.condition == "C1" else 2026)
    hist = {}  # CH: one index per historical List year
    out = ROOT / "results/runs" / f"{a.model.replace(':', '_').replace('/', '_')}__{a.condition}__{a.tag}.jsonl"
    done = set()  # an item is redone if its question text changed since it was run
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("parsed") is not None or r.get("final_failure") or r.get("raw"):  # unparsed text is a completed pass
                done.add((r["item_id"], r["repeat"], r.get("question")))
    digest = model_digest(a.model)
    its = items(a.subset, a.items)
    n = 0
    t0 = time.time()
    with open(out, "a") as f:
        for rep in range(a.repeats):
            for it in its:
                if (it["item_id"], rep, it["question"]) in done:
                    continue
                seed = SEED + rep
                ix = idx
                if a.condition == "CH":
                    ix = hist.setdefault(it["stale_year"], Index(it["stale_year"]))
                if ix:
                    hits = ix.search(it.get("retrieval_query") or it["question"], k=a.topk)
                    clauses = "\n\n".join(f"[{h['clause_ref']}] {h['text']}" for h in hits)
                    msgs = [{"role": "system", "content": prompts["system_rag"]},
                            {"role": "user", "content": prompts["user_rag"].format(clauses=clauses, question=it["question"])}]
                else:
                    hits = []
                    msgs = [{"role": "system", "content": prompts["system_closed"]},
                            {"role": "user", "content": prompts["user_closed"].format(question=it["question"])}]
                try:
                    res = chat(a.model, msgs, schema, temperature=a.temperature, seed=seed, num_predict=512)
                    fail = None
                except RuntimeError as e:
                    res, fail = {"raw": "", "parsed": None}, str(e)
                rec = {"item_id": it["item_id"], "question": it["question"], "group": it["group"], "model": a.model, "digest": digest,
                       "condition": a.condition, "tag": a.tag, "repeat": rep, "seed": seed,
                       "temperature": a.temperature, "topk": a.topk if ix else 0, "prompt_version": PROMPT_VERSION,
                       "retrieved": [{k: h[k] for k in h if k != "text"} for h in hits],
                       "timestamp": dt.datetime.now().isoformat(timespec="seconds"), **res,
                       "final_failure": fail}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0:
                    print(f"{a.model} {a.condition} {a.tag}: {n} calls, {(time.time() - t0) / n:.1f}s/call", flush=True)
    print(f"done {a.model} {a.condition} {a.tag}: {n} new calls")


if __name__ == "__main__":
    main()
