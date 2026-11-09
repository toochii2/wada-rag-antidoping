"""Write results/run_manifest.json: models, digests, parameters, seeds, dates, prompt versions."""
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runs = defaultdict(lambda: {"calls": 0, "first": None, "last": None})
for f in sorted((ROOT / "results/runs").glob("*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        k = (r["model"], r["digest"], r["condition"], r["tag"], r["temperature"], r["topk"], r["prompt_version"])
        d = runs[k]
        d["calls"] += 1
        d["seeds"] = sorted(set(d.get("seeds", [])) | {r["seed"]})
        d["parse_failures"] = d.get("parse_failures", 0) + (r.get("parsed") is None)
        d["first"] = min(filter(None, [d["first"], r["timestamp"]]))
        d["last"] = max(filter(None, [d["last"], r["timestamp"]]))
prompt_hash = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((ROOT / "prompts/v1").glob("*"))}
passb = [json.loads(l) for y in (2026, 2027) for l in open(ROOT / f"data/processed/passB_raw_{y}.jsonl")] \
    if (ROOT / "data/processed/passB_raw_2027.jsonl").exists() else []
try:
    ov = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip() or \
        subprocess.run(["ollama", "--version"], capture_output=True, text=True).stderr.strip()
except Exception:
    ov = "unknown"
man = {
    "hardware": "Apple M4 Pro, 24 GB unified memory", "platform": platform.platform(), "python": platform.python_version(),
    "ollama": ov, "num_ctx": 8192, "num_predict": 512, "think": False, "base_seed": 42,
    "embedding_model": "bge-m3", "retrieval": "BM25 (rank_bm25 Okapi) + bge-m3 cosine, RRF k=60, top-k 6 (ablation 3, 10)",
    "prompt_version": "v1", "prompt_sha256_16": prompt_hash,
    "pass_b_extraction": {"model": passb[0]["model"] if passb else None, "temperature": 0, "seed": 42,
                          "chunks": len(passb)},
    "runs": [{"model": k[0], "digest": k[1], "condition": k[2], "tag": k[3], "temperature": k[4], "topk": k[5],
              "prompt_version": k[6], **v} for k, v in sorted(runs.items())],
}
json.dump(man, open(ROOT / "results/run_manifest.json", "w"), indent=1)
print(f"manifest: {len(man['runs'])} run groups")
