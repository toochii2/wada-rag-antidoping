"""Hybrid clause retrieval per List year: BM25 plus bge-m3 dense vectors, fused by
reciprocal rank fusion (k = 60). Embeddings are cached in data/processed/emb_<year>.npy."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import embed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "bge-m3"
RRF_K = 60


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("ɑ", "a").replace("ß", "b"))


def chunk_text(c):
    return f"{c['clause_ref']} {c['section_title']}. {c['text']}"


QCACHE = ROOT / "data/processed/query_emb_cache.json"
_qcache = None


def query_vector(query):
    """Query embeddings are precomputed (see precompute()), so the embedding model never has to be
    loaded next to a large chat model during a run."""
    global _qcache
    if _qcache is None:
        _qcache = json.load(open(QCACHE)) if QCACHE.exists() else {}
    if query not in _qcache:
        _qcache[query] = embed([query], EMBED_MODEL)[0]
        json.dump(_qcache, open(QCACHE, "w"))
    return np.array(_qcache[query], dtype=np.float32)


def precompute():
    items = [json.loads(l) for l in open(ROOT / "benchmark/items.jsonl")]
    sc = ROOT / "benchmark/items_status_change.jsonl"
    if sc.exists():
        items += [json.loads(l) for l in open(sc)]
    qs = [i.get("retrieval_query") or i["question"] for i in items]
    cache = json.load(open(QCACHE)) if QCACHE.exists() else {}
    todo = [q for q in dict.fromkeys(qs) if q not in cache]
    for j in range(0, len(todo), 32):
        for q, v in zip(todo[j:j + 32], embed(todo[j:j + 32], EMBED_MODEL)):
            cache[q] = v
    json.dump(cache, open(QCACHE, "w"))
    for y in (2021, 2023, 2026, 2027):
        if (ROOT / f"data/processed/chunks_{y}.jsonl").exists():
            Index(y)  # builds and caches the clause embeddings
    print(f"query embeddings cached: {len(cache)}")


class Index:
    def __init__(self, year):
        self.year = year
        self.chunks = [json.loads(l) for l in open(ROOT / f"data/processed/chunks_{year}.jsonl")]
        self.texts = [chunk_text(c) for c in self.chunks]
        self.bm25 = BM25Okapi([tok(t) for t in self.texts])
        path = ROOT / f"data/processed/emb_{year}.npy"
        ids_path = ROOT / f"data/processed/emb_{year}_ids.json"
        ids = [c["chunk_id"] for c in self.chunks]
        if path.exists() and ids_path.exists() and json.load(open(ids_path)) == ids:
            self.emb = np.load(path)
        else:
            self.emb = np.array(embed(self.texts, EMBED_MODEL), dtype=np.float32)
            np.save(path, self.emb)
            json.dump(ids, open(ids_path, "w"))
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)

    def search(self, query, k=6):
        bm = self.bm25.get_scores(tok(query))
        q = query_vector(query)
        dn = self.emb @ (q / np.linalg.norm(q))
        rb = {i: r for r, i in enumerate(np.argsort(-bm))}
        rd = {i: r for r, i in enumerate(np.argsort(-dn))}
        fused = {i: 1 / (RRF_K + rb[i] + 1) + 1 / (RRF_K + rd[i] + 1) for i in range(len(self.chunks))}
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{"rank": n + 1, "chunk_id": self.chunks[i]["chunk_id"], "clause_ref": self.chunks[i]["clause_ref"],
                 "year": self.year, "section": self.chunks[i]["section"], "rrf": round(fused[i], 5),
                 "bm25_rank": rb[i] + 1, "dense_rank": rd[i] + 1, "text": self.chunks[i]["text"]} for n, i in enumerate(top)]


if __name__ == "__main__":
    precompute()
