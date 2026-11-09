"""Thin Ollama client: structured JSON output, thinking off, resumable callers."""
import json
import time

import requests

OLLAMA = "http://localhost:11434"


def chat(model, messages, schema, temperature=0.0, seed=42, num_ctx=8192, num_predict=1024,
         timeout=600, retries=3):
    body = {
        "model": model, "messages": messages, "format": schema, "stream": False,
        "think": False, "keep_alive": "30m",
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
            if r.status_code == 400 and "think" in r.text:
                body.pop("think")  # model without a thinking switch
                r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "")
            if not data.get("done") or not content:
                # Ollama answers 200 with an empty, unfinished message when the runner crashed
                # (for example GPU out of memory); treat that as a failure and retry
                raise RuntimeError(f"empty or unfinished response: {str(data)[:200]}")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            return {"raw": content, "parsed": parsed,
                    "eval_count": data.get("eval_count"), "prompt_eval_count": data.get("prompt_eval_count"),
                    "total_duration_s": (data.get("total_duration") or 0) / 1e9}
        except Exception as e:  # network hiccup or model reload
            last = e
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after {retries} attempts: {last}")


def embed(texts, model="bge-m3"):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": model, "input": texts}, timeout=600)
    r.raise_for_status()
    return r.json()["embeddings"]


def model_digest(model):
    r = requests.get(f"{OLLAMA}/api/tags", timeout=30)
    for m in r.json().get("models", []):
        if m["name"] == model or m["model"] == model:
            return m.get("digest")
    return None
