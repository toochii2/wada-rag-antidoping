"""Pass B: rule-table extraction with a local LLM, chunk by chunk.

Independent of Pass A: the model sees only the clause text (plus the NOTE,
EXCEPTIONS and footnote clauses of the same section, for conditions) and a JSON
schema. Temperature 0, thinking off. Resumable: one JSONL line per chunk.

Usage: python src/rules_pass_b.py <year> [model]
Writes data/processed/passB_raw_<year>.jsonl and rules_<year>_passB.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3:30b-a3b"

SCHEMA = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "substance": {"type": "string"},
            "synonyms": {"type": "array", "items": {"type": "string"}},
            "is_class": {"type": "boolean"},
            "status": {"type": "string", "enum": ["all_times", "in_competition", "particular_sports",
                                                  "not_prohibited"]},
            "route_exceptions": {"type": "string"},
            "dose_thresholds": {"type": "string"},
            "other_conditions": {"type": "string"},
        },
        "required": ["substance", "synonyms", "is_class", "status", "route_exceptions",
                     "dose_thresholds", "other_conditions"]}}},
    "required": ["rows"],
}

PROMPT_VERSION = "v2"  # v1 omitted examples listed after "e.g."; see DECISIONS.md
SYSTEM = """You extract a rule table from one clause of the WADA Prohibited List.
Return every substance, named class of substances, or method that is NAMED in the CLAUSE text, one row each.
Be exhaustive: every bullet point is its own row, and every example given after "e.g.", "such as" or
"including" is its own row, as well as the class it belongs to. A clause can contain 60 or more rows.
- substance: the name as written (without asterisks); put names given in brackets in synonyms.
- is_class: true for a named group (e.g. "growth hormone fragments"), false for a single substance.
- status: "all_times" if the section is prohibited at all times; "in_competition" if prohibited in-competition only;
  "particular_sports" for beta-blockers in section P1; "not_prohibited" if the clause says the item is not prohibited,
  is an exception, or is in the Monitoring Program and not considered prohibited.
- route_exceptions, dose_thresholds, other_conditions: copy any condition that applies to that item from the
  CLAUSE or the RELATED NOTES (routes that are permitted, urinary thresholds, maximum inhaled doses, sports,
  "in males", "by intravenous administration"). Use "" if none.
Do not add substances that are not written in the CLAUSE. Do not use outside knowledge."""

SECTION_STATUS_TEXT = {
    "all_times": "This section is PROHIBITED AT ALL TIMES (in- and out-of-competition).",
    "in_competition": "This section is PROHIBITED IN-COMPETITION.",
    "particular_sports": "This section is PROHIBITED IN PARTICULAR SPORTS.",
}
SKIP = {"INTRO", "S1", "S2", "S2.1", "S2.2", "S4", "S4.4", "M1", "M2", "M3", "S6", "S3.note", "S5.note"}


def main():
    year = int(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    chunks = [json.loads(l) for l in open(ROOT / f"data/processed/chunks_{year}.jsonl")]
    raw_path = ROOT / f"data/processed/passB_raw_{year}.jsonl"
    done = set()
    if raw_path.exists():
        done = {(json.loads(l)["chunk_id"], json.loads(l).get("text_hash")) for l in open(raw_path)}
    with open(raw_path, "a") as f:
        for c in chunks:
            th = hashlib.sha1(c["text"].encode()).hexdigest()[:12]
            if c["clause_ref"] in SKIP or (c["chunk_id"], th) in done:
                continue
            sec = c["section"]
            related = [x for x in chunks if x["section"] == sec and x["chunk_id"] != c["chunk_id"]
                       and any(k in x["clause_ref"] for k in ("note", "exceptions", "footnotes"))]
            user = (f"YEAR: {year}\nSECTION: {sec} {c['section_title']}. "
                    f"{SECTION_STATUS_TEXT.get(c['section_status'], '')}\n"
                    f"CLAUSE [{c['clause_ref']}]:\n{c['text']}\n\n"
                    "RELATED NOTES (for conditions only):\n"
                    + ("\n".join(f"[{x['clause_ref']}] {x['text']}" for x in related) or "none"))
            out = chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       SCHEMA, temperature=0.0, seed=42, num_predict=8192, num_ctx=12288)
            f.write(json.dumps({"chunk_id": c["chunk_id"], "clause_ref": c["clause_ref"], "page": c["page"],
                                "section": sec, "model": model, "prompt_version": PROMPT_VERSION, "text_hash": th, **out}, ensure_ascii=False) + "\n")
            f.flush()
            print(c["chunk_id"], len((out["parsed"] or {}).get("rows", [])), flush=True)
    rows = []
    current = {(c["chunk_id"], hashlib.sha1(c["text"].encode()).hexdigest()[:12]) for c in chunks}
    for l in open(raw_path):
        r = json.loads(l)
        if (r["chunk_id"], r.get("text_hash")) not in current:
            continue  # output for an older version of the chunk
        for x in (r["parsed"] or {}).get("rows", []):
            rows.append(dict(year=year, substance=x["substance"].strip(), synonyms="; ".join(x["synonyms"]),
                             is_class=x["is_class"], clause_ref=r["clause_ref"], section=r["section"],
                             page=r["page"], status=x["status"], route_exceptions=x["route_exceptions"],
                             dose_thresholds=x["dose_thresholds"], other_conditions=x["other_conditions"]))
    df = pd.DataFrame(rows).drop_duplicates(subset=["substance", "clause_ref"])
    df.insert(0, "rule_id", [f"B{year}-{i:04d}" for i in range(len(df))])
    df.to_csv(ROOT / f"data/processed/rules_{year}_passB.csv", index=False)
    print(f"{year}: {len(df)} Pass B rows")


if __name__ == "__main__":
    main()
