"""Row-level delta between the 2026 and 2027 rule tables, cross-checked against
the 2027 Summary of Major Modifications and the 2027 redline (tracked changes,
read from the coloured spans of the redline PDF). News coverage is not used.

Writes data/processed/list_delta_2026_2027.csv and data/processed/redline_fragments_2027.json
"""
import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_rules import norm, text_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REDLINE_COLOUR = 0xC75500  # colour of inserted and deleted text in the WADA redline


def redline_fragments():
    d = pymupdf.open(ROOT / "data/raw/2027_list_redline.pdf")
    frags = []
    for pno, p in enumerate(d):
        if pno >= 20:  # index pages repeat the body
            break
        line_txt = []
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"] if s["color"] == REDLINE_COLOUR)
                if t.strip():
                    line_txt.append(t.strip())
        if line_txt:
            frags.append({"page": pno + 1, "text": " ".join(line_txt)})
    return frags


def summary_text():
    d = pymupdf.open(ROOT / "data/raw/2027_summary_of_modifications.pdf")
    return " ".join(" ".join(p.get_text() for p in d).split())


def cond(r):
    return " | ".join(x for x in (str(r.get("route_exceptions", "")), str(r.get("dose_thresholds", "")),
                                  str(r.get("other_conditions", "")), str(r.get("sports", ""))) if x.strip())


def cond_norm(s):
    s = text_norm(s.replace("2026", "YYYY").replace("2027", "YYYY"))
    return re.sub(r"[^a-z0-9 ]", "", s)


def main():
    r26 = pd.read_csv(ROOT / "data/processed/rules_2026.csv").fillna("")
    r27 = pd.read_csv(ROOT / "data/processed/rules_2027.csv").fillna("")
    # rows confirmed by the deterministic pass; conditions compared on the Pass A text, because the
    # LLM pass words the same condition differently from one year to the next
    r26 = r26[r26["benchmark_eligible"].astype(str) == "True"]
    r27 = r27[r27["benchmark_eligible"].astype(str) == "True"]
    pa = {y: {norm(x["substance"]): x for x in pd.read_csv(ROOT / f"data/processed/rules_{y}_passA.csv").fillna("")
              .to_dict("records")} for y in (2026, 2027)}

    def with_a(rec, y):
        k = norm(rec["substance"])
        src = pa[y].get(k, rec)
        return {**rec, **{c: src.get(c, "") for c in ("route_exceptions", "dose_thresholds", "other_conditions", "sports")}}
    a = {norm(x["substance"]): with_a(x, 2026) for x in r26.to_dict("records")}
    b = {norm(x["substance"]): with_a(x, 2027) for x in r27.to_dict("records")}
    summ = summary_text()
    summ_n = text_norm(summ)
    frags = redline_fragments()
    red_n = text_norm(" ".join(f["text"] for f in frags))
    rows = []
    for k in list(dict.fromkeys(list(a) + list(b))):
        x, y = a.get(k), b.get(k)
        if x and y:
            if x["status"] != y["status"]:
                ctype = "changed_status"
            elif cond_norm(cond(x)) != cond_norm(cond(y)):
                ctype = "changed_condition"
            else:
                continue
        else:
            ctype = "added" if y else "removed"
        name = (y or x)["substance"]
        rows.append({"change_type": ctype, "substance": name,
                     "clause_ref_2026": x["clause_ref"] if x else "", "clause_ref_2027": y["clause_ref"] if y else "",
                     "status_2026": x["status"] if x else "not_named", "status_2027": y["status"] if y else "not_named",
                     "conditions_2026": cond(x) if x else "", "conditions_2027": cond(y) if y else "",
                     "in_summary": text_norm(norm(name)) in summ_n,
                     "in_redline": text_norm(norm(name)) in red_n})
    # clause-level wording changes, so that route and sport changes are captured even
    # when no named row changes
    c26 = {json.loads(l)["clause_ref"]: json.loads(l)["text"] for l in open(ROOT / "data/processed/chunks_2026.jsonl")}
    c27 = {json.loads(l)["clause_ref"]: json.loads(l)["text"] for l in open(ROOT / "data/processed/chunks_2027.jsonl")}
    for ref in sorted(set(c26) | set(c27)):
        t1 = c26.get(ref, "").replace("2026", "YYYY")
        t2 = c27.get(ref, "").replace("2027", "YYYY")
        if t1 == t2:
            continue
        sm = difflib.SequenceMatcher(None, t1.split(), t2.split())
        ins = " ".join(" ".join(t2.split()[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op in ("insert", "replace"))
        dele = " ".join(" ".join(t1.split()[i1:i2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op in ("delete", "replace"))
        rows.append({"change_type": "clause_text_changed", "substance": "", "clause_ref_2026": ref if ref in c26 else "",
                     "clause_ref_2027": ref if ref in c27 else "", "status_2026": "", "status_2027": "",
                     "conditions_2026": dele[:300], "conditions_2027": ins[:300],
                     "in_summary": ref.split(".")[0] in summ, "in_redline": bool(ins) and text_norm(ins[:30]) in red_n})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/processed/list_delta_2026_2027.csv", index=False)
    with open(ROOT / "data/processed/redline_fragments_2027.json", "w") as f:
        json.dump(frags, f, indent=1, ensure_ascii=False)
    print(df[["change_type", "substance", "clause_ref_2027", "status_2026", "status_2027", "in_summary",
              "in_redline"]].to_string())


if __name__ == "__main__":
    main()
