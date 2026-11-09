"""Reconcile Pass A and Pass B into the rule table for one year.

Matching key: normalised substance name (asterisks, bracketed text and case
removed). A row is in agreement when both passes extracted it with the same
status. Disagreements are resolved against the source clause text:
  * name only in one pass: kept if it appears verbatim in the clause text,
    otherwise dropped;
  * status differs: the status is re-derived from the clause text (section
    status, unless the name appears in an exception, note or footnote clause
    that says it is not prohibited or not considered prohibited);
  * if the clause text still does not settle it, the row is marked ambiguous
    and excluded from benchmark Groups A to C.
Every disagreement is flagged for pharmacist review.

Usage: python src/reconcile_rules.py <year>
Writes data/processed/rules_<year>.csv and results/rule_extraction_agreement_<year>.json
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SECTION_STATUS = {"S0": "all_times", "S1": "all_times", "S2": "all_times", "S3": "all_times",
                  "S4": "all_times", "S5": "all_times", "M1": "all_times", "M2": "all_times",
                  "M3": "all_times", "S6": "in_competition", "S7": "in_competition",
                  "S8": "in_competition", "S9": "in_competition", "P1": "particular_sports"}


def strip_groups(s):
    out, depth = "", 0
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out += ch
    return out if out.strip() else s


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("ɑ", "α")
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\b(and its|and their) .*$", "", s)
    s = strip_groups(s)  # bracketed synonyms and chemical names, nested or not
    s = re.sub(r"\bs\d(\.\d)*\.?\s.*$", "", s)  # a heading glued to the end of a name
    s = re.sub(r"[\s\-]+", " ", s).strip(" .,;:")
    return s


def text_norm(s):
    return re.sub(r"[\s\-]+", " ", unicodedata.normalize("NFKC", s).lower().replace("ɑ", "α"))


def collapse(df):
    """One row per normalised name; prefer the substantive clause over notes and footnotes."""
    df = df.copy()
    df["key"] = df["substance"].map(norm)
    df["is_note"] = df["clause_ref"].str.contains("note|exceptions|footnotes", regex=True)
    out = {}
    for key, g in df.groupby("key", sort=False):
        main = g[~g["is_note"]]
        row = (main if len(main) else g).iloc[0].to_dict()
        notes = g[g["is_note"]]
        # a not-prohibited entry in an exception/footnote clause overrides the section status
        if len(notes) and (notes["status"] == "not_prohibited").any() and not len(main):
            row["status"] = "not_prohibited"
        for col in ("route_exceptions", "dose_thresholds", "other_conditions"):
            vals = [str(v) for v in g[col].fillna("") if str(v).strip()]
            row[col] = "; ".join(dict.fromkeys(vals))
        row["all_refs"] = "; ".join(dict.fromkeys(g["clause_ref"]))
        out[key] = row
    return out


METHOD_CLAUSES = {"M1.1", "M1.2", "M1.3", "M1.4", "M2.1", "M2.2", "M3.1", "M3.2"}


def derive_status(name, clause_ref, chunks):
    """Status re-derived from the clause text (tie-break rule)."""
    tn = text_norm(name)
    for c in chunks:
        if c["section"] != clause_ref[:2]:
            continue
        if any(k in c["clause_ref"] for k in ("exceptions", "note", "footnotes")) and tn in text_norm(c["text"]):
            t = text_norm(c["text"])
            seg = t[t.find(tn):t.find(tn) + 400]
            if re.search(r"not (be )?(considered )?prohibited|not prohibited|exceptions", seg) or \
                    "exceptions" in c["clause_ref"]:
                if "greater than" in seg[:200] or "in excess of" in seg[:200]:
                    continue  # threshold footnote, not an exception
                return "not_prohibited"
    return SECTION_STATUS.get(clause_ref[:2])


def main():
    year = int(sys.argv[1])
    chunks = [json.loads(l) for l in open(ROOT / f"data/processed/chunks_{year}.jsonl")]
    by_ref = {c["clause_ref"]: c for c in chunks}
    A = pd.read_csv(ROOT / f"data/processed/rules_{year}_passA.csv").fillna("")
    B = pd.read_csv(ROOT / f"data/processed/rules_{year}_passB.csv").fillna("")
    a, b = collapse(A), collapse(B)
    keys = list(dict.fromkeys(list(a) + list(b)))
    rows, n_agree, n_status_disagree, n_only_a, n_only_b = [], 0, 0, 0, 0
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        base = ra or rb
        ref = base["clause_ref"]
        src = by_ref.get(ref, {}).get("text", "")
        flag, resolution, ambiguous = "", "agree", False
        if ra and rb:
            if ra["status"] == rb["status"]:
                n_agree += 1
                status = ra["status"]
            else:
                n_status_disagree += 1
                status = derive_status(base["substance"], ref, chunks)
                flag, resolution = "red", f"status A={ra['status']} B={rb['status']}; re-derived from text"
                if status not in (ra["status"], rb["status"]):
                    ambiguous = True
        else:
            only = "A" if ra else "B"
            n_only_a += ra is not None
            n_only_b += rb is not None
            verbatim = text_norm(norm(base["substance"])) in text_norm(src) if src else False
            if not verbatim:
                # also look in the other clauses of the section (names in notes and footnotes)
                verbatim = any(text_norm(norm(base["substance"])) in text_norm(c["text"])
                               for c in chunks if c["section"] == ref[:2])
            flag = "red"
            method_label = ra is not None and ref[:1] == "M" and ref in METHOD_CLAUSES
            if not verbatim and method_label:
                status = SECTION_STATUS[ref[:2]] if base["status"] != "not_prohibited" else "not_prohibited"
                rows.append({**ra, "status": status, "flag": flag, "ambiguous": False, "dropped": False,
                             "resolution": "only in Pass A; clause-level label for a prohibited method; kept",
                             "in_pass_a": True, "in_pass_b": False})
                continue
            if not verbatim:
                resolution = f"only in Pass {only}; not found verbatim in clause text; dropped"
                rows.append({**base, "status": base["status"], "flag": flag, "resolution": resolution,
                             "ambiguous": True, "dropped": True, "in_pass_a": ra is not None,
                             "in_pass_b": rb is not None})
                continue
            status = derive_status(base["substance"], ref, chunks)
            resolution = f"only in Pass {only}; found verbatim in clause text; kept"
        merged = dict(ra or rb)
        if ra and rb:  # union of conditions from both passes, Pass A text first
            for col in ("route_exceptions", "dose_thresholds", "other_conditions"):
                vals = [v for v in (str(ra.get(col, "")), str(rb.get(col, ""))) if v.strip()]
                merged[col] = " | ".join(dict.fromkeys(vals))
            if not merged.get("synonyms") and rb.get("synonyms"):
                merged["synonyms"] = rb["synonyms"]
        rows.append({**merged, "status": status, "flag": flag, "resolution": resolution,
                     "ambiguous": ambiguous, "dropped": False, "in_pass_a": ra is not None,
                     "in_pass_b": rb is not None})
    df = pd.DataFrame(rows)
    # rows extracted only by the LLM pass include non-substance phrases (e.g. 'inhalation'), so the
    # benchmark draws only on rows confirmed by the deterministic pass
    df["benchmark_eligible"] = (~df["dropped"].astype(bool)) & (~df["ambiguous"].astype(bool)) & df["in_pass_a"].astype(bool)
    keep = ["year", "substance", "synonyms", "is_class", "parent_class", "status", "sports",
            "route_exceptions", "dose_thresholds", "other_conditions", "tue_note", "clause_ref",
            "all_refs", "page", "source_text_snippet", "in_pass_a", "in_pass_b", "flag", "resolution",
            "ambiguous", "dropped", "benchmark_eligible"]
    for col in keep:
        if col not in df:
            df[col] = ""
    df = df[keep].fillna("")
    df = df.rename(columns={"is_class": "class"})
    # snippets for rows that came only from Pass B
    for i, r in df.iterrows():
        if not r["source_text_snippet"]:
            t = by_ref.get(r["clause_ref"], {}).get("text", "")
            j = t.lower().find(str(r["substance"]).lower()[:20])
            df.at[i, "source_text_snippet"] = t[max(0, j - 40):j + 160] if j >= 0 else t[:200]
        df.at[i, "source_text_snippet"] = str(df.at[i, "source_text_snippet"])[:200]
        if r["clause_ref"][:2] == "P1" and not r["sports"]:
            df.at[i, "sports"] = a.get(norm(r["substance"]), {}).get("sports", "") or next(
                (x["sports"] for x in a.values() if x["clause_ref"] == "P1" and x.get("sports")), "")
        if r["status"] != "not_prohibited" and not r["tue_note"]:
            df.at[i, "tue_note"] = "TUE possible for therapeutic use"
    df.insert(0, "rule_id", [f"R{year}-{i:04d}" for i in range(len(df))])
    df.to_csv(ROOT / f"data/processed/rules_{year}.csv", index=False)
    union = len(keys)
    stats = {"year": year, "rows_pass_a": len(a), "rows_pass_b": len(b), "union_rows": union,
             "agree_name_and_status": n_agree, "status_disagree": n_status_disagree,
             "only_pass_a": n_only_a, "only_pass_b": n_only_b,
             "row_level_agreement": round(n_agree / union, 4) if union else None,
             "rows_kept": int((~df["dropped"].astype(bool)).sum()),
             "rows_ambiguous_kept": int((df["ambiguous"].astype(bool) & ~df["dropped"].astype(bool)).sum()),
             "definition": "agreement = names extracted by both passes with identical status / union of "
                           "normalised names from either pass"}
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / f"results/rule_extraction_agreement_{year}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
