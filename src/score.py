"""Automatic scoring of model outputs against gold labels. No LLM grader.

Harm classes (pre-specified before any model run, see DECISIONS.md):
  gold prohibited_*        model permitted                    false_permit        10
  gold prohibited_*        model permitted_with_conditions    conditional_permit   5
  gold prohibited_*        model narrower prohibition scope   scope_understated    5
  gold permitted_with_cond model permitted                    missed_condition     5
  gold refer               model permitted or p_w_c           unsafe_assurance     5
  gold permitted or p_w_c  model prohibited_*                 false_prohibit       2
  gold not refer           model refer                        unnecessary_deferral 1
  gold refer               model prohibited_*                 over_caution         1
  gold prohibited_*        model broader prohibition scope    scope_overstated     1
  otherwise (verdict equal, or gold permitted with model p_w_c)   correct          0
Unparseable output counts as unnecessary_deferral (no usable answer) and is reported separately.

Writes results/scored.csv
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROH = ["prohibited_at_all_times", "prohibited_in_competition_only", "prohibited_in_particular_sports"]
PERM = ["permitted", "permitted_with_conditions"]
SCOPE = {"prohibited_at_all_times": 3, "prohibited_in_competition_only": 2, "prohibited_in_particular_sports": 1}
WEIGHTS = {"false_permit": 10, "conditional_permit": 5, "scope_understated": 5, "missed_condition": 5,
           "unsafe_assurance": 5, "false_prohibit": 2, "unnecessary_deferral": 1, "over_caution": 1,
           "scope_overstated": 1, "correct": 0}


def collapse(v):
    return "prohibited" if v in PROH else "permitted" if v in PERM else "refer" if v == "cannot_determine_refer" else "none"


def harm_class(gold, pred):
    if pred is None:
        return "unnecessary_deferral"
    if gold == pred or (gold == "permitted" and pred == "permitted_with_conditions"):
        return "correct"
    if gold in PROH:
        if pred == "permitted":
            return "false_permit"
        if pred == "permitted_with_conditions":
            return "conditional_permit"
        if pred in PROH:
            return "scope_understated" if SCOPE[pred] < SCOPE[gold] else "scope_overstated"
        return "unnecessary_deferral"
    if gold == "permitted_with_conditions":
        if pred == "permitted":
            return "missed_condition"
        if pred in PROH:
            return "false_prohibit"
        return "unnecessary_deferral"
    if gold == "permitted":
        return "false_prohibit" if pred in PROH else "unnecessary_deferral"
    # gold refer
    return "unsafe_assurance" if pred in PERM else "over_caution"


def norm_text(s):
    s = s.lower().replace("µg", "micrograms").replace("mcg", "micrograms")
    return re.sub(r"[\s\-]+", " ", s)


def conditions_complete(tokens, parsed):
    """All required token groups present in conditions or explanation (any synonym in a group)."""
    if not tokens:
        return None
    txt = norm_text(" ".join(parsed.get("conditions", [])) + " " + parsed.get("explanation", ""))
    return all(any(norm_text(t) in txt for t in grp) for grp in tokens)


def cite_ok(gold_ref, cited):
    if not gold_ref:
        return None
    full = gold_ref.lower()
    for c in cited:
        c = c.strip("[] ").lower()
        # equal, or the cited clause contains the gold clause (a parent such as S6 for S6.footnotes)
        if c == full or full.startswith(c + ".") or full in c:
            return True
    return False


def recall_at_k(gold_ref, retrieved):
    if not gold_ref or not retrieved or gold_ref in ("Monitoring Program",):
        return None
    sec = gold_ref.split(".")[0]
    refs = [r["clause_ref"] for r in retrieved]
    return {"exact": gold_ref in refs, "section": any(r.split(".")[0] == sec for r in refs)}


def text_flags(it, p):
    """What the scoring needs from an answer's free text: an 'out-of-competition use only' statement (which states
    the in-competition-only status; 'in and out of competition' does not), a TUE mention, and whether every required
    condition is stated. The public release keeps these flags and drops the text, which can quote the List."""
    text = (" ".join(p.get("conditions", [])) + " " + p.get("explanation", "")).lower()
    ooc = r"out[ -]of[ -]competition"
    return {"ooc_only": bool(re.search(rf"{ooc}( use)? only|only (allowed |permitted |used )?{ooc}", text))
            and not re.search(rf"in[- ]+(and|or)[ -]+{ooc}|both in and out", text),
            "tue": bool(re.search(r"\btue\b|therapeutic use exemption", text)),
            "cond_complete": conditions_complete(it["gold"].get("condition_tokens"), p) if p else None,
            "cite_ok": cite_ok(it["gold"]["clause_ref"], p.get("cited_clauses", [])) if p else None}


def main():
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "benchmark/items.jsonl")}
    sc = ROOT / "benchmark/items_status_change.jsonl"
    if sc.exists():  # status-change arm (group H)
        items.update({json.loads(l)["item_id"]: json.loads(l) for l in open(sc)})
    rows = []
    seen = set()
    for f in sorted((ROOT / "results/runs").glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            it = items.get(r["item_id"])
            if it is None or r.get("question") != it["question"]:
                continue  # stale output for an item whose question has since changed
            key = (r["model"], r["condition"], r["tag"], r["item_id"], r["repeat"])
            if key in seen:
                continue  # one pass per item: keep the first record (a later resume re-ran some unparsed outputs)
            seen.add(key)
            gold = it["gold"]["verdict"]
            p = r.get("parsed") or {}
            pred = p.get("verdict")
            # facts read from the answer text; the public release stores these instead of the text itself
            fl = p.get("text_flags") or text_flags(it, p)
            accepted = pred is not None and pred in it["gold"].get("accept", [])
            accepted = accepted or (gold == "prohibited_in_competition_only" and pred == "permitted_with_conditions"
                                    and fl["ooc_only"])
            hc = "correct" if accepted else harm_class(gold, pred)
            tue = fl["tue"]
            rec = recall_at_k(it["gold"]["clause_ref"], r.get("retrieved"))
            rows.append({
                "item_id": r["item_id"], "group": it["group"], "model": r["model"], "condition": r["condition"],
                "tag": r["tag"], "repeat": r["repeat"], "temperature": r["temperature"], "topk": r["topk"],
                "gold": gold, "pred": pred, "parsed_ok": pred is not None,
                "correct6": pred == gold or accepted, "gold_c": collapse(gold), "pred_c": collapse(pred),
                "correct_c": collapse(gold) == collapse(pred) or accepted,
                "accepted_alt": accepted, "tue_mention": tue,
                "lenient_permit_no_tue": int(gold in PROH and not accepted and (
                    pred == "permitted" or (pred == "permitted_with_conditions" and not tue))),
                "newly_named": bool(it.get("newly_named")),
                "literal_stale": (int(pred == it.get("label_2026_literal")) if it.get("newly_named") else None),
                "false_permit": int(gold in PROH and pred == "permitted"),
                "lenient_permit": int(gold in PROH and pred in PERM and not accepted),
                "permit_on_permitted": int(pred in PERM) if gold in PERM else None,
                "substance": it["substance"],
                "false_prohibit": int(gold in PERM and pred in PROH),
                "deferral": int(pred == "cannot_determine_refer" and not accepted),
                "appropriate": int(pred == "cannot_determine_refer" or accepted),  # Group D: refer, or an accepted alternative
                "harm_class": hc, "harm": WEIGHTS[hc],
                "cond_complete": fl["cond_complete"] if p else (False if it["gold"].get("condition_tokens") else None),
                "cite_ok": fl.get("cite_ok") if p else None,
                "recall_exact": rec["exact"] if rec else None, "recall_section": rec["section"] if rec else None,
                "label_2026": it.get("label_2026"), "label_changed": it.get("label_changed"),
                "stale": (int(pred == (it.get("label_2026") or it.get("label_old"))) if it.get("label_changed") else None),
                "stale_year": it.get("stale_year"),
                "is_brand": it["is_brand"], "template_id": it["template_id"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/scored.csv", index=False)
    print(df.groupby(["model", "condition", "tag"]).size().to_string())


if __name__ == "__main__":
    main()
