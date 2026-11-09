"""Status-change arm: items whose status really changed between an older List and 2027.

Two documented changes (WADA explanatory notes):
  * glucocorticoids by any injectable route (e.g. intra-articular, epidural, peritendinous) are prohibited
    in-competition from 1 January 2022; the 2021 List prohibited only oral, intravenous, intramuscular
    and rectal routes;
  * tramadol is prohibited in-competition from 1 January 2024; the 2023 List did not name it (S7 is a
    closed list).
Every question keeps a 2027 date frame. The stale index for each item is the List immediately before
the change (2021 or 2023). Controls ask about substances whose status is the same in both Lists.

Writes benchmark/items_status_change.jsonl
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ["under the rules in force on 1 January 2027", "for the 2027 season (rules in force from 1 January 2027)",
          "under the 2027 rules that apply from 1 January 2027"]
IC, P, PC = "prohibited_in_competition_only", "permitted", "permitted_with_conditions"
HARM = {IC: "false_permit", P: "false_prohibit", PC: "missed_condition"}
# conditions follow the main benchmark: other glucocorticoid routes are "not prohibited when used within licensed doses"
COND = {IC: ["in-competition"], PC: ["inhaled route", "within licensed doses"], P: []}


def chunk(year, ref):
    return next(json.loads(l)["text"] for l in open(ROOT / f"data/processed/chunks_{year}.jsonl")
                if json.loads(l)["clause_ref"] == ref)


def check_sources():
    """The labels rest on these readings of the List text; stop if a source does not say so."""
    s9_21, s9_27 = chunk(2021, "S9").lower(), chunk(2027, "S9").lower()
    assert "oral, intravenous, intramuscular or rectal route" in s9_21 and "injectable" not in s9_21
    assert "any injectable" in s9_27
    for y in (2021, 2023):
        assert "tramadol" not in chunk(y, "S7").lower() and "not limited" not in chunk(y, "S7").lower()
    assert "tramadol" in chunk(2027, "S7").lower()


def main():
    check_sources()
    items = []

    def add(template, q, sub, surface, gold, old, stale_year, ref, changed, brand=""):
        n = len(items)
        frame = FRAMES[n % 3]
        question = q.format(frame=frame)
        rq = re.sub(r"\s+([?.,])", r"\1", re.sub(r"\s+", " ", q.replace("{frame}", "")).strip())
        items.append({
            "item_id": f"H{n + 1:03d}", "group": "H", "template_id": template, "question": question[0].upper() + question[1:],
            "retrieval_query": rq[0].upper() + rq[1:], "date_frame": "2027-01-01", "substance": sub, "surface_name": surface,
            "is_brand": bool(brand), "brand": brand, "context": {}, "rule_id": "",
            "gold": {"verdict": gold, "conditions": COND[gold], "clause_ref": ref,
                     "harm_class_if_wrong": HARM[gold], "condition_tokens": []},
            "label_old": old, "stale_year": stale_year, "label_changed": changed, "label_2026": None,
            "newly_named": False, "note": "status-change arm"})

    gc_t = {"H1": "Can I have {act} on the day of my competition {frame}?",
            "H2": "My physio suggests {act} during the competition period. Is that allowed {frame}?",
            "H3": "Is {act} allowed in-competition {frame}?"}
    routes = ["an intra-articular injection of {x} into my knee", "an epidural injection of {x} for my back",
              "a peritendinous injection of {x} around my Achilles tendon"]
    k = 0
    for sub in ["triamcinolone acetonide", "methylprednisolone", "betamethasone", "dexamethasone"]:
        for r in routes:
            tid = list(gc_t)[k % 3]
            brand = "Kenalog" if sub.startswith("triamcinolone") and "knee" in r else ""
            act = r.format(x=brand or sub)
            add(tid, gc_t[tid].replace("{act}", act), sub, brand or sub, IC, P, 2021, "S9", True, brand)
            k += 1
    for tid, q in [("H4", "Can I take tramadol for pain on race day {frame}?"),
                   ("H5", "My doctor prescribed tramadol after an injury. Can I take it during the competition {frame}?"),
                   ("H4", "Is tramadol allowed in-competition {frame}?"),
                   ("H5", "I'm a cyclist. Can I use tramadol during a stage race {frame}?"),
                   ("H4", "Will tramadol taken at 00:30 on the day of my match cause an anti-doping violation {frame}?"),
                   ("H5", "Is it OK to take tramadol painkillers between rounds of a tournament {frame}?"),
                   ("H4", "Can I take a tramadol and paracetamol tablet on competition day {frame}?"),
                   ("H5", "As a footballer, can I take tramadol before kick-off {frame}?")]:
        add(tid, q, "tramadol", "tramadol", IC, P, 2023, "S7", True)
    for q, sub, gold, year, ref in [
            ("Can I take prednisolone tablets on race day {frame}?", "prednisolone", IC, 2021, "S9"),
            ("Can I take dexamethasone tablets during the competition {frame}?", "dexamethasone", IC, 2021, "S9"),
            ("Can I use my fluticasone inhaler on competition day {frame}?", "fluticasone", PC, 2021, "S9"),
            ("Is a budesonide inhaler allowed in-competition {frame}?", "budesonide", PC, 2021, "S9"),
            ("Can I take oxycodone on race day {frame}?", "oxycodone", IC, 2023, "S7"),
            ("Is morphine allowed in-competition {frame}?", "morphine", IC, 2023, "S7"),
            ("Can I take codeine for a headache on competition day {frame}?", "codeine", P, 2023, "S7"),
            ("Is co-codamol (codeine and paracetamol) allowed on race day {frame}?", "codeine", P, 2023, "S7")]:
        add("H6", q, sub, sub, gold, gold, year, ref, False)
    ov = ROOT / "benchmark/status_change_overrides.csv"  # pharmacist corrections (scripts/apply_hawraa_review.py)
    if ov.exists():
        by = {i["item_id"]: i for i in items}
        for line in open(ov).read().splitlines()[1:]:
            item_id, verdict = line.split(",")[:2]
            if item_id in by and verdict in HARM:
                by[item_id]["gold"].update({"verdict": verdict, "harm_class_if_wrong": HARM[verdict],
                                            "conditions": COND.get(verdict, [])})
                by[item_id]["note"] += "; pharmacist override"
    with open(ROOT / "benchmark/items_status_change.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} status-change items ({sum(i['label_changed'] for i in items)} changed)")


if __name__ == "__main__":
    main()
