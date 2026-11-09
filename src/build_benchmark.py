"""Generate the benchmark from the rule tables. Gold labels are computed, not judged.

Groups
  A fixed status       substance rows with no conditions
  B context-dependent  substance x competition period, route, dose, sport or sex
  C revision           entities in the 2026 to 2027 delta, asked with a 2027 date frame
  D not settleable     unnamed products and unlisted research compounds (refer), and substances the
                       List, Summary or Monitoring Program state are not prohibited (permitted)

Label rule for a substance that a year's List does not name (used for 2026 labels of Group C):
if it is not named and only an open class ("including, but not limited to") could cover it,
the List text alone cannot settle the question, so the label is cannot_determine_refer.

Writes benchmark/items.jsonl, benchmark/stratification.csv, benchmark/items_for_review.xlsx
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_rules import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20270101
FRAMES = ["under the rules in force on 1 January 2027", "for the 2027 season (rules in force from 1 January 2027)",
          "under the 2027 rules that apply from 1 January 2027"]
VERDICT = {"all_times": "prohibited_at_all_times", "in_competition": "prohibited_in_competition_only",
           "particular_sports": "prohibited_in_particular_sports"}
HARM = {"prohibited_at_all_times": "false_permit", "prohibited_in_competition_only": "false_permit",
        "prohibited_in_particular_sports": "false_permit", "permitted_with_conditions": "missed_condition",
        "permitted": "false_prohibit", "cannot_determine_refer": "unsafe_assurance"}
ENDURANCE_SPORTS = ["cycling", "swimming", "rowing", "track and field", "football", "triathlon", "rugby", "tennis"]


class Rules:
    def __init__(self, year):
        df = pd.read_csv(ROOT / f"data/processed/rules_{year}.csv").fillna("")
        df = df[df["benchmark_eligible"].astype(str) == "True"].copy()
        # labels use the deterministic Pass A condition text, not the LLM wording merged in for review
        pa = {(norm(r["substance"]), r["clause_ref"]): r for r in
              pd.read_csv(ROOT / f"data/processed/rules_{year}_passA.csv").fillna("").to_dict("records")}
        for col in ("route_exceptions", "dose_thresholds", "other_conditions", "sports"):
            df[col] = [pa.get((norm(r["substance"]), r["clause_ref"]), {}).get(col, r[col]) for r in df.to_dict("records")]
        self.df = df
        self.by = {norm(r["substance"]): r for r in self.df.to_dict("records")}
        for r in self.df.to_dict("records"):
            for s in str(r["synonyms"]).split(";"):
                if s.strip() and norm(s) not in self.by:
                    self.by[norm(s)] = r

    def get(self, name):
        return self.by.get(norm(name))

    def need(self, name):
        r = self.get(name)
        if r is None:
            raise KeyError(f"{name} not in rule table")
        if str(r["ambiguous"]).lower() == "true":
            raise KeyError(f"{name} is ambiguous in the rule table")
        return r


def brands(generic, bmap):
    g = norm(generic)
    return [b for b, gen in bmap if norm(gen) == g]


class Gen:
    def __init__(self):
        self.items, self.rng = [], random.Random(SEED)
        bm = pd.read_csv(ROOT / "benchmark/brand_map.csv").fillna("")
        self.bmap = [(r["brand"], r["generic_inn_list_name"]) for r in bm.to_dict("records") if r["status"] == "kept"]
        self.brand_used = {}

    def surface(self, generic, allow_brand=True, first_brand=False):
        """Generic name, or an unambiguous brand for about half the items of a substance that has one."""
        bs = brands(generic, self.bmap) if allow_brand else []
        if bs:
            k = self.brand_used.get(generic, 0) + (1 if first_brand else 0)
            self.brand_used[generic] = k + 1 - (1 if first_brand else 0)
            if k % 2 == 1:
                b = bs[(k // 2) % len(bs)]
                return f"{b}", True, b
        return generic, False, ""

    def add(self, group, template, question, substance, surface, is_brand, brand, verdict, conditions,
            clause_ref, rule, context=None, label_2026=None, cond_tokens=None, note=""):
        frame = self.rng.choice(FRAMES)
        q = question.replace("{frame}", frame)
        q = q[0].upper() + q[1:]
        # the date frame selects the index; it is not part of the retrieval query
        rq = re.sub(r"\s+", " ", question.replace("{frame}", "")).strip()
        rq = re.sub(r"\s+([?.,:])", r"\1", rq)
        rq = rq[0].upper() + rq[1:]
        self.items.append({
            "item_id": f"{group}{len([i for i in self.items if i['group'] == group]) + 1:03d}",
            "group": group, "template_id": template, "question": q, "retrieval_query": rq, "date_frame": "2027-01-01",
            "substance": substance, "surface_name": surface, "is_brand": is_brand, "brand": brand,
            "context": context or {}, "rule_id": rule["rule_id"] if rule else "",
            "gold": {"verdict": verdict, "conditions": conditions, "clause_ref": clause_ref,
                     "harm_class_if_wrong": HARM[verdict], "condition_tokens": cond_tokens or []},
            "label_2026": label_2026, "label_changed": (label_2026 is not None and label_2026 != verdict),
            "note": note})


# ---------------- Group A ----------------
A_TEMPLATES = {
    "A1": "Is {x} allowed in sport {frame}?",
    "A2": "My doctor prescribed {x}. Am I allowed to take it as a tested athlete {frame}?",
    "A3": "Quick check {frame}: is {x} on the WADA Prohibited List?",
    "A4": "I'm a {sport} athlete who gets drug tested. Can I use {x} {frame}?",
}
A_QUOTA = {"S0": 3, "S1": 11, "S2": 8, "S3": 4, "S4": 8, "S5": 7, "S6": 11, "S7": 5, "S8": 1, "P1": 2}
A_EXCLUDE = {"salbutamol", "formoterol", "salmeterol", "vilanterol", "cathine", "ephedrine", "methylephedrine",
             "pseudoephedrine", "epinephrine", "albumin", "dextran", "hydroxyethyl starch", "mannitol",
             "eplerenone", "finerenone", "imlunestrant", "seladelpar", "slu pp 332", "slu pp 915", "δ8 thc",
             "δ10 thc", "hexahydrocannabinols", "ab fubinaca", "5f adb", "δ9 thc", "cocaine", "diamorphine",
             "methylenedioxymethamphetamine", "strychnine"}


def plain_name(n):
    return (len(n) <= 28 and not re.search(r"\d,\d|\(|\[|ß|ɑ|α|^\d", n) and n.lower() == n.lower().strip()
            and not re.search(r"(?i)\b(and|or|of|the|by|in|with)\b", n))


def group_a(g, R):
    rows = [r for r in R.df.to_dict("records")
            if r["status"] in VERDICT and str(r["class"]).lower() != "true" and str(r["ambiguous"]).lower() != "true"
            and not r["route_exceptions"] and not r["dose_thresholds"]
            and not (r["other_conditions"] and r["other_conditions"] != "when administered exogenously")
            and plain_name(r["substance"]) and norm(r["substance"]) not in A_EXCLUDE
            and not r["clause_ref"].startswith(("S9", "M"))]
    rng = random.Random(SEED + 1)
    for sec, k in A_QUOTA.items():
        pool = sorted([r for r in rows if r["clause_ref"].startswith(sec)], key=lambda r: r["substance"].lower())
        # prefer substances with an unambiguous brand, which athletes are more likely to meet
        branded = [r for r in pool if brands(r["substance"], g.bmap)]
        rest = [r for r in pool if r not in branded]
        rng.shuffle(rest)
        pick = (branded + rest)[:k]
        for i, r in enumerate(pick):
            name = r["substance"][0].lower() + r["substance"][1:] if not r["substance"][:2].isupper() else r["substance"]
            surf, isb, b = g.surface(name, first_brand=(i % 2 == 0))
            tid = ["A1", "A2", "A3", "A4"][i % 4] if sec != "P1" else ["A1", "A2", "A3"][i % 3]
            q = A_TEMPLATES[tid].format(x=surf, sport=rng.choice(ENDURANCE_SPORTS), frame="{frame}")
            v = VERDICT[r["status"]]
            conds = {"in_competition": ["in-competition only"], "particular_sports": ["particular sports only"]}.get(
                r["status"], [])
            g.add("A", tid, q, r["substance"], surf, isb, b, v, conds, r["clause_ref"], r,
                  cond_tokens=[["in-competition", "in competition"]] if r["status"] == "in_competition" else [])


# ---------------- Group B ----------------
def s9_routes(R):
    r = R.need("prednisolone")
    m = re.search(r"prohibited only by (.*?) route; not prohibited by (.*)", r["other_conditions"])
    proh = m.group(1).lower() if m else ""
    ok = m.group(2).lower() if m else ""
    return proh, ok


ROUTES = {  # route phrase, route key, List words that must appear in the prohibited or permitted route text
    "oral": ("tablets", "oral"), "injection": ("an injection", "injectable"), "rectal": ("a suppository", "rectal"),
    "inhaled": ("an inhaler", "inhaled"), "nasal": ("a nasal spray", "intranasal"), "dermal": ("a cream on my skin", "dermal"),
    "eye": ("eye drops", "ophthalmological"), "ear": ("ear drops", "otic"), "vaginal": ("a vaginal pessary", "vaginal"),
}
PERIODS = {"IC": ("on race day", "in-competition"), "IC2": ("at 00:30 on the day of my competition", "in-competition"),
           "OOC": ("three weeks before my next competition", "out-of-competition")}
B_GC_TEMPLATES = {
    "B1": "Can I take {x} as {route} {period} {frame}?",
    "B2": "I've been prescribed {x} ({route}). I want to use it {period}. Is that allowed {frame}?",
    "B3": "{period_cap}, can I use {x} in the form of {route} {frame}?",
}


def route_label(route_key, period_key, proh, ok, year_ok=None):
    word = ROUTES[route_key][1]
    if word in proh:
        if PERIODS[period_key][1] == "in-competition":
            return "prohibited_in_competition_only", ["in-competition", f"{word} route"]
        return "permitted_with_conditions", ["out-of-competition only", "must not be used in-competition"]
    # the List prohibits only the routes it names; every other route is "not prohibited when used
    # within the manufacturer's licensed doses and therapeutic indications"
    return "permitted_with_conditions", [f"{word} route", "within licensed doses"]


def group_b(g, R):
    proh, ok = s9_routes(R)
    rng = random.Random(SEED + 2)
    gcs = ["prednisolone", "prednisone", "dexamethasone", "methylprednisolone", "budesonide", "fluticasone",
           "hydrocortisone", "betamethasone", "triamcinolone acetonide", "beclometasone", "mometasone"]
    combos = [("prednisolone", "oral", "IC"), ("prednisolone", "oral", "OOC"), ("prednisone", "oral", "IC2"),
              ("dexamethasone", "injection", "IC"), ("methylprednisolone", "injection", "OOC"),
              ("triamcinolone acetonide", "injection", "IC"), ("budesonide", "rectal", "IC"),
              ("hydrocortisone", "rectal", "OOC"), ("budesonide", "inhaled", "IC"), ("fluticasone", "inhaled", "IC"),
              ("beclometasone", "inhaled", "IC2"), ("mometasone", "nasal", "IC"), ("fluticasone", "nasal", "IC"),
              ("hydrocortisone", "dermal", "IC"), ("betamethasone", "dermal", "IC"), ("dexamethasone", "eye", "IC"),
              ("betamethasone", "ear", "IC"), ("budesonide", "oral", "IC"), ("dexamethasone", "oral", "OOC"),
              ("prednisolone", "eye", "IC2"), ("prednisolone", "injection", "OOC"), ("methylprednisolone", "oral", "IC"),
              ("hydrocortisone", "oral", "IC2"), ("mometasone", "inhaled", "OOC"), ("triamcinolone acetonide", "dermal", "IC"),
              ("betamethasone", "rectal", "IC")]
    assert all(c[0] in gcs for c in combos)
    for i, (sub, route, per) in enumerate(combos):
        r = R.need(sub)
        v, conds = route_label(route, per, proh, ok)
        if v is None:
            continue
        surf, isb, b = g.surface(sub, allow_brand=False)
        tid = ["B1", "B2", "B3"][i % 3]
        pp = PERIODS[per][0]
        q = B_GC_TEMPLATES[tid].format(x=surf, route=ROUTES[route][0], period=pp, period_cap=pp[0].upper() + pp[1:],
                                       frame="{frame}")
        tokens = [[ROUTES[route][1], route]] + ([["in-competition", "in competition", "competition"]]
                                                if "in-competition" in conds else [])
        if v == "permitted_with_conditions" and "out-of-competition only" in conds:
            tokens = [["out-of-competition", "out of competition", "outside competition", "in-competition", "in competition"]]
        g.add("B", tid, q, sub, surf, isb, b, v, conds, "S9" if ROUTES[route][1] in proh else "S9.note", r, context={"route": route, "period": PERIODS[per][1]}, cond_tokens=tokens)

    # beta-2 agonists: route and dose thresholds read from the rule row
    def dose(sub):
        t = R.need(sub)["dose_thresholds"]
        m24 = re.search(r"maximum (?:delivered dose of )?(\d+) micrograms over 24 hours", t)
        mw = re.search(r"not to exceed (\d+) micrograms over (\d+) hours", t)
        return int(m24.group(1)), (int(mw.group(1)), int(mw.group(2))) if mw else None
    b2 = []
    for sub, unit in [("salbutamol", 100), ("formoterol", 12), ("salmeterol", 50)]:
        mx, (wmax, wh) = dose(sub)
        ok_daily = (mx // 2 // unit) * unit
        b2.append((sub, f"{ok_daily // unit} puffs of {unit} micrograms spread over the day ({ok_daily} micrograms in 24 hours)",
                   "permitted_with_conditions", ["inhaled route", f"maximum {mx} micrograms over 24 hours"],
                   [["inhaled", "inhal"], [str(mx), str(wmax)]]))
        over = wmax + unit * 2
        b2.append((sub, f"{over} micrograms by inhaler within {wh // 2} hours before a hard session",
                   "prohibited_at_all_times", ["exceeds inhaled dose limit", f"{wmax} micrograms over {wh} hours"],
                   [[str(wmax), str(mx)]]))
    mxv, _ = dose("vilanterol") if R.need("vilanterol")["dose_thresholds"] else (None, None)
    if mxv:
        b2.append(("vilanterol", f"{mxv} micrograms once a day by inhaler", "permitted_with_conditions",
                   ["inhaled route", f"maximum {mxv} micrograms over 24 hours"], [["inhaled", "inhal"], [str(mxv)]]))
    b2 += [("salbutamol", "tablets by mouth for my breathing", "prohibited_at_all_times", ["only inhaled use is permitted"],
            [["inhaled", "inhal", "oral"]]),
           ("terbutaline", "by inhaler before training", "prohibited_at_all_times", ["no inhaled exception for terbutaline"],
            []),
           ("salbutamol", "1600 micrograms by inhaler all within one hour", "prohibited_at_all_times",
            ["exceeds inhaled dose limit"], [["600", "8 hours"]])]
    B2_T = {"B4": "Can I use {x} {how} {frame}?", "B5": "My asthma plan says {x}, {how}. Is that within the anti-doping rules {frame}?",
            "B6": "Is it OK to take {x} {how} as a tested athlete {frame}?"}
    for i, (sub, how, v, conds, tok) in enumerate(b2):
        r = R.need(sub)
        surf, isb, b = g.surface(sub, allow_brand=False)
        tid = list(B2_T)[i % 3]
        g.add("B", tid, B2_T[tid].format(x=surf, how=how, frame="{frame}"), sub, surf, isb, b, v, conds,
              "S3.exceptions" if v == "permitted_with_conditions" else "S3", r,
              context={"route": "inhaled" if "inhal" in how else "oral", "dose": how}, cond_tokens=tok)

    # stimulants, narcotics, cannabinoids, S5 exceptions: period and route
    B3_T = {"B7": "Can I take {x} {ctx} {frame}?", "B8": "I'm thinking of using {x} {ctx}. Would that break anti-doping rules {frame}?",
            "B9": "{ctx_cap}: is {x} OK for a tested athlete {frame}?"}
    sc = []
    for sub, why in [("pseudoephedrine", "for a cold on race day"), ("ephedrine", "in a cold remedy on race day"),
                     ("methylephedrine", "in a cough syrup during competition"),
                     ("cathine", "in a weight-loss product during competition")]:
        r = R.need(sub)
        thr = re.search(r"greater than ([\d.]+ micrograms per millilitre)", r["dose_thresholds"])
        sc.append((sub, why, "prohibited_in_competition_only",
                   ["in-competition", f"urinary threshold {thr.group(1) if thr else ''}"],
                   [["in-competition", "in competition", "race day", "competition"]], "S6.B"))
    sc.append(("pseudoephedrine", "for a blocked nose a week before my competition", "permitted_with_conditions",
               ["out-of-competition only"], [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S6.B"))
    sc += [("epinephrine", "as part of a local anaesthetic at the dentist on race day", "permitted_with_conditions",
            ["local administration"], [["local"]], "S6.footnotes"),
           ("epinephrine", "by auto-injector during my competition", "prohibited_in_competition_only",
            ["in-competition", "TUE"], [["in-competition", "in competition", "competition"]], "S6.B"),
           ("oxymetazoline", "as a nasal spray on race day", "permitted_with_conditions", ["nasal use"], [["nasal"]], "S6.exceptions"),
           ("xylometazoline", "as a nasal spray during competition", "permitted_with_conditions", ["nasal use"], [["nasal"]], "S6.exceptions"),
           ("methylphenidate", "for ADHD on competition day", "prohibited_in_competition_only", ["in-competition", "TUE"],
            [["in-competition", "in competition", "competition"]], "S6.B"),
           ("modafinil", "to stay alert during a training block with no competitions for two months", "permitted_with_conditions",
            ["out-of-competition only"], [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S6.A"),
           ("oxycodone", "for pain after surgery, five weeks before my next competition", "permitted_with_conditions",
            ["out-of-competition only"], [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S7"),
           ("oxycodone", "for a sore knee on race day", "prohibited_in_competition_only", ["in-competition"],
            [["in-competition", "in competition", "competition"]], "S7"),
           ("tramadol", "for pain during a stage race", "prohibited_in_competition_only", ["in-competition"],
            [["in-competition", "in competition", "competition"]], "S7"),
           ("tramadol", "for back pain during an off-season training camp", "permitted_with_conditions", ["out-of-competition only"],
            [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S7"),
           ("morphine", "in hospital after a crash, two months before my next race", "permitted_with_conditions",
            ["out-of-competition only"], [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S7"),
           ("fentanyl", "as a pain patch during a multi-day competition", "prohibited_in_competition_only", ["in-competition"],
            [["in-competition", "in competition", "competition"]], "S7"),
           ("hydromorphone", "after an operation, six weeks before my next event", "permitted_with_conditions",
            ["out-of-competition only"], [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S7"),
           ("methylphenidate", "for ADHD during the off-season with no events planned", "permitted_with_conditions",
            ["out-of-competition only"], [["out-of-competition", "out of competition", "in-competition", "in competition"]], "S6.B"),
           ("dorzolamide", "as eye drops for glaucoma", "permitted_with_conditions", ["topical ophthalmic use"],
            [["ophthalmic", "eye"]], "S5.exceptions"),
           ("acetazolamide", "tablets for altitude sickness at a training camp", "prohibited_at_all_times", ["prohibited at all times"],
            [], "S5"),
           ("felypressin", "in a dental anaesthetic", "permitted_with_conditions", ["local dental anaesthesia"],
            [["dental", "local"]], "S5.exceptions")]
    for i, (sub, ctx, v, conds, tok, ref) in enumerate(sc):
        r = R.need(sub)
        surf, isb, b = g.surface(sub)
        tid = list(B3_T)[i % 3]
        g.add("B", tid, B3_T[tid].format(x=surf, ctx=ctx, ctx_cap=ctx[0].upper() + ctx[1:], frame="{frame}"), sub, surf,
              isb, b, v, conds, ref, r, context={"situation": ctx}, cond_tokens=tok)

    # beta-blockers by sport and period, sports read from the P1 rule row
    P1_T = {"B10": "I compete in {sport}. Can I take {x} {per} {frame}?",
            "B11": "My cardiologist prescribed {x}. I do {sport}. Is it allowed {per} {frame}?",
            "B12": "As a {sport} athlete, am I allowed {x} {per} {frame}?"}
    sports = [s.strip() for s in R.need("propranolol")["sports"].split(";") if s.strip()]
    listed = {re.sub(r"\s*\(.*", "", s).strip().lower(): ("Out-of-Competition" in s) for s in sports}
    pairs = [("archery", "IC"), ("archery", "OOC"), ("golf", "IC"), ("golf", "OOC"), ("darts", "IC"), ("darts", "OOC"),
             ("shooting", "OOC"), ("automobile", "IC"), ("mini-golf", "IC"), ("underwater sports", "OOC"),
             ("marathon running", "IC"), ("swimming", "IC"), ("cycling", "OOC"), ("tennis", "IC"), ("shooting", "IC"),
             ("automobile", "OOC"), ("golf", "IC"), ("rowing", "IC")]
    subs = ["propranolol", "atenolol", "bisoprolol", "metoprolol", "nebivolol"]
    for i, (sport, per) in enumerate(pairs):
        sub = subs[i % len(subs)]
        r = R.need(sub)
        key = sport.lower()
        if key in listed:
            if per == "IC" or listed[key]:
                v, conds = "prohibited_in_particular_sports", [f"prohibited in {sport}"] + (
                    ["also out-of-competition"] if listed[key] else ["in-competition"])
                tok = [[sport.split()[0]], ["out-of-competition", "out of competition", "at all times"] if listed[key]
                       else ["in-competition", "in competition", "competition"]]
            else:
                v, conds = "permitted_with_conditions", ["out-of-competition only", f"prohibited in-competition in {sport}"]
                tok = [["out-of-competition", "out of competition", "in-competition", "in competition"]]
        else:
            v, conds, tok = "permitted", [], []
        surf, isb, b = g.surface(sub)
        tid = list(P1_T)[i % 3]
        pp = {"IC": "during competition", "OOC": "in training, out of competition"}[per]
        shown = {"automobile": "motor racing (FIA automobile)", "underwater sports": "freediving"}.get(sport, sport)
        g.add("B", tid, P1_T[tid].format(x=surf, sport=shown, per=pp, frame="{frame}"), sub, surf, isb, b, v, conds,
              "P1", r, context={"sport": sport, "period": per}, cond_tokens=tok)
        if v == "prohibited_in_particular_sports":  # the question names the sport, so the athlete-level verdict is right too
            g.items[-1]["gold"]["accept"] = ["prohibited_at_all_times" if listed[key] else "prohibited_in_competition_only"]

    # S2.2.1 applies in males only; M2.2 infusions
    sx = [("chorionic gonadotrophin", "male", "prohibited_at_all_times", ["prohibited in males"], [["male"]]),
          ("chorionic gonadotrophin", "female", "permitted", [], []),
          ("luteinizing hormone", "female", "permitted", [], []),
          ("leuprorelin", "male", "prohibited_at_all_times", ["prohibited in males"], [["male"]]),
          ("leuprorelin", "female", "permitted", [], [])]
    for i, (sub, sex, v, conds, tok) in enumerate(sx):
        r = R.need(sub)
        surf, isb, b = g.surface(sub)
        q = (f"I'm a {sex} athlete and my fertility clinic wants me to use {surf}. Is that allowed {{frame}}?" if i % 2 == 0
             else f"As a {sex} competitor, can I be treated with {surf} {{frame}}?")
        g.add("B", "B13" if i % 2 == 0 else "B14", q, sub, surf, isb, b, v, conds, "S2.2.1", r,
              context={"sex": sex}, cond_tokens=tok)
    m22 = R.need(norm("Intravenous infusions and/or injections of more than 100 mL per 12-hour period"))
    for i, (ctx, v, conds, tok) in enumerate([
            ("a 500 mL saline drip at a wellness clinic to rehydrate after a race", "prohibited_at_all_times",
             ["more than 100 mL per 12 hours outside hospital treatment"], [["100"]]),
            ("a 1 litre drip while admitted to hospital after a crash", "permitted_with_conditions",
             ["hospital treatment exception"], [["hospital"]]),
            ("a single 20 mL vitamin injection into a vein (the vitamin itself is not prohibited)", "permitted_with_conditions",
             ["no more than 100 mL per 12 hours"], [["100"]]),
            ("two 250 mL iron infusions on the same afternoon at a private clinic", "prohibited_at_all_times",
             ["more than 100 mL per 12 hours outside hospital treatment"], [["100"]])]):
        q = [f"Can I have {ctx} {{frame}}?", f"Is {ctx} allowed for a tested athlete {{frame}}?"][i % 2]
        g.add("B", "B15" if i % 2 == 0 else "B16", q, "intravenous infusion", "IV infusion", False, "", v, conds, "M2.2",
              m22, context={"setting": ctx}, cond_tokens=tok)


# ---------------- Group C ----------------
REVISION = [  # entity, how to ask, 2027 rule name
    ("eplerenone", "eplerenone"), ("finerenone", "finerenone"), ("imlunestrant", "imlunestrant"),
    ("seladelpar", "seladelpar"), ("SLU-PP-332", "SLU-PP-332"), ("SLU-PP-915", "SLU-PP-915"),
    ("HHC (hexahydrocannabinol)", "Hexahydrocannabinols"), ("AB-FUBINACA", "AB-FUBINACA"), ("5F-ADB", "5F-ADB")]
C_T = {"C1": "Is {x} prohibited {frame}?", "C2": "I was told {x} isn't on the list. Can I use it {frame}?",
       "C3": "My doctor wants to start me on {x}. Will that be a problem in drug testing {frame}?",
       "C4": "Someone at my club uses {x}. Is it allowed in sport {frame}?"}


# Revision items: substances first named in 2027. The 2026 List already covered each one through a
# class it states universally (Summary 2027: "added as examples"), so a pharmacist reads the 2026 status
# from that class. Each class phrase is checked against the 2026 clause text at build time.
CLASS_2026 = {
    "eplerenone": ("S5", "All diuretics and masking agents", "all_times"),
    "finerenone": ("S5", "All diuretics and masking agents", "all_times"),
    "imlunestrant": ("S4.2", "ANTI-ESTROGENIC SUBSTANCES", "all_times"),
    "seladelpar": ("S4.4.1", "Peroxisome proliferator-activated receptor delta", "all_times"),
    "SLU-PP-332": ("S0", "Any pharmacological substance which is not addressed", "all_times"),
    "SLU-PP-915": ("S0", "Any pharmacological substance which is not addressed", "all_times"),
    "Hexahydrocannabinols": ("S8", "All natural and synthetic cannabinoids", "in_competition"),
    "AB-FUBINACA": ("S8", "All natural and synthetic cannabinoids", "in_competition"),
    "5F-ADB": ("S8", "All natural and synthetic cannabinoids", "in_competition"),
}


def label_2026_by_class(name):
    ref, phrase, status = CLASS_2026[name]
    text = " ".join(json.loads(l)["text"] for l in open(ROOT / "data/processed/chunks_2026.jsonl")
                    if json.loads(l)["clause_ref"] == ref)
    assert phrase in text, (name, ref, phrase)
    return VERDICT[status], ref


def label_unnamed(name, R_old):
    """2026 label for an entity the 2026 List does not name: open class only, so refer."""
    r = R_old.get(name)
    if r is not None and str(r["ambiguous"]).lower() != "true":
        return VERDICT.get(r["status"], "permitted")
    return "cannot_determine_refer"


def group_c(g, R, R_old):
    for k, (ask, rname) in enumerate(REVISION):
        r = R.need(rname)
        v = VERDICT[r["status"]]
        literal = label_unnamed(rname, R_old)  # text-literal reading (sensitivity only)
        old, class_ref = label_2026_by_class(rname)
        tids = ["C1", "C2", "C3", "C4"] if not ask.startswith(("HHC", "AB-", "5F", "SLU")) else ["C1", "C2", "C4", "C1"]
        for j, tid in enumerate(tids):
            surf, isb, b = g.surface(ask) if tid in ("C3",) or j % 2 else (ask, False, "")
            if tid == "C1" and j == 3:
                q = f"Has anything changed for {surf} for 2027? Can I use it {{frame}}?"
            else:
                q = C_T[tid].format(x=surf, frame="{frame}")
            conds = ["in-competition only"] if r["status"] == "in_competition" else []
            g.add("C", tid, q, rname, surf, isb, b, v, conds, r["clause_ref"], r, label_2026=old,
                  cond_tokens=[["in-competition", "in competition"]] if conds else [],
                  note=f"newly named in 2027; covered in 2026 by class wording in {class_ref}")
            g.items[-1]["newly_named"] = True
            g.items[-1]["label_2026_literal"] = literal
    # clarifications whose label did not change (control items)
    proh, ok = s9_routes(R)
    R_old_routes = s9_routes(R_old)[1]
    for i, (sub, per) in enumerate([("hydrocortisone", "IC"), ("clotrimazole and hydrocortisone", None), ("betamethasone", "IC2")]):
        if per is None:
            continue
        v, conds = route_label("vaginal", per, proh, ok)
        old, _ = route_label("vaginal", per, s9_routes(R_old)[0], R_old_routes)
        r = R.need(sub)
        q = [f"Can I use {sub} as a vaginal pessary on race day {{frame}}?",
             f"My GP prescribed a {sub} vaginal cream. I compete tomorrow. Is it allowed {{frame}}?"][i % 2]
        g.add("C", "C5" if i % 2 == 0 else "C6", q, sub, sub, False, "", v, conds, "S9.note", r,
              label_2026=old, context={"route": "vaginal"}, cond_tokens=[["vaginal"]],
              note="2027 Summary: vaginal administration added as a permitted route")
    cbd = R.need("cannabidiol")
    for i, q in enumerate(["Is CBD oil allowed on competition day {frame}?",
                           "Can I use cannabidiol (CBD) balm for recovery during a tournament {frame}?"]):
        g.add("C", "C7", q, "cannabidiol", "CBD", False, "", "permitted", [], cbd["clause_ref"], cbd,
              label_2026=label_unnamed("cannabidiol", R_old) if R_old.get("cannabidiol") is None else "permitted",
              note="wording changed from Exceptions to Note")
        if i == 0:  # CBD oil products can contain THC
            g.items[-1]["gold"]["accept"] = ["cannot_determine_refer"]
    for i, sub in enumerate(["propranolol", "metoprolol"]):
        r = R.need(sub)
        q = [f"I play billiards competitively. Can I take {sub} during a tournament {{frame}}?",
             f"Is {sub} allowed in snooker and pool events (WCBS) in competition {{frame}}?"][i]
        g.add("C", "C8", q, sub, sub, False, "", "prohibited_in_particular_sports", ["prohibited in billiards", "in-competition"],
              "P1", r, label_2026="prohibited_in_particular_sports", context={"sport": "billiards"},
              cond_tokens=[["billiards", "snooker", "pool", "wcbs"]], note="2027 removed 'all disciplines' from WCBS")
        g.items[-1]["gold"]["accept"] = ["prohibited_in_competition_only"]
    for i, x in enumerate(["Δ8-THC", "delta-8 THC gummies"]):
        r = R.need("Δ8-THC")
        q = [f"Is {x} prohibited in competition {{frame}}?", f"Can I take {x} at 00:30 on the day of a match {{frame}}?"][i]
        g.add("C", "C9", q, "Δ8-THC", x, False, "", "prohibited_in_competition_only", ["in-competition"], "S8", r,
              label_2026="prohibited_in_competition_only", cond_tokens=[["in-competition", "in competition", "competition"]],
              note="named in 2027; covered in 2026 by 'natural and synthetic tetrahydrocannabinols'")


# ---------------- Group D ----------------
D_SETTLEABLE = {
    "ACP-105 from an online research chemical seller": ("prohibited_at_all_times", "S1.2", "SARM; S1.2 names the SARM class"),
    "GW0742 for endurance": ("prohibited_at_all_times", "S4.4.1", "PPARdelta agonist; S4.4.1 names the class"),
    "a 'research use only' peptide called 5-amino-1MQ": ("prohibited_at_all_times", "S0", "non-approved substance; S0"),
    "a new stimulant sold as 'DMHA' in a supplement": ("prohibited_in_competition_only", "S6.B", "DMHA is octodrine, named in S6.B"),
}
def group_d(g, R):
    permitted = [  # names the List, a footnote, the S8 note or the Monitoring Program state are not prohibited
        ("caffeine", "caffeine tablets before a race", "S6.footnotes"), ("caffeine", "a strong espresso on race morning", "S6.footnotes"),
        ("nicotine", "nicotine pouches during a tournament", "S6.footnotes"), ("bupropion", "bupropion to stop smoking", "S6.footnotes"),
        ("phenylephrine", "a cold remedy with phenylephrine on race day", "S6.footnotes"),
        ("synephrine", "a supplement listing synephrine (bitter orange)", "S6.footnotes"),
        ("phenylpropanolamine", "phenylpropanolamine during competition", "S6.footnotes"),
        ("clonidine", "clonidine on competition day", "S6.exceptions"), ("guanfacine", "guanfacine for ADHD on race day", "S6.exceptions"),
        ("Supplemental oxygen by inhalation", "supplemental oxygen from a mask between rounds", "M1.2"),
        ("caffeine", "caffeine gum during a match", "S6.footnotes"), ("nicotine", "a nicotine vape the night before a race", "S6.footnotes"),
    ]
    mon_text = " ".join(" ".join(__import__("pymupdf").open(ROOT / "data/raw/2027_monitoring_program.pdf")[0].get_text().split()).split())
    monitored = [("ecdysterone", "an ecdysterone supplement"), ("dihydrocodeine", "dihydrocodeine for a cough on race day"),
                 ("tapentadol", "tapentadol after an injury, during competition"), ("semaglutide", "semaglutide injections for weight"),
                 ("tirzepatide", "tirzepatide"), ("hypoxen", "Hypoxen capsules")]
    D_P = {"D1": "Can I take {ctx} {frame}?", "D2": "Is {ctx} OK for a drug-tested athlete {frame}?",
           "D3": "Would {ctx} cause a problem in an anti-doping test {frame}?"}
    i = 0
    for sub, ctx, ref in permitted:
        r = R.need(sub)
        tid = list(D_P)[i % 3]
        g.add("D", tid, D_P[tid].format(ctx=ctx, frame="{frame}"), sub, sub, False, "", "permitted", [], ref, r,
              note="List states not prohibited")
        if sub == "synephrine":  # a supplement product carries contamination risk, so refer is also right
            g.items[-1]["gold"]["accept"] = ["cannot_determine_refer"]
        i += 1
    for sub, ctx in monitored:
        assert sub.lower() in mon_text.lower(), sub
        tid = list(D_P)[i % 3]
        surf, isb, b = g.surface(sub)
        ctx2 = ctx.replace(sub, surf) if isb else ctx
        g.add("D", tid, D_P[tid].format(ctx=ctx2, frame="{frame}"), sub, surf, isb, b, "permitted", [],
              "Monitoring Program", None, note="2027 Monitoring Program: monitored, not on the Prohibited List")
        # the Monitoring Program is not in the retrieval index, so deferring is also right
        g.items[-1]["gold"]["accept"] = ["cannot_determine_refer"]
        i += 1
    # S7 is a closed list (no 'including, but not limited to'): an unlisted opioid is not prohibited
    s7 = " ".join(json.loads(l)["text"] for l in open(ROOT / "data/processed/chunks_2027.jsonl") if '"S7"' in l)
    assert "not limited to" not in s7 and "similar" not in s7
    for sub, ctx in [("codeine", "co-codamol (codeine and paracetamol) on race day"), ("codeine", "codeine linctus during a tournament")]:
        tid = list(D_P)[i % 3]
        g.add("D", tid, D_P[tid].format(ctx=ctx, frame="{frame}"), sub, sub, False, "", "permitted", [], "S7", None,
              note="S7 is a closed list; codeine removed from the Monitoring Program in 2027")
        i += 1
    refer = [  # the List text cannot settle these
        "a pre-workout powder I bought at my gym", "a 'fat burner' capsule I ordered online",
        "a protein shake from a brand I don't recognise", "herbal energy tablets my cousin brought from abroad",
        "a 'test booster' supplement from a sports shop", "an unlabelled tablet a teammate gave me for recovery",
        "a 'research use only' peptide called 5-amino-1MQ", "ACP-105 from an online research chemical seller",
        "a nootropic called 'Noopept' for focus", "a homeopathic remedy for joint pain",
        "a new stimulant sold as 'DMHA' in a supplement", "a 'mushroom coffee' blend with added extracts",
        "GW0742 for endurance", "a 'sleep gummy' with herbal extracts", "an Ayurvedic tonic for stamina",
        "a muscle cream my physio brought back from a trip abroad", "an imported 'energy shot' with ingredients only in Chinese",
        "a collagen drink marketed to athletes", "an injectable 'recovery cocktail' from a wellness clinic (contents unknown)",
        "a weight-loss tea from social media", "a 'methylated' energy supplement with unknown ingredients",
        "beet juice shots with added 'performance blend'", "an imported cold remedy whose label I can't read",
        "a peptide nasal spray called Semax", "an amino acid powder bought from a market stall",
        "a 'liver detox' supplement", "a traditional herbal cold remedy containing ephedra-type plants",
        "a 'natural' anabolic supplement advertised as 'plant sterols'"]
    D_R = {"D4": "Can I use {ctx} {frame}?", "D5": "Is {ctx} safe for anti-doping {frame}?",
           "D6": "Will I test positive if I take {ctx} {frame}?", "D7": "Is {ctx} allowed in sport {frame}?"}
    for j, ctx in enumerate(refer):
        tid = list(D_R)[j % 4]
        g.add("D", tid, D_R[tid].format(ctx=ctx, frame="{frame}"), ctx, ctx, False, "", "cannot_determine_refer", [], "",
              None, note="not settleable from the List text")
        alt = D_SETTLEABLE.get(ctx)
        if alt:  # the List settles these by class or name; refer is also accepted because product contents are uncertain
            gd = g.items[-1]["gold"]
            gd.update({"verdict": alt[0], "clause_ref": alt[1], "harm_class_if_wrong": HARM[alt[0]],
                       "accept": ["cannot_determine_refer"],
                       "conditions": ["in-competition"] if alt[0] == "prohibited_in_competition_only" else []})
            g.items[-1]["note"] = alt[2]


def apply_item_overrides(items):
    path = ROOT / "benchmark/item_overrides.csv"
    if not path.exists():
        return
    ov = pd.read_csv(path, dtype=str).fillna("")
    by = {i["item_id"]: i for i in items}
    for o in ov.to_dict("records"):
        it = by.get(o["item_id"])
        if it is None:
            print("item override target not found:", o)
            continue
        if o["field"] == "drop" and o["value"].lower() in ("yes", "y", "true"):
            items.remove(it)
        elif o["field"] == "conditions":
            it["gold"]["conditions"] = [x.strip() for x in o["value"].split("|") if x.strip()]
        else:
            it["gold"][o["field"]] = o["value"]
            if o["field"] == "verdict":
                it["gold"]["harm_class_if_wrong"] = HARM[o["value"]]
        it["note"] = (it["note"] + "; pharmacist override").strip("; ")


def main():
    R, R_old = Rules(2027), Rules(2026)
    g = Gen()
    group_a(g, R)
    group_b(g, R)
    group_c(g, R, R_old)
    group_d(g, R)
    apply_item_overrides(g.items)
    with open(ROOT / "benchmark/items.jsonl", "w") as f:
        for it in g.items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    df = pd.DataFrame([{**{k: v for k, v in it.items() if k not in ("gold", "context")}, **{
        "verdict": it["gold"]["verdict"], "clause_ref": it["gold"]["clause_ref"]}} for it in g.items])
    strat = df.pivot_table(index="group", columns="verdict", values="item_id", aggfunc="count", fill_value=0)
    strat["total"] = strat.sum(axis=1)
    strat["share"] = (strat["total"] / strat["total"].sum()).round(3)
    strat["brand_items"] = df.groupby("group")["is_brand"].sum()
    strat.loc["total"] = strat.sum(numeric_only=True)
    strat.to_csv(ROOT / "benchmark/stratification.csv")
    print(strat.to_string())
    # 30% stratified sample for the pharmacist's spot-check
    rs = random.Random(SEED + 9)
    sample = []
    for (grp, v), sub in df.groupby(["group", "verdict"]):
        ids = sorted(sub["item_id"])
        rs.shuffle(ids)
        sample += ids[:max(1, round(0.3 * len(ids)))]
    rev = df[df["item_id"].isin(sample)][["item_id", "group", "question", "substance", "verdict", "clause_ref", "label_2026"]].copy()
    rev["conditions"] = rev["item_id"].map({it["item_id"]: "; ".join(it["gold"]["conditions"]) for it in g.items})
    rev["correct? (Y/N)"], rev["correction"], rev["comment"] = "", "", ""
    with pd.ExcelWriter(ROOT / "benchmark/items_for_review.xlsx") as w:
        rev.to_excel(w, sheet_name="items_sample", index=False)
        for ws in w.sheets.values():
            for col, width in zip("ABCDEFGHIJK", [9, 6, 70, 22, 30, 14, 28, 30, 12, 30, 30]):
                ws.column_dimensions[col].width = width
    print(f"{len(g.items)} items; review sample {len(rev)}")


if __name__ == "__main__":
    main()
