"""Pass A: deterministic rule-table extraction from clause chunks.

Regex and layout rules only, no model. One row per named substance, named
class or method. Conditions (route exceptions, urinary thresholds, sports,
in-males, exogenous administration) are read from the NOTE, EXCEPTIONS and
footnote clauses of the same year.

Usage: python src/rules_pass_a.py <year>
Writes data/processed/rules_<year>_passA.csv
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STATUS = {"S0": "all_times", "S1": "all_times", "S2": "all_times", "S3": "all_times",
          "S4": "all_times", "S5": "all_times", "M1": "all_times", "M2": "all_times",
          "M3": "all_times", "S6": "in_competition", "S7": "in_competition",
          "S8": "in_competition", "S9": "in_competition", "P1": "particular_sports"}

DROP_SENTENCES = [
    r"All prohibited (?:substances|methods) in this class are [^•]*?Substances\.",
    r"All prohibited methods in this class are non-Specified except [^•]*?Methods\.",
    r"Prohibited substances in classes S4\.1 and S4\.2 [^•]*?non-Specified Substances\.",
    r"A stimulant not expressly listed in this section is a Specified Substance\.",
    r"Substances? of Abuse in this section: [^•]*?(?=(?:Stimulants include|The following|Tetrahydro|All |•|$))",
    r"[Ii]ncluding,? but not limited to:?",
    r"and other substances with (?:a )?similar chemical structure[^•]*?effect\(s\)[^•.]*\.?",
    r"and other growth factors or growth factor modulators[^•]*?switching\.",
    r"The following (?:substances|hormone and metabolic modulators|narcotics)[^•]*?are prohibited\.",
    r"The following(?:, with the potential to enhance sport performance,)? are prohibited:",
    r"Stimulants include:",
    r"All [^•]*?(?:optical isomers|beta-2 agonists)[^•]*?are prohibited\.",
    r"Anabolic agents are prohibited\.", r"When administered exogenously,",
    r"Beta-blockers are prohibited In-Competition only[^•]*?\(\*\)\.",
    r"\*Also prohibited Out-of-Competition",
    r"All glucocorticoids are prohibited when administered by any injectable[^•]*?route\.",
    r"Tetrahydrocannabinols \(THC\), as well as semi-synthetic[^•]*?are prohibited, e\.g\.",
    r"All natural and synthetic cannabinoids are prohibited, e\.g\.",
]
TITLES_UC = ["GROWTH FACTORS, RELATED SUBSTANCES, AND MIMETICS (continued)", "NON-APPROVED SUBSTANCES", "ANABOLIC AGENTS", "PEPTIDE HORMONES, GROWTH FACTORS, RELATED SUBSTANCES, AND MIMETICS",
             "BETA-2 AGONISTS", "HORMONE AND METABOLIC MODULATORS", "DIURETICS AND MASKING AGENTS",
             "STIMULANTS", "NARCOTICS", "CANNABINOIDS", "GLUCOCORTICOIDS", "BETA-BLOCKERS", "(continued)"]


def split_top(s, seps=(";", ",")):
    """Split on separators outside brackets. A comma before a digit is a chemical locant, not a separator."""
    out, depth, buf = [], 0, ""
    for i, ch in enumerate(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        nxt = s[i + 1:].lstrip()[:1]
        if ch in seps and depth == 0 and not (ch == "," and s[i + 1:i + 2].isdigit()):
            out.append(buf)
            buf = ""
        else:
            buf += ch
    out.append(buf)
    return [x.strip() for x in out if x.strip()]


def split_and(s):
    """Split 'A and B' outside brackets, but not 'X and its analogues'."""
    parts, depth, i, buf = [], 0, 0, ""
    while i < len(s):
        ch = s[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if depth == 0 and s.startswith(" and ", i) and not re.match(
                r" and (its|their|other|related|l-isomer)", s[i:]):
            parts.append(buf)
            buf = ""
            i += 5
            continue
        buf += ch
        i += 1
    parts.append(buf)
    return [re.sub(r"^and ", "", p.strip()) for p in parts if p.strip()]


def trailing_group(item):
    """Return (base, inner) if item ends with a bracketed group, matching brackets backwards."""
    if not item or item[-1] not in ")]":
        return None
    depth = 0
    for i in range(len(item) - 1, -1, -1):
        if item[i] in ")]":
            depth += 1
        elif item[i] in "([":
            depth -= 1
            if depth == 0:
                base = item[:i].strip()
                # a bracket glued to the name is part of a chemical name, e.g. 'bis(2-fluorophenyl)'
                if not base or (i > 0 and item[i - 1] not in " "):
                    return None
                return base, item[i + 1:-1]
    return None


def name_and_synonyms(item):
    item = re.sub(r"\*+", "", item.strip().strip(".").strip()).strip()
    item = re.sub(r"^and ", "", item)
    item = re.sub(r" and related substances$", "", item)
    m = re.match(r"^(.*?) as found in (\w+)\s*\((.*)\)$", item)
    if m:
        return m.group(1).strip(), [m.group(2)] + [x.strip() for x in m.group(3).split(",")]
    tail = ""
    m = re.match(r"^(.*[\)\]])\s+(and (?:its|their) [\w ]+)$", item)
    if m:
        item, tail = m.group(1), " " + m.group(2)
    tg = trailing_group(item)
    if tg:
        base, inner = tg
        syn = [x.strip() for x in re.split(r"[,;]\s*(?=[A-Za-zɑαß(\[])", inner) if x.strip()]
        return (base + tail).strip(), syn
    return (item + tail).strip(), []


def first_group(item):
    """Span of the first top-level bracket group, or None."""
    depth, start = 0, None
    for i, ch in enumerate(item):
        if ch in "([":
            if depth == 0:
                start = i
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                return start, i
    return None


CLASS_WORDS = re.compile(r"(?i)\b(and its|and their|derivatives|analogues|agonists|inhibitors|antagonists|"
                         r"factors|antibodies|modulators|competitors|products|peptides|receptors|"
                         r"stabilizers|activators|expanders|vaptans|diuretics|agents|proteins|"
                         r"cannabinoids|hexahydrocannabinols|glucocorticoids|corticotrophins|insulins)\b")


def expand(item):
    """Turn a bullet or inline item into (name, synonyms, parent_class, is_class) tuples."""
    item = item.strip().strip(".").strip()
    if not item:
        return []
    # 'Class (ABBR) and its analogues (e.g. a, b)': examples in the group that closes the item
    tg = trailing_group(item)
    if tg and tg[1].startswith("e.g.") and first_group(item)[0] < len(tg[0]):
        n, s = name_and_synonyms(tg[0])
        rows = [(n, s, None, True)]
        for x in split_top(tg[1][len("e.g."):], (",", ";")):
            for y in split_and(x):
                yn, ys = name_and_synonyms(y)
                rows.append((yn, ys, n, False))
        return rows
    # 'Class [ABBR, e.g. a, b and c]' or 'Class (e.g. a, b)' where that group closes the item
    fg = first_group(item)
    if fg and fg[1] == len(item) - 1:
        inner = item[fg[0] + 1:fg[1]]
        m = re.match(r"^([^,]*?,\s*)?e\.g\.\s*(.*)$", inner)
        if m:
            cls = item[:fg[0]].strip()
            abbr = (m.group(1) or "").strip(", ").strip()
            n, s = name_and_synonyms(cls)
            rows = [(n, s + ([abbr] if abbr else []), None, True)]
            for x in split_top(m.group(2), (",", ";")):
                for y in split_and(x):
                    yn, ys = name_and_synonyms(y)
                    rows.append((yn, ys, n, False))
            return rows
    kws = [(item.find(k), k) for k in (", e.g.", " e.g.", " such as:", " such as") if k in item]
    kws = [(pos, k) for pos, k in kws if not fg or pos < fg[0] or pos > fg[1]]
    if kws:
        pos, kw = min(kws)
        cls, rest = item[:pos].strip().rstrip(","), item[pos + len(kw):]
        lead = split_and(cls)
        rows = []
        for x in lead[:-1]:  # 'A and modified B products, e.g. ...': A is a separate item
            xn, xs = name_and_synonyms(x)
            rows.append((xn, xs, None, False))
        n, s = name_and_synonyms(lead[-1] if lead else cls)
        rows.append((n, s, None, True))
        for x in split_top(rest, (";",) if ";" in rest else (",",)):
            for z in expand_inline(x):
                rows.append((z[0], z[1], z[2] or n, z[3]))
        return rows
    out = []
    for y in split_and(item):
        n, s = name_and_synonyms(y)
        out.append((n, s, None, bool(CLASS_WORDS.search(n))))
    return out


def top_level(x):
    """Text of x with bracketed parts removed."""
    out, depth = "", 0
    for ch in x:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out += ch
    return out


def expand_inline(x):
    x = x.strip()
    segs = split_top(x, (",",))
    if "e.g." in top_level(x) or "such as" in top_level(x):
        return expand(x)
    if len(segs) == 1:
        return expand(x)
    out = []
    for y in segs:
        out.extend(expand(y))
    return out


def s0_items(t):
    t = t.split("including but not limited to", 1)[1]
    out = []
    for x in split_top(t, (",",)):
        for y in split_and(x):
            out.extend(expand(y.strip().rstrip(".")))
    return out


def m12_items(t):
    """M1.2 is a sentence, not a list: method, named agents, and an explicit exclusion."""
    rows = [("Artificially enhancing the uptake, transport or delivery of oxygen", [], None, True)]
    body = t.split("limited to:", 1)[1]
    excl = re.search(r"excluding ([^.]*)\.", body)
    body = body[:excl.start()] if excl else body
    for x in split_top(body.strip().rstrip(","), (";",)):
        for z in expand_inline(x):
            rows.append((z[0], z[1], z[2] or rows[0][0], z[3]))
    if excl:
        rows.append(("__EXCLUDED__" + excl.group(1).strip()[0].upper() + excl.group(1).strip()[1:], [], None, False))
    return rows


def body_items(c):
    t = c["text"]
    if c["clause_ref"] == "S0" and "including but not limited to" in t:
        return s0_items(t)
    if c["clause_ref"] == "M1.2":
        return m12_items(t)
    t = re.sub(r"^(?:S\d(?:\.\d){1,2}\.?|S6\.[AB]:)\s+", "", t.strip())
    toks = t.split(" ")
    def caps(tok):
        return tok != "•" and re.search(r"[A-Z]", tok) and tok == tok.upper() and not re.search(r"\d", tok)
    while len(toks) > 1 and caps(toks[0]) and caps(toks[1]):
        toks.pop(0)  # leftover upper-case heading lines
        if len(toks) == 1 or not caps(toks[1]):
            toks.pop(0)
            break
    t = " ".join(toks)
    for pat in DROP_SENTENCES:
        t = re.sub(pat, " ", t)
    for h in TITLES_UC:
        t = t.replace(h, " ")
    t = re.sub(r"\s+", " ", t).strip()
    items = [x for x in re.split(r"•| – ", t)]
    out = []
    for it in items:
        it = it.strip().rstrip(".").strip()
        if not it or it.endswith("such as:") and len(it) < 60:
            if it:
                n, s = name_and_synonyms(it[:-len("such as:")].strip())
                out.append((n, s, None, True))
            continue
        out.extend(expand_inline(it) if c["clause_ref"] not in ("S1.1", "S4.1", "S6.A", "S6.B", "S7", "S9", "P1", "S3")
                   else expand(it))
    return out


def conditions(chunks):
    """Read per-substance conditions from NOTE / EXCEPTIONS / footnote clauses."""
    by = {c["clause_ref"]: c for c in chunks}
    cond = {}

    def add(name, field, value, ref):
        d = cond.setdefault(name.strip().lower(), {})
        d.setdefault(field, []).append(value)
        d.setdefault("refs", []).append(ref)

    ex = by.get("S3.exceptions", {}).get("text", "")
    for m in re.finditer(r"Inhaled (\w+): (maximum [^•]*)", ex):
        add(m.group(1), "route_exceptions", "inhaled", "S3.exceptions")
        add(m.group(1), "dose_thresholds", "inhaled " + m.group(2).strip(), "S3.exceptions")
    note = by.get("S3.note", {}).get("text", "")
    for m in re.finditer(r"(salbutamol|formoterol) in excess of ([\d ]+ng/mL)", note):
        add(m.group(1), "dose_thresholds", f"urine > {m.group(2).strip()} is an AAF", "S3.note")
    fn = by.get("S6.footnotes", {}).get("text", "")
    for seg in re.split(r"\*+ ", fn):
        seg = seg.strip()
        if not seg:
            continue
        head, _, rule = seg.partition(":")
        rule = rule.strip()
        if "Monitoring Program" in rule:
            for x in split_top(head.replace(", and ", ", "), (",",)):
                for y in split_and(x):
                    add(y, "monitoring_only", rule, "S6.footnotes")
            continue
        names = re.sub(r"\([^)]*\)", "", head)
        names = [n.strip() for n in split_and(names.replace(" and its l-isomer", "")) if n.strip()]
        for n in names:
            if "greater than" in rule:
                add(n, "dose_thresholds", "urine " + rule[rule.find("greater than"):].rstrip("."), "S6.footnotes")
            elif "Not prohibited in local administration" in rule:
                add(n, "route_exceptions", rule.rstrip("."), "S6.footnotes")
    exc = by.get("S6.exceptions", {}).get("text", "")
    if "Clonidine" in exc:
        for n in ("clonidine", "guanfacine"):
            add(n, "exception", "listed under S6 Exceptions (not prohibited)", "S6.exceptions")
    m = re.search(r"Imidazoline derivatives for ([^(]*?) use \(e\.g\. ([^)]*)\)", exc)
    if m:
        for x in split_top(m.group(2), (",",)):
            add(x, "route_exceptions", f"{m.group(1)} use not prohibited", "S6.exceptions")
    s5 = by.get("S5.exceptions", {}).get("text", "")
    for x in split_top(re.sub(r"\(e\.g\.[^)]*\)", "", s5.split("•")[1] if "•" in s5 else ""), (";",)):
        if "carbonic" not in x:
            add(x.strip().strip("and ").strip(), "exception", "listed under S5 Exceptions", "S5.exceptions")
    m = re.search(r"carbonic anhydrase inhibitors \(e\.g\. ([^)]*)\)", s5)
    if m:
        for y in split_and(m.group(1).replace(",", " and ")):
            add(y, "route_exceptions", "topical ophthalmic administration not prohibited", "S5.exceptions")
    if "felypressin" in s5:
        add("felypressin", "route_exceptions", "local administration in dental anaesthesia not prohibited",
            "S5.exceptions")
    cbd = (by.get("S8.exceptions") or by.get("S8.note") or {}).get("text", "")
    if "Cannabidiol" in cbd:
        add("cannabidiol", "exception", "not prohibited", "S8.exceptions" if "S8.exceptions" in by else "S8.note")
    return cond


def sports(chunks):
    p1 = next(c for c in chunks if c["clause_ref"] == "P1")["text"]
    seg = p1.split("Including, but not limited to")[0]
    out = []
    for b in seg.split("•")[1:]:
        b = b.replace("*Also prohibited Out-of-Competition", "").strip()
        ooc = "*" in b
        out.append(b.replace("*", "").strip() + (" [also Out-of-Competition]" if ooc else ""))
    return out


def s9_routes(chunks):
    s9 = next(c for c in chunks if c["clause_ref"] == "S9")["text"]
    m = re.search(r"administered by any (.*?) route\.", s9)
    note = next(c for c in chunks if c["clause_ref"] == "S9.note")["text"]
    m2 = re.search(r"\(including (.*?)\) are not prohibited", note)
    return (m.group(1) if m else ""), (m2.group(1) if m2 else "")


METHODS = {
    "M1.1": "Administration or reintroduction of blood or red blood cell products; withdrawal of blood",
    "M1.2": None, "M1.3": "Intravascular manipulation of blood or blood components",
    "M1.4": "Re-breathing systems to deliver carbon monoxide",
    "M2.1": "Tampering with Samples (substitution, adulteration)",
    "M2.2": "Intravenous infusions and/or injections of more than 100 mL per 12-hour period",
    "M3.1": "Use of nucleic acids or nucleic acid analogues (gene editing, silencing, transfer)",
    "M3.2": "Use of normal or genetically modified cells or cell components",
}


def build(year):
    chunks = [json.loads(l) for l in open(ROOT / f"data/processed/chunks_{year}.jsonl")]
    cond = conditions(chunks)
    sp = sports(chunks)
    s9_proh, s9_ok = s9_routes(chunks)
    rows = []
    skip = {"INTRO", "S1", "S2", "S2.1", "S2.2", "S4", "S4.4", "M1", "M2", "M3", "S6",
            "S3.note", "S3.exceptions", "S5.note", "S5.exceptions", "S6.exceptions",
            "S6.footnotes", "S8.note", "S8.exceptions", "S9.note"}
    for c in chunks:
        ref = c["clause_ref"]
        if ref in skip:
            continue
        sec = ref[:2]
        if sec in ("M1", "M2", "M3") and METHODS.get(ref):
            items = [(METHODS[ref], [], None, True)]
        else:
            items = body_items(c)
        for name, syn, parent, is_class in items:
            name = re.sub(r"\s+", " ", name).strip(" .:;,")
            name = re.sub(r"(?<=[0-9])- (?=[a-z])", "-", name)
            if len(name) < 2 or name.lower() in ("and", "e.g"):
                continue
            if sec == "P1" and not name.lower().endswith("lol"):
                continue
            status = STATUS[sec]
            if name.startswith("__EXCLUDED__"):
                name, status = name[len("__EXCLUDED__"):], "not_prohibited"
            rows.append(dict(year=year, substance=name, synonyms="; ".join(s for s in syn if s),
                             parent_class=parent or "", is_class=is_class, clause_ref=ref,
                             section=sec, page=c["page"], status=status))
    # extra rows for names that exist only in exception / footnote clauses
    present = {r["substance"].lower() for r in rows}
    for n, d in cond.items():
        n = n.strip()
        if n not in present and n:
            ref = d["refs"][0]
            rows.append(dict(year=year, substance=n[:1].upper() + n[1:], synonyms="", parent_class="",
                             is_class=False, clause_ref=ref, section=ref[:2], page=next(
                                 c["page"] for c in chunks if c["clause_ref"] == ref),
                             status="not_prohibited"))
    text_by_ref = {c["clause_ref"]: c["text"] for c in chunks}
    for r in rows:
        d = cond.get(r["substance"].lower(), {})
        r["route_exceptions"] = "; ".join(d.get("route_exceptions", []))
        r["dose_thresholds"] = "; ".join(d.get("dose_thresholds", []))
        other = []
        if d.get("monitoring_only"):
            other.append("Monitoring Program, not considered a Prohibited Substance")
        if d.get("exception"):
            other.append("; ".join(d["exception"]))
        if r["section"] == "S9":
            other.append(f"prohibited only by {s9_proh} route; not prohibited by {s9_ok}")
        if r["clause_ref"] == "S2.2.1":
            other.append("prohibited in males")
        if r["clause_ref"] == "S1.1":
            other.append("when administered exogenously")
        if "intravenous administration" in text_by_ref.get(r["clause_ref"], "") and r["parent_class"].startswith(
                "Plasma expanders"):
            other.append("by intravenous administration")
        r["other_conditions"] = "; ".join(other)
        r["sports"] = "; ".join(sp) if r["section"] == "P1" else ""
        if d.get("monitoring_only") or d.get("exception"):
            r["status"] = "not_prohibited"
        r["condition_refs"] = "; ".join(sorted(set(d.get("refs", []))))
        r["tue_note"] = ("TUE possible for therapeutic use" if r["status"] not in ("not_prohibited",)
                         else "")
        snip = text_by_ref.get(r["clause_ref"], "")
        k = snip.lower().find(r["substance"].lower()[:20])
        r["source_text_snippet"] = snip[max(0, k - 40):k + 160] if k >= 0 else snip[:200]
    df = pd.DataFrame(rows).drop_duplicates(subset=["substance", "clause_ref"])
    df.insert(0, "rule_id", [f"A{year}-{i:04d}" for i in range(len(df))])
    return df


def main():
    year = int(sys.argv[1])
    df = build(year)
    out = ROOT / f"data/processed/rules_{year}_passA.csv"
    df.to_csv(out, index=False)
    print(f"{year}: {len(df)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
