"""Parse a WADA Prohibited List PDF into clause-level chunks.

One chunk per numbered subsection (e.g. S1.1, S2.2.3, M1.2, S6.B) or per
NOTE / EXCEPTIONS / footnote block. Blocks are read in positional order, so
headings precede the bullets they govern.

Usage: python src/parse_list.py <year>
Writes data/processed/chunks_<year>.jsonl
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]

SECTION_TITLES = {
    "S0": "Non-approved substances", "S1": "Anabolic agents",
    "S2": "Peptide hormones, growth factors, related substances, and mimetics",
    "S3": "Beta-2 agonists", "S4": "Hormone and metabolic modulators",
    "S5": "Diuretics and masking agents", "M1": "Manipulation of blood and blood components",
    "M2": "Chemical and physical manipulation", "M3": "Gene and cell doping",
    "S6": "Stimulants", "S7": "Narcotics", "S8": "Cannabinoids",
    "S9": "Glucocorticoids", "P1": "Beta-blockers", "INTRO": "Introduction",
}
SECTION_STATUS = {
    "S0": "all_times", "S1": "all_times", "S2": "all_times", "S3": "all_times",
    "S4": "all_times", "S5": "all_times", "M1": "all_times", "M2": "all_times",
    "M3": "all_times", "S6": "in_competition", "S7": "in_competition",
    "S8": "in_competition", "S9": "in_competition", "P1": "particular_sports",
    "INTRO": "",
}
# Subsection header, e.g. "S1.1.", "S2.1.3", "S4.4.1", "M1.2.", "S6.A:"
SUB_RE = re.compile(r"^(S\d(?:\.\d){1,2}|M\d\.\d|S6\.[AB])[\.:]?(?:\s|$)")
PAGE_SECTION_RE = re.compile(r"^(S\d|P1)$|^(S\d|P1)\s")
BOILERPLATE = ("PROHIBITED AT ALL TIMES", "PROHIBITED IN-COMPETITION",
               "PROHIBITED IN PARTICULAR SPORTS", "(continued)")


def clean(t):
    t = t.replace("​", "").replace("\t", " ")
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", t)  # control characters left by bullet glyphs
    # re-join words hyphenated over a line break: drop the hyphen inside a word,
    # keep it in chemical names (before a digit or bracket)
    t = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", t)
    t = re.sub(r"-\n", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(?<=[a-zA-Z0-9ßɑ])- (?!and |or )(?=[a-z0-9ɑß(\[])", "-", t)
    t = re.sub(r"ɑ - (?=[a-z])", "ɑ-", t)
    return t


def is_separator(t, x0, x1, width):
    return (SUB_RE.match(t) or t in ("NOTE", "EXCEPTIONS") or t.startswith(("Including", "*"))
            or (x1 - x0) > 0.6 * width or t.isupper())


def column_order(blocks, width):
    """Within each band between full-width or heading blocks, read column by column."""
    out, band = [], []

    def emit():
        if not band:
            return
        xs = sorted({round(b[0]) for b in band})
        clusters, start = {}, None
        for x in xs:
            if start is None or x - start > 60:
                start = x
            clusters[x] = start
        band.sort(key=lambda b: (clusters[round(b[0])], b[1]))
        out.extend(band)
        band.clear()

    for b in blocks:
        t = clean(b[4])
        if is_separator(t, b[0], b[2], width):
            emit()
            out.append(b)
        else:
            band.append(b)
    emit()
    return out


def page_section(blocks, pno, prev):
    """Section code for a page, from its running header."""
    for b in blocks[:4]:
        t = clean(b[4])
        m = PAGE_SECTION_RE.match(t)
        if m:
            return m.group(1) or m.group(2)
        if t.startswith("PROHIBITED METHODS"):
            return "M"
    return prev


def parse(year):
    path = ROOT / f"data/raw/{year}_list_clean.pdf"
    if not path.exists():  # older Lists for the status-change arm
        path = ROOT / f"data/raw/historical/{year}_list_clean.pdf"
    doc = pymupdf.open(path)
    chunks, cur, section = [], None, None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = cur["text"].strip()
            chunks.append(cur)
        cur = None

    def start(clause_ref, heading, page, sec):
        flush()
        nonlocal cur
        cur = {"year": year, "section": sec, "clause_ref": clause_ref,
               "heading": heading, "page": page, "text": ""}

    for pno in range(2, doc.page_count):  # skip cover and contents
        page = doc[pno]
        blocks = [b for b in page.get_text("blocks", sort=True) if clean(b[4])]
        blocks = column_order(blocks, page.rect.width)
        text0 = clean(blocks[0][4]) if blocks else ""
        if text0.startswith("INDEX") or any(clean(b[4]) == "INDEX" for b in blocks[:3]):
            break
        pg = pno + 1
        if pg == 3:
            section = "INTRO"
            start("INTRO", "Introduction and definitions", pg, "INTRO")
        else:
            new_sec = page_section(blocks, pno, section)
            if new_sec != section and new_sec != "M":
                section = new_sec
                start(section, SECTION_TITLES.get(section, section), pg, section)
            elif new_sec == "M" and not (section or "").startswith("M"):
                section = "M1"
        for b in blocks:
            t = clean(b[4])
            if re.fullmatch(r"\d{1,2}", t):  # page number
                continue
            if pg == 3 and ("Published by" in t or t.startswith(("THE ", "WORLD ANTI", "VALID",
                            "World Anti-Doping Agency", "Place Victoria", "Montr", "H3C", "URL",
                            "Tel", "Fax", "E-mail"))):
                continue
            if PAGE_SECTION_RE.match(t) and len(t) < 60 and pg != 3:
                continue  # running header (section code + title)
            if t.startswith(("METABOLIC MODULATORS", "STIMULANTS (continued)",
                             "ANABOLIC AGENTS (continued)", "PROHIBITED METHODS")) and len(t) < 60:
                continue
            m = SUB_RE.match(t)
            if m:
                ref = m.group(1)
                sec = ref[:2]
                if sec.startswith("M"):
                    section = sec
                if cur and cur["clause_ref"] == ref:
                    cur["text"] += " " + t
                    continue
                if " (continued)" in t:
                    start(ref, t.replace(" (continued)", ""), pg, sec)
                    continue
                start(ref, t, pg, sec)
                cur["text"] = t
                continue
            if t in ("NOTE", "EXCEPTIONS"):
                start(f"{section}.{t.lower()}", t.title(), pg, section)
                continue
            if t.startswith("*") and section in ("S6", "P1"):
                if section == "P1":  # legend for the sport list, keep with the section
                    cur["text"] += " " + t
                    continue
                if not (cur and cur["clause_ref"] == f"{section}.footnotes"):
                    start(f"{section}.footnotes", "Footnotes", pg, section)
                cur["text"] += " " + t
                continue
            if t.startswith("M") and re.match(r"^M\d\. ", t):
                section = t[:2]
                start(section, t, pg, section)
                continue
            if any(t.startswith(x) or t == x for x in BOILERPLATE) and len(t) < 70:
                if cur:
                    cur.setdefault("status_line", t)
                continue
            if cur is None:
                start(section, SECTION_TITLES.get(section, section), pg, section)
            cur["text"] += " " + t
    flush()
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{year}-{i:03d}-{c['clause_ref']}"
        c["section_title"] = SECTION_TITLES.get(c["section"], c["section"])
        c["section_status"] = SECTION_STATUS.get(c["section"], "")
    return chunks


def main():
    year = int(sys.argv[1])
    chunks = parse(year)
    out = ROOT / f"data/processed/chunks_{year}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{year}: {len(chunks)} chunks -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
