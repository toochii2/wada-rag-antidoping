"""Map brand names to generic ingredients through openFDA and RxNorm.

A brand is kept only when it is unambiguous: RxNorm resolves it to exactly one
ingredient, openFDA labels for that brand name list that same single
ingredient, and the ingredient (after US-to-INN name mapping) is named in the
2027 rule table or in the List's exception and Monitoring Program text.

Writes benchmark/brand_map.csv
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    "Ventolin", "ProAir", "Proventil", "Pulmicort", "Flovent", "Medrol", "Decadron", "Rayos", "Inderal",
    "Tenormin", "Lopressor", "Toprol-XL", "Bystolic", "Lasix", "Aldactone", "Inspra", "Kerendia", "Diamox",
    "Trusopt", "Azopt", "Afrin", "Ritalin", "Provigil", "Vyvanse", "Arimidex", "Femara", "Aromasin",
    "Soltamox", "Evista", "Ultram", "OxyContin", "EpiPen", "Zyban", "Wellbutrin", "Ozempic", "Mounjaro",
    "Epidiolex", "Catapres", "Intuniv", "Livdelzi", "Inluriyo", "Orserdu", "Anadrol-50", "Oxandrin",
    "Sudafed", "Qvar", "Arnuity Ellipta", "Entocort EC", "Uceris", "Cortef", "Kenalog",
    "Androgel", "Testim", "Xyosted", "Novarel", "Pregnyl", "Lupron Depot", "Zoladex", "Egrifta",
    "Retacrit", "Epogen", "Aranesp", "Jesduvroq", "Reblozyl", "Increlex", "Serevent", "Striverdi Respimat",
    "Arcapta", "Xopenex", "Brovana", "Perforomist", "Nuvigil", "Sunosi", "Desoxyn", "Evekeo", "Adipex-P",
    "Dilaudid", "Demerol", "Methadose", "Subutex", "Duragesic", "Bumex", "Demadex", "Zaroxolyn",
    "Dyrenium", "Microzide", "Thalitone", "Vaprisol", "Samsca", "DDAVP", "Nocdurna", "Fareston",
    "Faslodex", "Osphena", "Clomid", "Kapvay", "Lumify", "Visine", "Nexiclon", "Chantix", "Nicorette",
    "Nucynta", "Wegovy", "Zepbound", "Trulicity", "Dexedrine", "Zenzedi", "Focalin", "Mydayis",
    "Kerlone", "Corgard", "Betapace", "Coreg", "Trandate", "Sectral", "Zebeta", "Timoptic", "Betoptic",
]
US_TO_INN = {"albuterol": "salbutamol", "levalbuterol": "levosalbutamol", "epinephrine": "epinephrine",
             "clomiphene": "clomifene", "beclomethasone": "beclometasone", "betamethasone": "betamethasone", "amphetamine": "amfetamine", "dextroamphetamine": "amfetamine",
             "methamphetamine": "metamfetamine", "meperidine": "pethidine", "ethacrynic acid": "etacrynic acid",
             "chlorthalidone": "chlortalidone", "torsemide": "torasemide", "tetrahydrozoline": "tetryzoline",
             "glyburide": "glibenclamide", "dexmethylphenidate": "methylphenidate",
             "triamcinolone": "triamcinolone acetonide", "armodafinil": "modafinil",
             "somatropin": "growth hormone", "chorionic gonadotropin": "chorionic gonadotrophin",
             "leuprolide": "leuprorelin", "epoetin alfa": "erythropoietins", "epoetin alfa-epbx": "erythropoietins",
             "darbepoetin alfa": "darbepoetins", "testosterone cypionate": "testosterone",
             "testosterone enanthate": "testosterone", "nicotine polacrilex": "nicotine",
             "tapentadol": "tapentadol", "nebivolol": "nebivolol", "dihydrocodeine": "dihydrocodeine"}
# salts and esters to strip before matching
SALTS = [" hydrochloride", " sulfate", " sulphate", " tartrate", " succinate", " citrate", " maleate",
         " acetate", " phosphate", " sodium", " hydrobromide", " dimesylate", " bitartrate", " mesylate",
         " propionate", " furoate", " dipropionate", " fumarate", " xinafoate", " cypionate", " enanthate",
         " polacrilex", " besylate", " potassium", " calcium", " anhydrous", " monohydrate", " hcl"]


def strip_salt(s):
    s = s.lower().strip()
    for x in SALTS:
        s = s.replace(x, "")
    return s.strip()


def getj(url, params, tries=5):
    """GET JSON with backoff; the public APIs sometimes answer a burst with a non-JSON page."""
    for k in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 404:
            return {}
        try:
            return r.json()
        except ValueError:
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"no JSON from {url}")


def rxnorm_ingredients(brand):
    ids = getj("https://rxnav.nlm.nih.gov/REST/rxcui.json", {"name": brand, "search": 2}).get(
        "idGroup", {}).get("rxnormId", [])
    ings = set()
    for i in ids[:3]:
        rr = getj(f"https://rxnav.nlm.nih.gov/REST/rxcui/{i}/related.json", {"tty": "IN MIN"})
        for g in rr.get("relatedGroup", {}).get("conceptGroup", []):
            for c in g.get("conceptProperties", []) or []:
                ings.add((g["tty"], c["name"].lower()))
    return ids, ings


def openfda_generics(brand):
    js = getj("https://api.fda.gov/drug/label.json", {"search": f'openfda.brand_name:"{brand}"', "limit": 20})
    gens = set()
    for res in js.get("results", []):
        for g in res.get("openfda", {}).get("generic_name", []):
            gens.add(g.lower())
    return gens


def main():
    rules = pd.read_csv(ROOT / "data/processed/rules_2027.csv", dtype=str).fillna("")
    named = {str(x).lower() for x in rules["substance"]}
    for syn in rules["synonyms"]:
        named |= {s.strip().lower() for s in str(syn).split(";") if s.strip()}
    mon = " ".join(json.loads(l)["text"].lower() for l in open(ROOT / "data/processed/chunks_2027.jsonl"))
    out = []
    for brand in CANDIDATES:
        try:
            ids, ings = rxnorm_ingredients(brand)
            gens = openfda_generics(brand)
        except Exception as e:  # network failure: record and move on
            out.append({"brand": brand, "status": f"error {e}"})
            continue
        in_ = sorted({n for t, n in ings if t == "IN"})
        multi = sorted({n for t, n in ings if t == "MIN"})
        gen_set = {strip_salt(g) for g in gens}
        rx_single = in_[0] if len(in_) == 1 and not multi else None
        fda_single = next(iter(gen_set)) if len(gen_set) == 1 else None
        generic = strip_salt(rx_single) if rx_single else None
        agree = bool(generic and fda_single and (generic == fda_single or generic in fda_single
                                                or fda_single in generic))
        inn = US_TO_INN.get(generic, generic) if generic else None
        on_list = bool(inn and (inn in named or inn in mon))
        status = ("kept" if (agree and on_list) else
                  "rejected: not single-ingredient in RxNorm" if not rx_single else
                  "rejected: openFDA missing or disagrees" if not agree else
                  "rejected: ingredient not named in 2027 List text")
        out.append({"brand": brand, "rxnorm_rxcui": ";".join(ids[:3]), "rxnorm_ingredients": ";".join(in_ + multi),
                    "openfda_generic_names": ";".join(sorted(gens))[:200], "generic_us": generic or "",
                    "generic_inn_list_name": inn or "", "status": status})
        time.sleep(0.6)
        print(brand, status, inn)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "benchmark/brand_map.csv", index=False)
    print((df["status"] == "kept").sum(), "brands kept of", len(df))


if __name__ == "__main__":
    main()
