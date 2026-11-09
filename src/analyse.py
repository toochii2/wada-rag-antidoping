"""Metrics, statistics, figures and LaTeX tables from results/scored.csv.

Every number used in the paper is written to results/metrics.json and to
paper/generated/numbers.tex (LaTeX macros); tables go to paper/generated/*.tex
and figures to paper/figures/*.pdf.
"""
import json
import string
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, kendalltau  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"
B = 10000
RNG = np.random.default_rng(20261031)
COND_NAME = {"C0": "Closed-book", "C1": "RAG, current List", "C2": "RAG, stale List"}
# Okabe-Ito colour-blind-safe palette
COL = {"C0": "#E69F00", "C1": "#0072B2", "C2": "#D55E00"}
PROH = ["prohibited_at_all_times", "prohibited_in_competition_only", "prohibited_in_particular_sports"]
MODEL_ORDER = ["qwen3:30b-a3b", "qwen3:14b", "gemma3:12b", "llama3.1:8b"]
WEIGHT_SETS = {
    "base": {"false_permit": 10, "conditional_permit": 5, "scope_understated": 5, "missed_condition": 5,
             "unsafe_assurance": 5, "false_prohibit": 2, "unnecessary_deferral": 1, "over_caution": 1,
             "scope_overstated": 1, "correct": 0},
}
for fp in (5, 20):
    WEIGHT_SETS[f"fp{fp}"] = {**WEIGHT_SETS["base"], "false_permit": fp}
WEIGHT_SETS["flat"] = {k: (0 if k == "correct" else 1) for k in WEIGHT_SETS["base"]}
WEIGHT_SETS["mc2_fp20"] = {**WEIGHT_SETS["base"], "false_permit": 20, "missed_condition": 2,
                           "conditional_permit": 10, "scope_understated": 10}


def model_label(m):
    return {"qwen3:30b-a3b": "Qwen3-30B-A3B", "qwen3:14b": "Qwen3-14B", "gemma3:12b": "Gemma3-12B",
            "llama3.1:8b": "Llama3.1-8B"}.get(m, m)


def _cluster_draws(values, clusters):
    """B bootstrap draws of (sum, count) for one stratum, resampling whole clusters (items sharing a substance)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters if clusters is not None else np.arange(len(values)))
    ids, inv = np.unique(clusters, return_inverse=True)
    ok = ~np.isnan(values)
    csum = np.bincount(inv, weights=np.where(ok, values, 0.0), minlength=len(ids))
    ccnt = np.bincount(inv, weights=ok.astype(float), minlength=len(ids))
    idx = RNG.integers(0, len(ids), size=(B, len(ids)))
    return csum[idx].sum(axis=1), ccnt[idx].sum(axis=1)


def boot_mean(values, strata, clusters=None):
    """95% CI of a pooled mean, resampling within strata; with clusters, whole clusters (substances) are resampled.
    NaN values (rows outside the metric's denominator) keep their place but do not count."""
    values, strata = np.asarray(values, dtype=float), np.asarray(strata)
    clusters = None if clusters is None else np.asarray(clusters)
    sums, counts = np.zeros(B), np.zeros(B)
    for g in np.unique(strata):
        m = strata == g
        s_, c_ = _cluster_draws(values[m], None if clusters is None else clusters[m])
        sums += s_
        counts += c_
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means = means[~np.isnan(means)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))) if len(means) else (np.nan, np.nan)


def boot_did(diff_a, diff_b, cl_a=None, cl_b=None):
    """95% CI of mean(diff_b) - mean(diff_a), resampling substance clusters within each group."""
    sa, ca = _cluster_draws(diff_a, cl_a)
    sb, cb = _cluster_draws(diff_b, cl_b)
    d = sb / cb - sa / ca
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def wilson(k, n, z=1.959964):
    """Wilson score interval for a single proportion (better than the percentile bootstrap near 0 and 1)."""
    if not n:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def metric_values(d, col, mask):
    """Per-row values of a rate metric: the column inside the denominator, NaN outside it."""
    v = d[col].astype(float).where(d[col].notna())
    if mask is not None:
        v = v.where(mask(d))
    return v.values


def rate(col, mask=None):
    def f(d):
        d = d if mask is None else d[mask(d)]
        x = d[col].dropna()
        return float(x.astype(float).mean()) if len(x) else np.nan
    return f


def mcnemar(a, b):
    """Exact McNemar on paired binary vectors; returns b10, b01, p."""
    b10 = int(((a == 1) & (b == 0)).sum())
    b01 = int(((a == 0) & (b == 1)).sum())
    n = b10 + b01
    p = binomtest(b10, n, 0.5).pvalue if n else 1.0
    return b10, b01, float(p)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    m = len(ps)
    run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj.tolist()


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "results/scored.csv")
    items = [json.loads(l) for l in open(ROOT / "benchmark/items.jsonl")]
    main_ = df[(df["tag"] == "main")].reset_index(drop=True)
    models = [m for m in MODEL_ORDER if m in set(main_["model"])] + sorted(set(main_["model"]) - set(MODEL_ORDER))
    M = {"n_items": len(items), "models": models, "per": {}, "tests": {}, "h2": {}, "sensitivity": {}}
    is_proh = lambda d: d["gold"].isin(PROH)  # noqa: E731
    metric_defs = {
        "acc6": ("correct6", None), "acc_collapsed": ("correct_c", None),
        "permit_on_permitted": ("permit_on_permitted", lambda d: d["gold"].isin(["permitted", "permitted_with_conditions"])),
        "false_permit": ("false_permit", is_proh), "lenient_permit": ("lenient_permit", is_proh),
        "false_prohibit": ("false_prohibit", lambda d: d["gold"].isin(["permitted", "permitted_with_conditions"])),
        "appropriate_deferral": ("appropriate", lambda d: (d["group"] == "D") & (d["gold"] == "cannot_determine_refer")),
        "unnecessary_deferral": ("deferral", lambda d: d["group"].isin(["A", "B", "C"])),
        "cond_complete": ("cond_complete", lambda d: d["group"] == "B"),
        "cite_ok": ("cite_ok", None), "recall_exact": ("recall_exact", None), "recall_section": ("recall_section", None),
        "nn_defer": ("deferral", lambda d: (d["group"] == "C") & (d["newly_named"] == True)),  # noqa: E712
        "nn_fp": ("false_permit", lambda d: (d["group"] == "C") & (d["newly_named"] == True)),  # noqa: E712
        "lenient_no_tue": ("lenient_permit_no_tue", is_proh),
        "defer_proh": ("deferral", is_proh),
        "harm": ("harm", None), "parse_fail": ("parsed_ok", None),
    }
    for m in models:
        for c in ("C0", "C1", "C2"):
            d = main_[(main_["model"] == m) & (main_["condition"] == c)]
            if d.empty:
                continue
            res = {"n": len(d)}
            for k, (col, mask) in metric_defs.items():
                f = rate(col, mask)
                v = f(d)
                if k == "parse_fail":
                    res[k] = 1 - v
                    continue
                res[k] = v
                if np.isnan(v):
                    continue
                if k == "harm":
                    res[k + "_ci"] = boot_mean(metric_values(d, col, mask), d["group"].values, d["substance"].values)
                elif k in ("false_permit", "lenient_permit", "lenient_no_tue", "acc6", "acc_collapsed", "nn_defer", "nn_fp",
                           "appropriate_deferral", "unnecessary_deferral", "permit_on_permitted"):
                    x = metric_values(d, col, mask)
                    x = x[~np.isnan(x)]
                    res[k + "_ci"] = wilson(int(x.sum()), len(x))
            for g in "ABCD":
                dg = d[d["group"] == g]
                res[f"harm_{g}"] = float(dg["harm"].mean()) if len(dg) else np.nan
                res[f"fp_{g}"] = rate("false_permit", is_proh)(dg) if len(dg) else np.nan
            res["harm_classes"] = d["harm_class"].value_counts().to_dict()
            M["per"][f"{m}|{c}"] = res
    # paired McNemar on false-permit indicator among gold-prohibited items
    fam = {"C0vC1": [], "C1vC2": [], "C0vC2": []}
    for m in models:
        for name, (x, y) in {"C0vC1": ("C0", "C1"), "C1vC2": ("C1", "C2"), "C0vC2": ("C0", "C2")}.items():
            a = main_[(main_["model"] == m) & (main_["condition"] == x) & main_["gold"].isin(PROH)].set_index("item_id")
            b = main_[(main_["model"] == m) & (main_["condition"] == y) & main_["gold"].isin(PROH)].set_index("item_id")
            ids = a.index.intersection(b.index)
            if not len(ids):
                continue
            va, vb = a.loc[ids, "false_permit"], b.loc[ids, "false_permit"]
            b10, b01, p = mcnemar(va, vb)
            rd = float(va.mean() - vb.mean())
            orr = (b10 + 0.5) / (b01 + 0.5)
            fam[name].append((m, {"n_pairs": len(ids), "only_first": b10, "only_second": b01, "p": p,
                                  "risk_difference": rd, "paired_or_haldane": orr,
                                  "rate_first": float(va.mean()), "rate_second": float(vb.mean())}))
    for name, lst in fam.items():
        adj = holm([x[1]["p"] for x in lst]) if lst else []
        for (m, r), pa in zip(lst, adj):
            r["p_holm"] = pa
            M["tests"][f"{m}|{name}"] = r
    # H3 (stale answers on changed-label revision items) and H4 (appropriate deferral on Group D refer items)
    other = {"nndefer": ("deferral", lambda d: (d["group"] == "C") & (d["newly_named"] == True),  # noqa: E712
                         {"C1vC2": ("C1", "C2"), "C0vC1": ("C0", "C1")}),
             "lp": ("lenient_permit", lambda d: d["gold"].isin(PROH), {"C0vC1": ("C0", "C1")}),
             "lpnotue": ("lenient_permit_no_tue", lambda d: d["gold"].isin(PROH), {"C0vC1": ("C0", "C1")}),
             "nnfp": ("false_permit", lambda d: (d["group"] == "C") & (d["newly_named"] == True),  # noqa: E712
                      {"C1vC2": ("C1", "C2")}),
             "defer": ("appropriate", lambda d: (d["group"] == "D") & (d["gold"] == "cannot_determine_refer"),
                       {"C0vC1": ("C0", "C1")}),
             "accs": ("correct_c", lambda d: d["group"].notna(), {"C0vC1": ("C0", "C1"), "C1vC2": ("C1", "C2")}),
             "nnaccs": ("correct_c", lambda d: (d["group"] == "C") & (d["newly_named"] == True),  # noqa: E712
                        {"C1vC2": ("C1", "C2")}),
             "pp": ("permit_on_permitted", lambda d: d["gold"].isin(["permitted", "permitted_with_conditions"]),
                    {"C0vC1": ("C0", "C1")})}
    for fam_name, (col, mask, pairs) in other.items():
        for name, (x, y) in pairs.items():
            lst = []
            for m in models:
                a = main_[(main_["model"] == m) & (main_["condition"] == x)]
                b = main_[(main_["model"] == m) & (main_["condition"] == y)]
                a, b = a[mask(a)].set_index("item_id"), b[mask(b)].set_index("item_id")
                ids = a.index.intersection(b.index)
                if not len(ids):
                    continue
                va, vb = a.loc[ids, col].astype(float), b.loc[ids, col].astype(float)
                b10, b01, p = mcnemar(va, vb)
                lst.append((m, {"n_pairs": len(ids), "only_first": b10, "only_second": b01, "p": p,
                                "rate_first": float(va.mean()), "rate_second": float(vb.mean()),
                                "risk_difference": float(va.mean() - vb.mean())}))
            for (m, r), pa in zip(lst, holm([x[1]["p"] for x in lst]) if lst else []):
                r["p_holm"] = pa
                M["tests"][f"{m}|{fam_name}_{name}"] = r
    sub_of = {i["item_id"]: i["substance"] for i in items}
    # primary endpoint: paired risk difference C0 - C1 with a substance-cluster bootstrap interval
    M["fp_rd"] = {}
    for m in models:
        d = main_[(main_["model"] == m) & main_["gold"].isin(PROH)]
        piv = d.pivot_table(index=["item_id", "group"], columns="condition", values="false_permit").dropna().reset_index()
        for x, y in (("C0", "C1"), ("C1", "C2")):
            diff = (piv[x] - piv[y]).values
            M["fp_rd"][f"{m}|{x}-{y}"] = {"rd": float(diff.mean()), "ci": boot_mean(
                diff, piv["group"].values, piv["item_id"].map(sub_of).values)}
    # safety-verdict accuracy, C1 - C2: substance-cluster CI, and the difference outside the newly named items
    M["accs_rd"] = {}
    for m in models:
        d = main_[main_["model"] == m]
        piv = d.pivot_table(index=["item_id", "group"], columns="condition", values="correct_c").dropna().reset_index()
        diff = (piv["C1"].astype(float) - piv["C2"].astype(float)).values
        nnm = piv["item_id"].map(lambda i: bool(next(x for x in items if x["item_id"] == i).get("newly_named"))).values
        M["accs_rd"][m] = {"rd": float(diff.mean()), "ci": boot_mean(diff, piv["group"].values, piv["item_id"].map(sub_of).values),
                           "rd_not_newly_named": float(diff[~nnm].mean())}
    # substance-level sign tests next to the item-level McNemar tests (items sharing a substance are not independent)
    M["subst_tests"] = {}
    for fam_name, col, mask, (x, y), up_is in [
            ("accs_C1vC2", "correct_c", None, ("C1", "C2"), "first"), ("accs_C0vC1", "correct_c", None, ("C0", "C1"), "second"),
            ("fp_C0vC1", "false_permit", "proh", ("C0", "C1"), "first")]:
        lst = []
        for m in models:
            d0 = main_[main_["model"] == m]
            if mask == "proh":
                d0 = d0[d0["gold"].isin(PROH)]
            piv = d0.pivot_table(index="substance", columns="condition", values=col, aggfunc="mean")
            diff = (piv[x].astype(float) - piv[y].astype(float)).dropna()
            a_, b_ = int((diff > 0).sum()), int((diff < 0).sum())
            p = binomtest(a_, a_ + b_, 0.5).pvalue if a_ + b_ else 1.0
            lst.append((m, {"first_higher": a_, "second_higher": b_, "p": float(p)}))
        for (m, r), pa in zip(lst, holm([z[1]["p"] for z in lst])):
            r["p_holm"] = pa
            M["subst_tests"][f"{m}|{fam_name}"] = r
    # newly named items: 9 substances x 4 phrasings, so test at substance level (sign test on C2 - C1)
    M["nn_sign"] = {}
    nn = main_[(main_["group"] == "C") & (main_["newly_named"] == True)]  # noqa: E712
    lst = []
    for m in models:
        piv = nn[nn["model"] == m].pivot_table(index="substance", columns="condition", values="deferral")
        diff = (piv["C2"] - piv["C1"]).dropna()
        up, down = int((diff > 0).sum()), int((diff < 0).sum())
        p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
        lst.append((m, {"substances": int(len(diff)), "higher_with_2026": up, "lower_with_2026": down, "p": float(p)}))
    for (m, r), pa in zip(lst, holm([x[1]["p"] for x in lst])):
        r["p_holm"] = pa
        M["nn_sign"][m] = r
    # deferral on newly named items by whether the gold section was retrieved (pooled over models)
    M["nn_defer_by_retrieval"] = {}
    for (c, hit), g in nn[nn["condition"] != "C0"].groupby(["condition", "recall_section"]):
        M["nn_defer_by_retrieval"][f"{c}|{'hit' if hit else 'miss'}"] = {"defer": float(g["deferral"].mean()),
                                                                       "n": int(len(g))}
    # H2: false-permit advantage of RAG (C0 - C1) in Group B versus Group A, bootstrap over substance clusters
    for m in models:
        d = main_[(main_["model"] == m) & main_["gold"].isin(PROH) & main_["group"].isin(["A", "B"])]
        if d.empty or not {"C0", "C1"} <= set(d["condition"]):
            continue
        piv = d.pivot_table(index=["item_id", "group"], columns="condition", values="false_permit").dropna().reset_index()
        piv["diff"] = piv["C0"] - piv["C1"]
        def did(x):
            ga, gb = x[x["group"] == "A"]["diff"], x[x["group"] == "B"]["diff"]
            return gb.mean() - ga.mean() if len(ga) and len(gb) else np.nan
        est = did(piv)
        piv["substance"] = piv["item_id"].map(sub_of)
        ci = boot_did(piv[piv.group == "A"]["diff"], piv[piv.group == "B"]["diff"],
                      piv[piv.group == "A"]["substance"], piv[piv.group == "B"]["substance"])
        M["h2"][m] = {"adv_A": float(piv[piv.group == "A"]["diff"].mean()), "adv_B": float(piv[piv.group == "B"]["diff"].mean()),
                      "did": float(est), "ci": ci, "n_A": int((piv.group == "A").sum()), "n_B": int((piv.group == "B").sum())}
    # H2 and H1 for the harm score as well (all items)
    for m in models:
        d = main_[(main_["model"] == m) & main_["condition"].isin(["C0", "C1"]) & main_["group"].isin(["A", "B"])]
        piv = d.pivot_table(index=["item_id", "group"], columns="condition", values="harm").dropna().reset_index()
        if piv.empty or not {"C0", "C1"} <= set(piv.columns):
            continue
        piv["diff"] = piv["C0"] - piv["C1"]
        M["h2"].setdefault(m, {})["harm_adv_A"] = float(piv[piv.group == "A"]["diff"].mean())
        M["h2"][m]["harm_adv_B"] = float(piv[piv.group == "B"]["diff"].mean())
    # paired harm differences between conditions (exploratory), bootstrap over items stratified by group
    M["harm_diff"] = {}
    for m in models:
        d = main_[main_["model"] == m]
        piv = d.pivot_table(index=["item_id", "group"], columns="condition", values="harm").dropna().reset_index()
        for x, y in (("C0", "C1"), ("C1", "C2")):
            if not {x, y} <= set(piv.columns):
                continue
            f = (lambda a, b: (lambda z: float((z[a] - z[b]).mean())))(x, y)
            M["harm_diff"][f"{m}|{x}-{y}"] = {"diff": f(piv), "ci": boot_mean((piv[x] - piv[y]).values, piv["group"].values,
                                                                             piv["item_id"].map(sub_of).values)}
    # exploratory diagnostics: false permits by retrieval hit and by brand-name phrasing
    M["diag"] = {"fp_by_retrieval": {}, "fp_by_brand": {}, "fp_changed_items": {}}
    P = main_[main_["gold"].isin(PROH)]
    for (m, c, hit), g in P[P["condition"] != "C0"].groupby(["model", "condition", "recall_section"]):
        M["diag"]["fp_by_retrieval"][f"{m}|{c}|{'hit' if hit else 'miss'}"] = {"fp": float(g["false_permit"].mean()),
                                                                             "lenient": float(g["lenient_permit"].mean()),
                                                                             "n": int(len(g))}
    for (c, b), g in P.groupby(["condition", "is_brand"]):
        M["diag"]["fp_by_brand"][f"{c}|{'brand' if b else 'generic'}"] = {"fp": float(g["false_permit"].mean()),
                                                                          "lenient": float(g["lenient_permit"].mean()),
                                                                          "n": int(len(g))}
    ch = main_[(main_["group"] == "C") & (main_["newly_named"] == True)]  # noqa: E712
    for (m, c), g in ch.groupby(["model", "condition"]):
        M["diag"]["fp_changed_items"][f"{m}|{c}"] = {"fp": float(g["false_permit"].mean()),
                                                    "lenient": float(g["lenient_permit"].mean()),
                                                    "defer": float(g["deferral"].mean()), "n": int(len(g))}
    # trivial baselines: one fixed answer for every item
    from score import harm_class, WEIGHTS as HW
    M["baselines"] = {}
    for v in ("cannot_determine_refer", "prohibited_at_all_times", "permitted"):
        hs, fp, lp, n_p = [], 0, 0, 0
        for it in items:
            g = it["gold"]["verdict"]
            hc = "correct" if v in it["gold"].get("accept", []) else harm_class(g, v)
            hs.append(HW[hc])
            if g in PROH:
                n_p += 1
                fp += v == "permitted"
                lp += v in ("permitted", "permitted_with_conditions")
        from score import collapse
        M["baselines"][v] = {"harm": float(np.mean(hs)), "false_permit": fp / n_p, "acc6": float(np.mean(
            [it["gold"]["verdict"] == v or v in it["gold"].get("accept", []) for it in items])), "acc_collapsed": float(np.mean(
            [collapse(it["gold"]["verdict"]) == collapse(v) or v in it["gold"].get("accept", []) for it in items]))}
    # how often the stale-index answers mention the 2026 date, and how many of those defer
    M["diag"]["date_mentions"] = {}
    items_by = {i["item_id"]: i for i in items}
    for f in sorted((ROOT / "results/runs").glob("*__C2__main.jsonl")):
        seen, n, n26, n26ref, intro = set(), 0, 0, 0, 0
        for l in open(f):
            r = json.loads(l)
            if r["item_id"] in seen or r.get("question") != items_by[r["item_id"]]["question"]:
                continue
            seen.add(r["item_id"])
            n += 1
            intro += any(h["clause_ref"] == "INTRO" for h in r["retrieved"])
            if "2026" in (r.get("raw") or ""):
                n26 += 1
                n26ref += (r.get("parsed") or {}).get("verdict") == "cannot_determine_refer"
        M["diag"]["date_mentions"][r["model"]] = {"n": n, "mention_2026": n26, "mention_and_refer": n26ref,
                                                 "intro_retrieved": intro}
    # status-change arm (pre-specified in DECISIONS.md): historical index (CH) versus current index (C1)
    sc = df[df["tag"] == "statuschange"]
    if not sc.empty:
        S = {"per": {}, "tests": {}, "family": {}}
        chg = sc[sc["stale"].notna()]
        ctl = sc[sc["stale"].isna()]
        for (m, c), g in chg.groupby(["model", "condition"]):
            k, n = int(g["false_permit"].sum()), len(g)
            S["per"][f"{m}|{c}"] = {"fp": k / n, "fp_ci": wilson(k, n), "lp": float(g["lenient_permit"].mean()),
                                   "defer": float(g["deferral"].mean()), "n": n,
                                   "recall_section": float(g["recall_section"].dropna().astype(float).mean())
                                   if g["recall_section"].notna().any() else None,
                                   "control_acc": float(ctl[(ctl.model == m) & (ctl.condition == c)]["correct_c"].mean())}
        for (c, y), g in chg.groupby(["condition", "stale_year"]):
            S["family"][f"{c}|{int(y)}"] = {"fp": float(g["false_permit"].mean()), "n": int(len(g)),
                                            "defer": float(g["deferral"].mean()), "acc": float(g["correct_c"].mean())}
        for (c, hit), g in chg[chg.condition != "C0"].groupby(["condition", "recall_section"]):
            S["family"][f"{c}|{'hit' if hit else 'miss'}"] = {"fp": float(g["false_permit"].mean()), "n": int(len(g)),
                                                            "defer": float(g["deferral"].mean()),
                                                            "acc": float(g["correct_c"].mean())}
        for name, (x, y) in {"C1vCH": ("C1", "CH"), "C0vC1": ("C0", "C1")}.items():
            lst = []
            for m in models:
                a = chg[(chg.model == m) & (chg.condition == x)].set_index("item_id")["false_permit"]
                b = chg[(chg.model == m) & (chg.condition == y)].set_index("item_id")["false_permit"]
                ids = a.index.intersection(b.index)
                if len(ids):
                    b10, b01, p = mcnemar(a.loc[ids], b.loc[ids])
                    lst.append((m, {"n_pairs": len(ids), "only_first": b10, "only_second": b01, "p": p,
                                    "rate_first": float(a.loc[ids].mean()), "rate_second": float(b.loc[ids].mean())}))
            for (m, r), pa in zip(lst, holm([x[1]["p"] for x in lst]) if lst else []):
                r["p_holm"] = pa
                S["tests"][f"{m}|{name}"] = r
        comp = {}
        fam_of = chg["stale_year"].map({2021: "gc", 2023: "tram"})
        for (c, f_), g in chg.assign(fam=fam_of).groupby(["condition", "fam"]):
            v = g["pred"].fillna("none")
            comp[f"{c}|{f_}"] = {"permit": int(v.isin(["permitted", "permitted_with_conditions"]).sum()),
                                 "prohibited_ic": int((v == "prohibited_in_competition_only").sum()),
                                 "prohibited_other": int(v.isin(["prohibited_at_all_times", "prohibited_in_particular_sports"]).sum()),
                                 "refer": int((v == "cannot_determine_refer").sum()), "none": int((v == "none").sum()),
                                 "n": int(len(g)), "recall": float(g["recall_section"].dropna().astype(float).mean())
                                 if g["recall_section"].notna().any() else None}
        S["composition"] = comp
        S["c0_permit_by_model"] = {m: int(chg[(chg.model == m) & (chg.condition == "C0")]["pred"].isin(
            ["permitted", "permitted_with_conditions"]).sum()) for m in models}
        pooled = {c: chg[chg.condition == c]["false_permit"].mean() for c in ("C0", "C1", "CH")}
        S["pooled_fp"] = {c: float(v) for c, v in pooled.items()}
        M["statuschange"] = S
    # mixed-effects logistic regression: false permit ~ condition x group + model, item random effect
    M["mixed"] = mixed_model(main_)
    # weight sensitivity: rankings of model x condition cells
    cells = [(m, c) for m in models for c in ("C0", "C1", "C2") if f"{m}|{c}" in M["per"]]
    ranks = {}
    for ws, w in WEIGHT_SETS.items():
        scores = {f"{m}|{c}": float(main_[(main_.model == m) & (main_.condition == c)]["harm_class"].map(w).mean())
                  for m, c in cells}
        ranks[ws] = scores
    base = ranks["base"]
    sens = {}
    for ws, sc in ranks.items():
        keys = list(base)
        tau = kendalltau([base[k] for k in keys], [sc[k] for k in keys]).statistic if len(keys) > 2 else np.nan
        best_cond = {m: min(("C0", "C1", "C2"), key=lambda c: sc.get(f"{m}|{c}", np.inf)) for m in models}
        best_model = {c: min(models, key=lambda m: sc.get(f"{m}|{c}", np.inf)) for c in ("C0", "C1", "C2")}
        sens[ws] = {"scores": sc, "kendall_tau_vs_base": float(tau), "best_condition_per_model": best_cond,
                    "best_model_per_condition": best_model}
    M["sensitivity"] = sens
    M["sensitivity_rank_change"] = any(
        s["best_condition_per_model"] != sens["base"]["best_condition_per_model"] or
        s["best_model_per_condition"] != sens["base"]["best_model_per_condition"] for s in sens.values())
    # consistency run
    cons = df[df["tag"] == "consistency"]
    if not cons.empty:
        M["consistency"] = {}
        for (m, c), d in cons.groupby(["model", "condition"]):
            agree = d.groupby("item_id")["pred"].agg(lambda x: x.nunique() == 1 and x.notna().all())
            maj_fp = d.groupby("item_id")["false_permit"].max()
            M["consistency"][f"{m}|{c}"] = {"items": int(len(agree)), "all_repeats_agree": float(agree.mean()),
                                            "any_repeat_false_permit": float(maj_fp[d.groupby("item_id")["gold"].first().isin(PROH)].mean())}
    abl = df[df["tag"].str.startswith("k")]
    if not abl.empty:
        M["ablation"] = {}
        for (m, t), d in abl.groupby(["model", "tag"]):
            M["ablation"][f"{m}|{t}"] = {"false_permit": rate("false_permit", is_proh)(d), "harm": float(d["harm"].mean()),
                                         "acc6": float(d["correct6"].mean()), "recall_exact": rate("recall_exact")(d)}
    with open(ROOT / "results/metrics.json", "w") as f:
        json.dump(M, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    figures(main_, M, models)
    tables(items, M, models)
    macros(M, models, items)
    print("analysis written")


def mixed_model(d):
    out = {}
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        x = d[d["gold"].isin(PROH) & d["condition"].isin(["C0", "C1", "C2"])].copy()
        x["model"] = x["model"].astype("category")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = BinomialBayesMixedGLM.from_formula(
                "false_permit ~ C(condition, Treatment('C0')) * C(group) + C(model)", {"item": "0 + C(item_id)"}, x)
            fit = mdl.fit_vb()
        names = mdl.exog_names
        out = {"method": "BinomialBayesMixedGLM (variational Bayes), item random intercept",
               "converged": None, "note": "variational fit; statsmodels gives no convergence diagnostic",
               "n": int(len(x)),
               "coef": {n: {"mean": float(m), "sd": float(s)} for n, m, s in zip(names, fit.fe_mean, fit.fe_sd)},
               "item_re_sd_log": float(fit.vcp_mean[0])}
    except Exception as e:  # report and fall back to stratified McNemar
        out = {"converged": False, "error": str(e)[:300], "fallback": "stratified McNemar only"}
    return out


def figures(d, M, models):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    # Figure 2: strict and lenient false-permit rates by model x condition with 95% CIs
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True)
    w = 0.26
    for ax, (key, title) in zip(axes, [("false_permit", "False permit (verdict 'permitted')"),
                                       ("lenient_permit", "Any permit (incl. with conditions)")]):
        for j, c in enumerate(("C0", "C1", "C2")):
            xs, ys, lo, hi = [], [], [], []
            for i, m in enumerate(models):
                r = M["per"].get(f"{m}|{c}")
                if not r:
                    continue
                xs.append(i + (j - 1) * w)
                ys.append(100 * r[key])
                lo.append(100 * (r[key] - r[key + "_ci"][0]))
                hi.append(100 * (r[key + "_ci"][1] - r[key]))
            ax.bar(xs, ys, w, color=COL[c], label=COND_NAME[c], yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.5})
        ax.set_title(title, fontsize=7)
        ax.set_xticks(range(len(models)), [model_label(m).replace("-", "\n", 1) for m in models], fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of prohibited items")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_false_permit.pdf")
    plt.close(fig)
    # Figure 3: harm-weighted score by group x condition (mean over models)
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    for j, c in enumerate(("C0", "C1", "C2")):
        ys, lo, hi = [], [], []
        for g in "ABCD":
            x = d[(d.condition == c) & (d.group == g)]
            ys.append(x["harm"].mean())
            ci = boot_mean(x["harm"].values, x["model"].values, x["substance"].values) if len(x) else (np.nan, np.nan)
            lo.append(ys[-1] - ci[0])
            hi.append(ci[1] - ys[-1])
        ax.bar(np.arange(4) + (j - 1) * w, ys, w, color=COL[c], label=COND_NAME[c], yerr=[lo, hi], capsize=2,
               error_kw={"lw": 0.6})
    ax.set_xticks(range(4), ["A fixed", "B context", "C revision", "D not settleable"])
    ax.set_ylabel("Mean harm score per item")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_harm_by_group.pdf")
    plt.close(fig)
    # Figure 4: revision items, stale-answer and false-permit rates
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2), sharey=True)
    for ax, (col, title) in zip(axes, [("deferral", "Deferral, newly named"), ("false_permit", "False permits")]):
        for j, c in enumerate(("C0", "C1", "C2")):
            ys = []
            for m in models:
                x = d[(d.model == m) & (d.condition == c) & (d.group == "C") & (d.newly_named == True)]  # noqa: E712
                ys.append(100 * x[col].astype(float).mean() if len(x) else np.nan)
            ax.bar(np.arange(len(models)) + (j - 1) * w, ys, w, color=COL[c], label=COND_NAME[c])
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(models)), [model_label(m).replace("-", "\n", 1) for m in models], fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of changed-label items")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_revision.pdf")
    plt.close(fig)
    results_figure(d, M, models)
    design_figure()


def results_figure(d, M, models):
    """One 2x2 figure: (a) strict false permits, (b) any permit, (c) harm by group, (d) deferral on newly named items."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.2))
    w = 0.26
    err = {"lw": 0.6, "capthick": 0.6, "ecolor": "#333333"}
    labels = [model_label(m).replace("-A3B", "").replace("-", "\n", 1) for m in models]
    plotted = {}  # every value the figure draws, saved for the self-contained figure script

    def bars(ax, j, c, ys, lo=None, hi=None):
        panel = "abcd"[list(axes.flat).index(ax)]
        plotted.setdefault(panel, {})[c] = {"y": [round(float(v), 4) for v in ys],
                                            "lo": None if lo is None else [round(float(v), 4) for v in lo],
                                            "hi": None if hi is None else [round(float(v), 4) for v in hi]}
        x = np.arange(len(ys)) + (j - 1) * w
        ax.bar(x, ys, w, color=COL[c], edgecolor="white", linewidth=0.3, label=COND_NAME[c], zorder=2,
               yerr=None if lo is None else [lo, hi], capsize=1.5, error_kw=err)

    for ax, key, title, ylab in [(axes[0, 0], "false_permit", "False permit (primary)", "% of prohibited items"),
                                 (axes[0, 1], "acc_collapsed", "Safety-verdict accuracy", "% of all items")]:
        for j, c in enumerate(("C0", "C1", "C2")):
            ys, lo, hi = [], [], []
            for m in models:
                r = M["per"][f"{m}|{c}"]
                ys.append(100 * r[key])
                lo.append(100 * (r[key] - r[key + "_ci"][0]))
                hi.append(100 * (r[key + "_ci"][1] - r[key]))
            bars(ax, j, c, ys, lo, hi)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(models)), labels)
    base = 100 * M["baselines"]["prohibited_at_all_times"]["acc_collapsed"]
    axes[0, 1].axhline(base, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=3, label="Always \u2018prohibited\u2019")
    ax = axes[1, 0]
    S = M.get("statuschange")
    if S:  # status-change arm: what models answered on items whose status changed, pooled over models
        cats = [("permit", "Permitted (wrong)", "#B2182B"), ("prohibited_ic", "Prohibited in-comp. (right)", "#2166AC"),
                ("prohibited_other", "Prohibited, wider scope", "#92C5DE"), ("refer", "Refer", "#BDBDBD")]
        xs, xl = [], []
        for fi, (f_, fl) in enumerate((("gc", "Glucocorticoid inj. (2022)"), ("tram", "Tramadol (2024)"))):
            for ci, c in enumerate(("C0", "C1", "CH")):
                v = S["composition"][f"{c}|{f_}"]
                x0 = fi * 4.2 + ci
                bottom = 0.0
                for key, lab_, col in cats:
                    h = 100 * v[key] / v["n"]
                    ax.bar(x0, h, 0.82, bottom=bottom, color=col, edgecolor="white", linewidth=0.3, zorder=2,
                           label=lab_ if (fi == 0 and ci == 0) else None)
                    bottom += h
                xs.append(x0)
                xl.append({"C0": "CB", "C1": "Now", "CH": "Old"}[c])
            ax.text(fi * 4.2 + 1, -19, fl, ha="center", va="top", fontsize=6.6)
        ax.set_xticks(xs, xl)
        ax.tick_params(axis="x", labelsize=6.6)
        ax.set_title("Status-change items: verdicts", loc="left")
        ax.set_ylabel("% of answers")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
        ax.legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2,
                  handlelength=0.9, columnspacing=0.8, handletextpad=0.4)
    else:
        for j, c in enumerate(("C0", "C1", "C2")):
            ys, lo, hi = [], [], []
            for g in "ABCD":
                x = d[(d.condition == c) & (d.group == g)]
                ys.append(x["harm"].mean())
                ci = boot_mean(x["harm"].values, x["model"].values, x["substance"].values)
                lo.append(ys[-1] - ci[0])
                hi.append(ci[1] - ys[-1])
            bars(ax, j, c, ys, lo, hi)
        ax.set_title("Mean harm by benchmark group", loc="left")
        ax.set_ylabel("Mean harm per item")
        ax.set_xticks(range(4), ["A\nfixed", "B\ncontext", "C\nrevision", "D not\nsettleable"])
    ax = axes[1, 1]
    for j, c in enumerate(("C0", "C1", "C2")):
        ys, lo, hi = [], [], []
        for m in models:
            x = d[(d.model == m) & (d.condition == c) & (d.group == "C") & (d.newly_named == True)]  # noqa: E712
            v = x["deferral"].astype(float).values
            ys.append(100 * v.mean())
            ci = 100 * np.array(boot_mean(v, np.zeros(len(v)), x["substance"].values))
            lo.append(max(ys[-1] - ci[0], 0))
            hi.append(max(ci[1] - ys[-1], 0))
        bars(ax, j, c, ys, lo, hi)
    ax.set_title("Deferral, newly named items", loc="left")
    ax.set_ylabel("% deferred")
    ax.set_xticks(range(len(models)), labels)
    for ax, letter in zip(axes.flat, "abcd"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_ylim(bottom=0)
        ax.set_title(rf"$\bf{{{letter}}}$   " + ax.get_title(loc="left"), loc="left")
    axes[0, 0].set_ylim(0, 75)
    axes[0, 0].set_yticks(range(0, 76, 25))
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.tick_params(axis="x", labelsize=6.8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    h2, lab2 = axes[0, 1].get_legend_handles_labels()
    j = next(i for i, x in enumerate(lab2) if x.startswith("Always"))
    h, lab = h + [h2[j]], lab + [lab2[j]]
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0, w_pad=1.2)
    fig.savefig(FIG / "fig2_results.pdf")
    plt.close(fig)
    plotted["baseline_always_prohibited"] = round(float(base), 4)
    plotted["models"] = [model_label(m) for m in models]
    if S:
        plotted["c_composition"] = S["composition"]
    json.dump(plotted, open(ROOT / "results/fig2_values.json", "w"), indent=1)


def design_figure():
    """Fig. 1: study design. Every box is sized from its text; a check fails the build if any text comes within
    PAD of a box edge, so labels never touch the lines."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42})
    n_main = sum(1 for _ in open(ROOT / "benchmark/items.jsonl"))
    sc = ROOT / "benchmark/items_status_change.jsonl"
    n_sc = sum(1 for _ in open(sc)) if sc.exists() else 0
    W, H = 4.8, 1.80
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    INK, EDGE, FILL = "#222222", "#555555", "#F5F5F5"
    tint = {"C0": "#FCEFD6", "C1": "#D9EAF5", "C2": "#F9E1D6", "CH": "#F9E1D6"}
    PAD = 0.045  # minimum clearance (inches) between any text and its box edge
    boxes = []   # (x, y, w, h, [text artists])

    def box(x, y, w, h, title, lines=(), fc=FILL, ec=EDGE, title_size=7.2, body_size=6.4, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045", fc=fc, ec=ec, lw=lw,
                                    zorder=2))
        arts = []
        n = len(lines)
        line_h = body_size / 72 * 1.28
        block = title_size / 72 * 1.25 + (0.035 + n * line_h if n else 0)
        top = y + h / 2 + block / 2
        arts.append(ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_size, fontweight="bold",
                            color=INK, zorder=3))
        yy = top - title_size / 72 * 1.25 - 0.035
        for ln in lines:
            arts.append(ax.text(x + w / 2, yy, ln, ha="center", va="top", fontsize=body_size, color=INK, zorder=3))
            yy -= line_h
        boxes.append((x, y, w, h, arts))

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.75, color="#333333",
                                     shrinkA=0, shrinkB=0, zorder=1))

    gap = 0.13
    bw = (W - 0.04 - 3 * gap) / 4
    xs = [0.02 + i * (bw + gap) for i in range(4)]
    # row 1: sources and labels
    r1y, r1h = 1.14, 0.62
    box(xs[0], r1y, bw, r1h, "WADA Lists (PDF)", ["2026 and 2027;", "2021 and 2023 for the", "status-change arm"])
    box(xs[1], r1y, bw, r1h, "Rule tables", ["deterministic parser", "+ LLM extraction,", "reconciled"])
    box(xs[2], r1y, bw, r1h, "Benchmark", [f"{n_main} questions (A to D),", f"{n_sc} status-change (H);", "rule-derived labels"])
    box(xs[3], r1y, bw, r1h, "List changes", ["2026 to 2027 delta,", "checked against the", "WADA Summary"])
    # row 2: answering and scoring
    r2y, r2h = 0.03, 0.87
    box(xs[0], r2y, bw, r2h, "Clause index", ["one chunk per clause;", "BM25 + bge-m3,", "fused by RRF, top 6"])
    conds = [("C0", "C0  Closed-book"), ("C1", "C1  RAG, 2027 List"), ("C2", "C2  RAG, 2026 List"),
             ("CH", "CH  RAG, older List")]
    ch, cg = 0.18, (r2h - 4 * 0.18) / 3
    cy = {}
    for i, (key, label) in enumerate(conds):
        y = r2y + r2h - (i + 1) * ch - i * cg
        cy[key] = y + ch / 2
        box(xs[1], y, bw, ch, label, (), fc=tint[key], ec=COL.get(key, COL["C2"]), title_size=6.2, lw=0.8)
    box(xs[2], r2y, bw, r2h, "4 local LLMs", ["Qwen3-30B-A3B", "Qwen3-14B", "Gemma3-12B", "Llama3.1-8B"])
    box(xs[3], r2y, bw, r2h, "Automatic scoring", ["false permits (primary),", "safety-verdict accuracy", "harm and deferral"])
    # arrows, all in the white gaps between boxes
    mid1 = r1y + r1h / 2
    arrow((xs[0] + bw, mid1), (xs[1], mid1))
    arrow((xs[1] + bw, mid1), (xs[2], mid1))
    arrow((xs[3], mid1), (xs[2] + bw, mid1))
    arrow((xs[0] + bw / 2, r1y), (xs[0] + bw / 2, r2y + r2h))
    arrow((xs[2] + bw / 2, r1y), (xs[2] + bw / 2, r2y + r2h))
    for key in ("C1", "C2", "CH"):
        arrow((xs[0] + bw, cy[key]), (xs[1], cy[key]))
    for key in ("C0", "C1", "C2", "CH"):
        arrow((xs[1] + bw, cy[key]), (xs[2], cy[key]))
    arrow((xs[2] + bw, r2y + r2h / 2), (xs[3], r2y + r2h / 2))

    # check: every label sits inside its box with at least PAD clearance on all sides
    fig.canvas.draw()
    inv = ax.transData.inverted()
    renderer = fig.canvas.get_renderer()
    for x, y, w, h, arts in boxes:
        for a in arts:
            bb = a.get_window_extent(renderer)
            (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
            if not (x0 >= x + PAD and x1 <= x + w - PAD and y0 >= y + PAD * 0.8 and y1 <= y + h - PAD * 0.8):
                raise ValueError(f"Fig. 1 label '{a.get_text()}' is too close to its box edge")
    fig.savefig(FIG / "fig1_design.pdf")
    plt.close(fig)


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def tables(items, M, models):
    df = pd.DataFrame([{"group": i["group"], "verdict": i["gold"]["verdict"], "brand": i["is_brand"]} for i in items])
    gname = {"A": "A & Fixed status", "B": "B & Context-dependent", "C": "C & Revision", "D": "D & Not settleable"}
    vcols = [("prohibited_at_all_times", "AT"), ("prohibited_in_competition_only", "IC"),
             ("prohibited_in_particular_sports", "PS"), ("permitted", "P"), ("permitted_with_conditions", "PC"),
             ("cannot_determine_refer", "R")]
    lines = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{6}{c}{Gold verdict} & \\", r"\cmidrule(lr){4-9}",
             r"\multicolumn{2}{@{}l}{Group} & $n$ & " + " & ".join(a for _, a in vcols) + r" & Brand \\", r"\midrule"]
    for g in "ABCD":
        x = df[df.group == g]
        lines.append(f"{gname[g]} & {len(x)} & " + " & ".join(str(int((x.verdict == v).sum())) for v, _ in vcols)
                     + f" & {int(x.brand.sum())} " + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{@{}l}{Total}" + f" & {len(df)} & " + " & ".join(str(int((df.verdict == v).sum())) for v, _ in vcols)
                 + f" & {int(df.brand.sum())} " + r"\\")
    sc_path = ROOT / "benchmark/items_status_change.jsonl"
    if sc_path.exists():
        sc = pd.DataFrame([{"verdict": i["gold"]["verdict"], "brand": i["is_brand"]} for i in map(json.loads, open(sc_path))])
        lines.append(r"\midrule")
        lines.append(r"H & Status change (arm) & " + f"{len(sc)} & " + " & ".join(str(int((sc.verdict == v).sum())) for v, _ in vcols)
                     + f" & {int(sc.brand.sum())} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_benchmark.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{@{}llrrrcrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Accuracy} & \multicolumn{3}{c}{Permissive error} & & \multicolumn{2}{c}{Deferral} & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){9-10}",
             r"Model & Cond. & 6-way & Safety & FP & 95\% CI & LP & FPr & D & A--C & Harm \\", r"\midrule"]
    for k, m in enumerate(models):
        rows = [(c, M["per"].get(f"{m}|{c}")) for c in ("C0", "C1", "C2")]
        rows = [(c, r) for c, r in rows if r]
        best = min((float(pct(r["false_permit"])) for _, r in rows))
        best_acs = max((float(pct(r["acc_collapsed"])) for _, r in rows))
        if k:
            lines.append(r"\addlinespace")
        for c, r in rows:
            ci = r.get("false_permit_ci", (np.nan, np.nan))
            fp = pct(r["false_permit"])
            fp = rf"\textbf{{{fp}}}" if float(fp) == best else fp
            t01 = M["tests"].get(f"{m}|C0vC1", {}).get("p_holm", 1)
            if c == "C1" and t01 < 0.05 and M["subst_tests"].get(f"{m}|fp_C0vC1", {}).get("p_holm", 1) < 0.05:
                fp += r"$^{a}$"
            acs = pct(r["acc_collapsed"])
            acs = rf"\textbf{{{acs}}}" if float(acs) == best_acs else acs
            if c == "C1":
                marks = [s for s, key in (("a", "accs_C0vC1"), ("b", "accs_C1vC2"))
                         if M["tests"].get(f"{m}|{key}", {}).get("p_holm", 1) < 0.05
                         and M["subst_tests"].get(f"{m}|{key}", {}).get("p_holm", 1) < 0.05]
                if marks:
                    acs += r"$^{" + ",".join(marks) + "}$"
            lines.append(f"{model_label(m) if c == 'C0' else ''} & {c} & {pct(r['acc6'])} & {acs} & {fp} & "
                         f"{pct(ci[0])}--{pct(ci[1])} & {pct(r['lenient_permit'])} & {pct(r['false_prohibit'])} & "
                         f"{pct(r['appropriate_deferral'])} & {pct(r['unnecessary_deferral'])} & "
                         f"{r['harm']:.2f} " + r"\\")
    lines.append(r"\midrule")
    for v, name in (("prohibited_at_all_times", "Always ``prohibited''"), ("cannot_determine_refer", "Always ``refer''")):
        b = M["baselines"][v]
        lp_ = "0.0"
        fpr = "100.0" if v.startswith("prohibited") else "0.0"
        dD = "0.0" if v.startswith("prohibited") else "100.0"
        dac = "0.0" if v.startswith("prohibited") else "100.0"
        lines.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{name}}}}} & {pct(b['acc6'])} & {pct(b['acc_collapsed'])} & 0.0 & & "
                     f"{lp_} & {fpr} & {dD} & {dac} & {b['harm']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_main.tex").write_text("\n".join(lines) + "\n")
    # harm taxonomy (contribution 1), classes grouped by base weight
    tiers = [("false_permit", "False permit: a prohibited item answered ``permitted''"),
             ("conditional_permit", "Conditional permit on a prohibited item; scope understated (e.g.\\ in-competition "
              "for an at-all-times substance); missed condition; unsafe assurance on an unsettleable item"),
             ("false_prohibit", "False prohibit: a permitted item answered prohibited"),
             ("unnecessary_deferral", "Unnecessary deferral; over-caution on an unsettleable item; scope overstated")]
    W = WEIGHT_SETS
    lines = [r"\begin{tabular}{@{}p{0.62\textwidth}rrrr@{}}", r"\toprule",
             r"Error classes & Base & FP\,5 & FP\,20 & Flat \\", r"\midrule"]
    for key, text in tiers:
        lines.append(f"{text} & {W['base'][key]} & {W['fp5'][key]} & {W['fp20'][key]} & {W['flat'][key]} " + r"\\")
    lines += [r"\midrule", r"Kendall $\tau$ of the model--condition ranking against the base weights & 1.00 & "
              + " & ".join(f"{M['sensitivity'][w]['kendall_tau_vs_base']:.2f}" for w in ("fp5", "fp20", "flat")) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_harm.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{@{}lrll@{}}", r"\toprule", r"Weight set & $\tau$ vs base & Best condition (per model) & Best model (C1) \\",
             r"\midrule"]
    desc = {"base": "Base (FP 10)", "fp5": "FP 5", "fp20": "FP 20", "flat": "All errors 1", "mc2_fp20": "FP 20, MC 2, CP 10"}
    for ws, s in M["sensitivity"].items():
        short = {"qwen3:30b-a3b": "Q30", "qwen3:14b": "Q14", "gemma3:12b": "G12", "llama3.1:8b": "L8"}
        bc = ", ".join(f"{short.get(m, m)} {c}" for m, c in s["best_condition_per_model"].items())
        lines.append(f"{desc.get(ws, ws)} & {s['kendall_tau_vs_base']:.2f} & {bc} & {model_label(s['best_model_per_condition'].get('C1', ''))} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")


def macros(M, models, items):
    out = []

    seen = set()

    def mac(name, val):
        words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
                 "7": "Seven", "8": "Eight", "9": "Nine"}
        if "_" in name:  # test names such as nndefer_C1vC2: spell digits so names stay distinct
            name = "".join(words.get(ch, ch) for ch in name)
        name = "".join(ch for ch in name if ch in string.ascii_letters)
        if name in seen:
            raise ValueError(f"duplicate macro {name}")
        seen.add(name)
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    mac("NItems", M["n_items"])
    for g in "ABCD":
        mac(f"NGroup{g}", sum(1 for i in items if i["group"] == g))
    mac("NChanged", sum(1 for i in items if i.get("label_changed")))
    tag = {"qwen3:30b-a3b": "QwenThirty", "qwen3:14b": "QwenFourteen", "gemma3:12b": "Gemma", "llama3.1:8b": "Llama"}
    for m in models:
        t = tag.get(m, "".join(ch for ch in m if ch.isalpha()).title())
        for c in ("C0", "C1", "C2"):
            r = M["per"].get(f"{m}|{c}")
            if not r:
                continue
            cc = {"C0": "Zero", "C1": "One", "C2": "Two"}[c]
            for k in ("acc6", "acc_collapsed", "false_permit", "lenient_permit", "false_prohibit", "appropriate_deferral",
                      "unnecessary_deferral", "cond_complete", "cite_ok", "recall_exact", "recall_section", "nn_defer",
                      "nn_fp", "lenient_no_tue", "permit_on_permitted", "defer_proh"):
                key = "".join(w.title() for w in k.split("_"))
                mac(f"{t}{cc}{key}", pct(r.get(k)))
            mac(f"{t}{cc}Harm", f"{r['harm']:.2f}")
            if "lenient_permit_ci" in r:
                mac(f"{t}{cc}LenientLo", pct(r["lenient_permit_ci"][0]))
                mac(f"{t}{cc}LenientHi", pct(r["lenient_permit_ci"][1]))
            if "harm_ci" in r:
                mac(f"{t}{cc}HarmLo", f"{r['harm_ci'][0]:.2f}")
                mac(f"{t}{cc}HarmHi", f"{r['harm_ci'][1]:.2f}")
            if "false_permit_ci" in r:
                mac(f"{t}{cc}FalsePermitLo", pct(r["false_permit_ci"][0]))
                mac(f"{t}{cc}FalsePermitHi", pct(r["false_permit_ci"][1]))
        for name in ("C0vC1", "C1vC2", "C0vC2"):
            r = M["tests"].get(f"{m}|{name}")
            if r:
                mac(f"{t}{name.replace('v', 'vs').replace('0', 'Zero').replace('1', 'One').replace('2', 'Two')}P", f"{r['p_holm']:.3f}")
                mac(f"{t}{name.replace('v', 'vs').replace('0', 'Zero').replace('1', 'One').replace('2', 'Two')}RD", pct(r["risk_difference"]))
    for name in ("nndefer_C1vC2", "nndefer_C0vC1", "nnfp_C1vC2", "defer_C0vC1", "lp_C0vC1", "lpnotue_C0vC1",
                 "accs_C0vC1", "accs_C1vC2", "nnaccs_C1vC2", "pp_C0vC1"):
        for m in models:
            r = M["tests"].get(f"{m}|{name}")
            if r:
                t = tag.get(m)
                mac(f"{t}_{name}P", f"{r['p_holm']:.3f}")
    for m in models:
        t = tag.get(m)
        for k, v in M.get("harm_diff", {}).items():
            if k.startswith(m + "|"):
                nm = k.split("|")[1].replace("C0", "Zero").replace("C1", "One").replace("C2", "Two").replace("-", "vs")
                mac(f"{t}HarmDiff{nm}", f"{v['diff']:.2f}")
                mac(f"{t}HarmDiff{nm}Lo", f"{v['ci'][0]:.2f}")
                mac(f"{t}HarmDiff{nm}Hi", f"{v['ci'][1]:.2f}")
        h = M["h2"].get(m, {})
        if "did" in h:
            mac(f"{t}DiD", pct(h["did"]))
            mac(f"{t}DiDLo", pct(h["ci"][0]))
            mac(f"{t}DiDHi", pct(h["ci"][1]))
            mac(f"{t}AdvA", pct(h["adv_A"]))
            mac(f"{t}AdvB", pct(h["adv_B"]))
    for m in models:
        t = tag.get(m)
        for k, v in M["fp_rd"].items():
            if k.startswith(m + "|"):
                nm = k.split("|")[1].replace("C0", "Zero").replace("C1", "One").replace("C2", "Two").replace("-", "vs")
                mac(f"{t}FpRD{nm}", pct(v["rd"]))
                mac(f"{t}FpRD{nm}Lo", pct(v["ci"][0]))
                mac(f"{t}FpRD{nm}Hi", pct(v["ci"][1]))
        s = M["nn_sign"].get(m)
        if s:
            mac(f"{t}NnSignUp", s["higher_with_2026"])
            mac(f"{t}NnSignN", s["higher_with_2026"] + s["lower_with_2026"])
            mac(f"{t}NnSignP", f"{s['p_holm']:.3f}")
        for c in ("C1", "C2"):
            r = M["per"].get(f"{m}|{c}")
            nnacc = M["tests"].get(f"{m}|nnaccs_C1vC2")
            if nnacc:
                mac(f"{t}NnAcc{ {'C1': 'One', 'C2': 'Two'}[c] }", pct(nnacc["rate_first" if c == "C1" else "rate_second"]))
        r = M["per"].get(f"{m}|C1")
        if r and "acc_collapsed_ci" in r:
            for c in ("C0", "C1", "C2"):
                rr = M["per"][f"{m}|{c}"]
                cc = {"C0": "Zero", "C1": "One", "C2": "Two"}[c]
                mac(f"{t}{cc}AccSafeLo", pct(rr["acc_collapsed_ci"][0]))
                mac(f"{t}{cc}AccSafeHi", pct(rr["acc_collapsed_ci"][1]))
    for c in ("C1", "C2"):
        for hm in ("hit", "miss"):
            v = M["nn_defer_by_retrieval"].get(f"{c}|{hm}")
            if v:
                mac(f"NnDefer{ {'C1': 'One', 'C2': 'Two'}[c] }{hm.title()}", pct(v["defer"]))
    ps = [M["tests"][f"{m}|accs_C1vC2"]["p_holm"] for m in models if f"{m}|accs_C1vC2" in M["tests"]]
    mac("AccsOneTwoPMax", f"{max(ps):.3f}")
    ps = [M["tests"][f"{m}|accs_C0vC1"]["p_holm"] for m in models[:2] if f"{m}|accs_C0vC1" in M["tests"]]
    mac("AccsZeroOneQwenPMax", f"{max(ps):.3f}")
    prev = ROOT / "results/metrics_before_label_corrections.json"  # archived metrics before the round-3 label corrections
    if prev.exists():
        P = json.load(open(prev))["per"]
        mac("MaxFpShift", f"{100 * max(abs(M['per'][k]['false_permit'] - P[k]['false_permit']) for k in M['per'] if k in P):.1f}")
    ar = M["accs_rd"]
    mac("AccsRdLoMin", pct(min(v["ci"][0] for v in ar.values())))
    mac("AccsRdNotNnMax", pct(max(abs(v["rd_not_newly_named"]) for v in ar.values())))
    mac("MaxAccSafe", pct(max(v["acc_collapsed"] for v in M["per"].values())))
    if prev.exists():
        mac("MaxAccShift", f"{100 * max(abs(M['per'][k]['acc6'] - P[k]['acc6']) for k in M['per'] if k in P):.1f}")
    nps = [M["nn_sign"][m]["p_holm"] for m in models if m in M["nn_sign"]]
    mac("NnSignPMax", f"{max(nps):.3f}")
    mac("NnSignPMin", f"{min(nps):.3f}")
    S = M.get("statuschange")
    if S:
        for m in models:
            t = tag.get(m)
            for c in ("C0", "C1", "CH"):
                r = S["per"].get(f"{m}|{c}")
                if r:
                    cc = {"C0": "Zero", "C1": "One", "CH": "Hist"}[c]
                    mac(f"{t}Sc{cc}Fp", pct(r["fp"]))
                    mac(f"{t}Sc{cc}Lp", pct(r["lp"]))
                    mac(f"{t}Sc{cc}CtlAcc", pct(r["control_acc"]))
                    if r["recall_section"] is not None:
                        mac(f"{t}Sc{cc}Recall", pct(r["recall_section"]))
            for name in ("C1vCH", "C0vC1"):
                r = S["tests"].get(f"{m}|{name}")
                if r:
                    nm = {"C1vCH": "OneHist", "C0vC1": "ZeroOne"}[name]
                    mac(f"{t}ScP{nm}", f"{r['p_holm']:.3f}")
                    mac(f"{t}ScDisc{nm}", f"{r['only_first']}/{r['only_second']}")
        for c in ("C0", "C1", "CH"):
            mac(f"ScPooled{ {'C0': 'Zero', 'C1': 'One', 'CH': 'Hist'}[c] }Fp", pct(S["pooled_fp"][c]))
        for k, v in S["family"].items():
            c, y = k.split("|")
            nm = f"ScFam{ {'C0': 'Zero', 'C1': 'One', 'CH': 'Hist'}[c] }{ {'2021': 'Gc', '2023': 'Tram', 'hit': 'Hit', 'miss': 'Miss'}[y] }"
            for kk, lab_ in (("fp", "Fp"), ("defer", "Defer"), ("acc", "Acc")):
                mac(nm + lab_, pct(v[kk]))
        mac("ScPooledHistDefer", pct(float(np.mean([S["per"][f"{m}|CH"]["defer"] for m in models]))))
        mac("ScPooledOneDefer", pct(float(np.mean([S["per"][f"{m}|C1"]["defer"] for m in models]))))
        mac("ScNChanged", S["per"][f"{models[0]}|C1"]["n"])
    for k, v in M.get("subst_tests", {}).items():
        m, fam_name = k.split("|")
        nm = {"accs_C1vC2": "SubAccsOneTwo", "accs_C0vC1": "SubAccsZeroOne", "fp_C0vC1": "SubFpZeroOne"}[fam_name]
        mac(f"{tag.get(m)}{nm}P", f"{v['p_holm']:.3f}")
        mac(f"{tag.get(m)}{nm}Count", f"{v['first_higher']} vs {v['second_higher']}")
    mx = M.get("mixed", {})
    for k, v in mx.get("coef", {}).items():
        if k == "C(condition, Treatment('C0'))[T.C1]:C(group)[T.B]":
            mac("MixedOneB", f"{v['mean']:.2f}")
            mac("MixedOneBSD", f"{v['sd']:.2f}")
    S = M.get("statuschange")
    if S:
        for k, v in S["composition"].items():
            c, f_ = k.split("|")
            nm = f"ScComp{ {'C0': 'Zero', 'C1': 'One', 'CH': 'Hist'}[c] }{ {'gc': 'Gc', 'tram': 'Tram'}[f_] }"
            for kk in ("permit", "prohibited_ic", "prohibited_other", "refer", "n"):
                mac(nm + "".join(w.title() for w in kk.split("_")), v[kk])
            if v["recall"] is not None:
                mac(nm + "Recall", pct(v["recall"]))
        for m, v in S["c0_permit_by_model"].items():
            mac(f"{tag.get(m)}ScZeroPermitN", v)
    for ws in ("fp5", "fp20", "flat", "mc2_fp20"):
        nm = {"fp5": "FpFive", "fp20": "FpTwenty", "flat": "Flat", "mc2_fp20": "Alt"}[ws]
        mac(f"Tau{nm}", f"{M['sensitivity'][ws]['kendall_tau_vs_base']:.2f}")
    dg = M["diag"]
    for k, v in dg["fp_by_retrieval"].items():
        m, c, hm = k.split("|")
        mac(f"{tag.get(m)}{ {'C1': 'One', 'C2': 'Two'}[c] }Fp{hm.title()}", pct(v["fp"]))
        mac(f"{tag.get(m)}{ {'C1': 'One', 'C2': 'Two'}[c] }N{hm.title()}", v["n"])
    for k, v in dg["fp_by_brand"].items():
        c, b = k.split("|")
        mac(f"Fp{ {'C0': 'Zero', 'C1': 'One', 'C2': 'Two'}[c] }{b.title()}", pct(v["fp"]))
        mac(f"Lenient{ {'C0': 'Zero', 'C1': 'One', 'C2': 'Two'}[c] }{b.title()}", pct(v["lenient"]))
    for k, v in dg["fp_changed_items"].items():
        m, c = k.split("|")
        cc = {"C0": "Zero", "C1": "One", "C2": "Two"}[c]
        mac(f"{tag.get(m)}{cc}ChangedFp", pct(v["fp"]))
        mac(f"{tag.get(m)}{cc}ChangedLenient", pct(v["lenient"]))
        mac(f"{tag.get(m)}{cc}ChangedDefer", pct(v["defer"]))
    for m, v in M["diag"]["date_mentions"].items():
        t = tag.get(m)
        mac(f"{t}DateMention", v["mention_2026"])
        mac(f"{t}DateMentionRefer", v["mention_and_refer"])
        mac(f"{t}IntroRetrieved", v["intro_retrieved"])
        mac(f"{t}TwoN", v["n"])
    pf = {m: sum(M["per"][f"{m}|{c}"]["parse_fail"] * M["per"][f"{m}|{c}"]["n"] for c in ("C0", "C1", "C2")
                 if f"{m}|{c}" in M["per"]) for m in models}
    mac("NParseFail", int(round(sum(pf.values()))))
    mac("NMainCalls", sum(M["per"][k]["n"] for k in M["per"]))
    mx = M.get("mixed", {})
    if mx.get("converged"):
        for k, v in mx["coef"].items():
            if k.startswith("C(condition") and ":" not in k:
                cc = "One" if "C1" in k else "Two"
                mac(f"Mixed{cc}", f"{v['mean']:.2f}")
                mac(f"Mixed{cc}SD", f"{v['sd']:.2f}")
            elif k.startswith("C(condition") and ":C(group)" in k:
                cc = "One" if "[T.C1]" in k else "Two"
                g = k[-2]
                mac(f"Mixed{cc}Group{g}", f"{v['mean']:.2f}")
                mac(f"Mixed{cc}Group{g}SD", f"{v['sd']:.2f}")
    mac("SensitivityRankChange", "yes" if M.get("sensitivity_rank_change") else "no")
    for k, v in M.get("consistency", {}).items():
        m, c = k.split("|")
        cc_ = {"C0": "Zero", "C1": "One"}[c]
        mac(f"{tag.get(m)}{cc_}ConsAgree", pct(v["all_repeats_agree"]))
        mac(f"{tag.get(m)}{cc_}ConsAnyFp", pct(v["any_repeat_false_permit"]))
    for k, v in M.get("ablation", {}).items():
        m, t = k.split("|")
        kk = {"k3": "KThree", "k10": "KTen"}[t]
        mac(f"{tag.get(m)}{kk}Fp", pct(v["false_permit"]))
        mac(f"{tag.get(m)}{kk}Recall", pct(v["recall_exact"]))
        mac(f"{tag.get(m)}{kk}Harm", f"{v['harm']:.2f}")
    cc = ROOT / "results/condition_check_agreement.json"
    if cc.exists():
        c = json.load(open(cc))
        mac("CondCheckN", c["n"])
        mac("CondCheckAgree", pct(c["agreement"]))
        mac("CondCheckKappa", f"{c['cohen_kappa']:.2f}")
        mac("CondCheckOver", c["auto_yes_manual_no"])
        mac("CondCheckUnder", c["auto_no_manual_yes"])
    bl = M["baselines"]
    mac("BaseReferHarm", f"{bl['cannot_determine_refer']['harm']:.2f}")
    mac("BaseProhHarm", f"{bl['prohibited_at_all_times']['harm']:.2f}")
    mac("BasePermHarm", f"{bl['permitted']['harm']:.2f}")
    mac("BaseProhAccSafe", pct(bl["prohibited_at_all_times"]["acc_collapsed"]))
    mac("BaseProhAccSix", pct(bl["prohibited_at_all_times"]["acc6"]))
    mac("BaseReferAccSafe", pct(bl["cannot_determine_refer"]["acc_collapsed"]))
    mac("NNewlyNamed", sum(1 for i in items if i.get("newly_named")))
    sc_path = ROOT / "benchmark/items_status_change.jsonl"
    if sc_path.exists():
        mac("NStatusItems", sum(1 for _ in open(sc_path)))
    mac("NAcceptAlt", sum(1 for i in items if "accept" in i["gold"]))
    mac("MinModelHarm", f"{min(v['harm'] for v in M['per'].values()):.2f}")
    # corpus, rule table and benchmark facts used in the Method section (optional inputs: these files exist only
    # after the rule-table step has been run, so a re-analysis of released outputs skips them)
    agp, v1p = ROOT / "results/rule_extraction_agreement.json", ROOT / "results/rule_extraction_agreement_passB_v1.json"
    if agp.exists():
        ag = json.load(open(agp))
        mac("AgreePooled", f"{100 * ag['pooled_row_level_agreement']:.1f}")
        for y in ("2026", "2027"):
            w = {"2026": "TwentySix", "2027": "TwentySeven"}[y]
            mac(f"Agree{w}", f"{100 * ag[y]['row_level_agreement']:.1f}")
            mac(f"Rows{w}", ag[y]["rows_kept"])
            mac(f"Union{w}", ag[y]["union_rows"])
            mac(f"StatusDisagree{w}", ag[y]["status_disagree"])
    if v1p.exists():
        mac("AgreeVOne", f"{100 * json.load(open(v1p))['pooled_row_level_agreement']:.1f}")
    for y in ("2026", "2027"):
        ch = ROOT / f"data/processed/chunks_{y}.jsonl"
        if ch.exists():
            mac(f"Chunks{ {'2026': 'TwentySix', '2027': 'TwentySeven'}[y] }", sum(1 for _ in open(ch)))
    bm = pd.read_csv(ROOT / "benchmark/brand_map.csv").fillna("")
    mac("BrandCandidates", len(bm))
    mac("BrandKept", int((bm["status"] == "kept").sum()))
    mac("NBrandItems", sum(1 for i in items if i["is_brand"]))
    mac("NConsistency", len(json.load(open(ROOT / "benchmark/subset_consistency.json"))))
    mac("NTemplates", len({i["template_id"] for i in items}))
    rv = ROOT / "benchmark/items_for_review.xlsx"
    if rv.exists():
        mac("NReviewSample", len(pd.read_excel(rv)))
    dp = ROOT / "data/processed/list_delta_2026_2027.csv"
    if dp.exists():
        delta = pd.read_csv(dp).fillna("")
        mac("DeltaAdded", int((delta["change_type"] == "added").sum()))
        mac("DeltaClauses", int((delta["change_type"] == "clause_text_changed").sum()))
    for g in "ABCD":
        mac(f"NProh{g}", sum(1 for i in items if i["group"] == g and i["gold"]["verdict"] in PROH))
    mac("NProhibited", sum(1 for i in items if i["gold"]["verdict"] in PROH))
    mac("NReferD", sum(1 for i in items if i["group"] == "D" and i["gold"]["verdict"] == "cannot_determine_refer"))
    rec = {}
    for c in ("C1", "C2"):
        x = [M["per"][f"{m}|{c}"].get("recall_exact") for m in models if f"{m}|{c}" in M["per"]]
        x = [v for v in x if v is not None and not np.isnan(v)]
        if x:
            mac(f"Recall{ {'C1': 'One', 'C2': 'Two'}[c] }", pct(float(np.mean(x))))
    (GEN / "numbers.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
