"""Combine the per-year extraction agreement files into results/rule_extraction_agreement.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = {}
for y in (2026, 2027):
    out[str(y)] = json.load(open(ROOT / f"results/rule_extraction_agreement_{y}.json"))
u = sum(v["union_rows"] for v in out.values())
a = sum(v["agree_name_and_status"] for v in out.values())
out["pooled_row_level_agreement"] = round(a / u, 4)
out["hard_stop_below_0.70"] = out["pooled_row_level_agreement"] < 0.70
out["limitation_0.70_to_0.85"] = 0.70 <= out["pooled_row_level_agreement"] < 0.85
json.dump(out, open(ROOT / "results/rule_extraction_agreement.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if not k.isdigit()}))
