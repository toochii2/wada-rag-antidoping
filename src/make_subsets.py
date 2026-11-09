"""25% stratified (group x gold verdict) subset of items for the consistency run."""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "benchmark/subset_consistency.json"
if out.exists():  # the subset is fixed once runs have used it
    print(f"consistency subset kept: {len(json.load(open(out)))} items")
    raise SystemExit
items = [json.loads(l) for l in open(ROOT / "benchmark/items.jsonl")]
strata = defaultdict(list)
for i in items:
    strata[(i["group"], i["gold"]["verdict"])].append(i["item_id"])
rng = random.Random(25)
pick = []
for k in sorted(strata):
    ids = sorted(strata[k])
    rng.shuffle(ids)
    pick += ids[:max(1, round(0.25 * len(ids)))]
json.dump(sorted(pick), open(out, "w"), indent=0)
print(f"consistency subset: {len(pick)} items")
