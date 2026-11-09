"""Apply pharmacist rule overrides (data/processed/rule_overrides.csv) to rules_<year>.csv.

Usage: python src/apply_overrides.py <year>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    year = int(sys.argv[1])
    path = ROOT / f"data/processed/rules_{year}.csv"
    ov_path = ROOT / "data/processed/rule_overrides.csv"
    if not ov_path.exists():
        print("no overrides")
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    ov = pd.read_csv(ov_path, dtype=str).fillna("")
    ov = ov[ov["year"] == str(year)]
    n = 0
    for o in ov.to_dict("records"):
        m = (df["substance"] == o["substance"]) & (df["clause_ref"] == o["clause_ref"])
        if not m.any():
            print("override target not found:", o)
            continue
        if o["field"] == "drop" and o["value"].lower() in ("yes", "true", "y"):
            df.loc[m, "dropped"] = "True"
        else:
            df.loc[m, o["field"]] = o["value"]
        df.loc[m, "resolution"] = (df.loc[m, "resolution"] + "; pharmacist override").str.strip("; ")
        n += 1
    df.to_csv(path, index=False)
    print(f"{year}: {n} overrides applied")


if __name__ == "__main__":
    main()
