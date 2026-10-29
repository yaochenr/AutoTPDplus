"""Publish the latest release to the AutoTPD+ site and rebuild its summary statistics.

Copies the release CSVs from data/release into docs/data (the files the site serves
for download), then writes docs/assets/stats.json for the Overview panel and the
Download dialog. Re-run after each monthly update:

    pixi run python scripts/build_site_stats.py
"""
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data" / "release"
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "assets" / "stats.json"

MODALITIES = ("PROTAC", "MG")
FILES = [f"{name}-{m}.csv" for name in ("AutoTPDplus", "AutoTPD") for m in MODALITIES]
TOP_N = 8
# Field-coverage chart: LLM-extracted rows vs rows inherited from the source database.
COVERAGE_FIELDS = {
    "Assay": "Assay",
    "Cell line": "Cell_Line",
    "DC50": "DC50",
    "DC50 time": "DC50_h",
    "Dmax": "Dmax",
    "Dmax time": "Dmax_h",
    "Dmax conc.": "Dmax_conc",
}


def top_counts(series, n=TOP_N):
    counts = series.dropna().value_counts()
    return [{"name": k, "count": int(v)} for k, v in counts.head(n).items()]


def pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def summarize(modality):
    plus = pd.read_csv(DATA / f"AutoTPDplus-{modality}.csv", low_memory=False)
    auto = pd.read_csv(DATA / f"AutoTPD-{modality}.csv", low_memory=False)

    # Source-database records carry their original Record_ID; LLM-extracted ones do not.
    extracted = plus[plus["Record_ID"].isna()]
    database = plus[plus["Record_ID"].notna()]
    # Mouse orthologues (Crbn, Vhl) count as the same recruiter.
    recruiters = plus["Recruiter_Gene"].str.upper()

    return {
        "records": len(plus),
        "publications": int(plus["DOI"].nunique()),
        "structures": int(plus["StdInChIKey"].nunique()),
        "targets": int(plus["Degradation_Target_Gene"].nunique()),
        "recruiters": int(recruiters.nunique()),
        "cell_lines": int(plus["Cell_Line_STD"].nunique()),
        "smiles_pct": pct(plus["SMILES"].notna().sum(), len(plus)),
        "autotpd": {"records": len(auto), "publications": int(auto["DOI"].nunique())},
        "extracted_records": len(extracted),
        "extracted_publications": int(extracted["DOI"].nunique()),
        "database_records": len(database),
        "database_publications": int(database["DOI"].nunique()),
        "top_targets": top_counts(plus["Degradation_Target_Gene"]),
        "top_recruiters": top_counts(recruiters, 5),
        "top_assays": top_counts(plus["Assay_STD"], 6),
        "coverage": [
            {
                "field": label,
                "extracted": pct(extracted[col].notna().sum(), len(extracted)),
                "database": pct(database[col].notna().sum(), len(database)),
            }
            for label, col in COVERAGE_FIELDS.items()
        ],
    }


def totals():
    frames = [pd.read_csv(DATA / f"AutoTPDplus-{m}.csv", low_memory=False) for m in MODALITIES]
    both = pd.concat(frames, ignore_index=True)
    return {
        "records": len(both),
        "publications": int(both["DOI"].str.lower().nunique()),
        "structures": int(both["StdInChIKey"].nunique()),
    }


def main():
    for name in FILES:
        shutil.copy2(RELEASE / name, DATA / name)

    stats = {m: summarize(m) for m in MODALITIES}
    stats["total"] = totals()
    stats["updated"] = date.today().isoformat()
    stats["file_bytes"] = {name: (DATA / name).stat().st_size for name in FILES}
    OUT.write_text(json.dumps(stats, indent=1) + "\n")
    for m in MODALITIES:
        print(f"{m}: {stats[m]['records']} records from {stats[m]['publications']} publications")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
