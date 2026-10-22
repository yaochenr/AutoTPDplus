#!/usr/bin/env python















import os
import sys
import json
import shutil
import contextlib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd
import tpd_curator.pipeline as P
from tpd_curator.pipeline import run_evaluation

COHORTS = {
    "MG": dict(
        pred="data/evaluation/molecular_glues/llm/glue_data_llm.csv",
        gt="data/evaluation/molecular_glues/ground_truth/GT_Glues.csv",
        cache="semantic_cache/llm_gt_match_glue.json",
    ),
    "PROTAC": dict(
        pred="data/evaluation/protacs/llm/protac-data-llm.csv",
        gt="data/evaluation/protacs/ground_truth/PROTAC-GT.csv",
        cache="semantic_cache/llm_gt_match_protacs.json",
    ),
}
MODES = {"Experimental": False, "Mechanistic": True}
OUT = "output/results/exact_only_ablation"

_ORIG = P.check_semantic_match


def _exact_only(pred_value, labeled_value, feature=None, **kwargs):

    if pd.isna(pred_value) or pd.isna(labeled_value):
        return False
    return str(pred_value).strip().lower() == str(labeled_value).strip().lower()


def run_one(cohort, mode_name, ternary, variant, outdir):
    os.makedirs(outdir, exist_ok=True)
    c = COHORTS[cohort]
    if variant == "semantic":
        cache = os.path.join(outdir, "cache_copy.json")
        shutil.copy(c["cache"], cache)
        P.check_semantic_match = _ORIG
    else:
        cache = os.path.join(outdir, "unused.json")
        P.check_semantic_match = _exact_only
    config = {
        "data": {},
        "results": {"results_dir": outdir, "current_run_dir": outdir},
        "evaluation": {"cache_file_path": cache},
    }
    logpath = os.path.join(outdir, "run.log")
    try:
        with open(logpath, "w") as lg, contextlib.redirect_stdout(lg):
            res = run_evaluation(
                config_file=config,
                pred_csv_path=c["pred"],
                labeled_csv_path=c["gt"],
                ternary_complex_level=ternary,
                output_dir=outdir,
            )
    finally:
        P.check_semantic_match = _ORIG
    r = res["part1_record_level"]
    f = res["part2_entry_level"]
    return {
        "record": [r["precision_records_micro"], r["recall_records_micro"], r["f1_score_records_micro"]],
        "field": [f["precision_entry_micro"], f["recall_entry_micro"], f["f1_score_entry_micro"]],
        "record_counts": [r["total_tp"], r["total_fp"], r["total_fn"]],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for cohort in COHORTS:
        for mode_name, ternary in MODES.items():
            for variant in ("semantic", "exact"):
                key = f"{cohort}/{mode_name}/{variant}"
                outdir = os.path.join(OUT, f"{cohort}_{mode_name}_{variant}")
                print(f"running {key} ...", flush=True)
                results[key] = run_one(cohort, mode_name, ternary, variant, outdir)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


    def fmt(t):
        return f"P={t[0]:.2f} R={t[1]:.2f} F1={t[2]:.3f}"

    print("\n" + "=" * 78)
    print("record/field F1 — WITH semantic matching vs EXACT-only")
    print("=" * 78)
    lines = ["| Cohort | Mode | Level | With semantic | Exact-only | ΔF1 |",
             "|---|---|---|---|---|---|"]
    for cohort in COHORTS:
        for mode_name in MODES:
            sem = results[f"{cohort}/{mode_name}/semantic"]
            exa = results[f"{cohort}/{mode_name}/exact"]
            for level in ("record", "field"):
                s, e = sem[level], exa[level]
                d = s[2] - e[2]
                row = f"| {cohort} | {mode_name} | {level} | {fmt(s)} | {fmt(e)} | -{d:.3f} |"
                lines.append(row)
                print(f"  {cohort:6} {mode_name:12} {level:6}  with:{fmt(s)}   exact:{fmt(e)}   dF1=-{d:.3f}")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "table.md"), "w") as fh:
        fh.write(md)
    print(f"\nwritten: {OUT}/results.json, {OUT}/table.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
