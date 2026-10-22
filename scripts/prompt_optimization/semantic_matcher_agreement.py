#!/usr/bin/env python

import os
import sys
import json
import csv
import time
import argparse
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from tpd_curator.llm.run_llm_api import run_llm_api

CACHES = {
    "MG": "semantic_cache/llm_gt_match_glue.json",
    "PROTAC": "semantic_cache/llm_gt_match_protacs.json",
}
OUT_DIR = "output/results/semantic_matcher_agreement"


def load_entries(path):
    with open(path) as fh:
        data = json.load(fh)
    rows = []
    for doi, lst in data.items():
        if not isinstance(lst, list):
            continue
        for e in lst:
            pred = e.get("pred_value")
            labeled = e.get("labeled_value")
            if pred is None or labeled is None:
                continue
            rows.append({
                "doi": doi,
                "pred": str(pred).strip(),
                "labeled": str(labeled).strip(),
                "feature": e.get("feature"),
                "final": e.get("result"),
            })
    return rows


def matcher_prompt(pred, labeled, feature):
    if not feature:
        feature = "compound, protein or cell line"
    return f"""
    Please determine whether {pred} and {labeled} refer to the same {feature}.
    Please only answer "yes" or "no". If they are referring to the same thing, respond 'yes'; otherwise, respond 'no', don't add any extra words.
    """


def call_matcher(pred, labeled, feature, retries=3):
    prompt = matcher_prompt(pred, labeled, feature)
    for _ in range(retries):
        try:
            result, _ = run_llm_api(custom_prompt=prompt, paper_text=None)
            if result and isinstance(result, str):
                return "yes" if "yes" in result.lower() else "no"
            time.sleep(5)
        except Exception as exc:
            print(f"    matcher call failed: {exc}; retrying...")
            time.sleep(8)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only replay N unique pairs per cohort (smoke test)")
    ap.add_argument("--consistency-passes", type=int, default=2,
                    help="extra matcher passes to gauge self-consistency "
                         "(applied to disagreements + a random agreement sample)")
    ap.add_argument("--sample-agree", type=int, default=40,
                    help="how many agreeing pairs to also re-run for the "
                         "self-consistency estimate")
    args = ap.parse_args()

    import random
    random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    per_pair_rows = []
    raw = {}

    for cohort, path in CACHES.items():
        print(f"\n==== {cohort}: {path} ====")
        entries = load_entries(path)

        uniq = {}
        for e in entries:
            key = (e["pred"], e["labeled"], e["feature"])
            uniq.setdefault(key, {"final_results": [], "doi": e["doi"]})
            uniq[key]["final_results"].append(e["final"])
        keys = list(uniq.keys())
        if args.limit:
            keys = keys[:args.limit]
        print(f"  {len(entries)} entries -> {len(keys)} unique pairs to replay")

        for i, key in enumerate(keys, 1):
            pred, labeled, feature = key
            auto1 = call_matcher(pred, labeled, feature)
            finals = uniq[key]["final_results"]
            final = Counter(finals).most_common(1)[0][0]
            print(f"  [{cohort} {i}/{len(keys)}] {feature}: "
                  f"{pred[:30]!r} vs {labeled[:30]!r} -> auto={auto1} final={final}")
            per_pair_rows.append({
                "cohort": cohort, "feature": feature,
                "pred": pred, "labeled": labeled,
                "auto1": auto1, "auto_extra": [], "final": final,
                "n_entries": len(finals),
            })

    disagree = [r for r in per_pair_rows if r["auto1"] is not None and r["auto1"] != r["final"]]
    agree = [r for r in per_pair_rows if r["auto1"] is not None and r["auto1"] == r["final"]]
    sample_agree = random.sample(agree, min(args.sample_agree, len(agree)))
    recheck = disagree + sample_agree
    print(f"\n==== self-consistency: {args.consistency_passes} extra pass(es) on "
          f"{len(disagree)} disagreements + {len(sample_agree)} sampled agreements ====")
    for r in recheck:
        extra = []
        for _ in range(args.consistency_passes):
            extra.append(call_matcher(r["pred"], r["labeled"], r["feature"]))
        r["auto_extra"] = extra

    def stable(r):
        calls = [r["auto1"]] + r["auto_extra"]
        calls = [c for c in calls if c is not None]
        return len(set(calls)) <= 1 if calls else None

    lines = ["# Semantic matcher — agreement with expert-curated caches\n"]
    lines.append("Reconstructed by replaying the temperature=0 matcher on the "
                 "stored comparison pairs and diffing against the final "
                 "human-reviewed caches.\n")
    for cohort in CACHES:
        rows = [r for r in per_pair_rows if r["cohort"] == cohort]
        usable = [r for r in rows if r["auto1"] is not None]
        n = len(usable)
        agr = sum(1 for r in usable if r["auto1"] == r["final"])
        dis = n - agr
        n_entries = sum(r["n_entries"] for r in usable)
        entries_agr = sum(r["n_entries"] for r in usable if r["auto1"] == r["final"])
        lines.append(f"\n## {cohort}")
        lines.append(f"- unique pairs replayed: {n}")
        lines.append(f"- matcher agreed with final curated: {agr} "
                     f"({100*agr/n:.1f}%)" if n else "- no pairs")
        lines.append(f"- corrected by expert (auto != final): {dis} "
                     f"({100*dis/n:.1f}%)" if n else "")
        lines.append(f"- weighted by cache entries: {entries_agr}/{n_entries} "
                     f"= {100*entries_agr/n_entries:.1f}% agreement" if n_entries else "")

        checked = [r for r in rows if r["auto_extra"]]
        stab = [r for r in checked if stable(r) is True]
        if checked:
            lines.append(f"- self-consistency (rechecked pairs stable): "
                         f"{len(stab)}/{len(checked)} = {100*len(stab)/len(checked):.1f}%")

        featdis = Counter(r["feature"] for r in usable if r["auto1"] != r["final"])
        if featdis:
            lines.append("- disagreements by feature: "
                         + ", ".join(f"{k}:{v}" for k, v in featdis.most_common()))

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as fh:
        fh.write(summary)

    with open(os.path.join(OUT_DIR, "per_pair.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "feature", "pred", "labeled",
                    "auto1", "auto_extra", "final", "agree", "n_entries"])
        for r in per_pair_rows:
            w.writerow([r["cohort"], r["feature"], r["pred"], r["labeled"],
                        r["auto1"], "|".join(str(x) for x in r["auto_extra"]),
                        r["final"], r["auto1"] == r["final"], r["n_entries"]])

    with open(os.path.join(OUT_DIR, "disagreements.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "feature", "pred", "labeled",
                    "auto1", "auto_extra", "final(curated)", "n_entries"])
        for r in per_pair_rows:
            if r["auto1"] is not None and r["auto1"] != r["final"]:
                w.writerow([r["cohort"], r["feature"], r["pred"], r["labeled"],
                            r["auto1"], "|".join(str(x) for x in r["auto_extra"]),
                            r["final"], r["n_entries"]])

    with open(os.path.join(OUT_DIR, "raw.json"), "w") as fh:
        json.dump(per_pair_rows, fh, indent=2)

    print("\n" + summary)
    print(f"Written to {OUT_DIR}/ (summary.md, per_pair.csv, disagreements.csv, raw.json)")

if __name__ == "__main__":
    main()
