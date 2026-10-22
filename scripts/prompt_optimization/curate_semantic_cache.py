#!/usr/bin/env python3



























import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_gepa_loocv as rg
from tpd_curator.capo import Capo
from tpd_curator import pipeline as _pipeline

METRIC_KEYS = rg.METRIC_KEYS


def evaluate_paper(capo, paper, work_dir, reextract):

    doi = paper["doi"]
    pmc_id = paper.get("pmc_id")
    safe_doi = doi.replace("/", "_").replace(":", "_")
    paper_dir = work_dir / f"paper_{safe_doi}"
    have_pred = (paper_dir / "data_processed.csv").exists()

    if have_pred and not reextract:
        print(f"  [{doi}] reusing saved predictions -> re-evaluating only")
        ev = capo._re_evaluate_paper(paper_dir=paper_dir, doi=doi)
        metrics = capo._extract_metrics_from_evaluation(ev, doi)
    else:
        print(f"  [{doi}] extracting (3-round LLM) + evaluating ...")


        capo.config["data"]["output_dir"] = str(work_dir / f"clean_{safe_doi}")
        try:
            _pipeline.store.clear()
        except Exception:
            pass
        res = capo._extract_paper(doi=doi, pmc_id=pmc_id,
                                  output_dir=work_dir, round_dir=work_dir)
        metrics = res["metrics"]
    return metrics


def load_cache_index(cache_file):

    if not Path(cache_file).exists():
        return {}
    try:
        cache = json.loads(Path(cache_file).read_text())
    except Exception:
        return {}
    idx = {}
    for doi, entries in cache.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            key = (doi, e.get("feature"),
                   str(e.get("pred_value")), str(e.get("labeled_value")))
            idx[key] = str(e.get("result"))
    return idx


def show_cache_changes(before, after):


    added = [(k, after[k]) for k in after if k not in before]
    changed = [(k, before[k], after[k]) for k in after
               if k in before and after[k] != before[k]]

    print("\n" + "=" * 80)
    print("SEMANTIC MATCHES the LLM wrote THIS run (review these; a wrong 'no' "
          "fails the whole record)")
    print("=" * 80)

    if not added and not changed:
        print("\n(none - every comparison hit the existing cache; nothing re-judged)")
        return

    for (doi, feat, pred, lab), res in added:
        print(f"\nDOI {doi}")
        print(f"  [{feat}] result={res!r}")
        print(f"     pred = {pred!r}")
        print(f"     vs gt= {lab!r}")
    for (doi, feat, pred, lab), old, new in changed:
        print(f"\nDOI {doi}  (CHANGED {old!r} -> {new!r})")
        print(f"  [{feat}] pred={pred!r} vs gt={lab!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_path", type=Path, default=rg.DEFAULT_CONFIG)
    p.add_argument("--prompt", type=Path, default=rg.DEFAULT_SEED_PROMPT,
                   help="Prompt to extract with (default: the seed initial_prompt.md).")
    p.add_argument("--gt_csv", type=Path, default=rg.DEFAULT_GT_CSV)
    p.add_argument("--cache_file", type=Path, default=rg.DEFAULT_CACHE)
    p.add_argument("--filter_file", type=Path, default=rg.DEFAULT_FILTER)
    p.add_argument("--work_dir", type=Path,
                   default=rg.REPO / "output/results/gepa_cache_curation",
                   help="Where predictions persist across review cycles (stable dir).")
    p.add_argument("--papers", type=int, nargs="*", default=None,
                   help="Paper indices to curate (default: all 7).")
    p.add_argument("--reextract", action="store_true",
                   help="Force fresh extraction even if predictions exist.")
    args = p.parse_args()

    for path in (args.config_path, args.prompt, args.gt_csv):
        if not Path(path).exists():
            print(f"Error: not found: {path}")
            sys.exit(1)

    papers = rg.load_papers(args.gt_csv, args.filter_file)
    if args.papers is not None:
        papers = [papers[i] for i in args.papers]

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache file : {args.cache_file}")
    print(f"Prompt     : {args.prompt}")
    print(f"Work dir   : {work_dir}")
    print(f"Papers     : {[p['doi'] for p in papers]}\n")

    capo = Capo(config_path=str(args.config_path),
                initial_prompt_path=str(args.prompt), resume=False)
    capo.config.setdefault("evaluation", {})
    capo.config["evaluation"]["labeled_data_path"] = str(args.gt_csv)
    capo.config["evaluation"]["cache_file_path"] = str(args.cache_file)



    cache_before = load_cache_index(args.cache_file)

    results = {}
    for paper in papers:
        results[paper["doi"]] = evaluate_paper(capo, paper, work_dir, args.reextract)


    print("\n" + "=" * 80)
    print("METRICS (with the current cache)")
    print("=" * 80)
    hdr = "DOI".ljust(38) + "".join(k.replace("part", "p").replace("_", "")[:9].rjust(11)
                                     for k in METRIC_KEYS)
    print(hdr)
    for doi, m in results.items():
        print(doi.ljust(38) + "".join(f"{m[k]:.3f}".rjust(11) for k in METRIC_KEYS))
    n = len(results)
    print("-" * len(hdr))
    print("MEAN".ljust(38) + "".join(
        f"{sum(m[k] for m in results.values()) / n:.3f}".rjust(11) for k in METRIC_KEYS))

    show_cache_changes(cache_before, load_cache_index(args.cache_file))

    print("\n" + "=" * 80)
    print("NEXT STEP (paused for review)")
    print("=" * 80)
    print(f"1. Open and correct wrong 'result' values in:\n     {args.cache_file}")
    print("2. Re-run this tool to re-evaluate with your corrections (extraction is skipped):")
    print("     pixi run python scripts/prompt_optimization/curate_semantic_cache.py")
    print("3. When the metrics look right, run the GEPA LOOCV (it reuses this cache):")
    print("     pixi run python scripts/prompt_optimization/run_gepa_loocv.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
