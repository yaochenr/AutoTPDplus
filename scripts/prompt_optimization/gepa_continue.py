#!/usr/bin/env python3

















import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tpd_curator.capo import Capo
from tpd_curator.gepa_adapter import PAUSE_MARKER, PAUSE_REVIEW, CONTINUE_SIGNAL

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PAUSE_DIR = REPO / "output/results/.gepa_pause"
METRIC_KEYS = ("part1_precision", "part1_recall", "part2_precision", "part2_recall")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pause_dir", type=Path, default=DEFAULT_PAUSE_DIR,
                   help="Rendezvous dir where GEPA wrote the pause marker.")
    p.add_argument("--verify", action="store_true",
                   help="Re-evaluate and print scores but DO NOT continue GEPA "
                        "(iterate on cache corrections). Omit to continue.")
    args = p.parse_args()

    marker_path = args.pause_dir / PAUSE_MARKER
    if not marker_path.exists():
        print(f"No active pause found at {marker_path}.")
        print("(GEPA is not currently paused, or --pause_dir is wrong.)")
        sys.exit(1)

    marker = json.loads(marker_path.read_text())
    papers = marker.get("papers", [])
    cache_file = marker.get("cache_file")
    config_path = marker.get("config_path")
    gt_csv = marker.get("gt_csv")
    seed_prompt = marker.get("seed_prompt")

    print(f"Paused at    : {marker.get('paused_at')}")
    print(f"Cache file   : {cache_file}")
    print(f"Review file  : {args.pause_dir / PAUSE_REVIEW}")
    print(f"Papers to re-evaluate: {[p['doi'] for p in papers]}\n")


    capo = Capo(config_path=str(config_path),
                initial_prompt_path=str(seed_prompt), resume=False)
    capo.config.setdefault("evaluation", {})
    capo.config["evaluation"]["labeled_data_path"] = str(gt_csv)
    capo.config["evaluation"]["cache_file_path"] = str(cache_file)

    print("Re-evaluating with the current (corrected) cache ...\n")
    hdr = "DOI".ljust(38) + "".join(k.replace("part", "p").replace("_", "")[:9].rjust(11)
                                     for k in METRIC_KEYS)
    print(hdr)
    rows = []
    for paper in papers:
        doi = paper["doi"]
        pdir = Path(paper["paper_dir"])
        try:
            ev = capo._re_evaluate_paper(paper_dir=pdir, doi=doi)
            m = capo._extract_metrics_from_evaluation(ev, doi)
            print(doi.ljust(38) + "".join(f"{m[k]:.3f}".rjust(11) for k in METRIC_KEYS))
            rows.append(m)
        except Exception as e:
            print(f"{doi.ljust(38)}  re-eval failed: {e}")
    if rows:
        n = len(rows)
        print("-" * len(hdr))
        print("MEAN".ljust(38) + "".join(
            f"{sum(r[k] for r in rows) / n:.3f}".rjust(11) for k in METRIC_KEYS))

    if args.verify:
        print("\n[--verify] Scores above reflect the current cache. GEPA stays PAUSED.")
        print("Edit the cache more and re-run, or run without --verify to continue.")
        return


    (args.pause_dir / CONTINUE_SIGNAL).write_text("continue\n", encoding="utf-8")
    print("\n▶ CONTINUE signal sent. The paused GEPA run will resume and re-evaluate "
          "with the corrected cache.")


if __name__ == "__main__":
    main()
