#!/usr/bin/env python3






















import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from tpd_curator.capo import Capo
from tpd_curator.gepa_adapter import (
    TpdExtractionAdapter,
    prompt_to_candidate,
    candidate_to_prompt,
)
from tpd_curator.llm.run_llm_api import run_llm_api

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "tpd_curator/config/evaluation_glues_config.yaml"
DEFAULT_SEED_PROMPT = REPO / "prompts/initial_prompt.md"
DEFAULT_GT_CSV = REPO / "data/evaluation/molecular_glues/ground_truth/GT_Glues.csv"
DEFAULT_CACHE = REPO / "semantic_cache/gepa_match_glue.json"
DEFAULT_FILTER = REPO / "data_source/250825_molglue_papers/paper_dois/avaliable_paper_ids.json"

METRIC_KEYS = ("part1_precision", "part1_recall", "part2_precision", "part2_recall")


def load_papers(gt_csv: Path, filter_file: Path):

    df = pd.read_csv(gt_csv)
    doi_col = "DOI" if "DOI" in df.columns else df.columns[0]
    dois = sorted(d for d in df[doi_col].dropna().unique().tolist() if str(d).strip())

    pmc_map = {}
    if filter_file.exists():
        for rec in json.loads(Path(filter_file).read_text()):
            if isinstance(rec, dict):
                pmc_map[rec.get("doi")] = rec.get("pmc_id")

    return [{"doi": d, "pmc_id": pmc_map.get(d)} for d in dois]


def make_reflection_lm():

    def _lm(prompt: str) -> str:
        text, _ = run_llm_api(custom_prompt=prompt, paper_text=None, temperature=1.0)
        if isinstance(text, (list, dict)):
            text = json.dumps(text, ensure_ascii=False)
        return text
    return _lm


def scalar(metrics: dict) -> float:
    return sum(metrics[k] for k in METRIC_KEYS) / len(METRIC_KEYS)


def metrics_row(metrics: dict) -> dict:
    row = {k: round(float(metrics[k]), 4) for k in METRIC_KEYS}
    row["mean"] = round(scalar(metrics), 4)
    return row


def run_fold(fold_idx, papers, args, reflection_lm, out_root):
    import gepa

    test = papers[fold_idx]
    dev = [p for i, p in enumerate(papers) if i != fold_idx]

    fold_dir = out_root / f"fold_{fold_idx}_{test['doi'].replace('/', '_')}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}\nFOLD {fold_idx}: test={test['doi']}\n  dev={[p['doi'] for p in dev]}\n{'='*80}")

    seed_prompt_text = Path(args.seed_prompt).read_text(encoding="utf-8")
    seed_candidate = prompt_to_candidate(seed_prompt_text)

    capo = Capo(
        config_path=str(args.config_path),
        initial_prompt_path=str(args.seed_prompt),
        resume=False,
    )
    capo.config.setdefault("evaluation", {})
    capo.config["evaluation"]["labeled_data_path"] = str(args.gt_csv)
    capo.config["evaluation"]["cache_file_path"] = str(args.cache_file)

    adapter = TpdExtractionAdapter(
        capo=capo,
        template_prompt=seed_prompt_text,
        work_dir=fold_dir / "rollouts",
        pause_on_new_matches=args.pause_on_new_matches,
        pause_dir=args.pause_dir,
        pause_context={
            "config_path": str(args.config_path),
            "gt_csv": str(args.gt_csv),
            "seed_prompt": str(args.seed_prompt),
            "fold": fold_idx,
        },
    )

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=dev,
        valset=dev,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        run_dir=str(fold_dir / "gepa"),
        seed=0,
        display_progress_bar=False,
    )

    best = result.best_candidate
    best_val = result.val_aggregate_scores[result.best_idx]
    best_prompt = candidate_to_prompt(best, seed_prompt_text)
    (fold_dir / "best_prompt.md").write_text(best_prompt, encoding="utf-8")


    gepa_eval = adapter.evaluate([test], best, capture_traces=False)
    gepa_test = gepa_eval.outputs[0]["metrics"]

    fold_record = {
        "fold": fold_idx,
        "test_doi": test["doi"],
        "dev_dois": [p["doi"] for p in dev],
        "gepa_val_score_on_dev": round(float(best_val), 4),
        "gepa_test": metrics_row(gepa_test),
        "best_prompt_path": str(fold_dir / "best_prompt.md"),
    }

    if args.eval_seed_baseline:
        seed_eval = adapter.evaluate([test], seed_candidate, capture_traces=False)
        fold_record["seed_test"] = metrics_row(seed_eval.outputs[0]["metrics"])

    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FOLD {fold_idx}] gepa_test={fold_record['gepa_test']}"
          + (f"  seed_test={fold_record['seed_test']}" if args.eval_seed_baseline else ""))
    return fold_record


def aggregate(folds, key):
    out = {}
    cols = METRIC_KEYS + ("mean",)
    for c in cols:
        vals = [f[key][c] for f in folds if key in f]
        if not vals:
            continue
        out[c] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    return out


def write_summary(folds, out_root, eval_seed):
    summary = {
        "num_folds": len(folds),
        "gepa": aggregate(folds, "gepa_test"),
        "folds": folds,
    }
    if eval_seed:
        summary["seed_baseline"] = aggregate(folds, "seed_test")
    (out_root / "loocv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = METRIC_KEYS + ("mean",)
    lines = ["# GEPA LOOCV summary (held-out, MG)\n",
             f"\nFolds: {len(folds)}\n",
             "\n## Held-out test metrics per fold (GEPA)\n",
             "\n| fold | test DOI | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for f in folds:
        g = f["gepa_test"]
        lines.append(f"| {f['fold']} | {f['test_doi']} | "
                     + " | ".join(f"{g[c]:.3f}" for c in cols) + " |")

    def agg_line(label, agg):
        return ("| " + label + " | "
                + " | ".join(f"{agg[c]['mean']:.3f}±{agg[c]['std']:.3f}" for c in cols) + " |")

    lines += ["\n## Aggregate (mean±std over held-out folds)\n",
              "\n| optimizer | " + " | ".join(cols) + " |",
              "|---|" + "---|" * len(cols),
              agg_line("GEPA", summary["gepa"])]
    if eval_seed:
        lines.append(agg_line("seed (initial prompt)", summary["seed_baseline"]))

    (out_root / "loocv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote: {out_root/'loocv_summary.md'}  and  {out_root/'loocv_summary.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--seed_prompt", type=Path, default=DEFAULT_SEED_PROMPT)
    p.add_argument("--gt_csv", type=Path, default=DEFAULT_GT_CSV)
    p.add_argument("--cache_file", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--filter_file", type=Path, default=DEFAULT_FILTER)
    p.add_argument("--max_metric_calls", type=int, default=36)
    p.add_argument("--reflection_minibatch_size", type=int, default=3)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None,
                   help="Fold indices to run (default: all). Use e.g. '--folds 0' for a smoke test.")
    p.add_argument("--no_eval_seed_baseline", dest="eval_seed_baseline",
                   action="store_false", default=True,
                   help="Skip evaluating the seed (initial) prompt on each held-out paper.")
    p.add_argument("--pause-on-new-matches", dest="pause_on_new_matches",
                   action="store_true", default=False,
                   help="Pause optimization whenever an evaluation produces NEW "
                        "semantic-match judgments, for human review. Resume with "
                        "scripts/prompt_optimization/gepa_continue.py. Same matching criteria as CAPO.")
    p.add_argument("--pause_dir", type=Path, default=REPO / "output/results/.gepa_pause",
                   help="Rendezvous dir for pause/continue signals (must match "
                        "gepa_continue.py --pause_dir).")
    args = p.parse_args()

    for path in (args.config_path, args.seed_prompt, args.gt_csv):
        if not Path(path).exists():
            print(f"Error: not found: {path}")
            sys.exit(1)

    papers = load_papers(args.gt_csv, args.filter_file)
    print(f"Loaded {len(papers)} MG ground-truth papers:")
    for i, pp in enumerate(papers):
        print(f"  [{i}] {pp['doi']}  (pmc={pp['pmc_id']})")

    fold_indices = args.folds if args.folds is not None else list(range(len(papers)))

    if args.output_dir is not None:
        out_root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%y%m%d_%H%M")
        out_root = REPO / "output" / "results" / f"gepa_loocv_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {out_root}\nFolds to run: {fold_indices}\n"
          f"Budget/fold: {args.max_metric_calls} metric calls\n")

    reflection_lm = make_reflection_lm()

    folds = []
    for idx in fold_indices:
        try:
            folds.append(run_fold(idx, papers, args, reflection_lm, out_root))
        except Exception as e:
            import traceback
            print(f"[FOLD {idx}] FAILED: {e}")
            traceback.print_exc()

    if folds:
        write_summary(folds, out_root, args.eval_seed_baseline)
    else:
        print("No folds completed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
