#!/usr/bin/env python3
"""Retrain the frozen pDC50/Dmax surrogate configurations on a new AutoTPD+ input.

The original, hash-matched training implementation is imported from the source
snapshot stored with the September 2026 benchmark.  The unchanged PROTAC-DB
predictions are reused after their input and test-set hashes are verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OLD_RESULTS = PROJECT / "results/tack_paper_config_benchmark"
TRAINING_SNAPSHOT = OLD_RESULTS / "source_snapshot/training"
INTERVAL_SNAPSHOT = OLD_RESULTS / "source_snapshot/interval"
TASKS = ("pdc50", "dmax")
AUTO = "AutoTPDplus"
DB = "PROTACDB"

sys.path.insert(0, str(TRAINING_SNAPSHOT))
import tack_paper_config_benchmark as benchmark  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autotpdplus",
        type=Path,
        default=PROJECT / "data/release/AutoTPDplus-PROTAC.csv",
    )
    parser.add_argument(
        "--protacdb",
        type=Path,
        default=PROJECT / "data/evaluation/protacs/baseline/PROTACDB-training.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PROJECT / "data/evaluation/protacs/ground_truth/PROTAC-test.csv",
    )
    parser.add_argument("--old-results", type=Path, default=OLD_RESULTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/tack_frozen_models_autotpdplus_20261027",
    )
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-dmax",
        action="store_true",
        help="Reuse 25 completed pDC50 fold artifacts in output-dir and train Dmax only.",
    )
    return parser.parse_args()


def verify_unchanged_inputs(args: argparse.Namespace, old_manifest: dict) -> None:
    recorded = old_manifest["input_sha256"]
    expected_db = next(value for path, value in recorded.items() if path.endswith("PROTACDB-training.csv"))
    expected_test = next(value for path, value in recorded.items() if path.endswith("PROTAC-test.csv"))
    actual_db = sha256(args.protacdb)
    actual_test = sha256(args.test)
    if actual_db != expected_db:
        raise ValueError(f"PROTAC-DB changed: expected {expected_db}, found {actual_db}")
    if actual_test != expected_test:
        raise ValueError(f"Held-out test changed: expected {expected_test}, found {actual_test}")


def training_counts(trains: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (corpus, task), frame in trains.items():
        rows.append(
            {
                "corpus": corpus,
                "task": task,
                "rows": len(frame),
                "compounds": len(np.unique(benchmark.compound_groups(frame))),
                "exact_primary": int(frame._Primary_Exact.sum()),
                "label_mean": float(frame[benchmark.LABEL[task]].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_dmax:
        raise ValueError(f"Use a new empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_dmax:
        completed_pdc50 = sorted(
            (args.output_dir / "final_models/autotpdplus/pdc50").glob("sample_*/test_predictions.csv")
        )
        if len(completed_pdc50) != args.samples:
            raise ValueError(
                f"Resume requires {args.samples} completed pDC50 folds, found {len(completed_pdc50)}"
            )

    old_manifest = json.loads((args.old_results / "run_manifest.json").read_text())
    verify_unchanged_inputs(args, old_manifest)
    frozen = json.loads((args.old_results / "selection.json").read_text())
    selection = {task: frozen[task] for task in TASKS}
    selection["selection_data"] = frozen["selection_data"]
    selection["external_test_used_for_selection"] = False

    os.environ["TACKAI_CACHE"] = str(PROJECT / "tmp/tack-official-runtime-cache")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    import torch

    torch.set_num_threads(args.threads)
    _, activity_labeler, load_yaml = benchmark.load_official_tack(benchmark.DEFAULT_TACK)

    raw_test = pd.read_csv(args.test)
    raw_auto = pd.read_csv(args.autotpdplus)
    raw_db = pd.read_csv(args.protacdb)
    filtered_auto, auto_audit = benchmark.filter_leakage(raw_auto, raw_test, AUTO)
    filtered_db, db_audit = benchmark.filter_leakage(raw_db, raw_test, DB)
    pd.DataFrame([asdict(auto_audit), asdict(db_audit)]).to_csv(
        args.output_dir / "leakage_audit.csv", index=False
    )

    trains: dict[tuple[str, str], pd.DataFrame] = {}
    for task in TASKS:
        trains[(AUTO, task)] = benchmark.prepare_task(
            filtered_auto, task, AUTO, activity_labeler, set(), False
        )
        trains[(DB, task)] = benchmark.prepare_task(
            filtered_db, task, DB, activity_labeler, set(), False
        )
    counts = training_counts(trains)
    counts.to_csv(args.output_dir / "sample_counts_training.csv", index=False)

    benchmark.cache_molecular_descriptors([filtered_auto, raw_test])
    auto_trains = {(corpus, task): frame for (corpus, task), frame in trains.items() if corpus == AUTO}
    train_tasks = ("dmax",) if args.resume_dmax else TASKS
    split_trains = {
        (corpus, task): frame
        for (corpus, task), frame in auto_trains.items()
        if task in train_tasks
    }
    splits = benchmark.build_splits(
        split_trains, args.samples, args.validation_fraction, args.seed, args.output_dir
    )
    configs = {task: benchmark.data_config(task, load_yaml, args, None) for task in TASKS}

    tests: dict[str, pd.DataFrame] = {}
    test_counts = []
    for task in TASKS:
        tests[task] = benchmark.prepare_task(raw_test, task, "held-out test", activity_labeler, set(), True)
        test_counts.append(
            {
                "task": task,
                "rows": len(tests[task]),
                "exact_primary": int(tests[task]._Primary_Exact.sum()),
                "doi_clusters": int(tests[task].DOI.nunique()),
            }
        )
    pd.DataFrame(test_counts).to_csv(args.output_dir / "sample_counts_test.csv", index=False)

    benchmark.TASKS = train_tasks
    benchmark.CORPORA = (AUTO,)
    newly_fitted_predictions = benchmark.final_fit(
        auto_trains, tests, splits, selection, configs, args.output_dir, args.samples
    )
    newly_fitted_log = pd.read_csv(args.output_dir / "final_training_log.csv")
    if args.resume_dmax:
        pdc50_predictions = pd.concat(
            [pd.read_csv(path) for path in completed_pdc50], ignore_index=True
        )
        pdc50_diagnostics = []
        for prediction_path in completed_pdc50:
            diagnostic_path = prediction_path.with_name("fit_diagnostics.json")
            pdc50_diagnostics.append(json.loads(diagnostic_path.read_text()))
        auto_predictions = pd.concat(
            [pdc50_predictions, newly_fitted_predictions], ignore_index=True
        )
        auto_log = pd.concat(
            [pd.DataFrame(pdc50_diagnostics), newly_fitted_log], ignore_index=True
        )
    else:
        auto_predictions = newly_fitted_predictions
        auto_log = newly_fitted_log

    old_predictions_path = args.old_results / "model_test_predictions.csv"
    old_predictions = pd.read_csv(old_predictions_path)
    db_predictions = old_predictions[
        old_predictions.task.isin(TASKS) & old_predictions.corpus.eq(DB)
    ].copy()
    expected_rows = sum(len(tests[task]) for task in TASKS) * args.samples
    if len(db_predictions) != expected_rows:
        raise ValueError(
            f"Incomplete frozen PROTAC-DB predictions: expected {expected_rows}, found {len(db_predictions)}"
        )
    combined = pd.concat([auto_predictions, db_predictions], ignore_index=True)
    combined.to_csv(args.output_dir / "model_test_predictions.csv", index=False)

    old_log = pd.read_csv(args.old_results / "final_training_log.csv")
    db_log = old_log[old_log.task.isin(TASKS) & old_log.corpus.eq(DB)].copy()
    auto_log["prediction_provenance"] = "refit_new_AutoTPDplus_input"
    db_log["prediction_provenance"] = "reused_hash_verified_unchanged_PROTACDB_input"
    pd.concat([auto_log, db_log], ignore_index=True).to_csv(
        args.output_dir / "final_training_log.csv", index=False
    )

    benchmark.TASKS = TASKS
    benchmark.CORPORA = (AUTO, DB)
    benchmark.evaluate(
        combined,
        tests,
        args.output_dir,
        args.samples,
        args.bootstrap,
        args.seed,
        selection,
        counts,
    )

    source_hashes = {
        str(path.relative_to(PROJECT)): sha256(path)
        for path in sorted(TRAINING_SNAPSHOT.glob("*.py"))
    }
    tack_commit = subprocess.check_output(
        ["git", "-C", str(benchmark.DEFAULT_TACK), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "status": "completed",
        "controlled_change": "AutoTPDplus training input only",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": {
            "autotpdplus": sha256(args.autotpdplus),
            "protacdb": sha256(args.protacdb),
            "test": sha256(args.test),
            "reused_protacdb_predictions": sha256(old_predictions_path),
            "cellosaurus": sha256(PROJECT / "tmp/tack-official-runtime-cache/cellosaurus.txt"),
        },
        "frozen_selection": {task: selection[task] for task in TASKS},
        "training_source_snapshot_sha256": source_hashes,
        "tack_git_commit": tack_commit,
        "protacdb_retrained": False,
        "protacdb_reuse_reason": "Input, test set, model configuration, seeds, and training implementation are unchanged.",
        "new_autotpdplus_models": len(TASKS) * args.samples,
        "ensemble_models_per_corpus_task": args.samples,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    snapshot_out = args.output_dir / "source_snapshot"
    shutil.copytree(TRAINING_SNAPSHOT, snapshot_out / "training")
    shutil.copytree(INTERVAL_SNAPSHOT, snapshot_out / "interval")
    shutil.copy2(Path(__file__), snapshot_out / Path(__file__).name)
    (args.output_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "tasks": list(TASKS),
                "samples_per_corpus_task": args.samples,
                "new_models_trained": len(TASKS) * args.samples,
                "selected": {task: selection[task]["id"] for task in TASKS},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Completed frozen-model retraining: {args.output_dir}")


if __name__ == "__main__":
    main()
