#!/usr/bin/env python3








from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tack_official_benchmark import (
    interval_bounds,
    interval_errors,
    interval_metric_values,
    markdown_table,
    paired_interval_bootstrap,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/tack_paper_config_benchmark"
TASKS = ("pdc50", "dmax")
CORPORA = ("AutoTPDplus", "PROTACDB")
KEY = "_Benchmark_Row_ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalized_operator(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def validate_long_predictions(long: pd.DataFrame) -> int:
    required = {
        "corpus", "task", "sample", KEY, "DOI", "Value_Operator",
        "_Primary_Exact", "y_true", "prediction",
    }
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    selected = long[long.task.isin(TASKS) & long.corpus.isin(CORPORA)].copy()
    if selected.empty:
        raise ValueError("No pDC50/Dmax predictions found")
    if selected.duplicated(["corpus", "task", "sample", KEY]).any():
        raise ValueError("Duplicate corpus/task/sample/test-row predictions")
    if not np.isfinite(selected[["y_true", "prediction"]].to_numpy(float)).all():
        raise ValueError("Non-finite regression target or prediction")

    sample_counts = selected.groupby(["corpus", "task"])["sample"].nunique()
    if sample_counts.nunique() != 1:
        raise ValueError(f"Inconsistent model counts: {sample_counts.to_dict()}")
    samples = int(sample_counts.iloc[0])


    for (corpus, task), part in selected.groupby(["corpus", "task"], sort=False):
        expected = part[part["sample"].eq(part["sample"].min())][KEY].tolist()
        for sample, sample_part in part.groupby("sample", sort=False):
            if sample_part[KEY].tolist() != expected:
                raise ValueError(f"Test row order differs for {corpus}/{task}/sample={sample}")
        checks = part.assign(_operator=_normalized_operator(part.Value_Operator)).groupby(KEY).agg(
            doi=("DOI", "nunique"), target=("y_true", "nunique"),
            operator=("_operator", "nunique"), exact=("_Primary_Exact", "nunique"),
        )
        if checks.to_numpy().max() != 1:
            raise ValueError(f"Test metadata differs across models for {corpus}/{task}")
    return samples


def ensemble_for(long: pd.DataFrame, corpus: str, task: str, samples: int) -> pd.DataFrame:
    part = long[long.corpus.eq(corpus) & long.task.eq(task)].copy()
    metadata = part[part["sample"].eq(part["sample"].min())][
        [KEY, "DOI", "Value_Operator", "_Primary_Exact", "y_true"]
    ].copy()
    pivot = part.pivot(index=KEY, columns="sample", values="prediction").reindex(metadata[KEY])
    if pivot.shape != (len(metadata), samples) or pivot.isna().any().any():
        raise ValueError(f"Incomplete prediction matrix for {corpus}/{task}: {pivot.shape}")
    metadata["prediction"] = pivot.mean(axis=1).to_numpy(float)
    metadata["prediction_sd"] = pivot.std(axis=1, ddof=1).to_numpy(float)
    return metadata


def align_ensembles(auto: pd.DataFrame, db: pd.DataFrame, task: str) -> pd.DataFrame:
    merged = auto.merge(db, on=KEY, how="inner", validate="one_to_one", suffixes=("_Auto", "_DB"))
    if len(merged) != len(auto) or len(merged) != len(db):
        raise ValueError(f"Corpus prediction alignment failed for {task}")
    if not np.allclose(merged.y_true_Auto, merged.y_true_DB, atol=0, rtol=0):
        raise ValueError(f"Targets differ between corpora for {task}")
    if not merged.DOI_Auto.astype(str).equals(merged.DOI_DB.astype(str)):
        raise ValueError(f"DOIs differ between corpora for {task}")
    if not _normalized_operator(merged.Value_Operator_Auto).equals(
            _normalized_operator(merged.Value_Operator_DB)):
        raise ValueError(f"Operators differ between corpora for {task}")
    if not merged._Primary_Exact_Auto.astype(bool).equals(
            merged._Primary_Exact_DB.astype(bool)):
        raise ValueError(f"Exact/censored flags differ between corpora for {task}")

    return pd.DataFrame({
        "task": task,
        KEY: merged[KEY],
        "DOI": merged.DOI_Auto,
        "y_true": merged.y_true_Auto.astype(float),
        "value_operator": merged.Value_Operator_Auto,
        "is_exact_primary": merged._Primary_Exact_Auto.astype(bool),
        "AutoTPDplus_prediction": merged.prediction_Auto.astype(float),
        "AutoTPDplus_model_sd": merged.prediction_sd_Auto.astype(float),
        "PROTACDB_prediction": merged.prediction_DB.astype(float),
        "PROTACDB_model_sd": merged.prediction_sd_DB.astype(float),
    })


def per_model_interval_metrics(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selected = long[long.task.isin(TASKS) & long.corpus.isin(CORPORA)]
    for (corpus, task, sample), part in selected.groupby(["corpus", "task", "sample"], sort=False):
        lower, upper = interval_bounds(
            task, part.y_true.to_numpy(float), part.Value_Operator.to_numpy(object))
        for metric, value in interval_metric_values(
                part.prediction.to_numpy(float), lower, upper).items():
            rows.append({
                "corpus": corpus, "task": task, "sample": int(sample),
                "evaluation": "interval-aware", "metric": metric, "value": value,
            })
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["corpus", "task", "evaluation", "metric"]).value.agg(
        count="count", mean="mean", sd="std").reset_index()
    return metrics, summary


def point_metrics(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "metrics_all.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path)
    result = metrics[
        metrics.task.isin(TASKS) & metrics.metric.isin(["RMSE", "MAE"])
    ].copy()
    if len(result) != 4:
        raise ValueError(f"Expected four point-valued regression rows, found {len(result)}")
    result["absolute_error_reduction"] = result.PROTACDB - result.AutoTPDplus
    result["relative_error_reduction_percent"] = (
        100.0 * result.absolute_error_reduction / result.PROTACDB)
    return result


def append_report(results_dir: Path, point: pd.DataFrame, interval: pd.DataFrame,
                  per_model_summary: pd.DataFrame, bootstrap: int) -> None:
    point_display = point[[
        "task", "metric", "AutoTPDplus", "PROTACDB", "absolute_error_reduction",
        "relative_error_reduction_percent", "ci95_low", "ci95_high", "n_test",
        "n_test_doi_clusters",
    ]]
    interval_display = interval.copy()
    interval_display["absolute_error_reduction"] = (
        interval_display.PROTACDB - interval_display.AutoTPDplus)
    interval_display["relative_error_reduction_percent"] = (
        100.0 * interval_display.absolute_error_reduction / interval_display.PROTACDB)
    interval_display = interval_display[[
        "task", "metric", "AutoTPDplus", "PROTACDB", "absolute_error_reduction",
        "relative_error_reduction_percent", "ci95_low", "ci95_high",
        "bootstrap_p_two_sided", "n_test", "n_test_doi_clusters", "n_exact", "n_censored",
    ]]
    section = "\n".join([
        "<!-- interval-aware-evaluation:start -->",
        "## Point-valued and interval-aware regression evaluation", "",
        "Point-valued RMSE/MAE use only exact (`=`/`~`) held-out measurements. "
        "Interval-aware RMSE/MAE additionally include `<` and `>` measurements; error is zero "
        "inside the reported interval and distance to the nearest bound outside it. pDC50 "
        "inequality directions are reversed after transforming DC50 concentration to potency.", "",
        "Positive paired deltas and error reductions favor AutoTPD+. Confidence intervals use "
        f"the same paired DOI-cluster bootstrap as the official benchmark ({bootstrap:,} draws).", "",
        "### Primary point-valued held-out performance", "",
        markdown_table(point_display, digits=4), "",
        "### Supplemental interval-aware held-out performance", "",
        markdown_table(interval_display, digits=4), "",
        "### Per-model interval-aware mean and SD", "",
        markdown_table(per_model_summary, digits=4), "",
        "The interval-aware comparison conditions on the fitted 25-model ensembles; it does not "
        "include uncertainty from refitting the training procedure.",
        "<!-- interval-aware-evaluation:end -->",
    ])
    summary_path = results_dir / "summary.md"
    original = summary_path.read_text() if summary_path.is_file() else ""
    start = "<!-- interval-aware-evaluation:start -->"
    end = "<!-- interval-aware-evaluation:end -->"
    if start in original and end in original:
        before = original.split(start, 1)[0].rstrip()
        after = original.split(end, 1)[1].lstrip()
        updated = before + "\n\n" + section + ("\n\n" + after if after else "")
    else:
        updated = original.rstrip() + "\n\n" + section
    summary_path.write_text(updated.rstrip() + "\n")

    (results_dir / "point_and_interval_evaluation.md").write_text(
        "# Point-valued and interval-aware held-out performance\n\n" +
        "## Point-valued\n\n" + markdown_table(point_display, digits=4) + "\n\n" +
        "## Interval-aware\n\n" + markdown_table(interval_display, digits=4) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    results_dir = args.results_dir.resolve()
    prediction_path = results_dir / "model_test_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    long = pd.read_csv(prediction_path)
    samples = validate_long_predictions(long)

    interval_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for task in TASKS:
        auto = ensemble_for(long, CORPORA[0], task, samples)
        db = ensemble_for(long, CORPORA[1], task, samples)
        merged = align_ensembles(auto, db, task)
        comparison, lower, upper = paired_interval_bootstrap(
            task,
            merged.y_true.to_numpy(float),
            merged.value_operator.to_numpy(object),
            merged.AutoTPDplus_prediction.to_numpy(float),
            merged.PROTACDB_prediction.to_numpy(float),
            merged.DOI.astype(str).to_numpy(),
            args.bootstrap,
            args.seed + 20_000,
        )
        merged["interval_lower"] = lower
        merged["interval_upper"] = upper
        merged["AutoTPDplus_interval_error"] = interval_errors(
            merged.AutoTPDplus_prediction.to_numpy(float), lower, upper)
        merged["PROTACDB_interval_error"] = interval_errors(
            merged.PROTACDB_prediction.to_numpy(float), lower, upper)
        interval_frames.append(comparison)
        prediction_frames.append(merged)

    interval = pd.concat(interval_frames, ignore_index=True)
    interval["absolute_error_reduction"] = interval.PROTACDB - interval.AutoTPDplus
    interval["relative_error_reduction_percent"] = (
        100.0 * interval.absolute_error_reduction / interval.PROTACDB)
    interval.to_csv(results_dir / "metrics_interval_aware.csv", index=False)

    point = point_metrics(results_dir)
    point.to_csv(results_dir / "metrics_point_valued.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        results_dir / "test_predictions_point_and_interval.csv", index=False)

    per_model, per_model_summary = per_model_interval_metrics(long)
    per_model.to_csv(results_dir / "per_model_interval_metrics.csv", index=False)
    per_model_summary.to_csv(
        results_dir / "per_model_interval_metric_summary.csv", index=False)
    append_report(results_dir, point, interval, per_model_summary, args.bootstrap)

    official_source = Path(__file__).with_name("tack_official_benchmark.py")
    audit = {
        "evaluation_only": True,
        "models_refitted": False,
        "prediction_input": str(prediction_path),
        "prediction_input_sha256": sha256(prediction_path),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "official_interval_source": str(official_source.resolve()),
        "official_interval_source_sha256": sha256(official_source),
        "imported_official_functions": [
            "interval_bounds", "interval_errors", "interval_metric_values",
            "paired_interval_bootstrap",
        ],
        "bootstrap_draws": args.bootstrap,
        "bootstrap_seed": args.seed + 20_000,
        "cluster_unit": "DOI",
        "ensemble": f"mean of {samples} saved model predictions per test record",
        "point_metrics_source": str((results_dir / "metrics_all.csv").resolve()),
    }
    (results_dir / "interval_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n")

    print("Point-valued metrics (unchanged):")
    print(point.to_string(index=False))
    print("\nInterval-aware metrics:")
    print(interval.to_string(index=False))


if __name__ == "__main__":
    main()
