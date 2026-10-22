#!/usr/bin/env python3






from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
CORPORA = ("AutoTPDplus", "PROTACDB")
DEFAULT_AUTO = ROOT / "data/release/AutoTPDplus-PROTAC-training.csv"
DEFAULT_PROTACDB = ROOT / "data/evaluation/protacs/baseline/PROTACDB-training.csv"
DEFAULT_TEST = ROOT / "data/evaluation/protacs/ground_truth/PROTAC-test.csv"

Task = Literal["pdc50", "dmax", "activity"]


@dataclass
class LeakageAudit:
    corpus: str
    rows_raw: int
    rows_matching_test_doi: int
    rows_removed_test_doi: int
    rows_removed_test_compound_after_doi: int
    rows_removed_invalid_smiles: int
    rows_after_filter: int
    compounds_after_filter: int


def file_hash(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower()


def filter_leakage(
    frame: pd.DataFrame,
    test: pd.DataFrame,
    corpus: str,
) -> tuple[pd.DataFrame, LeakageAudit]:

    from tack_holdout_training import compound_groups
    from tack_official_benchmark import canonical_smiles, normalize_doi

    test_dois = set(normalize_doi(test.DOI).dropna())
    test_keys = {
        value
        for value in _normal_text(test.Connectivity_Key).dropna()
        if value not in {"", "nan", "<na>"}
    }
    test_smiles = {canonical_smiles(value) for value in test.SMILES.dropna()}
    test_smiles.discard(None)

    doi = normalize_doi(frame.DOI).isin(test_dois).fillna(False)
    keys = _normal_text(frame.Connectivity_Key).isin(test_keys).fillna(False)
    smiles = frame.SMILES.map(canonical_smiles).isin(test_smiles).fillna(False)
    compound = keys | smiles
    result = frame.loc[~(doi | compound)].copy().reset_index(drop=True)

    canonical = result.SMILES.map(canonical_smiles)
    invalid = canonical.isna()
    result = result.loc[~invalid].copy().reset_index(drop=True)
    groups = compound_groups(result)
    return result, LeakageAudit(
        corpus=corpus,
        rows_raw=len(frame),
        rows_matching_test_doi=int(doi.sum()),
        rows_removed_test_doi=int(doi.sum()),
        rows_removed_test_compound_after_doi=int((compound & ~doi).sum()),
        rows_removed_invalid_smiles=int(invalid.sum()),
        rows_after_filter=len(result),
        compounds_after_filter=int(len(np.unique(groups))),
    )


def build_splits(
    trains: dict[tuple[str, Task], pd.DataFrame],
    samples: int,
    fraction: float,
    seed: int,
    output: Path,
) -> dict[tuple[str, Task], list[dict[str, Any]]]:

    from tack_holdout_training import compound_groups, compound_splits

    result: dict[tuple[str, Task], list[dict[str, Any]]] = {}
    for (corpus, task), frame in trains.items():
        splits = list(compound_splits(frame, samples, fraction, seed))
        result[(corpus, task)] = splits
        groups = compound_groups(frame)
        audit = []
        for split in splits:
            train_groups = set(groups[split["train_idx"]])
            validation_groups = set(groups[split["test_idx"]])
            if train_groups & validation_groups:
                raise AssertionError("Compound leakage in internal split")
            if task == "activity" and len(np.unique(frame.iloc[split["train_idx"]].Activity)) < 2:
                raise ValueError(
                    f"Activity training split has one class: {corpus}, sample {split['fold']}"
                )
            audit.append(
                {
                    "sample": split["fold"],
                    "train_rows": len(split["train_idx"]),
                    "validation_rows": len(split["test_idx"]),
                    "validation_fraction": len(split["test_idx"]) / len(frame),
                    "overlapping_compounds": 0,
                }
            )
        path = output / "splits" / corpus.lower() / task
        path.mkdir(parents=True, exist_ok=True)
        (path / "splits.json").write_text(
            json.dumps(
                {
                    "protocol": "random complete-compound groups; not scaffold grouping",
                    "row_id_by_position": frame._Benchmark_Row_ID.tolist(),
                    "splits": splits,
                },
                indent=2,
            )
            + "\n"
        )
        pd.DataFrame(audit).to_csv(path / "audit.csv", index=False)
    return result


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float = 0.8) -> float:
    precision, recall, _ = precision_recall_curve(y, score)
    eligible = recall[precision >= target]
    return float(np.max(eligible)) if len(eligible) else 0.0


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, pred).statistic if len(y) > 1 else np.nan
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "Spearman": float(rho),
    }


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    label = (pred >= 0.5).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y, pred)),
        "PR_AUC": float(average_precision_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, label)),
        "Recall_at_80pct_precision": recall_at_precision(y, pred),
        "LogLoss": float(log_loss(y, pred, labels=[0, 1])),
        "Accuracy": float(accuracy_score(y, label)),
    }


def metric_values(task: Task, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:

    return classification_metrics(y, pred) if task == "activity" else regression_metrics(y, pred)
