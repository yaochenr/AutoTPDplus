#!/usr/bin/env python3




















from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from tack_holdout_training import compound_groups, holm_adjust, paired_bootstrap
from tack_official_benchmark import (
    DEFAULT_CACHE,
    DEFAULT_PROTEINS,
    DEFAULT_TACK,
    ROOT,
    assign_scaffold_clusters,
    base_tack_frame,
    cache_molecular_descriptors,
    choose_uniprot,
    load_official_tack,
    markdown_table,
    operator_is_exact,
)
from tack_benchmark_common import (
    CORPORA,
    DEFAULT_AUTO,
    DEFAULT_PROTACDB,
    DEFAULT_TEST,
    build_splits,
    file_hash,
    filter_leakage,
    metric_values,
)


TASKS = ("pdc50", "dmax", "activity")
DEFAULT_OUTPUT = ROOT / "results/tack_paper_config_benchmark"
PAPER = "https://arxiv.org/abs/2605.19579"
LABEL = {"pdc50": "pDC50", "dmax": "Dmax", "activity": "Activity"}


PAPER_CONFIG = {
    "pdc50": {
        "model_type": "mlp", "data_yaml": "simple-onehot.yaml",
        "table2": "MLP (†): Cell OneHot | E3 OneHot | POI OneHot | Desc | Assay OneHot + Time",
    },
    "dmax": {
        "model_type": "xgboost", "data_yaml": "cell_text_poi_countvec-onehot.yaml",
        "table2": "XGBoost (†): Cell Text | E3 OneHot | POI Vec | Desc | Assay OneHot + Time",
    },
    "activity": {
        "model_type": "xgboost", "data_yaml": "bin-fp_cell_text_esms.yaml",
        "table2": "XGBoost (‡): Cell Text | E3 ESM-S-PCA | POI ESM-S-PCA | Desc+FP | Assay OneHot + Time",
    },
}
MLP_PAPER = {"hidden_dim": [256], "learning_rate": 0.00075, "dropout": 0.0}
DMAX_PAPER = {"learning_rate": 0.0159, "max_depth": 7, "min_child_weight": 1, "gamma": 0.005,
              "subsample": 0.982, "colsample_bytree": 0.916, "reg_alpha": 0.007, "reg_lambda": 0.005}
ACTIVITY_PAPER = {"learning_rate": 0.0018, "max_depth": 6, "min_child_weight": 1, "gamma": 0.120,
                  "subsample": 0.946, "colsample_bytree": 0.629, "reg_alpha": 0.447, "reg_lambda": 0.018}
STRONG_REGULARIZATION = {"min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 5.0}


def candidate_specs(threads: int, smoke: bool) -> dict[str, list[dict[str, Any]]]:

    mlp = []
    for hidden in ([256], [512, 256]):
        for lr in (0.00025, 0.00075, 0.002):
            for dropout in (0.0, 0.2):
                mlp.append({
                    "id": f"mlp_h{'-'.join(map(str, hidden))}_lr{lr:g}_do{dropout:g}",
                    "paper_anchor": hidden == MLP_PAPER["hidden_dim"] and lr == MLP_PAPER["learning_rate"]
                                    and dropout == MLP_PAPER["dropout"],
                    "model": {
                        "hidden_dim": hidden, "learning_rate": lr, "dropout": dropout,
                        "head_depth": 1, "head_dropout": 0.0, "task_type": "point",
                        "activation": "relu", "norm_type": "layer",
                        "lr_scheduler_type": "reduce_on_plateau",
                    },
                    "training": {"max_epochs": 20 if smoke else 200, "patience": 2 if smoke else 5},
                })

    def xgb_grid(task: str, paper: dict[str, float], rates: tuple[float, ...], depths: tuple[int, ...]):
        specs = []
        for depth in depths:
            for lr in rates:
                for strength in ("paper", "strong"):
                    params = dict(paper, max_depth=depth, learning_rate=lr)
                    if strength == "strong":
                        params.update(STRONG_REGULARIZATION)
                    specs.append({
                        "id": f"{task}_xgb_d{depth}_lr{lr:g}_{strength}reg",
                        "paper_anchor": depth == paper["max_depth"] and lr == paper["learning_rate"]
                                        and strength == "paper",
                        "model": {
                            **params, "tree_method": "hist", "n_jobs": threads,
                            "objective": "binary:logistic" if task == "activity" else "reg:squarederror",
                            "eval_metric": "logloss" if task == "activity" else "rmse",
                        },
                        "training": {"num_boost_round": 50 if smoke else 2000,
                                     "early_stopping_rounds": 5 if smoke else 30},
                    })
        return specs

    specs = {
        "pdc50": mlp,
        "dmax": xgb_grid("dmax", DMAX_PAPER, (0.0159, 0.05), (3, 5, 7)),
        "activity": xgb_grid("activity", ACTIVITY_PAPER, (0.0018, 0.01, 0.03), (3, 6)),
    }
    if smoke:
        specs = {task: [s for s in grid if s["paper_anchor"]] + [s for s in grid if not s["paper_anchor"]][:1]
                 for task, grid in specs.items()}
    for grid in specs.values():
        if sum(spec["paper_anchor"] for spec in grid) != 1:
            raise AssertionError("Each grid must contain exactly one paper-selected anchor")
    return specs


def prepare_task(source: pd.DataFrame, task: str, corpus: str, activity_labeler,
                 protein_ids: set[str], is_test: bool) -> pd.DataFrame:

    work = base_tack_frame(source, corpus)
    valid = work.SMILES.notna()
    if task == "pdc50":
        raw = pd.to_numeric(source.DC50, errors="coerce")
        valid &= raw.notna() & np.isfinite(raw) & (raw > 0)
        work["DC50"] = raw
        work["pDC50"] = 9.0 - np.log10(raw.where(raw > 0))
        work["Assay_Time"] = pd.to_numeric(source.DC50_h, errors="coerce")
        work["Value_Operator"] = source.DC50_operator.astype("string")
        work["_Primary_Exact"] = source.DC50_operator.map(operator_is_exact)
    elif task == "dmax":
        raw = pd.to_numeric(source.Dmax, errors="coerce")

        valid &= raw.notna() & np.isfinite(raw) & source.Degradation_Target_Sequence.notna()
        work["Dmax"] = raw.clip(0.0, 100.0)
        work["Assay_Time"] = pd.to_numeric(source.Dmax_h, errors="coerce")
        work["Value_Operator"] = source.Dmax_operator.astype("string")
        work["_Primary_Exact"] = source.Dmax_operator.map(operator_is_exact)
    else:
        labels = source.apply(lambda row: activity_labeler(pd.Series(
            {"DC50": row.get("DC50"), "Dmax": row.get("Dmax")})), axis=1)
        work["Activity"] = labels.astype(float)
        work["Assay_Time"] = pd.to_numeric(source.Dmax_h, errors="coerce").combine_first(
            pd.to_numeric(source.DC50_h, errors="coerce"))
        work["Value_Operator"] = ""
        work["_Primary_Exact"] = True
        work["POI_UniProt"] = source.Degradation_Target_Uniprot.map(lambda v: choose_uniprot(v, protein_ids))
        work["Ligase_UniProt"] = source.Recruiter_Uniprot.map(lambda v: choose_uniprot(v, protein_ids))

        valid &= labels.notna() & work.POI_UniProt.notna() & work.Ligase_UniProt.notna()
    result = work.loc[valid].copy().reset_index(drop=True)
    if is_test:
        result["SMILES_Scaffold_Cluster"] = -1
        result["SMILES_Butina_Cluster"] = -1
        result["SMILES_Held_Out"] = True
    else:
        result = assign_scaffold_clusters(result)
    if len(result) < 2:
        raise ValueError(f"Not enough usable {corpus}/{task} records")
    return result


def activity_pca_widths(trains, splits, samples: int) -> tuple[tuple[int, int], dict[str, Any]]:

    widths = np.array([44, 7])
    audit = []
    for corpus in CORPORA:
        frame = trains[(corpus, "activity")]
        for split in splits[(corpus, "activity")][:samples]:
            part = frame.iloc[split["train_idx"]]
            counts = np.array([part.POI_UniProt.nunique(), part.Ligase_UniProt.nunique()])
            if counts.min() < 2:
                raise ValueError("Activity PCA needs >= 2 distinct POI/E3 proteins per training partition")
            widths = np.minimum(widths, counts - 1)
            audit.append({"corpus": corpus, "sample": split["fold"],
                          "training_unique_poi": int(counts[0]), "training_unique_e3": int(counts[1])})
    used = (int(widths[0]), int(widths[1]))
    return used, {"paper_dimensions": [44, 7], "used_dimensions": list(used),
                  "policy": ("min(paper width, training-partition unique proteins - 1) across both corpora "
                             "and all splits; external test unused"), "training_counts": audit}


def data_config(task: str, load_yaml, args, pca: tuple[int, int] | None) -> dict[str, Any]:
    config = load_yaml(DEFAULT_TACK / "configs/data" / PAPER_CONFIG[task]["data_yaml"])
    config.update(normalize_labels=False, standardize_labels=False, impute_labels=False,
                  batch_size=64, num_workers=0, num_proc=1, is_bert_multitask=False)
    if task == "activity":
        assert pca is not None
        config.update(
            poi_embeddings_file=str(args.protein_embeddings), ligase_embeddings_file=str(args.protein_embeddings),
            poi_embeddings_format="h5", ligase_embeddings_format="h5",
            poi_embeddings_per_residue=False, ligase_embeddings_per_residue=False,
            poi_embeddings_id_type="uniprot", ligase_embeddings_id_type="uniprot",
            poi_pca_n_components=pca[0], ligase_pca_n_components=pca[1],
        )
    return config


def build_module(frame: pd.DataFrame, split: dict[str, Any], task: str, config: dict[str, Any],
                 seed: int, test: pd.DataFrame | None = None):

    import pytorch_lightning as pl
    from datasets import Dataset, DatasetDict
    from tackai.data.datamodule import DegradationComplexDataModule

    parts = {"train": frame.iloc[split["train_idx"]].reset_index(drop=True),
             "validation": frame.iloc[split["test_idx"]].reset_index(drop=True)}
    if test is not None:
        parts["test"] = test
    pl.seed_everything(seed, verbose=False)
    module = DegradationComplexDataModule(
        dataset=DatasetDict({k: Dataset.from_pandas(v, preserve_index=False) for k, v in parts.items()}),
        labels=[LABEL[task]], verbose=0, **copy.deepcopy(config),
    )
    module.setup()
    featurized = {"train": module.train_dataset, "validation": module.val_dataset}
    if test is not None:
        featurized["test"] = module.test_dataset
    for name, part in parts.items():
        if len(featurized[name]) != len(part):
            raise AssertionError(f"Featurization dropped rows in {name}")
        np.testing.assert_allclose(np.asarray(featurized[name].with_format("numpy")[LABEL[task]], float),
                                   part[LABEL[task]].to_numpy(float), rtol=1e-5, atol=1e-5)
    return module, parts


class XGBRunner:
    def __init__(self, module, has_test: bool):

        self.matrices = {"train": module.get_xgboost_dataset("train"),
                         "validation": module.get_xgboost_dataset("validation")}
        if has_test:
            self.matrices["test"] = module.get_xgboost_dataset("test")
        self.n_features = int(self.matrices["train"].num_col())

    def fit(self, spec, task: str, seed: int, directory: Path | None):
        import xgboost as xgb

        params = dict(copy.deepcopy(spec["model"]), seed=seed)
        model = xgb.train(params, self.matrices["train"], num_boost_round=spec["training"]["num_boost_round"],
                          evals=[(self.matrices["validation"], "validation")],
                          early_stopping_rounds=spec["training"]["early_stopping_rounds"], verbose_eval=False)
        end = int(model.best_iteration) + 1
        predictions = {name: model.predict(m, iteration_range=(0, end)) for name, m in self.matrices.items()}
        if task == "dmax":
            predictions = {k: np.clip(v, 0.0, 100.0) for k, v in predictions.items()}
        diagnostics = {"best_iteration": end, "max_rounds": spec["training"]["num_boost_round"],
                       "prediction_iteration_range": [0, end]}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            model.save_model(directory / "model.json")
        return predictions, diagnostics


class MLPRunner:
    def __init__(self, module, has_test: bool):
        self.module = module
        self.has_test = has_test
        self.n_features = int(module.get_total_feature_dim())
        if module.get_categorical_vocab_sizes():
            raise AssertionError("Paper pDC50 MLP uses one-hot, not categorical embeddings")

        self.tensors = {"train": self._materialize(module.train_dataset),
                        "validation": self._materialize(module.val_dataset)}
        if has_test:
            self.tensors["test"] = self._materialize(module.test_dataset)

    @staticmethod
    def _materialize(dataset):
        import torch

        columns = dataset.with_format("torch")[:]
        return {k: v if isinstance(v, torch.Tensor) else torch.stack(list(v)) for k, v in columns.items()}

    def loader(self, split: str, seed: int = 0):

        import torch
        from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

        tensors = self.tensors[split]
        size = len(next(iter(tensors.values())))

        class Batches(Dataset):
            def __len__(self):
                return size

            def __getitem__(self, rows):
                index = torch.as_tensor(rows)
                return {k: v[index] for k, v in tensors.items()}

        train = split == "train"
        order = (RandomSampler(range(size), generator=torch.Generator().manual_seed(seed)) if train
                 else SequentialSampler(range(size)))
        sampler = BatchSampler(order, batch_size=self.module.batch_size, drop_last=train)
        return DataLoader(Batches(), sampler=sampler, batch_size=None, num_workers=0)

    @staticmethod
    def _kaiming_init(model) -> None:

        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif name.startswith("model.mlp.0"):
                if len(param.shape) >= 2:
                    param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
            elif len(param.shape) >= 2:
                param.data.normal_(0, np.sqrt(2) / np.sqrt(param.shape[1]))

    def _predict(self, model, loader) -> np.ndarray:
        import torch

        model.eval()
        out = []
        with torch.no_grad():
            for batch in loader:
                out.append(model.predict_step(batch)["pred"].reshape(-1).cpu().numpy())
        return np.concatenate(out).astype(float)

    def fit(self, spec, task: str, seed: int, directory: Path | None):
        import pytorch_lightning as pl
        import torch
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from tackai.models.tack_model import TACKModel

        pl.seed_everything(seed, verbose=False)
        config = copy.deepcopy(spec["model"])
        config["mlp_depth"] = len(config["hidden_dim"])
        model = TACKModel(model_type="mlp", label_names=[LABEL[task]], input_dim=self.n_features, **config)
        self._kaiming_init(model)
        scratch = Path(tempfile.mkdtemp(prefix="mlp-ckpt-")) if directory is None else directory
        scratch.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(dirpath=scratch, filename="best", monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=spec["training"]["patience"]),
                       checkpoint],
            logger=False, max_epochs=spec["training"]["max_epochs"], gradient_clip_val=1.0,
            accelerator="cpu", devices=1, precision="32-true", enable_progress_bar=False,
            enable_model_summary=False, deterministic=True,
        )
        trainer.fit(model, train_dataloaders=self.loader("train", seed), val_dataloaders=self.loader("validation"))

        state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        predictions = {"validation": self._predict(model, self.loader("validation"))}
        if self.has_test:
            predictions["test"] = self._predict(model, self.loader("test"))
        diagnostics = {"best_iteration": int(state["epoch"]) + 1, "epochs_run": int(trainer.current_epoch),
                       "best_val_loss": float(checkpoint.best_model_score)}
        if directory is None:
            shutil.rmtree(scratch, ignore_errors=True)
        return predictions, diagnostics


def runner_for(task: str, module, has_test: bool):
    return MLPRunner(module, has_test) if PAPER_CONFIG[task]["model_type"] == "mlp" else XGBRunner(module, has_test)


def validation_score(task: str, parts: dict[str, pd.DataFrame], prediction: np.ndarray) -> dict[str, float]:
    validation = parts["validation"]
    if task == "activity":
        y = validation.Activity.to_numpy(int)
        pred = np.clip(np.asarray(prediction, float), 1e-7, 1 - 1e-7)
        prevalence = float(parts["train"].Activity.mean())
        return {"roc_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
                "logloss": float(log_loss(y, pred, labels=[0, 1])),
                "prevalence_baseline_logloss": float(log_loss(y, np.full(len(y), prevalence), labels=[0, 1])),
                "n_score_records": len(y)}
    primary = validation._Primary_Exact.astype(bool).to_numpy()
    y = validation.loc[primary, LABEL[task]].to_numpy(float)
    pred = np.asarray(prediction, float)[primary]
    mean = float(parts["train"][LABEL[task]].mean())
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "mae": float(mean_absolute_error(y, pred)),
            "mean_baseline_rmse": float(np.sqrt(mean_squared_error(y, np.full(len(y), mean)))),
            "mean_baseline_mae": float(mean_absolute_error(y, np.full(len(y), mean))),
            "n_score_records": int(len(y))}


def search(trains, splits, specs, configs, output: Path, samples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for corpus in CORPORA:
            frame = trains[(corpus, task)]
            for split in splits[(corpus, task)][:samples]:
                sample = split["fold"]
                print(f"SEARCH task={task} corpus={corpus} sample={sample + 1}/{samples}", flush=True)
                module, parts = build_module(frame, split, task, configs[task], split["model_seed"])
                runner = runner_for(task, module, has_test=False)
                for spec in specs[task]:
                    started = time.time()
                    predictions, diag = runner.fit(spec, task, split["model_seed"], None)
                    rows.append({"task": task, "candidate": spec["id"], "paper_anchor": spec["paper_anchor"],
                                 "corpus": corpus, "sample": sample, "best_iteration": diag["best_iteration"],
                                 "seconds": time.time() - started,
                                 **validation_score(task, parts, predictions["validation"])})
                pd.DataFrame(rows).to_csv(output / "selection_scores.csv", index=False)
                del module, runner
    return pd.DataFrame(rows)


def select(scores: pd.DataFrame, specs, output: Path) -> dict[str, Any]:

    selection, rankings = {}, []
    for task in TASKS:
        part = scores[scores.task.eq(task)].copy()
        if task == "activity":
            per_corpus = part.groupby(["candidate", "corpus"]).agg(score=("roc_auc", "mean"),
                                                                   tie=("logloss", "mean")).reset_index()
            rank = per_corpus.groupby("candidate").agg(selection_score=("score", "mean"),
                                                      tiebreak=("tie", "mean")).reset_index()
            rank = rank.sort_values(["selection_score", "tiebreak", "candidate"], ascending=[False, True, True])
            criterion = "mean validation ROC-AUC (higher better); tie-break mean log loss"
        else:
            part["rmse_ratio"] = part.rmse / part.mean_baseline_rmse
            per_corpus = part.groupby(["candidate", "corpus"]).agg(score=("rmse_ratio", "mean"),
                                                                   tie=("rmse", "mean")).reset_index()
            rank = per_corpus.groupby("candidate").agg(selection_score=("score", "mean"),
                                                      tiebreak=("tie", "mean")).reset_index()
            rank = rank.sort_values(["selection_score", "tiebreak", "candidate"])
            criterion = ("mean validation RMSE / split-specific training-mean RMSE on exact records "
                         "(lower better), averaged per corpus then across corpora")
        wide = per_corpus.pivot(index="candidate", columns="corpus", values="score").add_prefix("score_")
        rank = rank.merge(wide, on="candidate").reset_index(drop=True)
        rank.insert(0, "task", task)
        rank["rank"] = np.arange(1, len(rank) + 1)
        rankings.append(rank)
        chosen = next(spec for spec in specs[task] if spec["id"] == rank.iloc[0].candidate)
        selection[task] = {**chosen, "criterion": criterion}
    pd.concat(rankings, ignore_index=True).to_csv(output / "selection_ranking.csv", index=False)
    selection["selection_data"] = "internal compound-grouped 10% validation partitions of both corpora only"
    selection["external_test_used_for_selection"] = False
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def final_fit(trains, tests, splits, selection, configs, output: Path, samples: int) -> pd.DataFrame:
    rows, log = [], []
    keep = ["_Benchmark_Row_ID", "DOI", "Compound_Name", "SMILES", "POI_Name", "Ligase_Name",
            "Cell_Line", "Value_Operator", "_Primary_Exact"]
    for task in TASKS:
        spec = selection[task]
        for corpus in CORPORA:
            frame = trains[(corpus, task)]
            for split in splits[(corpus, task)][:samples]:
                sample = split["fold"]
                directory = output / "final_models" / corpus.lower() / task / f"sample_{sample:02d}"
                print(f"FINAL task={task} corpus={corpus} sample={sample + 1}/{samples} config={spec['id']}",
                      flush=True)
                module, parts = build_module(frame, split, task, configs[task], split["model_seed"], tests[task])
                runner = runner_for(task, module, has_test=True)
                started = time.time()
                predictions, diag = runner.fit(spec, task, split["model_seed"], directory)
                result = tests[task][keep].copy()
                result.insert(0, "corpus", corpus)
                result.insert(1, "task", task)
                result.insert(2, "sample", sample)
                result["y_true"] = tests[task][LABEL[task]].to_numpy(float)
                result["prediction"] = predictions["test"]
                result["training_mean_prediction"] = float(parts["train"][LABEL[task]].mean())
                result.to_csv(directory / "test_predictions.csv", index=False)
                rows.append(result)
                entry = {"corpus": corpus, "task": task, "sample": sample, "candidate": spec["id"],
                         "model_type": PAPER_CONFIG[task]["model_type"], "n_train": len(parts["train"]),
                         "n_validation": len(parts["validation"]), "n_features": runner.n_features,
                         "seconds": time.time() - started, **diag,
                         **{f"validation_{k}": v for k, v in
                            validation_score(task, parts, predictions["validation"]).items()}}
                (directory / "fit_diagnostics.json").write_text(json.dumps(entry, indent=2, default=float) + "\n")
                log.append(entry)
                del module, runner
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(output / "model_test_predictions.csv", index=False)
    pd.DataFrame(log).to_csv(output / "final_training_log.csv", index=False)
    return long


def metrics(task: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    values = metric_values(task, y, pred)
    if task == "activity":
        values["Brier"] = float(brier_score_loss(y, np.clip(pred, 0, 1)))
    return values


def evaluate(long: pd.DataFrame, tests, output: Path, samples: int, bootstrap: int, seed: int,
             selection: dict[str, Any], counts: pd.DataFrame) -> None:
    per_model, baselines, ensemble_rows, comparisons, merged_all = [], [], [], [], []
    ensembles = {}
    baseline_metrics = {"pdc50": ("RMSE", "MAE"), "dmax": ("RMSE", "MAE"), "activity": ("LogLoss", "Brier")}
    for task in TASKS:
        test = tests[task]
        primary = test._Primary_Exact.astype(bool).to_numpy() if task != "activity" else np.ones(len(test), bool)
        y = test.loc[primary, LABEL[task]].to_numpy(float)
        for corpus in CORPORA:
            part = long[(long.task == task) & (long.corpus == corpus)]
            if part["sample"].nunique() != samples:
                raise AssertionError(f"Expected {samples} models for {corpus}/{task}")
            pivot = part.pivot(index="_Benchmark_Row_ID", columns="sample", values="prediction").reindex(
                test._Benchmark_Row_ID)
            if pivot.isna().any().any():
                raise AssertionError("Incomplete test prediction matrix")
            ensemble = pivot.mean(axis=1).to_numpy()
            constant = float(part.groupby("sample").training_mean_prediction.first().mean())
            ensembles[(corpus, task)] = pd.DataFrame({
                "_Benchmark_Row_ID": test._Benchmark_Row_ID.to_numpy(), "DOI": test.DOI.to_numpy(),
                "y_true": test[LABEL[task]].to_numpy(float), "prediction": ensemble,
                "prediction_sd": pivot.std(axis=1, ddof=1).to_numpy() if samples > 1 else 0.0,
                "training_mean_prediction": constant, "is_exact_primary": primary})
            model_values = metrics(task, y, ensemble[primary])
            base_values = metrics(task, y, np.full(len(y), constant))
            for metric, value in model_values.items():
                ensemble_rows.append({"corpus": corpus, "task": task, "metric": metric, "value": value})
            for metric in baseline_metrics[task]:
                baselines.append({"corpus": corpus, "task": task, "metric": metric,
                                  "model_ensemble": model_values[metric],
                                  "training_mean_baseline": base_values[metric],
                                  "delta_favoring_model": base_values[metric] - model_values[metric]})
            for sample, sample_part in part.groupby("sample"):
                ordered = sample_part.set_index("_Benchmark_Row_ID").reindex(test._Benchmark_Row_ID)
                pred = ordered.prediction.to_numpy()[primary]
                for metric, value in metrics(task, y, pred).items():
                    per_model.append({"corpus": corpus, "task": task, "sample": sample,
                                      "predictor": PAPER_CONFIG[task]["model_type"], "metric": metric,
                                      "value": value})
                const = float(ordered.training_mean_prediction.iloc[0])
                for metric, value in metrics(task, y, np.full(len(y), const)).items():
                    if metric in baseline_metrics[task]:
                        per_model.append({"corpus": corpus, "task": task, "sample": sample,
                                          "predictor": "training_mean", "metric": metric, "value": value})
        auto, db = ensembles[(CORPORA[0], task)], ensembles[(CORPORA[1], task)]
        comparison = paired_bootstrap(task, y, auto.prediction.to_numpy()[primary],
                                      db.prediction.to_numpy()[primary],
                                      auto.DOI.astype(str).to_numpy()[primary], bootstrap, seed + 10_000)
        if task != "activity":
            comparison = comparison[comparison.metric.isin(["RMSE", "MAE"])].reset_index(drop=True)
        comparisons.append(comparison)
        merged = auto.rename(columns={"prediction": "AutoTPDplus_prediction",
                                      "prediction_sd": "AutoTPDplus_model_sd",
                                      "training_mean_prediction": "AutoTPDplus_training_mean"})
        merged["PROTACDB_prediction"] = db.prediction.to_numpy()
        merged["PROTACDB_model_sd"] = db.prediction_sd.to_numpy()
        merged["PROTACDB_training_mean"] = db.training_mean_prediction.to_numpy()
        merged["abs_error_AutoTPDplus"] = (merged.AutoTPDplus_prediction - merged.y_true).abs()
        merged["abs_error_PROTACDB"] = (merged.PROTACDB_prediction - merged.y_true).abs()
        merged.insert(0, "task", task)
        merged_all.append(merged)

    per_model = pd.DataFrame(per_model)
    per_model.to_csv(output / "per_model_metrics.csv", index=False)
    summary = per_model.groupby(["corpus", "task", "predictor", "metric"]).value.agg(
        count="count", mean="mean", sd="std").reset_index()
    summary.to_csv(output / "per_model_metric_summary.csv", index=False)
    ensemble_frame = pd.DataFrame(ensemble_rows)
    ensemble_frame.to_csv(output / "ensemble_metrics.csv", index=False)
    baseline_frame = pd.DataFrame(baselines)
    baseline_frame.to_csv(output / "training_mean_baseline_comparison.csv", index=False)
    comparison = pd.concat(comparisons, ignore_index=True)
    primary_rows = comparison.apply(
        lambda r: r.metric in ({"ROC_AUC"} if r.task == "activity" else {"RMSE", "MAE"}), axis=1)
    comparison["bootstrap_p_holm_primary"] = np.nan
    comparison.loc[primary_rows, "bootstrap_p_holm_primary"] = holm_adjust(
        comparison.loc[primary_rows, "bootstrap_p_two_sided"].to_numpy())
    comparison.to_csv(output / "metrics_all.csv", index=False)
    pd.concat(merged_all, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)

    table = summary_table(ensemble_frame, summary, baseline_frame, comparison, counts, selection)
    table.to_csv(output / "summary_table.csv", index=False)
    write_report(output, table, comparison, baseline_frame, summary, selection, samples)


def summary_table(ensemble, per_model, baselines, comparison, counts, selection) -> pd.DataFrame:

    headline = {"pdc50": ("RMSE", "MAE"), "dmax": ("RMSE", "MAE"), "activity": ("ROC_AUC", "LogLoss")}
    rows = []
    for task in TASKS:
        for metric in headline[task]:
            cmp = comparison[(comparison.task == task) & (comparison.metric == metric)].iloc[0]
            for corpus in CORPORA:
                ens = ensemble[(ensemble.task == task) & (ensemble.corpus == corpus) & (ensemble.metric == metric)]
                pm = per_model[(per_model.task == task) & (per_model.corpus == corpus)
                               & (per_model.metric == metric) & (per_model.predictor != "training_mean")].iloc[0]
                base = baselines[(baselines.task == task) & (baselines.corpus == corpus)
                                 & (baselines.metric == metric)]
                n = counts[(counts.task == task) & (counts.corpus == corpus)].iloc[0]
                rows.append({
                    "task": task, "model": PAPER_CONFIG[task]["model_type"], "config": selection[task]["id"],
                    "corpus": corpus, "n_train_rows": int(n.rows), "n_train_compounds": int(n.compounds),
                    "metric": metric, "ensemble": float(ens.value.iloc[0]),
                    "per_model_mean": float(pm["mean"]), "per_model_sd": float(pm["sd"]),
                    "training_mean_baseline": (float(base.training_mean_baseline.iloc[0]) if len(base)
                                               else (0.5 if metric == "ROC_AUC" else np.nan)),
                    "delta_AutoTPDplus_vs_PROTACDB": float(cmp.delta_favoring_AutoTPDplus),
                    "ci95_low": float(cmp.ci95_low), "ci95_high": float(cmp.ci95_high),
                    "n_test": int(cmp.n_test), "n_test_doi": int(cmp.n_test_doi_clusters),
                })
    return pd.DataFrame(rows)


def write_report(output, table, comparison, baselines, summary, selection, samples) -> None:
    compact = table.copy()
    compact["per_model"] = compact.apply(lambda r: f"{r.per_model_mean:.3f} ± {r.per_model_sd:.3f}", axis=1)
    compact["paired_delta (95% CI)"] = compact.apply(
        lambda r: f"{r.delta_AutoTPDplus_vs_PROTACDB:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]", axis=1)
    compact = compact[["task", "model", "metric", "corpus", "n_train_rows", "ensemble", "per_model",
                       "training_mean_baseline", "paired_delta (95% CI)", "n_test", "n_test_doi"]]
    lines = [
        "# AutoTPD+ versus PROTACDB: TACK paper best configurations", "",
        f"Configurations follow Table 2 of the TACK paper ({PAPER}):", "",
        *[f"- **{task}**: {PAPER_CONFIG[task]['table2']}; selected candidate `{selection[task]['id']}`"
          f"{' (paper hyperparameters)' if selection[task]['paper_anchor'] else ''}" for task in TASKS], "",
        f"{samples} compound-grouped ~90/10 splits per corpus/endpoint. The 10% validation part drove "
        "early stopping and candidate scoring; the external test played no role in selection. "
        "Positive paired deltas favor AutoTPD+ (lower error or higher ROC-AUC); 95% CIs come from a paired "
        "DOI-cluster bootstrap. Training-mean baseline = constant mean of the training labels "
        "(activity: training prevalence, ROC-AUC 0.5). Regression metrics use exact (`=`/`~`) test records.", "",
        "## Summary table", "", markdown_table(compact, 3), "",
        "## Paired comparison (all metrics)", "", markdown_table(comparison, 4), "",
        "## Model versus training-mean baseline (ensemble)", "", markdown_table(baselines, 4), "",
        "## Per-model mean and SD", "", markdown_table(summary, 4), "",
    ]
    (output / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autotpdplus", type=Path, default=DEFAULT_AUTO)
    parser.add_argument("--protacdb", type=Path, default=DEFAULT_PROTACDB)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--protein-embeddings", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Tiny grid and epochs; plumbing check only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1 or not 0 < args.validation_fraction < 1 or args.bootstrap < 1:
        raise ValueError("Invalid samples, validation fraction or bootstrap count")
    args.output_dir = args.output_dir.resolve()
    launcher = {"supervisor.pid", "training.pid", "run.log", "exit_code.txt"}
    if args.output_dir.exists() and {p.name for p in args.output_dir.iterdir()} - launcher:
        raise ValueError(f"Use a new empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TACKAI_CACHE"] = str(DEFAULT_CACHE)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import h5py
    import torch
    torch.set_num_threads(args.threads)

    specs = candidate_specs(args.threads, args.smoke)
    _, activity_labeler, load_yaml = load_official_tack(DEFAULT_TACK)
    with h5py.File(args.protein_embeddings, "r") as handle:
        protein_ids = set(handle.keys())
    raw_test = pd.read_csv(args.test)
    filtered, audits = {}, []
    for corpus, path in zip(CORPORA, (args.autotpdplus, args.protacdb)):
        filtered[corpus], audit = filter_leakage(pd.read_csv(path), raw_test, corpus)
        audits.append(asdict(audit))
    pd.DataFrame(audits).to_csv(args.output_dir / "leakage_audit.csv", index=False)

    trains, counts = {}, []
    for task in TASKS:
        for corpus in CORPORA:
            frame = prepare_task(filtered[corpus], task, corpus, activity_labeler, protein_ids, False)
            trains[(corpus, task)] = frame
            counts.append({"corpus": corpus, "task": task, "rows": len(frame),
                           "compounds": len(np.unique(compound_groups(frame))),
                           "exact_primary": int(frame._Primary_Exact.sum()),
                           "label_mean": float(frame[LABEL[task]].mean())})
    counts = pd.DataFrame(counts)
    counts.to_csv(args.output_dir / "sample_counts_training.csv", index=False)
    cache_molecular_descriptors([*filtered.values(), raw_test])
    splits = build_splits(trains, args.samples, args.validation_fraction, args.seed, args.output_dir)
    pca, pca_audit = activity_pca_widths(trains, splits, args.samples)
    (args.output_dir / "activity_pca_audit.json").write_text(json.dumps(pca_audit, indent=2) + "\n")
    configs = {task: data_config(task, load_yaml, args, pca if task == "activity" else None) for task in TASKS}

    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "input_sha256": {str(p.resolve()): file_hash(p) for p in
                         (args.autotpdplus, args.protacdb, args.test, args.protein_embeddings)},
        "code_sha256": {str(p): file_hash(p) for p in [Path(__file__), *sorted(Path(__file__).parent.glob("tack_*.py"))]},
        "paper": PAPER, "paper_table2": {t: PAPER_CONFIG[t]["table2"] for t in TASKS},
        "data_configs": configs, "candidates": specs,
        "labels": {"pdc50": "9 - log10(DC50 nM), unnormalized", "dmax": "Dmax clipped to 0-100, unnormalized",
                   "activity": "released TACK get_bin_label (binary classification)"},
        "split": f"{args.samples} distinct ~{1 - args.validation_fraction:.0%}/{args.validation_fraction:.0%} "
                 "complete-compound holdouts (canonical SMILES or Connectivity_Key); not scaffold grouping",
        "external_test_selection_role": "none",
    }, indent=2, default=str) + "\n")

    scores = search(trains, splits, specs, configs, args.output_dir, args.samples)
    selection = select(scores, specs, args.output_dir)
    print("SELECTION FROZEN " + " ".join(f"{t}={selection[t]['id']}" for t in TASKS), flush=True)


    tests, test_counts = {}, []
    for task in TASKS:
        tests[task] = prepare_task(raw_test, task, "held-out test", activity_labeler, protein_ids, True)
        test_counts.append({"task": task, "rows": len(tests[task]),
                            "exact_primary": int(tests[task]._Primary_Exact.sum()),
                            "doi_clusters": int(tests[task].DOI.nunique())})
    pd.DataFrame(test_counts).to_csv(args.output_dir / "sample_counts_test.csv", index=False)
    long = final_fit(trains, tests, splits, selection, configs, args.output_dir, args.samples)
    evaluate(long, tests, args.output_dir, args.samples, args.bootstrap, args.seed, selection, counts)
    marker = "SMOKE_COMPLETED.json" if args.smoke else "COMPLETED.json"
    (args.output_dir / marker).write_text(json.dumps({
        "status": "completed", "samples_per_corpus_task": args.samples, "selection_fits": len(scores),
        "final_models": len(TASKS) * len(CORPORA) * args.samples,
        "selected": {t: selection[t]["id"] for t in TASKS}}, indent=2) + "\n")
    print(f"Completed: {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
