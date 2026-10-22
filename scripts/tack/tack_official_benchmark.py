#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import h5py
import numpy as np
import pandas as pd

_rdkit_base = Path(sys.prefix) / "share" / "RDKit"

if (_rdkit_base / "Data" / "FragmentDescriptors.csv").is_file():
    os.environ["RDBASE"] = str(_rdkit_base)

from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TACK = Path("/Users/yaochenr/project/TACK")
DEFAULT_AUTO = ROOT / "data/release/AutoTPD-PROTAC.csv"
DEFAULT_PROTACDB = ROOT / "data/evaluation/protacs/baseline/PROTACDB_pmc.csv"
DEFAULT_TEST = ROOT / "data/evaluation/protacs/ground_truth/PROTAC-test.csv"
DEFAULT_OUTPUT = ROOT / "results/tack_repeated_holdout"
DEFAULT_CACHE = ROOT / "tmp/tack-official-runtime-cache"
DEFAULT_PROTEINS = ROOT / "tmp/tack-source/uniprot_embeddings.h5"
Task = Literal["pdc50", "dmax", "activity"]


@dataclass
class LeakageAudit:
    corpus: str
    rows_raw: int
    rows_matching_test_doi: int
    test_dois_matched: int
    rows_removed_test_doi: int
    rows_removed_test_compound_after_doi: int
    rows_after_filter: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tack-repo", type=Path, default=DEFAULT_TACK)
    parser.add_argument("--autotpdplus", type=Path, default=DEFAULT_AUTO)
    parser.add_argument("--protacdb", type=Path, default=DEFAULT_PROTACDB)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tack-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--protein-embeddings", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--tasks", nargs="+", choices=["pdc50", "dmax", "activity"],
                        default=["pdc50", "dmax", "activity"])
    parser.add_argument("--leakage-policy", choices=["doi", "doi+compound", "none"],
                        default="doi+compound")
    parser.add_argument("--split-strategy", choices=["repeated-holdout", "scaffold-cv"],
                        default="repeated-holdout")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--holdout-unit", choices=["row", "compound"], default="row",
                        help="Random records or complete compound groups (pDC50 control)")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--tuning-trials", type=int, default=20)
    parser.add_argument("--hyperparameter-mode", choices=["tuned", "fixed"], default="tuned",
                        help="Retune each holdout, or use the fixed paper-selected parameters")
    parser.add_argument("--split-reference", type=Path,
                        help="Completed control run: reuse its data/config; reuse splits unless changing holdout unit")
    parser.add_argument("--pdc50-model", choices=["xgboost", "mlp"], default="xgboost")
    parser.add_argument("--pdc50-target", choices=["dc50-quantile", "direct-pdc50"],
                        default="dc50-quantile",
                        help="Original DC50 label pipeline, or unnormalized 9-log10(DC50 nM)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument(
        "--fold-ids", type=int, nargs="+", default=None,
        help=("Optional zero-based fold IDs to run from TACK's n_splits × n_repeats "
              "generator, or sample IDs in repeated-holdout mode. Intended for disjoint parallel shards."),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-proc", type=int, default=1,
                        help=("TACK datamodule feature-worker processes; use 1 for "
                              "the Activity ESM-S configuration, whose HDF5/RDKit "
                              "objects are not safely pickleable"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--outer-workers", type=int, default=1,
                        help="Run the two corpus fits concurrently (use 2 on this Mac)")
    parser.add_argument("--batch-featurization", action="store_true",
                        help="Use TACK's batched, cache-sharing feature assembly")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_doi(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.strip().str.lower()
            .str.replace(r"^https?://(?:dx\.)?doi\.org/", "", regex=True)
            .str.replace(r"^doi:\s*", "", regex=True))


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def nonempty_set(series: pd.Series) -> set[str]:
    return {x for x in normalize_text(series).dropna() if x and x.lower() != "nan"}


def filter_leakage(frame: pd.DataFrame, test: pd.DataFrame, corpus: str,
                   policy: str) -> tuple[pd.DataFrame, LeakageAudit]:
    test_dois = set(normalize_doi(test["DOI"]).dropna())
    normalized_doi = normalize_doi(frame["DOI"])
    doi_mask = normalized_doi.isin(test_dois).fillna(False)
    compound_mask = (
        normalize_text(frame["Connectivity_Key"]).isin(nonempty_set(test["Connectivity_Key"]))
        | normalize_text(frame["SMILES"]).isin(nonempty_set(test["SMILES"]))
    ).fillna(False)
    remove = pd.Series(False, index=frame.index)
    if policy in {"doi", "doi+compound"}:
        remove |= doi_mask
    if policy == "doi+compound":
        remove |= compound_mask
    filtered = frame.loc[~remove].copy().reset_index(drop=True)
    return filtered, LeakageAudit(
        corpus=corpus,
        rows_raw=len(frame),
        rows_matching_test_doi=int(doi_mask.sum()),
        test_dois_matched=int(normalized_doi.loc[doi_mask].nunique()),
        rows_removed_test_doi=int(doi_mask.sum()) if policy != "none" else 0,
        rows_removed_test_compound_after_doi=(
            int((compound_mask & ~doi_mask).sum()) if policy == "doi+compound" else 0
        ),
        rows_after_filter=len(filtered),
    )


def canonical_smiles(value: Any) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    mol = Chem.MolFromSmiles(str(value))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold(value: str) -> str:
    result = MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=True)
    return result or f"ACYCLIC::{value}"


def cache_molecular_descriptors(frames: list[pd.DataFrame]) -> int:
    from tackai.data.embeddings.mol_embeddings import MolEmbedding

    smiles = {
        canonical_smiles(value)
        for frame in frames
        for value in frame["SMILES"].dropna()
    }
    smiles.discard(None)
    embedder = MolEmbedding(
        embeddings_type="rdkit_descriptors",
        load_from_cache=True,
        filename="rdkit_descriptors.npz",
    )
    missing = sorted(smiles - set(embedder.embeddings))
    print(f"RDKit descriptors: {len(Descriptors._descList)} features; "
          f"caching {len(missing)} of {len(smiles)} molecules", flush=True)
    if missing:
        embedder.transform(missing, update_cache=True)
    return len(smiles)


def validate_descriptor_cache(path: Path, expected_width: int) -> int:
    if not path.is_file():
        return 0
    invalid: list[tuple[str, tuple[int, ...]]] = []
    with np.load(path, allow_pickle=False) as cached:
        for smiles in cached.files:
            shape = tuple(np.asarray(cached[smiles]).shape)
            if shape != (expected_width,):
                invalid.append((smiles, shape))
                if len(invalid) == 5:
                    break
        count = len(cached.files)
    if invalid:
        examples = ", ".join(f"{smi!r}: {shape}" for smi, shape in invalid)
        raise RuntimeError(
            f"Stale or mixed descriptor cache at {path}; expected every vector "
            f"to have shape ({expected_width},). Invalid examples: {examples}. "
            "Use a fresh cache or archive this file."
        )
    return count

_BATCH_RAW_FEATURE_CACHE: dict[Any, np.ndarray] = {}


def install_tack_acceleration(batch_featurization: bool,
                              read_only_cell_cache: bool = False) -> None:

    if batch_featurization:
        from datasets import Dataset
        from tackai.data.datamodule import DegradationComplexDataModule

        if not getattr(DegradationComplexDataModule, "_benchmark_batch_patch", False):
            def featurize_dataset_batch(self, dataset, split_name="unknown", num_proc=None):
                examples = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)
                features_list = self.featurize_samples_batch(
                    examples, return_tensor="np", shared_cache=_BATCH_RAW_FEATURE_CACHE
                )
                tasks_list = []
                for sample, features in zip(examples, features_list):
                    features.update(self._featurize_and_normalize_labels(sample))
                    if self.is_bert_multitask and self.label_task_col in sample:
                        tasks_list.append(sample[self.label_task_col])
                ds = Dataset.from_list(features_list)
                if len(self.labels) == 1 and not self.impute_labels:
                    label = self.labels[0]
                    keep = [pd.notnull(ds[i][label]) for i in range(len(ds))]
                    ds = ds.filter(lambda x: pd.notnull(x[label]))
                    if tasks_list:
                        tasks_list = [task for task, retain in zip(tasks_list, keep) if retain]
                if split_name == "train":
                    self.train_tasks = tasks_list if tasks_list else None
                elif split_name == "validation":
                    self.val_tasks = tasks_list if tasks_list else None
                elif split_name == "test":
                    self.test_tasks = tasks_list if tasks_list else None
                return ds.with_format("torch")

            DegradationComplexDataModule.featurize_dataset = featurize_dataset_batch
            DegradationComplexDataModule._benchmark_batch_patch = True

    if read_only_cell_cache:
        from tackai.data.embeddings.cell_embeddings import CellEmbedding
        if not getattr(CellEmbedding, "_benchmark_read_only_patch", False):
            original_transform = CellEmbedding.transform

            def transform_without_save(self, *args, **kwargs):
                kwargs["update_cache"] = False
                return original_transform(self, *args, **kwargs)

            CellEmbedding.transform = transform_without_save
            CellEmbedding._benchmark_read_only_patch = True


def cache_cell_embeddings(frames: list[pd.DataFrame]) -> int:
    from tackai.data.embeddings.cell_embeddings import CellEmbedding
    values = sorted({str(value) for frame in frames
                     for value in frame["Cell_Line_ID"].dropna().unique()})
    embedder = CellEmbedding(
        embeddings_type="sentence_transformer", pooling="sum",
        load_from_cache=True,
        filename="cell_embeddings_model=sentence-transformer_pooling=sum.npz",
    )
    missing = [value for value in values if value not in embedder.embeddings]
    print(f"Cell-text embeddings: caching {len(missing)} of {len(values)} values", flush=True)
    if missing:
        embedder.transform(missing, update_cache=True)
    return len(values)


def choose_text(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    first = frame[primary].astype("string").str.strip()
    second = frame[fallback].astype("string").str.strip()
    missing = first.isna() | first.isin(["", "nan", "<NA>"])
    first.loc[missing] = second.loc[missing]
    return first


def choose_uniprot(value: Any, available: set[str] | None = None) -> str | None:
    if pd.isna(value):
        return None
    tokens = [x.strip() for x in str(value).replace(",", ";").split(";") if x.strip()]
    if available is None:
        return tokens[0] if tokens else None
    return next((x for x in tokens if x in available), None)


def operator_is_exact(value: Any) -> bool:
    return pd.isna(value) or str(value).strip() in {"", "~"}


def base_tack_frame(frame: pd.DataFrame, corpus: str) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["_Benchmark_Row_ID"] = np.arange(len(frame), dtype=int)
    out["DOI"] = normalize_doi(frame["DOI"])
    out["SMILES"] = frame["SMILES"].map(canonical_smiles)
    out["Connectivity_Key"] = normalize_text(frame["Connectivity_Key"])
    out["Compound_Name"] = frame["Compound_Name"].astype("string")
    out["POI_Name"] = choose_text(frame, "Degradation_Target_Gene", "Degradation_Target")
    out["POI_Sequence"] = frame["Degradation_Target_Sequence"].astype("string")
    out["POI_UniProt"] = frame["Degradation_Target_Uniprot"].map(choose_uniprot)
    out["Ligase_Name"] = choose_text(frame, "Recruiter_Gene", "Recruiter")
    out["Ligase_Sequence"] = frame["Recruiter_Sequence"].astype("string")
    out["Ligase_UniProt"] = frame["Recruiter_Uniprot"].map(choose_uniprot)
    out["Cell_Line"] = choose_text(frame, "Cell_Line_STD", "Cell_Line")
    out["Cell_Line_ID"] = choose_text(frame, "Cell_Line_STD_Accession", "Cell_Line_STD")
    out["Cell_Line_Species"] = frame["Cell_Line_Species"].astype("string")
    out["Assay"] = choose_text(frame, "Assay_STD", "Assay")
    out["Reference"] = out["DOI"]
    out["Database"] = corpus
    for col in ["POI_Name", "POI_Sequence", "Ligase_Name", "Ligase_Sequence",
                "Cell_Line", "Cell_Line_ID", "Assay"]:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def assign_scaffold_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    scaffolds = out["SMILES"].map(scaffold)
    mapping = {value: idx for idx, value in enumerate(sorted(scaffolds.unique()))}
    out["SMILES_Scaffold_Cluster"] = scaffolds.map(mapping).astype(int)
    out["SMILES_Butina_Cluster"] = out["SMILES_Scaffold_Cluster"]
    out["SMILES_Held_Out"] = False
    return out


def prepare_task(source: pd.DataFrame, task: Task, corpus: str,
                 activity_labeler: Callable[[pd.Series], float],
                 protein_ids: set[str], is_test: bool,
                 require_poi_sequence: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    work = base_tack_frame(source, corpus)
    valid = work["SMILES"].notna()
    coverage = {"rows_before_task_filter": len(work)}
    if task == "pdc50":
        valid &= source["DC50"].notna() & (source["DC50"].astype(float) > 0)
        work["DC50"] = source["DC50"].astype(float)
        work["Assay_Time"] = source["DC50_h"].astype(float)
        work["Value_Operator"] = source["DC50_operator"].astype("string")
        work["_Primary_Exact"] = source["DC50_operator"].map(operator_is_exact)
    elif task == "dmax":
        valid &= source["Dmax"].notna() & source["Degradation_Target_Sequence"].notna()
        work["Dmax"] = source["Dmax"].astype(float).clip(0, 100)
        work["Assay_Time"] = source["Dmax_h"].astype(float)
        work["Value_Operator"] = source["Dmax_operator"].astype("string")
        work["_Primary_Exact"] = source["Dmax_operator"].map(operator_is_exact)
    else:
        labels = source.apply(lambda row: activity_labeler(pd.Series(
            {"DC50": row.get("DC50"), "Dmax": row.get("Dmax")})), axis=1)
        work["Activity"] = labels.astype(float)
        valid &= labels.notna()
        work["Assay_Time"] = source["Dmax_h"].combine_first(source["DC50_h"])
        work["Value_Operator"] = ""
        work["_Primary_Exact"] = True
        work["POI_UniProt"] = source["Degradation_Target_Uniprot"].map(
            lambda value: choose_uniprot(value, protein_ids))
        work["Ligase_UniProt"] = source["Recruiter_Uniprot"].map(
            lambda value: choose_uniprot(value, protein_ids))
        coverage["rows_missing_poi_embedding"] = int((valid & work["POI_UniProt"].isna()).sum())
        coverage["rows_missing_ligase_embedding"] = int((valid & work["Ligase_UniProt"].isna()).sum())
        valid &= work["POI_UniProt"].notna() & work["Ligase_UniProt"].notna()
    if require_poi_sequence:
        valid &= source["Degradation_Target_Sequence"].notna()
    selected = work.loc[valid].copy().reset_index(drop=True)
    coverage["rows_after_task_filter"] = len(selected)
    coverage["unique_scaffolds"] = int(selected["SMILES"].map(scaffold).nunique())
    if is_test:
        selected["SMILES_Scaffold_Cluster"] = -1
        selected["SMILES_Butina_Cluster"] = -1
        selected["SMILES_Held_Out"] = True
    else:
        selected = assign_scaffold_clusters(selected)
    return selected, coverage


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_tack(tack_repo: Path) -> tuple[Callable[..., Any], Callable, Callable]:
    required = [
        tack_repo / "tackai/training.py",
        tack_repo / "tackai/data/datamodule.py",
        tack_repo / "scripts/train_models.py",
        tack_repo / "scripts/plot_evaluation_figures.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete TACK checkout; missing: {missing}")
    sys.path.insert(0, str(tack_repo))
    from tackai.config import load_config_from_yaml
    from tackai.training import run_cv_experiment

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tack_official_train_models", tack_repo / "scripts/train_models.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load TACK scripts/train_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return run_cv_experiment, module.get_bin_label, load_config_from_yaml


def best_configs(task: Task, args: argparse.Namespace,
                 load_yaml: Callable[[Path], dict]) -> tuple[str, list[str], dict, dict, str]:
    data_dir = args.tack_repo / "configs/data"
    if task == "pdc50" and args.pdc50_model == "mlp":
        data_config = load_yaml(data_dir / "simple-onehot.yaml")
        model_config = {
            "model_config": {
                "learning_rate": 0.00075,
                "hidden_dim": [256],
                "mlp_depth": 1,
                "head_depth": 1,
                "dropout": 0.0,
                "head_dropout": 0.0,
                "task_type": "point",
                "activation": "relu",
                "norm_type": "layer",
                "lr_scheduler_type": "reduce_on_plateau",
            },
            "training_config": {
                "max_epochs": 200,
                "patience": 5,
                "gradient_clip_val": 1.0,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        }
        method = "MLP-DC50 Cell-OneHot E3-OneHot Mol-Desc POI-OneHot Time"
        return "mlp", ["DC50"], data_config, model_config, method
    if task in {"pdc50", "dmax"}:
        data_config = load_yaml(data_dir / "cell_text_poi_countvec-onehot.yaml")
        model_config = {
            "model_config": {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "tree_method": "hist", "learning_rate": 0.0159,
                "max_depth": 7, "min_child_weight": 1, "gamma": 0.005,
                "subsample": 0.982, "colsample_bytree": 0.916,
                "reg_alpha": 0.007, "reg_lambda": 0.005,
                "n_jobs": args.threads,
            },
            "training_config": {"num_boost_round": 2000, "early_stopping_rounds": 30},
        }
        method = "XGB-DMAX Cell-Text E3-OneHot Mol-Desc POI-Vec Time"
        label = "DC50" if task == "pdc50" else "Dmax"
        if task == "pdc50" and getattr(args, "pdc50_target", "dc50-quantile") == "direct-pdc50":
            label = "pDC50"
            data_config.update(normalize_labels=False, standardize_labels=False)
        method = method.replace("XGB-DMAX", f"XGB-{label.upper()}")
        return "xgboost", [label], data_config, model_config, method

    data_config = load_yaml(data_dir / "bin-fp_cell_text_esms.yaml")
    data_config.update({
        "poi_embeddings_file": str(args.protein_embeddings),
        "ligase_embeddings_file": str(args.protein_embeddings),
        "poi_embeddings_format": "h5", "ligase_embeddings_format": "h5",
        "poi_embeddings_per_residue": False, "ligase_embeddings_per_residue": False,
        "poi_embeddings_id_type": "uniprot", "ligase_embeddings_id_type": "uniprot",
        "poi_pca_n_components": getattr(args, "activity_pca_dimensions", (44, 7))[0],
        "ligase_pca_n_components": getattr(args, "activity_pca_dimensions", (44, 7))[1],
    })
    model_config = {
        "model_config": {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "learning_rate": 0.0018,
            "max_depth": 6, "min_child_weight": 1, "gamma": 0.120,
            "subsample": 0.946, "colsample_bytree": 0.629,
            "reg_alpha": 0.447, "reg_lambda": 0.018,
            "n_jobs": args.threads,
        },
        "training_config": {"num_boost_round": 2000, "early_stopping_rounds": 30},
    }
    method = ("XGB-BIN Cell-Text E3-ESM-S FP Mol-Desc POI-ESM-S "
              "POI/E3-ESM-S-PCA Time")
    return "xgboost", ["Activity"], data_config, model_config, method


def run_official_training(corpus: str, task: Task, train: pd.DataFrame,
                          test: pd.DataFrame, args: argparse.Namespace,
                          run_cv_experiment: Callable[..., Any],
                          load_yaml: Callable[[Path], dict]) -> tuple[pd.DataFrame, dict]:
    if args.split_strategy == "repeated-holdout":
        from tack_holdout_training import run_repeated_holdout
        return run_repeated_holdout(corpus, task, train, test, args,
                                    run_cv_experiment, load_yaml, best_configs)
    from datasets import Dataset
    model_type, labels, data_config, model_config, method = best_configs(task, args, load_yaml)
    data_config = copy.deepcopy(data_config)
    data_config.update({"batch_size": args.batch_size, "num_workers": 0,
                        "num_proc": args.num_proc})
    run_dir = args.output_dir / "official_runs" / corpus.lower() / task
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOFFICIAL TACK RUN corpus={corpus} task={task} train={len(train)} "
          f"test={len(test)} method={method}", flush=True)
    official_globals = run_cv_experiment.__globals__
    original_splitter = official_globals["create_cv_splits"]
    selected_fold_ids = None if args.fold_ids is None else set(args.fold_ids)
    if selected_fold_ids is not None:
        def selected_splitter(*split_args: Any, **split_kwargs: Any):
            for split in original_splitter(*split_args, **split_kwargs):
                if split["fold"] in selected_fold_ids:
                    yield split
        official_globals["create_cv_splits"] = selected_splitter
    try:
        run_cv_experiment(
            dataset=Dataset.from_pandas(train, preserve_index=False),
            held_out_dataset=Dataset.from_pandas(test, preserve_index=False),
            model_type=model_type,
            data_config=data_config,
            model_config=copy.deepcopy(model_config),
            task_name="dc50" if task == "pdc50" else ("bin" if task == "activity" else task),
            labels=labels,
            group="scaffold",
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            checkpoints_dir=checkpoint_dir,
            results_dir=prediction_dir,
            tune_hyperparameters=False,
        )
    finally:
        official_globals["create_cv_splits"] = original_splitter
    files = sorted(prediction_dir.glob("*split=test.csv"))
    if selected_fold_ids is not None:
        files = [
            path for path in files
            if (match := re.search(r"-fold=(\d+)-split=test\.csv$", path.name))
            and int(match.group(1)) in selected_fold_ids
        ]
    expected = (args.n_splits * args.n_repeats
                if selected_fold_ids is None else len(selected_fold_ids))
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} test prediction files for {corpus}/{task}, "
                           f"found {len(files)}")
    frames = []
    for path in files:
        fold = pd.read_csv(path)
        required = {"_Benchmark_Row_ID", "DOI", "target", "pred", "fold"}
        if not required.issubset(fold.columns):
            raise ValueError(f"Malformed official TACK prediction file: {path}")
        fold["official_prediction_file"] = str(path)
        frames.append(fold)
    long = pd.concat(frames, ignore_index=True)
    if long.groupby("fold").size().nunique() != 1:
        raise AssertionError(f"Held-out row count changed across folds for {corpus}/{task}")
    provenance = {
        "corpus": corpus, "task": task, "paper_method": method,
        "official_model_type": model_type, "official_labels": labels,
        "data_config": data_config, "model_config": model_config,
        "prediction_files": len(files),
    }
    return long, provenance


def run_official_training_worker(payload: tuple[str, Task, pd.DataFrame,
                                                  pd.DataFrame, argparse.Namespace]):

    corpus, task, train, test, args = payload
    os.environ["TACKAI_CACHE"] = str(args.tack_cache)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    run_cv_experiment, _, load_yaml = load_official_tack(args.tack_repo)
    install_tack_acceleration(args.batch_featurization, read_only_cell_cache=True)
    return run_official_training(
        corpus, task, train, test, args, run_cv_experiment, load_yaml
    )


def pdc50_from_nm(values: np.ndarray) -> np.ndarray:
    return -np.log10(np.asarray(values, dtype=float) * 1e-9 + 1e-12)


def direct_pdc50_from_nm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Direct pDC50 requires finite positive DC50 values in nM")
    return 9.0 - np.log10(values)


def direct_pdc50_to_metric(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(-np.asarray(values, dtype=float) * np.log(10.0),
                        np.log(1e-12)) / np.log(10.0)


def interval_bounds(task: Task, values: np.ndarray,
                    operators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    values = np.asarray(values, dtype=float)
    operators = np.asarray(operators, dtype=object)
    lower = values.copy()
    upper = values.copy()
    for idx, raw_operator in enumerate(operators):
        operator = "" if pd.isna(raw_operator) else str(raw_operator).strip()
        if operator in {"", "~", "=", "==", "≈"}:
            continue
        if operator in {">", ">=", "≥"}:
            if task == "pdc50":
                lower[idx], upper[idx] = -np.inf, values[idx]
            elif task == "dmax":
                lower[idx], upper[idx] = values[idx], 100.0
            else:
                raise ValueError(f"Interval bounds are not defined for {task}")
        elif operator in {"<", "<=", "≤"}:
            if task == "pdc50":
                lower[idx], upper[idx] = values[idx], np.inf
            elif task == "dmax":
                lower[idx], upper[idx] = 0.0, values[idx]
            else:
                raise ValueError(f"Interval bounds are not defined for {task}")
        else:
            raise ValueError(f"Unsupported {task} endpoint operator: {raw_operator!r}")
    return lower, upper


def interval_errors(prediction: np.ndarray, lower: np.ndarray,
                    upper: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    return np.where(prediction < lower, lower - prediction,
                    np.where(prediction > upper, prediction - upper, 0.0))


def interval_metric_values(prediction: np.ndarray, lower: np.ndarray,
                           upper: np.ndarray) -> dict[str, float]:
    errors = interval_errors(prediction, lower, upper)
    return {
        "Interval_RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Interval_MAE": float(np.mean(errors)),
    }


def paired_interval_bootstrap(
    task: Task,
    values: np.ndarray,
    operators: np.ndarray,
    auto_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    clusters: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    lower, upper = interval_bounds(task, values, operators)
    observed_auto = interval_metric_values(auto_prediction, lower, upper)
    observed_baseline = interval_metric_values(baseline_prediction, lower, upper)
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters).astype(str)
    unique_clusters = np.unique(clusters)
    cluster_indices = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    }
    samples = {metric: [] for metric in observed_auto}
    for _ in range(n_boot):
        chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_indices[cluster] for cluster in chosen])
        auto = interval_metric_values(auto_prediction[idx], lower[idx], upper[idx])
        baseline = interval_metric_values(
            baseline_prediction[idx], lower[idx], upper[idx])
        for metric in samples:
            samples[metric].append(float(baseline[metric] - auto[metric]))

    exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
    rows = []
    for metric, auto_value in observed_auto.items():
        baseline_value = observed_baseline[metric]
        delta = baseline_value - auto_value
        boot = np.asarray(samples[metric], dtype=float)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

        left_tail = (int(np.sum(boot <= 0)) + 1) / (len(boot) + 1)
        right_tail = (int(np.sum(boot >= 0)) + 1) / (len(boot) + 1)
        p_two_sided = min(1.0, 2.0 * min(left_tail, right_tail))
        rows.append({
            "task": task,
            "metric": metric,
            "AutoTPDplus": auto_value,
            "PROTACDB": baseline_value,
            "delta_favoring_AutoTPDplus": delta,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_p_two_sided": p_two_sided,
            "n_test": len(values),
            "n_test_doi_clusters": len(unique_clusters),
            "n_exact": int(exact.sum()),
            "n_censored": int((~exact).sum()),
        })
    return pd.DataFrame(rows), lower, upper


def ensemble_predictions(long: pd.DataFrame, task: Task) -> pd.DataFrame:
    work = long.copy()
    if task == "pdc50":
        space = work.get("_Prediction_Space", pd.Series("dc50-nm", index=work.index))
        if space.eq("pdc50").all():
            work["metric_target"] = pdc50_from_nm(work["DC50"])
            work["metric_pred"] = direct_pdc50_to_metric(work["pred"])
        elif space.eq("dc50-nm").all():
            work["metric_target"] = pdc50_from_nm(work["target"])
            work["metric_pred"] = pdc50_from_nm(work["pred"])
        else:
            raise ValueError("Mixed or unknown pDC50 prediction spaces")
    else:
        work["metric_target"] = work["target"].astype(float)
        work["metric_pred"] = work["pred"].astype(float)
    return work.groupby("_Benchmark_Row_ID", sort=True).agg(
        DOI=("DOI", "first"), Compound_Name=("Compound_Name", "first"),
        SMILES=("SMILES", "first"), Connectivity_Key=("Connectivity_Key", "first"),
        value_operator=("Value_Operator", "first"),
        is_exact_primary=("_Primary_Exact", "first"), y_true=("metric_target", "first"),
        prediction=("metric_pred", "mean"), prediction_sd_across_folds=("metric_pred", "std"),
        n_fold_predictions=("metric_pred", "size"),
    ).reset_index()


def markdown_table(frame: pd.DataFrame, digits: int | None = None) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if digits is not None and isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def plot_results(output: Path, comparisons: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    selected = []
    for task, metric, label in [
        ("pdc50", "RMSE", "pDC50 RMSE"),
        ("dmax", "RMSE", "Dmax RMSE"),
        ("activity", "ROC_AUC", "Activity ROC-AUC"),
    ]:
        match = comparisons[(comparisons.task == task) & (comparisons.metric == metric)]
        if not match.empty:
            selected.append((task, label, match.iloc[0]))
    if not selected:
        return
    fig, axes = plt.subplots(2, len(selected), figsize=(max(4.3, 3.8 * len(selected)), 6.3),
                             height_ratios=[1.35, 1], squeeze=False)
    for col, (task, label, row) in enumerate(selected):
        top = axes[0, col]
        values = [row.AutoTPDplus, row.PROTACDB]
        bars = top.bar([0, 1], values, color=["#277DA1", "#F8961E"], width=0.62)
        top.set_xticks([0, 1], ["AutoTPDplus", "PROTACDB"])
        top.set_title(label)
        top.spines[["top", "right"]].set_visible(False)
        top.set_ylim(0, 1 if task == "activity" else max(values) * 1.25)
        if task == "activity":
            top.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        top.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        bottom = axes[1, col]
        delta, low, high = (float(row.delta_favoring_AutoTPDplus),
                            float(row.ci95_low), float(row.ci95_high))
        bottom.axvline(0, color="black", linewidth=1)
        bottom.hlines(0, low, high, color="#4D4D4D")
        bottom.plot(delta, 0, "o", color="#277DA1")
        padding = max((high - low) * 0.18, 0.01)
        bottom.set_xlim(low - padding, high + padding)
        bottom.set_yticks([])
        bottom.set_xlabel("Delta favoring AutoTPDplus")
        bottom.set_title(f"95% DOI-cluster CI: [{low:.3f}, {high:.3f}]", fontsize=10)
        bottom.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Official TACK benchmark", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "comparison_figure.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "comparison_figure.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(args: argparse.Namespace, audits: list[LeakageAudit], counts: pd.DataFrame,
                 comparisons: pd.DataFrame, interval_comparisons: pd.DataFrame,
                 provenance: list[dict], tack_commit: str) -> None:
    primary_names = {"pdc50": "RMSE", "dmax": "RMSE", "activity": "ROC_AUC"}
    primary = comparisons[
        comparisons.apply(lambda row: primary_names.get(row.task) == row.metric, axis=1)]
    supported = primary[(primary.ci95_low > 0) &
                        (primary.delta_favoring_AutoTPDplus > 0) &
                        (primary.bootstrap_p_holm_primary < 0.05)]
    conclusion = (
        "No primary endpoint meets both a positive DOI-bootstrap 95% CI and Holm-adjusted tail diagnostic < 0.05."
        if supported.empty
        else "AutoTPDplus improvements meeting the bootstrap CI and Holm diagnostic criteria: "
        + ", ".join(f"{row.task}/{row.metric}" for row in supported.itertuples())
        + "."
    )
    methods = pd.DataFrame([{
        "task": item["task"], "official_model": item["official_model_type"],
        "configured_method": item["paper_method"],
    } for item in provenance if item["corpus"] == "AutoTPDplus"])
    lines = [
        "# AutoTPDplus vs PROTACDB through official TACK code", "",
        f"TACK checkout: `{args.tack_repo}` at commit `{tack_commit}`.",
        (f"Official call: `tackai.training.run_cv_experiment`; "
         f"{args.samples} repeated {args.holdout_unit} holdouts, requested validation row fraction "
         f"{args.validation_fraction:g}, "
         f"{args.tuning_trials if args.hyperparameter_mode == 'tuned' else 0} Optuna trials per sample "
         f"(hyperparameter mode: {args.hyperparameter_mode}). "
         "Final models use only the training partition with validation early stopping."
         if args.split_strategy == "repeated-holdout" else
         f"Official call: `tackai.training.run_cv_experiment`; "
         f"{args.n_splits} scaffold folds × {args.n_repeats} repeats."),
        f"Leakage policy: `{args.leakage_policy}`. Positive deltas favor AutoTPDplus.",
        "", "## Bottom line", "", conclusion,
        "", "## Configured TACK methods", "", markdown_table(methods),
        "", "## Leakage audit", "", markdown_table(pd.DataFrame([asdict(x) for x in audits])),
        "", "## Samples passed to TACK", "", markdown_table(counts),
        "", "## Paired external-test results", "", markdown_table(comparisons, digits=4),
    ]
    if not interval_comparisons.empty:
        lines.extend([
            "", "## Supplemental interval-aware results", "",
            "Censored observations are treated as allowed intervals. Error is zero inside "
            "the reported interval and equals distance to the nearest bound outside it. "
            "These metrics supplement rather than replace exact-only RMSE/MAE.", "",
            markdown_table(interval_comparisons, digits=4),
        ])
    lines.extend([
        "", "## Evaluation details", "",
        ("- pDC50 is trained directly as 9-log10(DC50 in nM), with no label normalization or standardization. Early stopping uses pDC50 RMSE. The previous evaluator's 1e-12 M offset is retained only in reported metrics for comparability."
         if args.pdc50_target == "direct-pdc50" else
         "- DC50 labels are quantile-normalized by the official TACK pipeline, inverse-transformed to nM, then converted to pDC50 for evaluation."),
        "- Dmax is clipped to [0, 100], matching TACK's CLI preprocessing.",
        "- Regression primary metrics exclude censored test bounds; approximate (`~`) values are retained.",
        "- Supplemental interval-aware errors use all task-eligible exact and censored observations.",
        "- Confidence intervals resample paired DOI clusters on the fixed external test set; they condition on the fitted ensemble and do not measure all training-data uncertainty.",
        "- Bootstrap tail proportions are diagnostics, not exact hypothesis-test p-values. Holm-adjusted diagnostics are provided across primary metrics.",
        "- Per-repeat metrics and their mean/SD describe training-split variability. The overlapping training sets and shared test set are not 25 independent experiments.",
        ("- Compound holdout groups records sharing canonical SMILES or Connectivity_Key. Complete groups are selected using only random order and group size to approach the requested validation row fraction. Compound overlap is zero; scaffold overlap is allowed."
         if args.holdout_unit == "compound" else
         "- Repeated holdout uses random rows (stratified for Activity), so it does not enforce scaffold separation between training and validation."),
        "- The adapter handles schema conversion, leakage removal, scaffold IDs, ensembling, and metrics only; model fitting and fold prediction use official TACK code.",
    ])
    if "activity" in args.tasks:
        lines.extend([
            "- Activity labels are produced by official `scripts/train_models.py::get_bin_label`.",
            "- Activity retains the paper-selected ESM-S/PCA feature combination and the local UniProt embedding artifact. On smaller training corpora, PCA dimensions are capped to the common nonzero-rank bound across both corpora and every training partition; see activity_pca_audit.json.",
        ])
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    descriptor_count = len(Descriptors._descList)
    fragment_descriptor_count = sum(
        name.startswith("fr_") for name, _ in Descriptors._descList
    )
    if descriptor_count != 217 or fragment_descriptor_count != 85:
        raise RuntimeError(
            f"TACK's 217 RDKit descriptors were not loaded: found {descriptor_count} "
            f"total and {fragment_descriptor_count} fragment descriptors; "
            f"RDKit data directory is {RDConfig.RDDataDir}"
        )
    descriptor_cache = args.tack_cache / "rdkit_descriptors.npz"
    descriptor_cache_entries_before = validate_descriptor_cache(
        descriptor_cache, descriptor_count
    )
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if args.outer_workers not in {1, 2}:
        raise ValueError("outer-workers must be 1 or 2 (there are two corpora)")
    if args.samples < 1 or not 0 < args.validation_fraction < 1 or args.tuning_trials < 1:
        raise ValueError("samples/trials must be positive and validation-fraction in (0, 1)")
    if args.split_strategy == "repeated-holdout" and args.pdc50_model != "xgboost":
        raise ValueError("Repeated holdout requires shared XGBoost regression configuration")
    if args.pdc50_target == "direct-pdc50" and (
            args.split_strategy != "repeated-holdout" or args.pdc50_model != "xgboost"
            or args.hyperparameter_mode != "fixed"):
        raise ValueError("Direct pDC50 control requires fixed-parameter XGBoost repeated holdout")
    if args.holdout_unit == "compound" and (
            args.tasks != ["pdc50"] or args.pdc50_target != "direct-pdc50"
            or args.split_reference is None):
        raise ValueError("Compound control requires pDC50-only direct targets and a completed row-holdout reference")
    total_folds = (args.samples if args.split_strategy == "repeated-holdout"
                   else args.n_splits * args.n_repeats)
    if args.fold_ids is not None:
        if len(set(args.fold_ids)) != len(args.fold_ids):
            raise ValueError("--fold-ids must be unique")
        invalid = [fold for fold in args.fold_ids if not 0 <= fold < total_folds]
        if invalid:
            raise ValueError(f"--fold-ids out of range [0, {total_folds}): {invalid}")
    if args.n_splits < 2 or args.n_repeats < 1:
        raise ValueError("n-splits must be >=2 and n-repeats must be >=1")
    for path in [args.tack_repo, args.autotpdplus, args.protacdb, args.test,
                 args.tack_cache, args.protein_embeddings]:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "adapted_data").mkdir(parents=True, exist_ok=True)
    os.environ["TACKAI_CACHE"] = str(args.tack_cache)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tack-mpl")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    run_cv_experiment, activity_labeler, load_yaml = load_official_tack(args.tack_repo)
    install_tack_acceleration(args.batch_featurization)
    source_file = Path(inspect.getsourcefile(run_cv_experiment) or "")
    expected_source = (args.tack_repo / "tackai/training.py").resolve()
    if source_file.resolve() != expected_source:
        raise RuntimeError(f"TACK import resolved to {source_file}, expected {expected_source}")
    tack_commit = subprocess.check_output(
        ["git", "-C", str(args.tack_repo), "rev-parse", "HEAD"], text=True).strip()

    raw_train = {"AutoTPDplus": pd.read_csv(args.autotpdplus),
                 "PROTACDB": pd.read_csv(args.protacdb)}
    raw_test = pd.read_csv(args.test)
    required = {"DOI", "SMILES", "Connectivity_Key", "Compound_Name",
                "Degradation_Target", "Degradation_Target_Gene",
                "Degradation_Target_Uniprot", "Degradation_Target_Sequence",
                "Recruiter", "Recruiter_Gene", "Recruiter_Uniprot", "Recruiter_Sequence",
                "Cell_Line", "Cell_Line_STD", "Cell_Line_STD_Accession", "Cell_Line_Species",
                "Assay", "Assay_STD", "DC50", "DC50_h", "DC50_operator",
                "Dmax", "Dmax_h", "Dmax_operator"}
    for name, frame in {**raw_train, "test": raw_test}.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks required columns: {missing}")

    from tack_holdout_training import guard_run_manifest
    guard_run_manifest(args)
    filtered, audits = {}, []
    for corpus, frame in raw_train.items():
        filtered[corpus], audit = filter_leakage(frame, raw_test, corpus,
                                                 args.leakage_policy)
        audits.append(audit)
    cached_molecules = cache_molecular_descriptors([*filtered.values(), raw_test])
    pd.DataFrame([asdict(x) for x in audits]).to_csv(
        args.output_dir / "leakage_audit.csv", index=False)
    with h5py.File(args.protein_embeddings, "r") as handle:
        protein_ids = set(handle.keys())

    prepared_test: dict[Task, pd.DataFrame] = {}
    count_rows: list[dict[str, Any]] = []
    for task in args.tasks:
        prepared_test[task], coverage = prepare_task(
            raw_test, task, "held-out test", activity_labeler, protein_ids, True,
            require_poi_sequence=(task == "pdc50" and args.pdc50_model == "xgboost"))
        prepared_test[task].to_csv(
            args.output_dir / "adapted_data" / f"heldout_{task}.csv", index=False)
        count_rows.append({"corpus": "held-out test", "task": task,
                           "rows": len(prepared_test[task]),
                           "exact_primary": int(prepared_test[task]["_Primary_Exact"].sum()),
                           **coverage})

    prepared_train = {}
    for task in args.tasks:
        for corpus in ["AutoTPDplus", "PROTACDB"]:
            train, coverage = prepare_task(filtered[corpus], task, corpus,
                activity_labeler, protein_ids, False,
                require_poi_sequence=(task == "pdc50" and args.pdc50_model == "xgboost"))
            prepared_train[(corpus, task)] = train
            train.to_csv(args.output_dir / "adapted_data" /
                         f"{corpus.lower()}_{task}_train.csv", index=False)
            count_rows.append({"corpus": corpus, "task": task, "rows": len(train),
                               "exact_primary": int(train["_Primary_Exact"].sum()), **coverage})
    if "activity" in args.tasks and args.split_strategy == "repeated-holdout":
        from tack_holdout_training import shared_activity_pca_dimensions
        args.activity_pca_dimensions, pca_audit = shared_activity_pca_dimensions(
            {corpus: prepared_train[(corpus, "activity")]
             for corpus in ["AutoTPDplus", "PROTACDB"]}, args)
        (args.output_dir / "activity_pca_audit.json").write_text(
            json.dumps(pca_audit, indent=2) + "\n")
        print(f"Shared Activity POI/E3 PCA dimensions: {args.activity_pca_dimensions}", flush=True)
    pd.DataFrame(count_rows).to_csv(args.output_dir / "sample_counts.csv", index=False)
    cached_cell_values = 0
    if "dmax" in args.tasks:
        cached_cell_values = cache_cell_embeddings([
            prepared_train[(corpus, "dmax")] for corpus in ["AutoTPDplus", "PROTACDB"]
        ] + [prepared_test["dmax"]])
    if args.split_reference is not None:
        from tack_holdout_training import validate_reference_experiment
        validate_reference_experiment(args, prepared_train, prepared_test, load_yaml, best_configs)
    long_outputs, provenance = [], []
    ensembles: dict[tuple[str, Task], pd.DataFrame] = {}
    for task in args.tasks:
        corpora = ["AutoTPDplus", "PROTACDB"]
        if args.outer_workers == 1:
            task_results = [run_official_training(
                corpus, task, prepared_train[(corpus, task)], prepared_test[task], args,
                run_cv_experiment, load_yaml) for corpus in corpora]
        else:
            payloads = [
                (corpus, task, prepared_train[(corpus, task)], prepared_test[task], args)
                for corpus in corpora
            ]
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.outer_workers) as executor:
                task_results = list(executor.map(run_official_training_worker, payloads))
        for corpus, (long, run_info) in zip(corpora, task_results):
            long.insert(0, "corpus", corpus)
            long.insert(1, "benchmark_task", task)
            long_outputs.append(long)
            ensembles[(corpus, task)] = ensemble_predictions(long, task)
            provenance.append(run_info)

    from tack_holdout_training import paired_bootstrap, save_repeat_metrics
    comparisons, interval_comparisons, prediction_rows = [], [], []
    for task in args.tasks:
        auto = ensembles[("AutoTPDplus", task)]
        protacdb = ensembles[("PROTACDB", task)]
        key = "_Benchmark_Row_ID"
        merged = auto.merge(
            protacdb[[key, "prediction", "prediction_sd_across_folds"]],
            on=key, how="inner", validate="one_to_one",
            suffixes=("_AutoTPDplus", "_PROTACDB"))
        if len(merged) != len(prepared_test[task]):
            raise AssertionError(f"Prediction alignment failed for {task}")
        merged.insert(0, "task", task)
        merged = merged.rename(columns={
            "prediction_AutoTPDplus": "AutoTPDplus_prediction",
            "prediction_PROTACDB": "PROTACDB_prediction",
            "prediction_sd_across_folds_AutoTPDplus": "AutoTPDplus_fold_sd",
            "prediction_sd_across_folds_PROTACDB": "PROTACDB_fold_sd",
        })
        primary = (merged["is_exact_primary"].astype(bool).to_numpy()
                   if task != "activity" else np.ones(len(merged), dtype=bool))
        comparison = paired_bootstrap(
            task, merged.loc[primary, "y_true"].to_numpy(float),
            merged.loc[primary, "AutoTPDplus_prediction"].to_numpy(float),
            merged.loc[primary, "PROTACDB_prediction"].to_numpy(float),
            merged.loc[primary, "DOI"].astype(str).to_numpy(),
            args.bootstrap, args.seed + 10_000)
        if task != "activity":
            comparison = comparison[
                comparison["metric"].isin(["RMSE", "MAE"])
            ].reset_index(drop=True)
        comparison.to_csv(args.output_dir / f"metrics_{task}.csv", index=False)
        comparisons.append(comparison)
        print("\n" + comparison.to_string(index=False), flush=True)

        if task != "activity":
            interval_comparison, lower, upper = paired_interval_bootstrap(
                task,
                merged["y_true"].to_numpy(float),
                merged["value_operator"].to_numpy(object),
                merged["AutoTPDplus_prediction"].to_numpy(float),
                merged["PROTACDB_prediction"].to_numpy(float),
                merged["DOI"].astype(str).to_numpy(),
                args.bootstrap,
                args.seed + 20_000,
            )
            merged["interval_lower"] = lower
            merged["interval_upper"] = upper
            merged["AutoTPDplus_interval_error"] = interval_errors(
                merged["AutoTPDplus_prediction"].to_numpy(float), lower, upper)
            merged["PROTACDB_interval_error"] = interval_errors(
                merged["PROTACDB_prediction"].to_numpy(float), lower, upper)
            interval_comparisons.append(interval_comparison)
            print("\n" + interval_comparison.to_string(index=False), flush=True)
        prediction_rows.append(merged)

    comparison_df, count_df = pd.concat(comparisons, ignore_index=True), pd.DataFrame(count_rows)
    interval_comparison_df = (
        pd.concat(interval_comparisons, ignore_index=True)
        if interval_comparisons else pd.DataFrame()
    )
    from tack_holdout_training import holm_adjust
    primary_mask = comparison_df.apply(
        lambda row: row.metric in ({"ROC_AUC"} if row.task == "activity" else {"RMSE", "MAE"}), axis=1)
    comparison_df["bootstrap_p_holm_primary"] = np.nan
    primary_p = comparison_df.loc[primary_mask, "bootstrap_p_two_sided"].to_numpy()
    if args.pdc50_target == "direct-pdc50" and args.split_reference is not None:
        reference_frames, visited = [], set()
        reference_dir = args.split_reference
        while reference_dir is not None:
            reference_dir = reference_dir.resolve()
            if reference_dir in visited:
                raise ValueError("Cyclic reference experiment chain")
            visited.add(reference_dir)
            reference_frames.append(pd.read_csv(reference_dir / "metrics_all.csv"))
            reference_manifest = json.loads((reference_dir / "run_manifest.json").read_text())
            parent_reference = reference_manifest['arguments'].get('split_reference')
            reference_dir = Path(parent_reference) if parent_reference else None
        reference_metrics = pd.concat(reference_frames).drop_duplicates(['task', 'metric'], keep='first')
        unchanged = reference_metrics[
            ~reference_metrics.task.isin(args.tasks)
            & reference_metrics.apply(lambda row: row.metric in (
                {"ROC_AUC"} if row.task == "activity" else {"RMSE", "MAE"}), axis=1)]
        correction_p = np.r_[primary_p, unchanged.bootstrap_p_two_sided.to_numpy()]
    else:
        correction_p = primary_p
    comparison_df.loc[primary_mask, "bootstrap_p_holm_primary"] = holm_adjust(
        correction_p)[:len(primary_p)]
    (args.output_dir / "multiplicity_audit.json").write_text(json.dumps({
        "primary_comparison_count": len(correction_p),
        "current_comparison_count": len(primary_p),
        "raw_bootstrap_tail_diagnostics": correction_p.tolist(),
        "note": "Unchanged endpoints, if any, are inherited from the reference experiment chain.",
    }, indent=2) + "\n")
    comparison_df.to_csv(args.output_dir / "metrics_all.csv", index=False)
    rmse_mae_comparison = comparison_df[
        comparison_df["metric"].isin(["RMSE", "MAE"])
    ][["task", "metric", "AutoTPDplus", "PROTACDB"]].copy()
    rmse_mae_comparison["absolute_error_reduction"] = (
        rmse_mae_comparison["PROTACDB"] - rmse_mae_comparison["AutoTPDplus"]
    )
    rmse_mae_comparison["relative_error_reduction_percent"] = (
        100 * rmse_mae_comparison["absolute_error_reduction"]
        / rmse_mae_comparison["PROTACDB"]
    )
    rmse_mae_comparison.to_csv(
        args.output_dir / "rmse_mae_comparison.csv", index=False)
    interval_comparison_df.to_csv(
        args.output_dir / "metrics_interval_aware.csv", index=False)
    count_df.to_csv(args.output_dir / "sample_counts.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "test_predictions.csv", index=False)
    pd.concat(long_outputs, ignore_index=True).to_csv(
        args.output_dir / "official_fold_predictions.csv", index=False)
    official_paths = [args.tack_repo / "tackai/training.py",
                      args.tack_repo / "tackai/data/datamodule.py",
                      args.tack_repo / "scripts/train_models.py",
                      args.tack_repo / "scripts/plot_evaluation_figures.py"]
    code_audit = {
        "tack_repo": str(args.tack_repo.resolve()), "tack_git_commit": tack_commit,
        "run_cv_experiment_source": str(source_file.resolve()),
        "run_cv_experiment_module": run_cv_experiment.__module__,
        "official_source_sha256": {str(p.relative_to(args.tack_repo)): sha256(p)
                                   for p in official_paths},
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "rdkit_version": rdBase.rdkitVersion,
        "rdkit_data_dir": RDConfig.RDDataDir,
        "rdkit_descriptor_count": descriptor_count,
        "rdkit_fragment_descriptor_count": fragment_descriptor_count,
        "rdkit_descriptor_names_sha256": hashlib.sha256(
            "\n".join(name for name, _ in Descriptors._descList).encode("utf-8")
        ).hexdigest(),
        "descriptor_cache_entries_before": descriptor_cache_entries_before,
        "descriptor_cache_entries_after": validate_descriptor_cache(
            descriptor_cache, descriptor_count
        ),
        "cached_descriptor_molecules": cached_molecules,
        "cached_cell_text_values": cached_cell_values,
        "acceleration": {
            "batch_featurization": args.batch_featurization,
            "outer_workers": args.outer_workers,
            "threads_per_model": args.threads,
            "xgboost_matrix_scope": (
                "one DMatrix build per fold; fixed paper configuration has no "
                "hyperparameter candidates requiring further matrix reuse"
            ),
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "official_runs": provenance,
    }
    (args.output_dir / "official_code_audit.json").write_text(
        json.dumps(code_audit, indent=2) + "\n")
    save_repeat_metrics(pd.concat(long_outputs, ignore_index=True), args.output_dir)
    write_report(args, audits, count_df, comparison_df, interval_comparison_df,
                 provenance, tack_commit)
    plot_results(args.output_dir, comparison_df)
    if args.split_reference is not None:
        from tack_holdout_training import write_reference_comparison
        write_reference_comparison(args)
    (args.output_dir / "COMPLETED.json").write_text(json.dumps({
        "tasks": args.tasks, "split_strategy": args.split_strategy,
        "sample_ids": args.fold_ids if args.fold_ids is not None else list(range(total_folds)),
        "final_models": sum(p["prediction_files"] for p in provenance),
    }, indent=2) + "\n")

if __name__ == "__main__":
    main()
