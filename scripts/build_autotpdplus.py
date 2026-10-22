#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ID_COLUMN = "AutoTPD_ID"
ARTICLE_TYPE_COLUMN = "Source_Article_Type"
ID_WIDTH = 6

IDENTITY_COLUMNS = [
    "DOI",
    "Compound_Name",
    "Degradation_Target",
    "Cell_Line",
    "DC50",
    "DC50_units",
    "Dmax",
]

MODALITIES = {
    "MG": {"prefix": "AUTOTPD-MG", "renames": {"Standard_InChIKey": "StdInChIKey"}},
    "PROTAC": {"prefix": "AUTOTPD-PROTAC", "renames": {"Standard_InChIKey": "InChIKey"}},
}

RETIRED_COLUMNS = [ID_COLUMN, "DOI", "retired_at", "reason", "replaced_by_run"]
CONFLICT_COLUMNS = [
    "reason", "DOI", "Compound_Name", "Degradation_Target", "Cell_Line",
    "DC50", "DC50_units", "Dmax", "source", "existing_AutoTPD_IDs",
]
MISSING_DOI_TOKENS = {"", "unknown", "nan", "none", "null"}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:", "", text).strip()
    return "" if text in MISSING_DOI_TOKENS else text


def format_id(prefix: str, number: int) -> str:
    return f"{prefix}-{number:0{ID_WIDTH}d}"


def id_number(value: str, prefix: str) -> int:
    head, _, tail = str(value).rpartition("-")
    if head != prefix or not tail.isdigit():
        raise ValueError(f"{ID_COLUMN} {value!r} does not belong to series {prefix}")
    return int(tail)


def sort_by_id(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    order = df[ID_COLUMN].map(lambda value: id_number(value, prefix))
    return df.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def load_article_types(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("articles"), list):
        data = {
            article.get("doi"): article.get("article_type")
            for article in data["articles"]
            if article.get("article_type") not in (None, "", "unknown")
        }
    return {normalize_doi(doi): str(kind or "") for doi, kind in data.items() if normalize_doi(doi)}


def annotate_article_type(df: pd.DataFrame, article_types: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    looked_up = df["DOI"].map(lambda doi: article_types.get(normalize_doi(doi), ""))
    if ARTICLE_TYPE_COLUMN in df.columns:
        df[ARTICLE_TYPE_COLUMN] = df[ARTICLE_TYPE_COLUMN].where(df[ARTICLE_TYPE_COLUMN] != "", looked_up)
    else:
        df[ARTICLE_TYPE_COLUMN] = looked_up
    return df


def conform_schema(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dropped = [column for column in df.columns if column not in columns]
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns], dropped


def ordered_union(*column_lists: list[str]) -> list[str]:
    columns: list[str] = []
    for column_list in column_lists:
        columns.extend(column for column in column_list if column not in columns)
    return columns


def conflict_rows(df: pd.DataFrame, reason: str, existing_ids: dict[str, list[str]] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    out = pd.DataFrame({"reason": reason}, index=df.index)
    for column in CONFLICT_COLUMNS[1:-1]:
        out[column] = df[column] if column in df.columns else ""
    ids = existing_ids or {}
    out["existing_AutoTPD_IDs"] = df["DOI"].map(
        lambda doi: ";".join(ids.get(normalize_doi(doi), [])[:5])
    )
    return out[CONFLICT_COLUMNS].reset_index(drop=True)


def prepare_source(df: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    df = df.rename(columns={old: new for old, new in renames.items() if old in df.columns})
    return df.drop(columns=[ID_COLUMN], errors="ignore")


@dataclass
class BuildResult:
    release: pd.DataFrame
    next_id: int
    report: dict[str, Any]
    conflicts: pd.DataFrame
    retired: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RETIRED_COLUMNS))


def bootstrap(
    autotpd: pd.DataFrame,
    large_scale: list[pd.DataFrame],
    baseline: pd.DataFrame,
    *,
    modality: str,
    article_types: dict[str, str] | None = None,
) -> BuildResult:
    prefix = MODALITIES[modality]["prefix"]
    renames = MODALITIES[modality]["renames"]
    article_types = article_types or {}

    autotpd = autotpd.rename(columns={old: new for old, new in renames.items() if old in autotpd.columns})
    if ID_COLUMN not in autotpd.columns or (autotpd[ID_COLUMN] == "").any():
        raise ValueError("every AutoTPD row needs an AutoTPD_ID")
    if not autotpd[ID_COLUMN].is_unique:
        raise ValueError("AutoTPD_ID values in AutoTPD are not unique")
    autotpd_numbers = autotpd[ID_COLUMN].map(lambda value: id_number(value, prefix))

    autotpd_dois = set(autotpd["DOI"].map(normalize_doi)) - {""}

    conflicts = [pd.DataFrame(columns=CONFLICT_COLUMNS)]
    large_scale_kept: list[pd.DataFrame] = []
    large_scale_dois: set[str] = set()
    replaced_by_autotpd = 0
    for frame in large_scale:
        frame = prepare_source(frame, renames)
        doi = frame["DOI"].map(normalize_doi)
        missing = doi == ""
        conflicts.append(conflict_rows(frame[missing], "large_scale_missing_doi"))
        in_autotpd = doi.isin(autotpd_dois) & ~missing
        replaced_by_autotpd += int(in_autotpd.sum())

        in_earlier_batch = doi.isin(large_scale_dois) & ~missing & ~in_autotpd
        conflicts.append(conflict_rows(frame[in_earlier_batch], "large_scale_doi_in_earlier_batch"))
        keep = ~missing & ~in_autotpd & ~in_earlier_batch
        large_scale_kept.append(frame[keep])
        large_scale_dois |= set(doi[keep])
    large_scale_df = (
        pd.concat(large_scale_kept, ignore_index=True)
        if large_scale_kept else pd.DataFrame(columns=["DOI"])
    )

    baseline = prepare_source(baseline, renames)
    baseline_doi = baseline["DOI"].map(normalize_doi)
    baseline_replaced = baseline_doi.isin(autotpd_dois | large_scale_dois) & (baseline_doi != "")
    baseline_kept = baseline[~baseline_replaced]

    columns = ordered_union([ID_COLUMN], list(baseline.columns), list(autotpd.columns), [ARTICLE_TYPE_COLUMN])
    autotpd_part, _ = conform_schema(autotpd, columns)
    large_scale_part, large_scale_dropped = conform_schema(large_scale_df, columns)
    baseline_part, _ = conform_schema(baseline_kept, columns)

    next_number = int(autotpd_numbers.max()) + 1
    large_scale_part = large_scale_part.copy()
    large_scale_part[ID_COLUMN] = [format_id(prefix, next_number + i) for i in range(len(large_scale_part))]
    next_number += len(large_scale_part)
    baseline_part = baseline_part.copy()
    baseline_part[ID_COLUMN] = [format_id(prefix, next_number + i) for i in range(len(baseline_part))]
    next_number += len(baseline_part)

    release = pd.concat([autotpd_part, large_scale_part, baseline_part], ignore_index=True)
    release = annotate_article_type(release, article_types)
    release = sort_by_id(release, prefix)

    conflicts_df = pd.concat(conflicts, ignore_index=True)
    report = {
        "mode": "bootstrap",
        "modality": modality,
        "sources": {
            "autotpd": {"rows": len(autotpd), "dois": len(autotpd_dois)},
            "large_scale": {
                "rows_in": int(sum(len(frame) for frame in large_scale)),
                "rows_kept": len(large_scale_part),
                "dois_kept": len(large_scale_dois),
                "rows_superseded_by_autotpd": replaced_by_autotpd,
                "dropped_columns": large_scale_dropped,
            },
            "baseline": {
                "rows_in": len(baseline),
                "rows_kept": len(baseline_part),
                "rows_superseded": int(baseline_replaced.sum()),
            },
        },
        "release_rows": len(release),
        "id_range": [release[ID_COLUMN].iloc[0], release[ID_COLUMN].iloc[-1]] if len(release) else [],
        "next_id": next_number,
        "conflict_rows": conflicts_df["reason"].value_counts().to_dict(),
        "article_type_rows": release[ARTICLE_TYPE_COLUMN].replace("", "(unknown)").value_counts().to_dict(),
    }
    errors = validate_release(
        previous=None, release=release, prefix=prefix,
        retired_ids=set(), next_id_before=None, next_id_after=next_number,
    )
    report["validation_errors"] = errors
    return BuildResult(release=release, next_id=next_number, report=report, conflicts=conflicts_df)


def incremental(
    previous: pd.DataFrame,
    batch: pd.DataFrame,
    *,
    modality: str,
    next_id: int,
    run_label: str,
    retired_ids: set[str] | None = None,
    replace_dois: set[str] | None = None,
    rejected_dois: set[str] | None = None,
    article_types: dict[str, str] | None = None,
) -> BuildResult:
    prefix = MODALITIES[modality]["prefix"]
    renames = MODALITIES[modality]["renames"]
    retired_ids = retired_ids or set()
    replace_dois = {normalize_doi(doi) for doi in (replace_dois or set())} - {""}
    rejected_dois = {normalize_doi(doi) for doi in (rejected_dois or set())} - {""}
    article_types = article_types or {}

    if ID_COLUMN not in previous.columns:
        raise ValueError("previous release has no AutoTPD_ID column; run bootstrap first")

    batch = prepare_source(batch, renames)
    batch_doi = batch["DOI"].map(normalize_doi) if len(batch) else pd.Series(dtype=str)

    previous_doi = previous["DOI"].map(normalize_doi)
    ids_by_doi: dict[str, list[str]] = {}
    for doi, record_id in zip(previous_doi, previous[ID_COLUMN]):
        if doi:
            ids_by_doi.setdefault(doi, []).append(record_id)

    rejected = batch_doi.isin(rejected_dois)
    missing = (batch_doi == "") & ~rejected
    existing = batch_doi.isin(list(ids_by_doi)) & ~rejected & ~missing
    replacing = existing & batch_doi.isin(replace_dois)
    skipped_existing = existing & ~replacing
    candidates = batch[~rejected & ~missing & ~skipped_existing]
    duplicated = candidates.duplicated(keep="first")
    new_rows = candidates[~duplicated]

    conflicts_df = pd.concat([
        conflict_rows(batch[missing], "missing_doi"),
        conflict_rows(batch[skipped_existing], "doi_already_in_release", ids_by_doi),
    ], ignore_index=True)

    replaced_dois = set(batch_doi[replacing])
    requested_but_absent = sorted(replace_dois - replaced_dois)
    retire_mask = previous_doi.isin(replaced_dois)
    retired = pd.DataFrame({
        ID_COLUMN: previous.loc[retire_mask, ID_COLUMN],
        "DOI": previous.loc[retire_mask, "DOI"],
        "retired_at": run_label,
        "reason": "doi_replaced_by_reextraction",
        "replaced_by_run": run_label,
    })[RETIRED_COLUMNS].reset_index(drop=True)

    columns = list(previous.columns)
    if ARTICLE_TYPE_COLUMN not in columns:
        columns.append(ARTICLE_TYPE_COLUMN)
    new_part, dropped_columns = conform_schema(new_rows, columns)
    new_part = annotate_article_type(new_part, article_types)
    new_part[ID_COLUMN] = [format_id(prefix, next_id + i) for i in range(len(new_part))]
    next_id_after = next_id + len(new_part)

    kept_previous, _ = conform_schema(previous[~retire_mask], columns)
    release = sort_by_id(pd.concat([kept_previous, new_part], ignore_index=True), prefix)

    errors = validate_release(
        previous=previous, release=release, prefix=prefix,
        retired_ids=retired_ids | set(retired[ID_COLUMN]),
        next_id_before=next_id, next_id_after=next_id_after,
    )
    if requested_but_absent:
        errors.append(f"--replace_doi given for DOIs not in this batch: {requested_but_absent}")

    report = {
        "mode": "incremental",
        "modality": modality,
        "run_label": run_label,
        "previous_rows": len(previous),
        "batch_rows": len(batch),
        "batch_dois": len(set(batch_doi) - {""}),
        "rejected_by_reviewer_rows": int(rejected.sum()),
        "conflict_rows": conflicts_df["reason"].value_counts().to_dict(),
        "exact_duplicate_rows_skipped": int(duplicated.sum()),
        "replaced_dois": sorted(replaced_dois),
        "retired_rows": len(retired),
        "appended_rows": len(new_part),
        "appended_dois": len(set(new_part["DOI"].map(normalize_doi)) - {""}),
        "release_rows": len(release),
        "rows_reconcile": len(release) == len(previous) - len(retired) + len(new_part),
        "new_id_range": [new_part[ID_COLUMN].iloc[0], new_part[ID_COLUMN].iloc[-1]] if len(new_part) else [],
        "next_id_before": next_id,
        "next_id_after": next_id_after,
        "dropped_batch_columns": dropped_columns,
        "appended_article_type_rows": new_part[ARTICLE_TYPE_COLUMN].replace("", "(unknown)").value_counts().to_dict(),
        "validation_errors": errors,
    }
    return BuildResult(release=release, next_id=next_id_after, report=report, conflicts=conflicts_df, retired=retired)


def validate_release(
    *,
    previous: pd.DataFrame | None,
    release: pd.DataFrame,
    prefix: str,
    retired_ids: set[str],
    next_id_before: int | None,
    next_id_after: int,
) -> list[str]:
    errors: list[str] = []
    ids = release[ID_COLUMN]
    if (ids == "").any():
        errors.append(f"{int((ids == '').sum())} rows without {ID_COLUMN}")
    if not ids.is_unique:
        errors.append(f"duplicate {ID_COLUMN}: {sorted(ids[ids.duplicated()].unique())[:10]}")
    try:
        numbers = ids.map(lambda value: id_number(value, prefix))
    except ValueError as exc:
        return errors + [str(exc)]
    if len(numbers) and numbers.max() >= next_id_after:
        errors.append(f"next_id {next_id_after} is not above the largest ID {numbers.max()}")
    reused = sorted(set(ids) & retired_ids)
    if reused:
        errors.append(f"retired IDs present in release: {reused[:10]}")

    if previous is not None:
        previous_ids = set(previous[ID_COLUMN])
        release_ids = set(ids)
        vanished = sorted(previous_ids - release_ids - retired_ids)
        if vanished:
            errors.append(f"IDs dropped without retirement: {vanished[:10]}")
        new_ids = release_ids - previous_ids
        if next_id_before is not None:
            below = sorted(i for i in new_ids if id_number(i, prefix) < next_id_before)
            if below:
                errors.append(f"new IDs below next_id {next_id_before}: {below[:10]}")
            if len(new_ids) != next_id_after - next_id_before:
                errors.append("number of new IDs does not match the next_id advance")
        carried = previous[previous[ID_COLUMN].isin(release_ids)].set_index(ID_COLUMN)
        current = release.set_index(ID_COLUMN).loc[carried.index]
        identity = [column for column in IDENTITY_COLUMNS if column in carried.columns]
        changed = (carried[identity] != current[identity]).any(axis=1)
        if changed.any():
            errors.append(f"identity fields changed for frozen IDs: {sorted(changed[changed].index)[:10]}")
    return errors


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_retired(path: Path) -> pd.DataFrame:
    if path.exists():
        return read_table(path)
    return pd.DataFrame(columns=RETIRED_COLUMNS)


def read_doi_list(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        normalize_doi(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    } - {""}


def write_outputs(result: BuildResult, args: argparse.Namespace) -> None:
    write_table(result.release, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result.report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.conflicts:
        write_table(result.conflicts, args.conflicts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AutoTPD+ (bootstrap or monthly incremental merge).")
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--modality", choices=sorted(MODALITIES), required=True)
        p.add_argument("--output", type=Path, required=True, help="AutoTPD+ CSV to write.")
        p.add_argument("--state", type=Path, required=True, help="JSON holding next_id per modality.")
        p.add_argument("--article_types", type=Path, default=None, help="JSON {doi: article_type}.")
        p.add_argument("--report", type=Path, default=None)
        p.add_argument("--conflicts", type=Path, default=None)

    b = sub.add_parser("bootstrap", help="One-off: AutoTPD + large-scale + baseline -> AutoTPD+.")
    common(b)
    b.add_argument("--autotpd", type=Path, required=True)
    b.add_argument("--large_scale", type=Path, nargs="*", default=[])
    b.add_argument("--baseline", type=Path, required=True)
    b.add_argument("--force", action="store_true", help="Overwrite an existing next_id in --state.")

    i = sub.add_parser("incremental", help="Previous AutoTPD+ + one standardized batch -> AutoTPD+.")
    common(i)
    i.add_argument("--previous", type=Path, required=True)
    i.add_argument("--batch", type=Path, required=True)
    i.add_argument("--retired", type=Path, required=True, help="retired_ids.csv (appended to).")
    i.add_argument("--run_label", required=True, help="e.g. 2026-11")
    i.add_argument("--replace_doi", action="append", default=[], help="Replace this DOI's rows (repeatable).")
    i.add_argument("--rejected_dois", type=Path, default=None, help="Text file of reviewer-rejected DOIs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = load_state(args.state)
    next_ids = state.setdefault("next_id", {})
    article_types = load_article_types(args.article_types)

    if args.mode == "bootstrap":
        if args.modality in next_ids and not args.force:
            print(f"Error: {args.state} already has next_id for {args.modality}; "
                  "IDs are frozen. Pass --force only before the first release commit.")
            return 1
        result = bootstrap(
            read_table(args.autotpd),
            [read_table(path) for path in args.large_scale],
            read_table(args.baseline),
            modality=args.modality,
            article_types=article_types,
        )
    else:
        if args.modality not in next_ids:
            print(f"Error: no next_id for {args.modality} in {args.state}; run bootstrap first.")
            return 1
        retired = load_retired(args.retired)
        result = incremental(
            read_table(args.previous),
            read_table(args.batch),
            modality=args.modality,
            next_id=int(next_ids[args.modality]),
            run_label=args.run_label,
            retired_ids=set(retired[ID_COLUMN]),
            replace_dois=set(args.replace_doi),
            rejected_dois=read_doi_list(args.rejected_dois),
            article_types=article_types,
        )

    errors = result.report["validation_errors"]
    if errors:
        for error in errors:
            print(f"Validation error: {error}")
        return 1

    write_outputs(result, args)
    if args.mode == "incremental" and len(result.retired):
        write_table(pd.concat([load_retired(args.retired), result.retired], ignore_index=True), args.retired)
    next_ids[args.modality] = result.next_id
    save_state(args.state, state)

    report = result.report
    print(json.dumps({k: report[k] for k in report if k not in {"article_type_rows", "appended_article_type_rows"}},
                     indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
