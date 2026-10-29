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
ID_WIDTH = 6
RETIRED_RELEASE_COLUMNS = {
    "Source_Article_Type",
    "StdInChI",
    "InChIKey_Original",
    "Standard_InChIKey_Source",
}

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
    "MG": {
        "prefix": "AUTOTPD-MG",
        "legacy_original_key": "StdInChIKey",
        "baseline_smiles_source": "MolGlueDB",
    },
    "PROTAC": {
        "prefix": "AUTOTPD-PROTAC",
        "legacy_original_key": "InChIKey",
        "baseline_smiles_source": "PROTAC-DB 3.0",
    },
}

RETIRED_COLUMNS = [ID_COLUMN, "DOI", "retired_at", "reason", "replaced_by_run"]
CONFLICT_COLUMNS = [
    "reason", "DOI", "Compound_Name", "Degradation_Target", "Cell_Line",
    "DC50", "DC50_units", "Dmax", "Dmax_conc", "Dmax_conc_units",
    "source", "existing_AutoTPD_IDs",
]
MISSING_DOI_TOKENS = {"", "unknown", "nan", "none", "null"}
SUPPORTED_DC50_UNITS = frozenset({"nM"})
RECRUITER_UNIPROT_COLUMN = "Recruiter_Uniprot"
LEGACY_RECRUITER_UNIPROT_COLUMN = "Recruiter_UniProtID"
IUPAC_NAME_COLUMN = "IUPAC_Name"
IUPAC_NAME_SOURCE_COLUMN = "IUPAC_Name_Source"
STANDARD_INCHI_COLUMN = "StdInChI"
ORIGINAL_INCHIKEY_COLUMN = "InChIKey_Original"
SMILES_COLUMN = "SMILES"
SMILES_SOURCE_COLUMN = "SMILES_Source"
ACTIVITY_PARSE_STATUS_COLUMNS = {
    "DC50_parse_status": "dc50",
    "Dmax_parse_status": "dmax",
    "Dmax_conc_parse_status": "dmax_conc",
}
ACTIVITY_VALUE_COLUMNS = {
    "dc50": ["DC50", "DC50_range_min", "DC50_range_max"],
    "dmax": ["Dmax", "Dmax_range_min", "Dmax_range_max"],
    "dmax_conc": ["Dmax_conc", "Dmax_conc_range_min", "Dmax_conc_range_max"],
}
ACTIVITY_FIELD_COLUMNS = {
    "dc50": [
        *ACTIVITY_VALUE_COLUMNS["dc50"],
        "DC50_units", "DC50_h", "DC50_operator", "DC50_parse_status",
    ],
    "dmax": [
        *ACTIVITY_VALUE_COLUMNS["dmax"],
        "Dmax_h", "Dmax_operator", "Dmax_parse_status",
    ],
    "dmax_conc": [
        *ACTIVITY_VALUE_COLUMNS["dmax_conc"],
        "Dmax_conc_operator", "Dmax_conc_units", "Dmax_conc_parse_status",
    ],
}


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


def normalize_source_values(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize release-facing text without changing scientific values."""
    df = df.copy()
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip().fillna("")
    if "DOI" in df.columns:
        df["DOI"] = df["DOI"].map(normalize_doi)
    for column in (name for name in df.columns if name.endswith("_units")):
        df[column] = df[column].str.replace("μ", "µ", regex=False)
    return df


def normalize_legacy_dmax_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Repair reversed numeric Dmax ranges in releases created before validation."""
    if not {"Dmax_range_min", "Dmax_range_max"}.issubset(df.columns):
        return df

    df = df.copy()
    range_min = pd.to_numeric(df["Dmax_range_min"], errors="coerce")
    range_max = pd.to_numeric(df["Dmax_range_max"], errors="coerce")
    reversed_range = range_min.notna() & range_max.notna() & (range_min > range_max)
    if reversed_range.any():
        df.loc[reversed_range, "Dmax_range_min"] = range_max[reversed_range]
        df.loc[reversed_range, "Dmax_range_max"] = range_min[reversed_range]
    return df


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


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() != ""


def _activity_values(df: pd.DataFrame, field: str) -> tuple[pd.Series, pd.Series]:
    """Return masks for any populated value and any non-numeric populated value."""
    populated = pd.Series(False, index=df.index)
    non_numeric = pd.Series(False, index=df.index)
    for column in ACTIVITY_VALUE_COLUMNS[field]:
        if column not in df.columns:
            continue
        present = _nonempty(df[column])
        populated |= present
        non_numeric |= present & pd.to_numeric(df[column], errors="coerce").isna()
    return populated, non_numeric


def activity_quality_reasons(df: pd.DataFrame) -> pd.Series:
    """Return one release-blocking activity reason per row, or an empty string."""
    reasons = pd.Series("", index=df.index, dtype=object)

    for column, field in ACTIVITY_PARSE_STATUS_COLUMNS.items():
        if column not in df.columns:
            continue
        status = df[column].fillna("").astype(str).str.strip().str.lower()
        for rejected_status in ("unparsed", "recognised", "recognized"):
            rejected = (status == rejected_status) & (reasons == "")
            label = "recognised_non_numeric" if rejected_status.startswith("recogni") else "unparsed"
            reasons.loc[rejected] = f"activity_{field}_{label}"

    dc50, dc50_non_numeric = _activity_values(df, "dc50")
    dmax, dmax_non_numeric = _activity_values(df, "dmax")
    dmax_conc, dmax_conc_non_numeric = _activity_values(df, "dmax_conc")
    dc50_units = _nonempty(df["DC50_units"]) if "DC50_units" in df.columns else pd.Series(False, index=df.index)
    dmax_conc_units = (
        _nonempty(df["Dmax_conc_units"])
        if "Dmax_conc_units" in df.columns else pd.Series(False, index=df.index)
    )
    dc50_values = (
        pd.to_numeric(df["DC50"], errors="coerce")
        if "DC50" in df.columns else pd.Series(float("nan"), index=df.index)
    )
    dmax_values = (
        pd.to_numeric(df["Dmax"], errors="coerce")
        if "Dmax" in df.columns else pd.Series(float("nan"), index=df.index)
    )
    dmax_range_min = (
        pd.to_numeric(df["Dmax_range_min"], errors="coerce")
        if "Dmax_range_min" in df.columns else pd.Series(float("nan"), index=df.index)
    )
    dmax_range_max = (
        pd.to_numeric(df["Dmax_range_max"], errors="coerce")
        if "Dmax_range_max" in df.columns else pd.Series(float("nan"), index=df.index)
    )
    dc50_unit_values = (
        df["DC50_units"].fillna("").astype(str).str.strip()
        if "DC50_units" in df.columns else pd.Series("", index=df.index)
    )
    unsupported_dc50_units = dc50_units & ~dc50_unit_values.isin(SUPPORTED_DC50_UNITS)

    checks = [
        (~dc50 & ~dmax, "activity_missing_dc50_and_dmax"),
        (dc50_non_numeric, "activity_dc50_non_numeric"),
        (dmax_non_numeric, "activity_dmax_non_numeric"),
        (dmax_conc_non_numeric, "activity_dmax_conc_non_numeric"),
        (dc50_values <= 0, "activity_dc50_non_positive"),
        (dmax_values > 100, "activity_dmax_above_100"),
        (dmax_range_min > dmax_range_max, "activity_dmax_reversed_range"),
        (dc50 & ~dc50_units, "activity_dc50_missing_units"),
        (~dc50 & dc50_units, "activity_dc50_units_without_value"),
        (dc50 & unsupported_dc50_units, "activity_dc50_unsupported_units"),
        (dmax_conc & ~dmax_conc_units, "activity_dmax_conc_missing_units"),
        (~dmax_conc & dmax_conc_units, "activity_dmax_conc_units_without_value"),
    ]
    for invalid, reason in checks:
        reasons.loc[invalid & (reasons == "")] = reason

    return reasons


def _invalid_activity_field(reason: str) -> str | None:
    if reason.startswith("activity_dmax_conc_"):
        return "dmax_conc"
    if reason.startswith("activity_dc50_"):
        return "dc50"
    if reason.startswith("activity_dmax_"):
        return "dmax"
    return None


def _clear_invalid_activity(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Clear the measurement implicated by a release-blocking reason in place."""
    field = _invalid_activity_field(reason)
    if field is None:
        return

    columns = ACTIVITY_FIELD_COLUMNS[field]
    # A reversed range invalidates the bounds, but not an independently reported
    # central value.
    if reason == "activity_dmax_reversed_range":
        columns = ["Dmax_range_min", "Dmax_range_max", "Dmax_parse_status"]
    # Dmax_conc only qualifies Dmax, so it cannot outlive a cleared Dmax.
    elif field == "dmax":
        columns = [*columns, *ACTIVITY_FIELD_COLUMNS["dmax_conc"]]

    for column in columns:
        if column in df.columns:
            df.loc[mask, column] = ""


def reject_invalid_activity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Remove invalid measurements, rejecting only rows left without DC50 or Dmax.

    Conflicts retain the original row values for audit. A row can therefore have
    more than one conflict when multiple measurements need to be cleared.
    """
    cleaned = df.copy()
    conflicts: list[pd.DataFrame] = []

    # Clearing one invalid field can expose another reason that was hidden by the
    # one-reason-per-row priority in activity_quality_reasons, so iterate until all
    # remaining rows are clean or simply lack both primary activity measurements.
    while True:
        reasons = activity_quality_reasons(cleaned)
        actionable = (reasons != "") & (reasons != "activity_missing_dc50_and_dmax")
        if not actionable.any():
            break
        for reason in reasons[actionable].drop_duplicates():
            mask = reasons == reason
            conflicts.append(conflict_rows(df[mask], reason))
            _clear_invalid_activity(cleaned, mask, reason)

    dc50, _ = _activity_values(cleaned, "dc50")
    dmax, _ = _activity_values(cleaned, "dmax")
    rejected = ~dc50 & ~dmax
    if rejected.any():
        conflicts.append(conflict_rows(df[rejected], "activity_missing_dc50_and_dmax"))

    conflicts_df = concat_conflict_frames(conflicts)
    return cleaned[~rejected].copy(), conflicts_df, int(rejected.sum())


def merge_recruiter_uniprot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fold the legacy MolGlueDB recruiter accession into the canonical column."""
    if LEGACY_RECRUITER_UNIPROT_COLUMN not in df.columns:
        return df

    df = df.copy()
    legacy = df[LEGACY_RECRUITER_UNIPROT_COLUMN]
    if RECRUITER_UNIPROT_COLUMN in df.columns:
        canonical = df[RECRUITER_UNIPROT_COLUMN]
        missing = canonical.fillna("").astype(str).str.strip() == ""
        df.loc[missing, RECRUITER_UNIPROT_COLUMN] = legacy[missing]
    else:
        df[RECRUITER_UNIPROT_COLUMN] = legacy
    return df.drop(columns=LEGACY_RECRUITER_UNIPROT_COLUMN)


def fill_missing_iupac_name_source(df: pd.DataFrame) -> pd.DataFrame:
    """Mark pre-existing IUPAC names without provenance as paper-derived."""
    if IUPAC_NAME_COLUMN not in df.columns:
        return df

    df = df.copy()
    if IUPAC_NAME_SOURCE_COLUMN not in df.columns:
        df[IUPAC_NAME_SOURCE_COLUMN] = ""
    has_iupac_name = df[IUPAC_NAME_COLUMN].fillna("").astype(str).str.strip() != ""
    missing_source = df[IUPAC_NAME_SOURCE_COLUMN].fillna("").astype(str).str.strip() == ""
    df.loc[has_iupac_name & missing_source, IUPAC_NAME_SOURCE_COLUMN] = "PAPER"
    return df


def missing_smiles_source_mask(df: pd.DataFrame) -> pd.Series:
    if SMILES_COLUMN not in df.columns:
        return pd.Series(False, index=df.index)
    has_smiles = _nonempty(df[SMILES_COLUMN])
    if SMILES_SOURCE_COLUMN not in df.columns:
        return has_smiles
    return has_smiles & ~_nonempty(df[SMILES_SOURCE_COLUMN])


def fill_missing_smiles_source(df: pd.DataFrame, source: str | None) -> pd.DataFrame:
    """Attribute inherited baseline SMILES to the database that supplied them."""
    if not source or SMILES_COLUMN not in df.columns:
        return df
    df = df.copy()
    if SMILES_SOURCE_COLUMN not in df.columns:
        df[SMILES_SOURCE_COLUMN] = ""
    df.loc[missing_smiles_source_mask(df), SMILES_SOURCE_COLUMN] = source
    return df


def merge_column_aliases(
    df: pd.DataFrame,
    canonical: str,
    aliases: list[str],
) -> pd.DataFrame:
    """Merge legacy column aliases into one canonical release column."""
    df = df.copy()
    for alias in aliases:
        if alias == canonical or alias not in df.columns:
            continue
        if canonical not in df.columns:
            df = df.rename(columns={alias: canonical})
            continue
        missing = df[canonical].fillna("").astype(str).str.strip() == ""
        df.loc[missing, canonical] = df.loc[missing, alias]
        df = df.drop(columns=alias)
    return df


def rename_inchi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Standard InChI column names used by MG and PROTAC inputs."""
    return merge_column_aliases(df, STANDARD_INCHI_COLUMN, ["StdInChl", "InChI"])


def rename_inchikey_columns(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Separate the SMILES-derived standard key from the original source key."""
    df = merge_column_aliases(
        df,
        ORIGINAL_INCHIKEY_COLUMN,
        ["StdInChIKey_Original"],
    )

    # A frame with Compound_Key still uses the input/legacy schema. Move its
    # source-provided key aside before assigning the computed key's final name.
    if "Compound_Key" in df.columns:
        candidates = ["Standard_InChIKey", MODALITIES[modality]["legacy_original_key"]]
        for candidate in candidates:
            if candidate not in df.columns:
                continue
            if ORIGINAL_INCHIKEY_COLUMN not in df.columns:
                df = df.rename(columns={candidate: ORIGINAL_INCHIKEY_COLUMN})
                continue
            missing = df[ORIGINAL_INCHIKEY_COLUMN].fillna("").astype(str).str.strip() == ""
            df.loc[missing, ORIGINAL_INCHIKEY_COLUMN] = df.loc[missing, candidate]
            df = df.drop(columns=candidate)
        df = df.rename(columns={"Compound_Key": "StdInChIKey"})
    elif "Standard_InChIKey" in df.columns:
        df = merge_column_aliases(
            df,
            ORIGINAL_INCHIKEY_COLUMN,
            ["Standard_InChIKey"],
        )

    return df


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


def concat_conflict_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return (
        pd.concat(nonempty, ignore_index=True)
        if nonempty else pd.DataFrame(columns=CONFLICT_COLUMNS)
    )


def prepare_release(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    df = rename_inchi_columns(df)
    df = rename_inchikey_columns(df, modality)
    df = merge_recruiter_uniprot_columns(df)
    df = normalize_legacy_dmax_ranges(df)
    return df.drop(columns=[*RETIRED_RELEASE_COLUMNS, *ACTIVITY_PARSE_STATUS_COLUMNS], errors="ignore")


def prepare_source(
    df: pd.DataFrame,
    modality: str,
    *,
    default_smiles_source: str | None = None,
) -> pd.DataFrame:
    df = normalize_source_values(df)
    df = rename_inchi_columns(df)
    df = rename_inchikey_columns(df, modality)
    df = merge_recruiter_uniprot_columns(df)
    df = fill_missing_iupac_name_source(df)
    df = fill_missing_smiles_source(df, default_smiles_source)
    return df.drop(columns=[ID_COLUMN, *RETIRED_RELEASE_COLUMNS], errors="ignore")


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
) -> BuildResult:
    prefix = MODALITIES[modality]["prefix"]

    autotpd = normalize_source_values(autotpd)
    autotpd = rename_inchi_columns(autotpd)
    autotpd = rename_inchikey_columns(autotpd, modality)
    autotpd = merge_recruiter_uniprot_columns(autotpd)
    autotpd = autotpd.drop(columns=list(RETIRED_RELEASE_COLUMNS), errors="ignore")
    autotpd = fill_missing_iupac_name_source(autotpd)
    autotpd_reasons = activity_quality_reasons(autotpd)
    if (autotpd_reasons != "").any():
        counts = autotpd_reasons[autotpd_reasons != ""].value_counts().to_dict()
        raise ValueError(f"AutoTPD contains release-blocking activity values: {counts}")
    if ID_COLUMN not in autotpd.columns or (autotpd[ID_COLUMN] == "").any():
        raise ValueError("every AutoTPD row needs an AutoTPD_ID")
    if not autotpd[ID_COLUMN].is_unique:
        raise ValueError("AutoTPD_ID values in AutoTPD are not unique")
    autotpd_numbers = autotpd[ID_COLUMN].map(lambda value: id_number(value, prefix))

    autotpd_dois = set(autotpd["DOI"].map(normalize_doi)) - {""}

    conflicts: list[pd.DataFrame] = []
    large_scale_kept: list[pd.DataFrame] = []
    large_scale_dois: set[str] = set()
    replaced_by_autotpd = 0
    large_scale_activity_rejected = 0
    for frame in large_scale:
        frame = prepare_source(frame, modality)
        frame, activity_conflicts, activity_rejected = reject_invalid_activity(frame)
        large_scale_activity_rejected += activity_rejected
        conflicts.append(activity_conflicts)
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

    baseline_rows_in = len(baseline)
    baseline_missing_smiles_source = missing_smiles_source_mask(baseline)
    baseline = prepare_source(
        baseline,
        modality,
        default_smiles_source=MODALITIES[modality]["baseline_smiles_source"],
    )
    baseline, baseline_activity_conflicts, baseline_activity_rejected = reject_invalid_activity(baseline)
    conflicts.append(baseline_activity_conflicts)
    baseline_doi = baseline["DOI"].map(normalize_doi)
    baseline_replaced = baseline_doi.isin(autotpd_dois | large_scale_dois) & (baseline_doi != "")
    baseline_kept = baseline[~baseline_replaced]
    baseline_smiles_source_filled = int(
        baseline_missing_smiles_source.loc[baseline_kept.index].sum()
    )

    columns = [
        column
        for column in ordered_union([ID_COLUMN], list(baseline.columns), list(autotpd.columns))
        if column not in ACTIVITY_PARSE_STATUS_COLUMNS
    ]
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
    release = sort_by_id(release, prefix)

    conflicts_df = concat_conflict_frames(conflicts)
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
                "rows_rejected_activity_quality": large_scale_activity_rejected,
                "dropped_columns": large_scale_dropped,
            },
            "baseline": {
                "rows_in": baseline_rows_in,
                "rows_kept": len(baseline_part),
                "rows_superseded": int(baseline_replaced.sum()),
                "rows_rejected_activity_quality": baseline_activity_rejected,
                "smiles_source_filled": baseline_smiles_source_filled,
            },
        },
        "release_rows": len(release),
        "id_range": [release[ID_COLUMN].iloc[0], release[ID_COLUMN].iloc[-1]] if len(release) else [],
        "next_id": next_number,
        "conflict_rows": conflicts_df["reason"].value_counts().to_dict(),
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
) -> BuildResult:
    prefix = MODALITIES[modality]["prefix"]
    retired_ids = retired_ids or set()
    replace_dois = {normalize_doi(doi) for doi in (replace_dois or set())} - {""}
    rejected_dois = {normalize_doi(doi) for doi in (rejected_dois or set())} - {""}

    if ID_COLUMN not in previous.columns:
        raise ValueError("previous release has no AutoTPD_ID column; run bootstrap first")

    previous = prepare_release(previous, modality)
    batch_rows_in = len(batch)
    batch = prepare_source(batch, modality)
    batch, activity_conflicts, activity_rejected_rows = reject_invalid_activity(batch)
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

    conflicts_df = concat_conflict_frames([
        activity_conflicts,
        conflict_rows(batch[missing], "missing_doi"),
        conflict_rows(batch[skipped_existing], "doi_already_in_release", ids_by_doi),
    ])

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
    new_part, dropped_columns = conform_schema(new_rows, columns)
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
        "batch_rows": batch_rows_in,
        "batch_dois": len(set(batch_doi) - {""}),
        "rejected_by_reviewer_rows": int(rejected.sum()),
        "rejected_activity_quality_rows": activity_rejected_rows,
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

    activity_reasons = activity_quality_reasons(release)
    if (activity_reasons != "").any():
        errors.append(
            "release-blocking activity values: "
            f"{activity_reasons[activity_reasons != ''].value_counts().to_dict()}"
        )

    if "DOI" in release.columns:
        canonical_doi = release["DOI"].map(normalize_doi)
        noncanonical = release["DOI"].fillna("").astype(str) != canonical_doi
        if noncanonical.any():
            errors.append(f"non-canonical DOI values: {int(noncanonical.sum())}")

    string_columns = release.select_dtypes(include=["object", "string"]).columns
    whitespace_rows = pd.Series(False, index=release.index)
    for column in string_columns:
        values = release[column].fillna("").astype(str)
        whitespace_rows |= values != values.str.strip()
    if whitespace_rows.any():
        errors.append(f"rows with leading/trailing whitespace: {int(whitespace_rows.sum())}")

    micro_variant_rows = pd.Series(False, index=release.index)
    for column in (name for name in release.columns if name.endswith("_units")):
        micro_variant_rows |= release[column].fillna("").astype(str).str.contains("μ", regex=False)
    if micro_variant_rows.any():
        errors.append(f"rows using non-canonical Greek mu in units: {int(micro_variant_rows.sum())}")

    missing_smiles_sources = missing_smiles_source_mask(release)
    if missing_smiles_sources.any():
        errors.append(f"SMILES values without SMILES_Source: {int(missing_smiles_sources.sum())}")

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

    print(json.dumps(result.report, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
