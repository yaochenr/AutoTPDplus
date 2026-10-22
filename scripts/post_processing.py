#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd
import pubchempy as pcp
import requests

OPSIN_BASE = "https://www.ebi.ac.uk/opsin/ws"
MISSING_TOKENS = {"", "-", "n/a", "na", "nan", "none", "null", "unknown"}
SOURCE_COLUMNS = {
    "IUPAC_Name": "IUPAC_Name_Source",
    "SMILES": "SMILES_Source",
    "Standard_InChIKey": "Standard_InChIKey_Source",
}
EXTRACTION_FILE_PROVENANCE_COLUMNS = (
    "source_relative_path",
    "source_archive",
    "source_archive_member",
    "source_file_sha256",
)

_DASH_CHARS = "‐‑‒–—―−﹣－"
_DASH_TRANS = str.maketrans({char: "-" for char in _DASH_CHARS})

DEFAULT_PUBCHEM_BLOCKLIST = {
    "Mei", "Po", "Le", "LC-4", "LC-6", "D10", "D12", "D15", "A13",
    "13b", "SC5", "SC7", "SC10", "DBt-10", "PS-6", "MD9", "Cpd 1",
}


def has_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().casefold() not in MISSING_TOKENS


def normalize_query(value: Any) -> str:
    if not has_value(value):
        return ""
    return " ".join(str(value).translate(_DASH_TRANS).split())


def _missing_mask(series: pd.Series) -> pd.Series:
    return ~series.map(has_value)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_post_processed.csv")


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        frame = pd.read_csv(path)
        print(f"Loaded {len(frame):>6} rows: {path}")
        frames.append(frame)

    if not frames:
        raise ValueError("At least one input CSV is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def remove_empty_activity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = [column for column in ("DC50", "Dmax") if column not in df]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Cannot filter activity rows; missing column(s): {joined}")

    remove = _missing_mask(df["DC50"]) & _missing_mask(df["Dmax"])
    kept = df.loc[~remove].copy().reset_index(drop=True)
    rejected = df.loc[remove].copy().reset_index(drop=True)
    return kept, rejected


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Compound_Name", "IUPAC_Name", "SMILES"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required chemical column(s): {', '.join(missing)}")

    df = df.copy()
    for column in ("Standard_InChIKey", *SOURCE_COLUMNS.values()):
        if column not in df:
            df[column] = pd.NA

    for value_column in ("IUPAC_Name", "SMILES"):
        source_column = SOURCE_COLUMNS[value_column]
        mark = df[value_column].map(has_value) & _missing_mask(df[source_column])
        df.loc[mark, source_column] = "PAPER"
    return df


class LookupCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
                print(f"Loaded {len(self.data)} cached lookups: {path}")
            except (OSError, json.JSONDecodeError) as exc:
                print(f"WARNING: could not read lookup cache {path}: {exc}")

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, result: dict[str, Any]) -> None:
        if result.get("status") in {"ok", "not_found"}:
            self.data[key] = result

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _cached_lookup(
    cache: LookupCache,
    key: str,
    lookup: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    cached = cache.get(key)
    if cached is not None:
        return cached, True
    result = lookup()
    cache.put(key, result)
    return result, False


def query_opsin(iupac_name: str, session: requests.Session) -> dict[str, Any]:
    url = f"{OPSIN_BASE}/{quote(iupac_name, safe='')}.json"
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 404:
            return {"status": "not_found"}
        response.raise_for_status()
        payload = response.json()
        inchikey = payload.get("stdinchikey")
        smiles = payload.get("smiles")
        if not inchikey and not smiles:
            return {"status": "not_found"}
        return {
            "status": "ok",
            "inchikey": inchikey,
            "smiles": smiles,
        }
    except (requests.RequestException, ValueError) as exc:
        return {"status": "error", "error": str(exc)}


def _compound_result(compound: Any) -> dict[str, Any]:
    smiles = (
        getattr(compound, "smiles", None)
        or getattr(compound, "connectivity_smiles", None)
    )
    return {
        "status": "ok",
        "cid": getattr(compound, "cid", None),
        "iupac_name": getattr(compound, "iupac_name", None),
        "smiles": smiles,
        "inchikey": getattr(compound, "inchikey", None),
    }


def query_pubchem_name(query: str, session: requests.Session) -> dict[str, Any]:
    try:
        compounds = pcp.get_compounds(query, "name")
        if not compounds:
            return {"status": "not_found"}
        top = compounds[0]
        cid = getattr(top, "cid", None)
        if not cid:
            return {"status": "not_found"}

        response = session.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
            timeout=30,
        )
        response.raise_for_status()
        information = response.json().get("InformationList", {}).get("Information", [])
        synonyms = information[0].get("Synonym", []) if information else []
        expected = normalize_query(query).casefold()
        if not any(normalize_query(synonym).casefold() == expected for synonym in synonyms):
            return {"status": "not_found", "reason": "query is not an exact synonym"}
        return _compound_result(top)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def query_pubchem_smiles(smiles: str) -> dict[str, Any]:
    try:
        compounds = pcp.get_compounds(smiles, "smiles")
        if not compounds:
            return {"status": "not_found"}
        return _compound_result(compounds[0])
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def enrich_with_opsin(
    df: pd.DataFrame,
    cache: LookupCache,
    session: requests.Session,
    delay: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    failures: list[dict[str, Any]] = []
    stats = {"queries": 0, "cache_hits": 0, "matched": 0,
             "smiles_filled": 0, "inchikey_filled": 0}

    needs_smiles = _missing_mask(df["SMILES"])
    needs_key = _missing_mask(df["Standard_InChIKey"])
    candidates = df.loc[df["IUPAC_Name"].map(has_value) & (needs_smiles | needs_key)]
    groups = candidates.groupby(candidates["IUPAC_Name"].map(normalize_query), sort=False)

    print(f"OPSIN: {len(candidates)} rows, {groups.ngroups} unique IUPAC names")
    for iupac_name, rows in groups:
        key = f"opsin:name:{iupac_name}"
        result, cached = _cached_lookup(
            cache, key, lambda name=iupac_name: query_opsin(name, session)
        )
        stats["cache_hits" if cached else "queries"] += 1
        if not cached and delay:
            time.sleep(delay)

        if result.get("status") == "error":
            failures.append({"service": "OPSIN", "query_type": "IUPAC_Name",
                             "query": iupac_name, "error": result.get("error")})
            continue
        if result.get("status") != "ok":
            continue

        stats["matched"] += 1
        for index in rows.index:
            if not has_value(df.at[index, "SMILES"]) and has_value(result.get("smiles")):
                df.at[index, "SMILES"] = result["smiles"]
                df.at[index, "SMILES_Source"] = "OPSIN"
                stats["smiles_filled"] += 1
            if (not has_value(df.at[index, "Standard_InChIKey"])
                    and has_value(result.get("inchikey"))):
                df.at[index, "Standard_InChIKey"] = result["inchikey"]
                df.at[index, "Standard_InChIKey_Source"] = "OPSIN"
                stats["inchikey_filled"] += 1

    cache.save()
    return df, failures, stats


def _is_numeric_name(value: str) -> bool:
    return value.replace(".", "").replace("-", "").isdigit()


def _usable_compound_name(name: str, blocklist: set[str], minimum_length: int) -> bool:
    normalized_blocklist = {normalize_query(item).casefold() for item in blocklist}
    return bool(
        name
        and len(name) >= minimum_length
        and not _is_numeric_name(name)
        and name.casefold() not in normalized_blocklist
    )


def enrich_with_pubchem(
    df: pd.DataFrame,
    cache: LookupCache,
    session: requests.Session,
    delay: float,
    blocklist: set[str],
    minimum_name_length: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    failures: list[dict[str, Any]] = []
    stats = {"queries": 0, "cache_hits": 0, "matched": 0,
             "smiles_filled": 0, "iupac_filled": 0, "inchikey_filled": 0}

    needs = (
        _missing_mask(df["SMILES"])
        | _missing_mask(df["IUPAC_Name"])
        | _missing_mask(df["Standard_InChIKey"])
    )
    candidates = df.loc[needs]
    grouped: dict[tuple[str, str, str], list[Any]] = {}
    for index, row in candidates.iterrows():
        triple = (
            normalize_query(row.get("Compound_Name")),
            normalize_query(row.get("IUPAC_Name")),
            normalize_query(row.get("SMILES")),
        )
        grouped.setdefault(triple, []).append(index)

    print(f"PubChem: {len(candidates)} rows, {len(grouped)} unique query triples")
    for (compound_name, iupac_name, smiles), indices in grouped.items():
        attempts: list[tuple[str, str]] = []
        if _usable_compound_name(compound_name, blocklist, minimum_name_length):
            attempts.append(("Compound_Name", compound_name))
        if iupac_name and iupac_name.casefold() != compound_name.casefold():
            attempts.append(("IUPAC_Name", iupac_name))
        if smiles:
            attempts.append(("SMILES", smiles))

        result: dict[str, Any] | None = None
        for query_type, query in attempts:
            cache_key = f"pubchem:{query_type.lower()}:{query}"
            if query_type == "SMILES":
                lookup = lambda value=query: query_pubchem_smiles(value)
            else:
                lookup = lambda value=query: query_pubchem_name(value, session)
            current, cached = _cached_lookup(cache, cache_key, lookup)
            stats["cache_hits" if cached else "queries"] += 1
            if not cached and delay:
                time.sleep(delay)

            if current.get("status") == "error":
                failures.append({"service": "PubChem", "query_type": query_type,
                                 "query": query, "error": current.get("error")})
                continue
            if current.get("status") == "ok":
                result = current
                break

        if not result:
            continue
        stats["matched"] += 1
        for index in indices:
            for value_column, result_key, count_key in (
                ("SMILES", "smiles", "smiles_filled"),
                ("IUPAC_Name", "iupac_name", "iupac_filled"),
                ("Standard_InChIKey", "inchikey", "inchikey_filled"),
            ):
                if not has_value(df.at[index, value_column]) and has_value(result.get(result_key)):
                    df.at[index, value_column] = result[result_key]
                    df.at[index, SOURCE_COLUMNS[value_column]] = "PUBCHEM"
                    stats[count_key] += 1

    cache.save()
    return df, failures, stats


def reorder_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    for value_column, source_column in SOURCE_COLUMNS.items():
        if value_column not in columns or source_column not in columns:
            continue
        columns.remove(source_column)
        columns.insert(columns.index(value_column) + 1, source_column)
    return df[columns]


def run_post_processing(
    input_paths: list[Path],
    output_path: Path,
    *,
    drop_empty_activity: bool = True,
    run_opsin: bool = True,
    run_pubchem: bool = True,
    cache_path: Path | None = None,
    blocklist: set[str] | None = None,
    opsin_delay: float = 0.1,
    pubchem_delay: float = 0.3,
    minimum_name_length: int = 3,
    keep_source_provenance: bool = False,
) -> dict[str, Any]:
    df = load_inputs(input_paths)
    input_rows = len(df)
    removed = df.iloc[0:0].copy()
    if drop_empty_activity:
        df, removed = remove_empty_activity(df)
        print(f"Removed {len(removed)} rows with neither DC50 nor Dmax")

    dropped_provenance_columns: list[str] = []
    if not keep_source_provenance:
        dropped_provenance_columns = [
            column for column in EXTRACTION_FILE_PROVENANCE_COLUMNS if column in df.columns
        ]
        df = df.drop(columns=dropped_provenance_columns, errors="ignore")
        removed = removed.drop(columns=dropped_provenance_columns, errors="ignore")
        if dropped_provenance_columns:
            print("Dropped extraction file provenance: " + ", ".join(dropped_provenance_columns))

    df = prepare_columns(df)
    cache = LookupCache(cache_path)
    session = requests.Session()
    session.headers.update({"User-Agent": "tpd-curator-post-processing/1.0"})
    failures: list[dict[str, Any]] = []
    service_stats: dict[str, dict[str, int]] = {}

    if run_opsin:
        df, service_failures, service_stats["opsin"] = enrich_with_opsin(
            df, cache, session, opsin_delay
        )
        failures.extend(service_failures)
    if run_pubchem:
        df, service_failures, service_stats["pubchem"] = enrich_with_pubchem(
            df,
            cache,
            session,
            pubchem_delay,
            blocklist or set(DEFAULT_PUBCHEM_BLOCKLIST),
            minimum_name_length,
        )
        failures.extend(service_failures)

    df = reorder_provenance_columns(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = {
        "input_files": [str(path) for path in input_paths],
        "output_csv": str(output_path),
        "input_rows": input_rows,
        "output_rows": len(df),
        "removed_no_activity_rows": len(removed),
        "dropped_source_provenance_columns": dropped_provenance_columns,
        "lookup_failures": len(failures),
        "service_stats": service_stats,
        "remaining_missing": {
            column: int(_missing_mask(df[column]).sum())
            for column in ("IUPAC_Name", "SMILES", "Standard_InChIKey")
        },
    }
    print(f"Saved {len(df)} post-processed rows: {output_path}")
    if failures:
        print(f"Lookup failures (not written to disk): {len(failures)}")
    print("Remaining missing values: " + ", ".join(
        f"{column}={count}" for column, count in summary["remaining_missing"].items()
    ))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich LLM-extracted TPD records with OPSIN and PubChem before standardization."
    )
    parser.add_argument(
        "--input_csv", type=Path, nargs="+", required=True,
        help="One or more LLM-extracted CSV files to merge and post-process.",
    )
    parser.add_argument(
        "--output_csv", type=Path, default=None,
        help="Output CSV (default: <first-input>_post_processed.csv).",
    )
    parser.add_argument(
        "--keep_empty_activity", action="store_true",
        help="Keep rows where both DC50 and Dmax are empty.",
    )
    parser.add_argument(
        "--keep_source_provenance", action="store_true",
        help="Keep extraction-only source path/archive/hash columns in the output.",
    )
    parser.add_argument("--skip_opsin", action="store_true", help="Do not query OPSIN.")
    parser.add_argument("--skip_pubchem", action="store_true", help="Do not query PubChem.")
    parser.add_argument(
        "--cache_json", type=Path, default=None,
        help="Optional persistent API cache. No cache file is written by default.",
    )
    parser.add_argument(
        "--pubchem_blocklist", type=Path, default=None,
        help="Optional text file with one additional unsafe compound name per line.",
    )
    parser.add_argument("--opsin_delay", type=float, default=0.1)
    parser.add_argument("--pubchem_delay", type=float, default=0.3)
    parser.add_argument("--minimum_name_length", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_csv or _default_output_path(args.input_csv[0])
    cache_path = args.cache_json

    blocklist = set(DEFAULT_PUBCHEM_BLOCKLIST)
    if args.pubchem_blocklist:
        additions = {
            line.strip()
            for line in args.pubchem_blocklist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        blocklist.update(additions)

    run_post_processing(
        args.input_csv,
        output_path,
        drop_empty_activity=not args.keep_empty_activity,
        run_opsin=not args.skip_opsin,
        run_pubchem=not args.skip_pubchem,
        cache_path=cache_path,
        blocklist=blocklist,
        opsin_delay=args.opsin_delay,
        pubchem_delay=args.pubchem_delay,
        minimum_name_length=args.minimum_name_length,
        keep_source_provenance=args.keep_source_provenance,
    )

if __name__ == "__main__":
    main()
