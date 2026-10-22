"""Shared RDKit SMILES canonicalization.

Every dataset in this project - AutoTPD (LLM-extracted), the PROTAC-DB /
MolGlueDB baselines, and the manual ground truth - carries SMILES strings that
were written by different toolkits (PubChem, ChEMBL, the source databases, the
LLM). The same molecule therefore appears as different strings depending on the
source, which makes any string-level structure comparison meaningless.

This module is the single canonicalization function all of them go through:

    Raw SMILES
        |
        v
    RDKit parse            (Chem.MolFromSmiles)
        |
        v
    chemical standardization
        - Cleanup           sanitize, disconnect metals, normalize, reionize
        - largest fragment  drop salts / counterions
        - uncharge          neutralize where chemically sensible
        |
        v
    canonical SMILES       (isomeric: stereochemistry is preserved)

Stereochemistry is deliberately kept, because compound matching downstream
distinguishes stereoisomers that share a Connectivity_Key.

Used as a stage of scripts/data_standardization.py (which is the shared
standardization path for both AutoTPD and the baselines), and runnable on its
own for any other CSV:

    pixi run python -m tpd_curator.preprocessing.smiles_standardization \\
        --input_csv data/evaluation/protacs/ground_truth/PROTAC-GT.csv
"""

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit prints parse failures to stderr; we report them ourselves instead.
RDLogger.DisableLog("rdApp.*")

_MISSING_TOKENS = {"n/a", "na", "nan", "none", "-", ""}

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()

# Status codes returned per SMILES string.
MISSING = "missing"            # no SMILES to canonicalize
UNCHANGED = "unchanged"        # input string was already the canonical form
CANONICALIZED = "canonicalized"  # same molecule, different string
STANDARDIZED = "standardized"  # chemical standardization altered the structure
PARSE_FAILED = "parse_failed"  # RDKit could not read the string


def _is_missing(raw) -> bool:
    if raw is None or pd.isna(raw):
        return True
    return str(raw).strip().strip("()").lower() in _MISSING_TOKENS


@lru_cache(maxsize=None)
def _canonicalize(s: str) -> tuple[str | None, str]:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None, PARSE_FAILED

    parsed = Chem.MolToSmiles(mol)

    try:
        std = rdMolStandardize.Cleanup(mol)
        std = _LARGEST_FRAGMENT.choose(std)
        std = _UNCHARGER.uncharge(std)
    except Exception:
        # A molecule RDKit can read but not standardize still gets canonicalized.
        return parsed, CANONICALIZED if parsed != s else UNCHANGED

    canonical = Chem.MolToSmiles(std, isomericSmiles=True, canonical=True)

    if canonical != parsed:
        return canonical, STANDARDIZED
    if canonical != s:
        return canonical, CANONICALIZED
    return canonical, UNCHANGED


def canonicalize_smiles(raw) -> tuple[str | None, str]:
    """Canonicalize one SMILES string.

    Returns (canonical_smiles, status). On a parse failure the canonical value
    is None and the caller decides what to keep - data_standardization.py keeps
    the original string rather than dropping it.
    """
    if _is_missing(raw):
        return None, MISSING
    return _canonicalize(str(raw).strip())


@lru_cache(maxsize=None)
def _inchikey(s: str) -> str | None:
    canonical, _ = _canonicalize(s)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def inchikey_from_smiles(raw) -> str | None:
    """Full InChIKey of a SMILES string, via the same standardization path.

    Unlike Connectivity_Key (the first 14 characters, which encode connectivity
    only) this keeps the stereochemistry layer, so stereoisomers stay distinct.
    Computing it from the canonical SMILES means AutoTPD and the baselines get
    their compound key from one function rather than from each source database.
    """
    if _is_missing(raw):
        return None
    return _inchikey(str(raw).strip())


def add_compound_key(
    df: pd.DataFrame,
    smiles_column: str = "SMILES",
    key_column: str = "Compound_Key",
    verbose: bool = True,
) -> pd.DataFrame:
    """Add `key_column`, the InChIKey computed from `smiles_column`."""
    if smiles_column not in df.columns:
        if verbose:
            print(f"Column '{smiles_column}' not present, skipping compound key")
        return df

    df[key_column] = df[smiles_column].map(inchikey_from_smiles).astype("string")

    if verbose:
        n_smiles = df[smiles_column].map(lambda v: not _is_missing(v)).sum()
        n_key = df[key_column].notna().sum()
        print(f"{key_column} computed: {n_key}/{n_smiles} rows with a SMILES")
        print(f"  distinct compounds: {df[key_column].nunique()}")
        if "Connectivity_Key" in df.columns:
            sub = df.dropna(subset=[key_column, "Connectivity_Key"])
            collapsed = sub.groupby("Connectivity_Key")[key_column].nunique()
            n_groups = int((collapsed > 1).sum())
            if n_groups:
                print(f"  stereoisomer groups Connectivity_Key would merge: {n_groups}")

    return df


def canonicalize_smiles_column(
    df: pd.DataFrame,
    column: str = "SMILES",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonicalize `column` in place, keeping unparsable strings untouched.

    Returns the DataFrame and a status -> count summary.
    """
    if column not in df.columns:
        if verbose:
            print(f"Column '{column}' not present, skipping SMILES canonicalization")
        return df, {}

    results = df[column].map(canonicalize_smiles)
    canonical = results.map(lambda r: r[0])
    statuses = results.map(lambda r: r[1])

    # Missing stays missing; a string RDKit cannot parse is left as-is.
    keep_original = statuses.isin([MISSING, PARSE_FAILED])
    df[column] = canonical.where(~keep_original, df[column]).astype("string")

    counts = {k: int(v) for k, v in statuses.value_counts().items()}

    if verbose:
        n_smiles = len(df) - counts.get(MISSING, 0)
        print(f"Rows with a SMILES value: {n_smiles}/{len(df)}")
        print(f"  already canonical:  {counts.get(UNCHANGED, 0)}")
        print(f"  rewritten:          {counts.get(CANONICALIZED, 0)}")
        print(f"  chemically changed: {counts.get(STANDARDIZED, 0)}")
        print(f"  no SMILES:          {counts.get(MISSING, 0)}")
        n_failed = counts.get(PARSE_FAILED, 0)
        if n_failed:
            print(f"  PARSE FAILED:       {n_failed}  (original strings kept)")
            failed = sorted(set(df.loc[statuses == PARSE_FAILED, column].astype(str)))
            for s in failed:
                print(f"    WARNING: RDKit could not parse SMILES: {s!r}")

    return df, counts


def main():
    parser = argparse.ArgumentParser(
        description="Canonicalize the SMILES column of a CSV with the shared RDKit function."
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the CSV to canonicalize.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Where to write the result (default: overwrite the input).")
    parser.add_argument("--column", type=str, default="SMILES",
                        help="Name of the SMILES column (default: SMILES).")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        print(f"Error: input CSV does not exist - {input_csv}")
        raise SystemExit(1)

    output_csv = Path(args.output_csv) if args.output_csv else input_csv

    df = pd.read_csv(input_csv)
    df, _ = canonicalize_smiles_column(df, column=args.column)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()
