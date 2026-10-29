#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tpd_curator.pmc_miner.core.client import PMCClient
from tpd_curator.pmc_miner.core.storage import StorageManager
from tpd_curator.pmc_miner.mining.activity_screen import (
    ACCEPTED,
    ERROR,
    MANUAL,
    REJECTED,
    download_accepted_articles,
    iter_text_chunks,
    screen_catalog,
)
from tpd_curator.pmc_miner.mining.citation_registry import (
    add_pmc_search_results,
    load_registry_json,
    normalize_doi,
    write_registry,
)

PENDING ="pending_activity_screen"
REJECTED_BY_REVIEWER = "rejected_by_reviewer"
MENTION_POLICY = "dc50_dmax_mention_policy_v1"
ACCEPT_POLICY = "user_accept_dc50_dmax_mentions_v1"
DC50_RE = re.compile(r"\bDC[ _-]?50\b", re.IGNORECASE)
DMAX_RE = re.compile(r"\bD[ _-]?max\b", re.IGNORECASE)
MG_DATABASES = {"MGDB", "MGTbind"}
PROTAC_DATABASES = {"PROTAC-DB", "TACK2"}
MG_TITLE = re.compile(
    r"molecular\s+glue|glue[- ](?:degrad|induc|depend|mediat|enhanc)|"
    r"targeted\s+glue|covalent\s+glue|bivalent\s+glue|"
    r"monovalent\s+(?:direct\s+)?degrader|\bcelmods?\b|neosubstrate",
    re.IGNORECASE,
)
PROTAC_TITLE = re.compile(
    r"\bprotacs?\b|proteolysis[-\s]?targeting\s+chimer|heterobifunctional|"
    r"bifunctional\s+(?:molecule|degrader)",
    re.IGNORECASE,
)
MG_TEXT = re.compile(
    r"molecular\s+glue|glue[- ](?:degrader|induced|dependent|mediated)|"
    r"\bcelmods?\b|neosubstrate",
    re.IGNORECASE,
)
PROTAC_TEXT = re.compile(
    r"\bprotacs?\b|proteolysis[-\s]?targeting\s+chimer|heterobifunctional",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify(article: dict, paper_dir: Path) -> tuple[str, str, int, int]:
    sources = set(article.get("sources") or [])
    modalities = set(article.get("modalities") or [])
    from_mg_database = bool(sources & MG_DATABASES)
    from_protac_database = bool(sources & PROTAC_DATABASES)
    title = article.get("title") or ""
    mg_in_title = bool(MG_TITLE.search(title))
    protac_in_title = bool(PROTAC_TITLE.search(title))

    if from_mg_database and not from_protac_database:
        return "MG", "mg_database_source", 0, 0
    if from_protac_database and not from_mg_database:
        return "PROTAC", "protac_database_source", 0, 0
    if not from_mg_database and not from_protac_database:
        if "molecular_glue" in modalities and "protac" not in modalities:
            return "MG", "pmc_mg_query_only", 0, 0
        if "protac" in modalities and "molecular_glue" not in modalities:
            return "PROTAC", "pmc_protac_query_only", 0, 0
    if mg_in_title and not protac_in_title:
        return "MG", "title_mg_specific", 0, 0
    if protac_in_title and not mg_in_title:
        return "PROTAC", "title_protac_specific", 0, 0

    processed = json.loads((paper_dir / "processed.json").read_text(encoding="utf-8"))
    title_and_abstract = " ".join(
        [processed.get("title") or "", processed.get("abstract") or ""]
    )
    mg_mentions = len(MG_TEXT.findall(title_and_abstract))
    protac_mentions = len(PROTAC_TEXT.findall(title_and_abstract))
    if mg_mentions >= 2 and mg_mentions > protac_mentions:
        return "MG", "abstract_mg_dominant", mg_mentions, protac_mentions
    if protac_mentions >= 1 and protac_mentions > mg_mentions:
        return "PROTAC", "abstract_protac_dominant", mg_mentions, protac_mentions
    if from_mg_database:
        return "MG", "mixed_sources_forced_mg", mg_mentions, protac_mentions
    return "PROTAC", "broad_or_mixed_default_protac", mg_mentions, protac_mentions


def read_doi_lists(paths: List[Path]) -> set:
    dois = set()
    for path in paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    dois.add(normalize_doi(line))
    return dois - {""}


def mention_counts(processed) -> Dict[str, int]:
    counts = {"dc50": 0, "dmax": 0}
    for chunk in iter_text_chunks(processed):
        counts["dc50"] += len(DC50_RE.findall(chunk.text))
        counts["dmax"] += len(DMAX_RE.findall(chunk.text))
    return counts


def apply_manual_policies(records, cache: StorageManager) -> List[Dict[str, Any]]:
    decisions = []
    for record in records:
        if record.screening_status != MANUAL:
            continue
        processed = cache.load_processed_paper(record.pmcid)
        counts = mention_counts(processed) if processed is not None else {"dc50": 0, "dmax": 0}
        if processed is None:
            policy, status = "processed_paper_unavailable", MANUAL
        elif counts["dc50"] == 0 and counts["dmax"] == 0:
            policy, status = MENTION_POLICY, REJECTED
        else:
            policy, status = ACCEPT_POLICY, ACCEPTED
        if status != MANUAL:
            record.screening_status = status
            record.screening_reasons.append(f"policy:{policy}")
        decisions.append({
            "pmcid": record.pmcid, "doi": record.doi, "title": record.title,
            "article_type": record.article_type, "has_dc50": counts["dc50"] > 0,
            "has_dmax": counts["dmax"] > 0, "mention_count": counts["dc50"] + counts["dmax"],
            "policy": policy, "new_screening_status": status, "decided_at": _utc_now(),
        })
    return decisions


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    registry_path: Path = args.registry
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry_path, work_dir / "registry_before.json")

    registry, catalog_summary = load_registry_json(registry_path)
    by_pmcid = lambda: {r.pmcid: r for r in registry.records() if r.pmcid}

    rejected_dois = read_doi_lists(args.rejected_dois)
    vetoed, retried = [], []
    for record in registry.records():
        if record.doi in rejected_dois and record.screening_status != REJECTED_BY_REVIEWER:
            record.screening_status = REJECTED_BY_REVIEWER
            record.screening_reasons.append("reviewer_rejected_in_monthly_pr")
            vetoed.append(record.doi)
        elif record.screening_status == ERROR and args.retry_errors:
            record.screening_status = PENDING
            retried.append(record.pmcid)

    search_counts: Dict[str, int] = {}
    if not args.skip_search:
        search_counts = add_pmc_search_results(registry, PMCClient(), args.max_results_per_query)
    write_registry(registry, registry_path.parent, catalog_summary)

    pending = {r.pmcid for r in registry.records() if r.screening_status == PENDING and r.pmcid}
    logging.info("Pending activity screen: %d articles", len(pending))

    screening_dir = work_dir / "screening"
    screen_summary: Dict[str, Any] = {}
    if pending:
        result = screen_catalog(
            catalog_json=registry_path,
            output_dir=screening_dir,
            include_reviews=args.include_reviews,
            download_accepted=False,
            resume=False,
            download_images=False,
        )
        screen_summary = result["summary"]
        registry, catalog_summary = load_registry_json(registry_path)

    index = by_pmcid()
    records = [index[pmcid] for pmcid in sorted(pending) if pmcid in index]

    cache = StorageManager(screening_dir / "screening_cache")
    decisions = apply_manual_policies(records, cache)
    write_csv(work_dir / "policy_decisions.csv", decisions, [
        "pmcid", "doi", "title", "article_type", "has_dc50", "has_dmax", "mention_count",
        "policy", "new_screening_status", "decided_at",
    ])
    accepted = [r for r in records if r.screening_status == ACCEPTED and not r.confirmed_relevant]
    write_registry(registry, registry_path.parent, catalog_summary)

    papers_dir: Path = args.papers_dir
    staging = papers_dir / "_staging"
    downloads = (
        download_accepted_articles(accepted, staging, None, not args.no_images)
        if accepted else {"statuses": {}}
    )
    policy_by_pmcid = {d["pmcid"]: d["policy"] for d in decisions}
    lists: Dict[str, List[Dict[str, Any]]] = {"MG": [], "PROTAC": []}
    classification = []
    for record in accepted:
        source = Path(record.paper_path) if record.paper_path else staging / record.pmcid
        if downloads["statuses"].get(record.pmcid) not in {"success", "already_downloaded", "reused_local"}:
            continue
        modality, reason, mg_mentions, protac_mentions = classify(record.to_dict(), source)
        destination = papers_dir / ("MGs" if modality == "MG" else "PROTACs") / record.pmcid
        if source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(source), str(destination))
        record.paper_path = str(destination)
        entry = {
            "doi": record.doi, "pmc_id": record.pmcid, "title": record.title,
            "article_type": record.article_type, "modalities": sorted(record.modalities),
            "classification_reason": reason,
            "acceptance_policy": policy_by_pmcid.get(record.pmcid, "activity_rules"),
        }
        lists[modality].append(entry)
        classification.append({**entry, "modality": modality,
                               "mg_mentions": mg_mentions, "protac_mentions": protac_mentions})
    write_registry(registry, registry_path.parent, catalog_summary)

    write_json(work_dir / "accepted_mg.json", lists["MG"])
    write_json(work_dir / "accepted_protac.json", lists["PROTAC"])
    write_json(work_dir / "classification.json", classification)

    summary = {
        "run_label": args.run_label,
        "completed_at": _utc_now(),
        "registry": str(registry_path),
        "search_hits_per_query": search_counts,
        "reviewer_rejected_applied": sorted(vetoed),
        "screen_errors_retried": len(retried),
        "new_candidates": len(pending),
        "screening_status_counts": dict(Counter(r.screening_status for r in records)),
        "policy_counts": dict(Counter(d["policy"] for d in decisions)),
        "policy_accepted": [d for d in decisions if d["new_screening_status"] == ACCEPTED],
        "accepted": len(accepted),
        "download_statuses": dict(Counter(downloads["statuses"].values())),
        "download_failures": sorted(p for p, s in downloads["statuses"].items()
                                    if s not in {"success", "already_downloaded", "reused_local"}),
        "accepted_by_modality": {k: len(v) for k, v in lists.items()},
        "mixed_modality_articles": [c["pmc_id"] for c in classification if len(c["modalities"]) > 1],
        "screen_errors": sorted(r.pmcid for r in records if r.screening_status == ERROR),
        "screening_run": {k: screen_summary.get(k) for k in ("processed_now", "selected_status_counts")},
    }
    write_json(work_dir / "pmc_stage_summary.json", summary)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly PMC search, screening and download.")
    parser.add_argument("--registry", type=Path, required=True, help="Persistent unified_article_list.json")
    parser.add_argument("--papers_dir", type=Path, required=True, help="This month's paper root (MGs/, PROTACs/)")
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--rejected_dois", type=Path, nargs="*", default=[])
    parser.add_argument("--include_reviews", action="store_true",
                        help="Accept reviews with table evidence directly (default: via manual policy).")
    parser.add_argument("--retry_errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_search", action="store_true", help="Only process already pending records.")
    parser.add_argument("--max_results_per_query", type=int, default=None, help="Test cap per query.")
    parser.add_argument("--no_images", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(parse_args(argv))
    print(json.dumps({k: summary[k] for k in (
        "new_candidates", "screening_status_counts", "policy_counts", "accepted_by_modality",
        "download_failures",
    )}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
