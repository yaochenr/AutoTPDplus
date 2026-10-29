"""Unified command-line interface for pmc_miner.

Usage examples:
    python -m tpd_curator.pmc_miner.cli search --output-dir results/ --max-papers 100
    python -m tpd_curator.pmc_miner.cli doi    --input dois.csv --output-dir doi_results/ --paper-type molglue
    python -m tpd_curator.pmc_miner.cli catalog --data-dir data/ --output-dir catalog/ --resolve-pmc --include-search
    python -m tpd_curator.pmc_miner.cli images --data-dir  results/filtered_papers
    python -m tpd_curator.pmc_miner.cli image-summary --data-dir results/filtered_papers
"""

import argparse
from pathlib import Path

from tpd_curator.pmc_miner.core.config import BATCH_SIZE, MAX_PAPERS
from tpd_curator.pmc_miner.mining.doi_miner import DOIBasedMiner
from tpd_curator.pmc_miner.mining.search_miner import SearchBasedMiner
from tpd_curator.pmc_miner.utils.logging import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tpd_curator.pmc_miner.cli",
        description="Mine full-text papers from PMC Central.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Keyword-based mining with dual storage")
    s.add_argument("--output-dir", required=True, help="Root directory for mined data")
    s.add_argument("--max-papers", type=int, default=MAX_PAPERS)
    s.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    s.add_argument(
        "--init-dir-name",
        default="init_papers",
        help="Sub-directory inside --output-dir for ALL initially found papers",
    )
    s.add_argument(
        "--filtered-dir-name",
        default="filtered_papers",
        help="Sub-directory inside --output-dir for post-filtered papers",
    )
    s.add_argument(
        "--no-images",
        action="store_true",
        help="Skip figure image downloads (faster)",
    )
    s.add_argument(
        "--images-for",
        choices=["filtered", "init"],
        default="filtered",
        help="Download images only for filtered papers (default) or for every initial paper",
    )

    d = sub.add_parser("doi", help="DOI-based mining from a CSV of DOIs")
    d.add_argument("--input", required=True, help="CSV / text file of DOIs (one per line)")
    d.add_argument("--output-dir", required=True)
    d.add_argument(
        "--paper-type",
        default="generic",
        help="Label used in inventory/results filenames (e.g. molglue, protac)",
    )
    d.add_argument(
        "--no-images",
        action="store_true",
        help="Skip figure image downloads (faster)",
    )

    c = sub.add_parser(
        "catalog",
        help="Merge database citations and optional PMC keyword results into one paper list",
    )
    c.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing MGDB, MGTbind, TACK2, and PROTAC-DB exports",
    )
    c.add_argument("--output-dir", required=True)
    c.add_argument(
        "--resolve-pmc",
        action="store_true",
        help="Resolve seed DOI/PMID identifiers to PMC IDs and canonical metadata",
    )
    c.add_argument(
        "--include-search",
        action="store_true",
        help="Also run PROTAC and molecular-glue PMC Open Access keyword searches",
    )
    c.add_argument(
        "--max-results-per-query",
        type=int,
        default=None,
        help="Optional test cap for each PMC query; omit to retrieve every hit",
    )
    c.add_argument(
        "--download-confirmed",
        action="store_true",
        help="Download every database-confirmed article that is available in PMC OA",
    )
    c.add_argument(
        "--no-images",
        action="store_true",
        help="When downloading confirmed articles, skip figure images",
    )
    c.add_argument(
        "--reuse-dir",
        default=None,
        help="Existing data_source root whose complete PMC folders should be reused",
    )

    cd = sub.add_parser(
        "catalog-download",
        help="Download confirmed PMC OA articles from an existing unified catalog",
    )
    cd.add_argument("--catalog-json", required=True)
    cd.add_argument("--output-dir", required=True)
    cd.add_argument("--reuse-dir", default=None)
    cd.add_argument("--no-images", action="store_true")

    cs = sub.add_parser(
        "catalog-screen",
        help="Screen PMC OA catalog candidates for quantitative degradation activity",
    )
    cs.add_argument("--catalog-json", required=True)
    cs.add_argument("--output-dir", required=True)
    cs.add_argument("--year-from", type=int, default=None)
    cs.add_argument("--year-to", type=int, default=None)
    cs.add_argument("--include-reviews", action="store_true")
    cs.add_argument("--download-accepted", action="store_true")
    cs.add_argument("--reuse-dir", default=None)
    cs.add_argument("--no-images", action="store_true")
    cs.add_argument("--resume", action="store_true")

    i = sub.add_parser("images", help="Download figures for all papers in a directory")
    i.add_argument("--data-dir", required=True, help="Directory containing PMC* sub-dirs")

    sm = sub.add_parser("image-summary", help="Print image-download stats for a data dir")
    sm.add_argument("--data-dir", required=True)

    return parser


def _run_search(args: argparse.Namespace) -> None:
    miner = SearchBasedMiner(
        output_dir=args.output_dir,
        init_dir_name=args.init_dir_name,
        filtered_dir_name=args.filtered_dir_name,
        download_images=not args.no_images,
        images_for=args.images_for,
    )
    miner.mine_papers(max_papers=args.max_papers, batch_size=args.batch_size)


def _run_doi(args: argparse.Namespace) -> None:
    miner = DOIBasedMiner(
        output_dir=args.output_dir,
        paper_type=args.paper_type,
        download_images=not args.no_images,
    )
    miner.mine_from_csv(args.input)


def _run_catalog(args: argparse.Namespace) -> None:
    from tpd_curator.pmc_miner.mining.citation_registry import build_unified_registry

    result = build_unified_registry(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        resolve_pmc=args.resolve_pmc,
        include_search=args.include_search,
        max_results_per_query=args.max_results_per_query,
        download_confirmed=args.download_confirmed,
        download_images=not args.no_images,
        reuse_dir=Path(args.reuse_dir) if args.reuse_dir else None,
    )
    summary = result["summary"]
    print(f"Unified articles: {summary['unified_articles']}")
    print(f"Confirmed database articles: {summary['confirmed_relevant_articles']}")
    print(f"Articles with PMCID: {summary['articles_with_pmcid']}")
    print(f"Database/search overlap: {summary['database_and_search_overlap']}")
    print(f"CSV: {result['paths']['csv']}")


def _run_catalog_download(args: argparse.Namespace) -> None:
    from tpd_curator.pmc_miner.mining.citation_registry import download_existing_catalog

    result = download_existing_catalog(
        catalog_json=Path(args.catalog_json),
        output_dir=Path(args.output_dir),
        reuse_dir=Path(args.reuse_dir) if args.reuse_dir else None,
        download_images=not args.no_images,
    )
    stats = result['summary']['confirmed_downloads']
    print(f"Eligible confirmed PMC OA articles: {stats['eligible_confirmed_articles']}")
    print(f"Reused local articles: {stats['reused_local']}")
    print(f"Newly downloaded articles: {stats['newly_downloaded']}")


def _run_catalog_screen(args: argparse.Namespace) -> None:
    from tpd_curator.pmc_miner.mining.activity_screen import screen_catalog

    result = screen_catalog(
        catalog_json=Path(args.catalog_json),
        output_dir=Path(args.output_dir),
        year_from=args.year_from,
        year_to=args.year_to,
        include_reviews=args.include_reviews,
        download_accepted=args.download_accepted,
        reuse_dir=Path(args.reuse_dir) if args.reuse_dir else None,
        download_images=not args.no_images,
        resume=args.resume,
    )
    summary = result["summary"]
    print(f"Selected articles: {summary['selected_articles']}")
    print(f"Processed now: {summary['processed_now']}")
    print(f"Status counts: {summary['selected_status_counts']}")
    print(f"Cumulative screened: {summary['cumulative_screened_articles']}")
    print(f"Summary: {result['paths']['summary']}")


def _run_images(args: argparse.Namespace) -> None:
    from tpd_curator.pmc_miner.core.storage import StorageManager
    from tpd_curator.pmc_miner.utils.image_downloader import PMCImageDownloader

    storage = StorageManager(data_dir=Path(args.data_dir))
    PMCImageDownloader(storage).download_images_for_all_papers()


def _run_image_summary(args: argparse.Namespace) -> None:
    from tpd_curator.pmc_miner.utils.image_summary import main as run_summary

    run_summary(args.data_dir)


def main() -> None:
    setup_logging()
    args = _build_parser().parse_args()
    {
        "search": _run_search,
        "doi": _run_doi,
        "catalog": _run_catalog,
        "catalog-download": _run_catalog_download,
        "catalog-screen": _run_catalog_screen,
        "images": _run_images,
        "image-summary": _run_image_summary,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
