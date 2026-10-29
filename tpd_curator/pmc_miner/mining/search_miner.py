"""Keyword-search based mining with dual (init + filtered) storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tpd_curator.pmc_miner.core.client import PMCClient
from tpd_curator.pmc_miner.core.config import (
    BATCH_SIZE,
    MAX_PAPERS,
    MOLECULAR_GLUE_KEYWORDS,
    POST_FILTER_KEYWORDS,
)
from tpd_curator.pmc_miner.core.models import PaperMetadata, PaperRecord
from tpd_curator.pmc_miner.core.processor import TextProcessor
from tpd_curator.pmc_miner.core.storage import StorageManager
from tpd_curator.pmc_miner.utils.image_downloader import PMCImageDownloader


class PostFilter:
    """Check whether a paper's text matches any post-filter keyword."""

    def __init__(self, keywords: Optional[List[str]] = None):
        self.keywords = keywords or POST_FILTER_KEYWORDS

    def matches(self, paper: PaperRecord) -> Tuple[bool, Dict]:
        combined = ""
        if paper.metadata.abstract:
            combined += paper.metadata.abstract.lower() + " "
        if paper.processed and paper.processed.body_text:
            combined += paper.processed.body_text.lower()

        matched = {kw for kw in self.keywords if kw.lower() in combined}
        return bool(matched), {
            "total_unique_matches": list(matched),
            "match_count": len(matched),
        }


class SearchBasedMiner:
    """Two-stage pipeline: broad search, post-filter."""

    def __init__(
        self,
        output_dir: str,
        init_dir_name: str = "init_papers",
        filtered_dir_name: str = "filtered_papers",
        keywords: Optional[List[str]] = None,
        post_filter_keywords: Optional[List[str]] = None,
        download_images: bool = True,
        images_for: str = "filtered",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keywords = keywords or MOLECULAR_GLUE_KEYWORDS

        self.init_storage = StorageManager(
            data_dir=self.output_dir / init_dir_name,
            inventory_path=self.output_dir / f"inventory_{init_dir_name}.json",
            search_results_path=self.output_dir / "search_results.json",
        )
        self.filtered_storage = StorageManager(
            data_dir=self.output_dir / filtered_dir_name,
            inventory_path=self.output_dir / f"inventory_{filtered_dir_name}.json",
        )

        self.client = PMCClient()
        self.processor = TextProcessor()
        self.post_filter = PostFilter(post_filter_keywords)

        if images_for not in {"filtered", "init", "none"}:
            raise ValueError("images_for must be 'filtered', 'init', or 'none'")
        self.download_images = download_images and images_for != "none"
        self.images_for = images_for
        self.init_image_downloader = (
            PMCImageDownloader(self.init_storage)
            if self.download_images and images_for == "init"
            else None
        )
        self.filtered_image_downloader = (
            PMCImageDownloader(self.filtered_storage)
            if self.download_images and images_for == "filtered"
            else None
        )

    def mine_papers(
        self,
        max_papers: int = MAX_PAPERS,
        batch_size: int = BATCH_SIZE,
    ) -> Dict:
        session_start = datetime.now().isoformat()
        logging.info("Starting PMC search-based mining")
        logging.info(f"Output dir: {self.output_dir}")
        logging.info(f"Keywords: {self.keywords}")

        pmc_ids = self.client.search_papers(self.keywords, max_results=max_papers)
        if not pmc_ids:
            logging.error("No papers returned from PMC search")
            return {"initial_papers_found": 0}

        # PMCClient already retrieves ESummary records in API-sized batches.
        # The previous slice silently limited every run to the first
        # ``batch_size`` hits (100 by default), even when max_papers was larger.
        papers = self.client.get_paper_metadata(pmc_ids)
        if not papers:
            logging.error("No metadata retrieved for search hits")
            return {"initial_papers_found": len(pmc_ids), "initial_papers_stored": 0}

        logging.info(f"Processing {len(papers)} papers (of {len(pmc_ids)} hits)")

        initial_stored, filtered_stored, failed = [], [], []
        for i, paper_meta in enumerate(papers, 1):
            pmcid = paper_meta["pmcid"]
            logging.info(f"[{i}/{len(papers)}] {pmcid}")
            init_ok, filtered_ok = self._process_paper(paper_meta)
            if init_ok:
                initial_stored.append(pmcid)
            else:
                failed.append(pmcid)
            if filtered_ok:
                filtered_stored.append(pmcid)

        summary = {
            "session_start_time": session_start,
            "completion_time": datetime.now().isoformat(),
            "search_keywords": self.keywords,
            "post_filter_keywords": self.post_filter.keywords,
            "initial_papers_found": len(pmc_ids),
            "initial_papers_stored": len(initial_stored),
            "filtered_papers_stored": len(filtered_stored),
            "failed_papers": len(failed),
            "filter_success_rate": (
                len(filtered_stored) / len(initial_stored) * 100 if initial_stored else 0
            ),
            "storage_locations": {
                "init": str(self.init_storage.data_dir),
                "filtered": str(self.filtered_storage.data_dir),
            },
            "stored_papers": {
                "initial": initial_stored,
                "filtered": filtered_stored,
                "failed": failed,
            },
        }

        with open(self.init_storage.search_results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.init_storage.save_inventory()
        self.filtered_storage.save_inventory()
        self._write_stats_file(summary)
        logging.info(
            f"Done. initial={len(initial_stored)}, filtered={len(filtered_stored)}, "
            f"failed={len(failed)}"
        )
        return summary

    def _process_paper(self, paper_meta: Dict) -> Tuple[bool, bool]:
        pmcid = paper_meta["pmcid"]

        if self.init_storage.paper_exists(pmcid):
            existing = self.init_storage.load_paper_record(pmcid)
            if existing:
                matched, _ = self.post_filter.matches(existing)
                if matched and not self.filtered_storage.paper_exists(pmcid):
                    self.filtered_storage.save_paper_record(existing)
                return True, matched
            return True, False

        try:
            xml_content = self.client.get_full_text_xml(pmcid)
            if not xml_content:
                return False, False
            if not self.client.validate_full_text_availability(xml_content):
                logging.info(f"{pmcid} is not Open Access — skipping")
                return False, False

            processed = self.processor.process_xml(xml_content, pmcid)
            if not processed:
                return False, False

            xml_meta = processed.xml_metadata or {}
            metadata = PaperMetadata(
                pmcid=pmcid,
                pmid=paper_meta.get("pmid") or xml_meta.get("pmid", ""),
                doi=paper_meta.get("doi") or xml_meta.get("doi", ""),
                title=paper_meta.get("title", ""),
                authors=paper_meta.get("authors", []),
                journal=paper_meta.get("journal") or xml_meta.get("journal", ""),
                pubdate=paper_meta.get("pubdate") or xml_meta.get("pubdate", ""),
                abstract=processed.abstract or "",
            )

            record = PaperRecord(
                metadata=metadata,
                processed=processed,
                raw_xml=xml_content,
                status="completed",
            )

            if not self.init_storage.save_paper_record(record):
                return False, False

            matched, details = self.post_filter.matches(record)
            if matched:
                self.filtered_storage.save_paper_record(record)
                logging.info(f"{pmcid} passed filter on {details['total_unique_matches']}")

            self._download_images(pmcid, matched)
            return True, matched

        except Exception as e:
            logging.error(f"Error processing {pmcid}: {e}")
            return False, False

    def _download_images(self, pmcid: str, matched: bool) -> None:
        downloader = None
        if self.init_image_downloader is not None:
            downloader = self.init_image_downloader
        elif self.filtered_image_downloader is not None and matched:
            downloader = self.filtered_image_downloader
        if downloader is None:
            return
        try:
            downloader.download_paper_images(pmcid)
        except Exception as e:
            logging.warning(f"Image download error for {pmcid}: {e}")

    def _write_stats_file(self, summary: Dict) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for storage in (self.init_storage, self.filtered_storage):
            path = storage.data_dir / f"mining_stat_{timestamp}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write("PMC Search Mining Statistics\n")
                f.write(f"Session start: {summary['session_start_time']}\n")
                f.write(f"Completion: {summary['completion_time']}\n\n")
                f.write(f"Search keywords: {summary['search_keywords']}\n")
                f.write(f"Filter keywords: {summary['post_filter_keywords']}\n\n")
                f.write(f"Initial papers found: {summary['initial_papers_found']}\n")
                f.write(f"Initial stored: {summary['initial_papers_stored']}\n")
                f.write(f"Filtered stored: {summary['filtered_papers_stored']}\n")
                f.write(f"Failed: {summary['failed_papers']}\n")
                f.write(f"Filter success rate: {summary['filter_success_rate']:.1f}%\n")
