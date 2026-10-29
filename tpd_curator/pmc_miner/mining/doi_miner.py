"""DOI-driven mining: take a list of DOIs and mine each paper from PMC."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tpd_curator.pmc_miner.core.client import PMCClient
from tpd_curator.pmc_miner.core.models import PaperMetadata, PaperRecord
from tpd_curator.pmc_miner.core.processor import TextProcessor
from tpd_curator.pmc_miner.core.storage import StorageManager
from tpd_curator.pmc_miner.utils.image_downloader import PMCImageDownloader


def load_dois_from_csv(csv_file: str) -> List[str]:
    dois = []
    path = Path(csv_file)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            doi = raw.strip()
            if not doi or doi.startswith("#"):
                continue
            if doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            if doi.startswith("10."):
                dois.append(doi)
    logging.info(f"Loaded {len(dois)} DOIs from {csv_file}")
    return dois


class DOIBasedMiner:

    def __init__(
        self,
        output_dir: str,
        paper_type: str = "generic",
        download_images: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paper_type = paper_type
        self.download_images = download_images

        self.storage = StorageManager(
            data_dir=self.output_dir,
            inventory_path=self.output_dir / f"inventory_{paper_type}.json",
            search_results_path=self.output_dir / f"search_results_{paper_type}.json",
            logs_dir=self.output_dir / "logs",
        )
        self.client = PMCClient()
        self.processor = TextProcessor()
        self.image_downloader = PMCImageDownloader(self.storage) if download_images else None

    def mine_from_csv(self, csv_file: str) -> Dict:
        return self.mine_from_list(load_dois_from_csv(csv_file), source=str(csv_file))

    def mine_from_list(self, dois: List[str], source: Optional[str] = None) -> Dict:
        session_start = datetime.now().isoformat()
        logging.info(f"DOI-based mining ({self.paper_type}): {len(dois)} DOIs")

        stats = {
            "total_dois": len(dois),
            "found_pmc_ids": 0,
            "successfully_processed": 0,
            "failed_processing": 0,
            "not_in_pmc": 0,
            "not_open_access": 0,
            "already_processed": 0,
        }
        successful, failed, not_in_pmc, not_open_access = [], [], [], []

        for i, doi in enumerate(dois, 1):
            logging.info(f"[{i}/{len(dois)}] {doi}")
            pmc_id = self.client.doi_to_pmcid(doi)
            if not pmc_id:
                stats["not_in_pmc"] += 1
                not_in_pmc.append(doi)
                failed.append({"doi": doi, "error": "not found in PMC"})
                continue

            stats["found_pmc_ids"] += 1

            if self.storage.paper_exists(pmc_id):
                stats["already_processed"] += 1
                logging.info(f"{pmc_id} already processed, skipping")
                continue

            result = self._process_doi(doi, pmc_id)
            if result == "success":
                stats["successfully_processed"] += 1
                successful.append({"doi": doi, "pmc_id": pmc_id})
            elif result == "not_open_access":
                stats["not_open_access"] += 1
                not_open_access.append(doi)
            else:
                stats["failed_processing"] += 1
                failed.append({"doi": doi, "pmc_id": pmc_id, "error": result})

        self._write_excluded_dois(not_in_pmc, not_open_access, stats["total_dois"])
        summary = self._write_results(session_start, source, stats, successful, failed)
        self._log_summary(stats)
        return summary

    def _process_doi(self, doi: str, pmc_id: str) -> str:
        try:
            metadata_list = self.client.get_paper_metadata([pmc_id])
            if not metadata_list:
                return "metadata_failed"
            paper_meta = metadata_list[0]

            xml_content = self.client.get_full_text_xml(pmc_id)
            if not xml_content:
                return "no_xml"
            if not self.client.validate_full_text_availability(xml_content):
                return "not_open_access"

            processed = self.processor.process_xml(xml_content, pmc_id)
            if not processed:
                return "processing_failed"

            xml_meta = processed.xml_metadata or {}
            metadata = PaperMetadata(
                pmcid=pmc_id,
                pmid=paper_meta.get("pmid") or xml_meta.get("pmid", ""),
                doi=doi or paper_meta.get("doi") or xml_meta.get("doi", ""),
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
            if not self.storage.save_paper_record(record):
                return "storage_failed"

            if self.image_downloader:
                try:
                    self.image_downloader.download_paper_images(pmc_id)
                except Exception as e:
                    logging.warning(f"Image download error for {pmc_id}: {e}")

            return "success"
        except Exception as e:
            logging.error(f"Error processing {doi}: {e}")
            return "exception_error"

    def _write_excluded_dois(
        self, not_in_pmc: List[str], not_open_access: List[str], total: int
    ) -> None:
        path = self.output_dir / "excluded_dois.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {self.paper_type} DOIs excluded from processing\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(
                f"# Total excluded: {len(not_in_pmc) + len(not_open_access)} / {total}\n\n"
            )
            f.write(f"# === NOT FOUND IN PMC ({len(not_in_pmc)}) ===\n")
            for doi in not_in_pmc:
                f.write(f"{doi}\n")
            f.write(f"\n# === IN PMC BUT NOT OPEN ACCESS ({len(not_open_access)}) ===\n")
            for doi in not_open_access:
                f.write(f"{doi}\n")
        logging.info(f"Saved excluded DOIs to {path}")

    def _write_results(
        self,
        session_start: str,
        source: Optional[str],
        stats: Dict,
        successful: List[Dict],
        failed: List[Dict],
    ) -> Dict:
        summary = {
            "session_start_time": session_start,
            "completion_time": datetime.now().isoformat(),
            "source_file": source,
            "output_directory": str(self.output_dir),
            "paper_type": self.paper_type,
            "statistics": stats,
            "successful_papers": successful,
            "failed_papers": failed,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"mining_results_{timestamp}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        stats_path = self.output_dir / f"mining_stat_{timestamp}.txt"
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"DOI-based PMC {self.paper_type} Mining Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Session: {session_start} → {summary['completion_time']}\n")
            f.write(f"Source:  {source}\n")
            f.write(f"Output:  {self.output_dir}\n\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
            if stats["found_pmc_ids"]:
                rate = stats["successfully_processed"] / stats["found_pmc_ids"] * 100
                f.write(f"\nSuccess rate: {rate:.1f}%\n")
        return summary

    def _log_summary(self, stats: Dict) -> None:
        logging.info("=" * 60)
        logging.info(f"DOI-based mining complete ({self.paper_type})")
        for k, v in stats.items():
            logging.info(f"  {k}: {v}")
