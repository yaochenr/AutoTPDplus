"""File-based storage manager for PMC papers."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests

from tpd_curator.pmc_miner.core.models import PaperInventory, PaperMetadata, PaperRecord, ProcessedPaper


class StorageManager:
    """Per-paper file store with inventory index and optional supplementary fetch."""

    def __init__(
        self,
        data_dir: Path,
        inventory_path: Optional[Path] = None,
        search_results_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir)
        self.inventory_path = Path(inventory_path) if inventory_path else self.data_dir / "inventory.json"
        self.search_results_path = Path(search_results_path) if search_results_path else self.data_dir / "search_results.json"
        self.logs_dir = Path(logs_dir) if logs_dir else self.data_dir / "logs"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.inventory = PaperInventory.load_from_file(str(self.inventory_path))

    def get_paper_dir(self, pmcid: str) -> Path:
        # 先把已有的 PMC 前缀去掉,避免拼接出 PMCPMC... 这种错误目录名。
        clean_id = pmcid.replace('PMC', '')
        return self.data_dir / f"PMC{clean_id}"

    def paper_exists(self, pmcid: str) -> bool:
        paper_dir = self.get_paper_dir(pmcid)
        return paper_dir.exists() and (paper_dir / "metadata.json").exists()

    def save_paper_xml(self, pmcid: str, xml_content: str) -> bool:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            paper_dir.mkdir(exist_ok=True)

            xml_path = paper_dir / "full_text.xml"
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

            logging.info(f"Saved XML for PMC{pmcid}")
            return True

        except Exception as e:
            logging.error(f"Failed to save XML for PMC{pmcid}: {e}")
            return False

    def load_paper_xml(self, pmcid: str) -> Optional[str]:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            xml_path = paper_dir / "full_text.xml"

            if xml_path.exists():
                with open(xml_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return None

        except Exception as e:
            logging.error(f"Failed to load XML for PMC{pmcid}: {e}")
            return None

    def save_paper_metadata(self, pmcid: str, metadata: PaperMetadata) -> bool:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            paper_dir.mkdir(exist_ok=True)

            metadata_path = paper_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

            logging.info(f"Saved metadata for PMC{pmcid}")
            return True

        except Exception as e:
            logging.error(f"Failed to save metadata for PMC{pmcid}: {e}")
            return False

    def load_paper_metadata(self, pmcid: str) -> Optional[PaperMetadata]:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            metadata_path = paper_dir / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return PaperMetadata.from_dict(data)
            return None

        except Exception as e:
            logging.error(f"Failed to load metadata for PMC{pmcid}: {e}")
            return None

    def save_processed_paper(self, pmcid: str, processed: ProcessedPaper, download_supplements: bool = True) -> bool:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            paper_dir.mkdir(exist_ok=True)

            if download_supplements and processed.supplementary_info.get('has_supplements', False):
                self._download_supplementary_files(pmcid, processed)

            processed_path = paper_dir / "processed.json"
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(processed.to_dict(), f, indent=2, ensure_ascii=False)

            logging.info(f"Saved processed content for PMC{pmcid}")
            return True

        except Exception as e:
            logging.error(f"Failed to save processed content for PMC{pmcid}: {e}")
            return False

    def load_processed_paper(self, pmcid: str) -> Optional[ProcessedPaper]:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            processed_path = paper_dir / "processed.json"

            if processed_path.exists():
                with open(processed_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ProcessedPaper.from_dict(data)
            return None

        except Exception as e:
            logging.error(f"Failed to load processed content for PMC{pmcid}: {e}")
            return None

    def save_paper_record(self, paper: PaperRecord) -> bool:
        success = True
        pmcid = paper.metadata.pmcid

        if not self.save_paper_metadata(pmcid, paper.metadata):
            success = False

        if paper.raw_xml and not self.save_paper_xml(pmcid, paper.raw_xml):
            success = False

        if paper.processed and not self.save_processed_paper(pmcid, paper.processed):
            success = False

        self.inventory.add_paper(paper)

        return success

    def load_paper_record(self, pmcid: str) -> Optional[PaperRecord]:
        metadata = self.load_paper_metadata(pmcid)
        if not metadata:
            return None

        processed = self.load_processed_paper(pmcid)
        raw_xml = self.load_paper_xml(pmcid)

        inventory_paper = self.inventory.get_paper(pmcid)
        status = inventory_paper.status if inventory_paper else "unknown"
        error_message = inventory_paper.error_message if inventory_paper else None

        return PaperRecord(
            metadata=metadata,
            processed=processed,
            raw_xml=raw_xml,
            status=status,
            error_message=error_message
        )

    def get_all_paper_ids(self) -> List[str]:
        paper_ids = []
        if self.data_dir.exists():
            for paper_dir in self.data_dir.iterdir():
                if paper_dir.is_dir() and paper_dir.name.startswith('PMC'):
                    paper_ids.append(paper_dir.name)
        return sorted(paper_ids)

    def save_inventory(self) -> bool:
        try:
            self.inventory.save_to_file(str(self.inventory_path))
            return True
        except Exception as e:
            logging.error(f"Failed to save inventory: {e}")
            return False

    def save_search_results(self, results: Dict[str, Any]) -> bool:
        try:
            with open(self.search_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Failed to save search results: {e}")
            return False

    def load_search_results(self) -> Optional[Dict[str, Any]]:
        try:
            if self.search_results_path.exists():
                with open(self.search_results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logging.error(f"Failed to load search results: {e}")
            return None

    def download_image(self, pmcid: str, image_url: str, figure_id: str) -> Optional[str]:
        try:
            paper_dir = self.get_paper_dir(pmcid)
            images_dir = paper_dir / "images"
            images_dir.mkdir(exist_ok=True)

            # 相对路径的图片引用,需要拼接到 PMC 文章的 /bin/ 基础路径上。
            if not image_url.startswith('http'):
                clean_pmcid = pmcid.replace('PMC', '')
                base_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{clean_pmcid}/bin/"
                image_url = urljoin(base_url, image_url)

            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                for ext in ['.jpg', '.png', '.gif', '.tif', '.pdf']:
                    test_url = image_url + ext
                    if self._url_exists(test_url):
                        image_url = test_url
                        filename = f"{figure_id or 'figure'}{ext}"
                        break
                else:
                    filename = f"{figure_id or 'figure'}.jpg"

            response = requests.get(image_url, stream=True, timeout=30)
            response.raise_for_status()

            image_path = images_dir / filename
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Downloaded image for PMC{pmcid}: {filename}")
            return str(image_path.relative_to(paper_dir))

        except Exception as e:
            logging.warning(f"Failed to download image for PMC{pmcid} ({image_url}): {e}")
            return None

    def _url_exists(self, url: str) -> bool:
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def download_figure_images(self, pmcid: str, figures: List[Dict[str, str]]) -> List[Dict[str, str]]:
        updated_figures = []

        for figure in figures:
            updated_figure = figure.copy()
            image_url = figure.get('image_url', '')

            if image_url:
                local_path = self.download_image(pmcid, image_url, figure.get('id', ''))
                if local_path:
                    updated_figure['local_image_path'] = local_path

            updated_figures.append(updated_figure)

        return updated_figures

    def _download_supplementary_files(self, pmcid: str, processed: ProcessedPaper) -> None:
        try:
            # 延迟导入:避免模块加载时出现循环依赖。
            from tpd_curator.pmc_miner.core.supplementary_downloader import SupplementaryDownloader

            downloader = SupplementaryDownloader(self.data_dir)

            logging.info(f"Starting supplementary download for PMC{pmcid}")

            supp_info = downloader.download_supplementary_info(pmcid)

            if supp_info.has_supplements:
                processed.supplementary_info.update({
                    'download_status': supp_info.download_status,
                    'download_date': supp_info.download_date,
                    'total_files': supp_info.total_files,
                    'total_size_bytes': supp_info.total_size_bytes,
                    'downloaded_files': [
                        {
                            'id': f.id,
                            'filename': f.filename,
                            'caption': f.caption,
                            'file_type': f.file_type,
                            'size_bytes': f.size_bytes,
                            'local_path': f.local_path,
                            'content_summary': f.content_summary
                        }
                        for f in supp_info.files
                    ]
                })

                logging.info(f"Successfully downloaded {len(supp_info.files)} supplementary files for PMC{pmcid}")
            else:
                logging.info(f"No supplementary files found for PMC{pmcid}")

            downloader.cleanup_temporary_files(pmcid)

        except ImportError:
            logging.warning(f"SupplementaryDownloader not available - skipping supplementary download for PMC{pmcid}")
        except Exception as e:
            logging.error(f"Error downloading supplementary files for PMC{pmcid}: {e}")

    def has_supplementary_files(self, pmcid: str) -> bool:
        paper_dir = self.get_paper_dir(pmcid)
        supp_dir = paper_dir / "supplementary"
        return supp_dir.exists() and any(supp_dir.iterdir())

    def list_supplementary_files(self, pmcid: str) -> List[Dict[str, str]]:
        supp_files = []
        paper_dir = self.get_paper_dir(pmcid)
        supp_dir = paper_dir / "supplementary"

        if supp_dir.exists():
            metadata_file = supp_dir / "supplementary_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    return metadata.get('files', [])
                except Exception as e:
                    logging.warning(f"Failed to read supplementary metadata for PMC{pmcid}: {e}")

            for file_path in supp_dir.glob('*'):
                if file_path.is_file() and file_path.name != 'supplementary_metadata.json':
                    supp_files.append({
                        'filename': file_path.name,
                        'local_path': str(file_path),
                        'size_bytes': file_path.stat().st_size
                    })

        return supp_files

    def get_storage_summary(self) -> Dict[str, Any]:
        all_papers = self.get_all_paper_ids()
        inventory_summary = self.inventory.get_summary()

        complete_papers = 0
        papers_with_images = 0
        for pmcid in all_papers:
            paper_dir = self.get_paper_dir(pmcid)
            has_metadata = (paper_dir / "metadata.json").exists()
            has_xml = (paper_dir / "full_text.xml").exists()
            has_processed = (paper_dir / "processed.json").exists()
            has_images = (paper_dir / "images").exists()

            if has_metadata and (has_xml or has_processed):
                complete_papers += 1
            if has_images:
                papers_with_images += 1

        return {
            "total_directories": len(all_papers),
            "complete_papers": complete_papers,
            "papers_with_images": papers_with_images,
            "inventory_summary": inventory_summary,
            "storage_path": str(self.data_dir),
            "inventory_path": str(self.inventory_path)
        }
