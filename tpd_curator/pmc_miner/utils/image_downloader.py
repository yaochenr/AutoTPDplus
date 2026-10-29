"""Image downloader for PMC papers via the pmc-oa-opendata S3 bucket."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from tpd_curator.pmc_miner.core.storage import StorageManager


PMC_OA_S3_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
_IMAGE_EXT_PATTERN = re.compile(r'\.(jpe?g|png|gif|tiff?)$', re.IGNORECASE)


class PMCImageDownloader:
    """Downloads figure images from the PMC OA S3 bucket (HTML scraping is blocked by reCAPTCHA)."""

    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PMC-Image-Downloader/1.0 (mailto:research@example.com)'
        })
        self._last_page_was_stub = False

    def _list_s3_article_objects(self, pmcid: str) -> Optional[Dict[str, object]]:
        try:
            list_url = f"{PMC_OA_S3_BUCKET_URL}?list-type=2&prefix={pmcid}"
            response = self.session.get(list_url, timeout=30)
            if response.status_code != 200:
                logging.warning(f"S3 list failed for {pmcid}: HTTP {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, 'xml')
            keys = [k.get_text() for k in soup.find_all('Key')]
            if not keys:
                return None

            version_re = re.compile(rf'^{re.escape(pmcid)}\.(\d+)/')
            versions = {int(m.group(1)) for k in keys if (m := version_re.match(k))}
            if not versions:
                return None

            latest = max(versions)
            prefix = f"{pmcid}.{latest}/"
            article_keys = [k for k in keys if k.startswith(prefix)]

            xml_key = next(
                (k for k in article_keys if k.lower().endswith(('.nxml', '.xml'))),
                None,
            )
            image_keys = [k for k in article_keys if _IMAGE_EXT_PATTERN.search(k)]

            return {
                'version': latest,
                'prefix': prefix,
                'xml_key': xml_key,
                'image_keys': image_keys,
            }
        except Exception as e:
            logging.warning(f"Failed to list S3 objects for {pmcid}: {e}")
            return None

    def _fetch_xml_captions(self, xml_url: str) -> Dict[str, Dict[str, str]]:
        try:
            response = self.session.get(xml_url, timeout=30)
            if response.status_code != 200:
                return {}
            soup = BeautifulSoup(response.content, 'xml')
            mapping: Dict[str, Dict[str, str]] = {}

            for fig in soup.find_all('fig'):
                graphic = fig.find('graphic')
                if not graphic:
                    continue
                href = graphic.get('xlink:href') or graphic.get('href') or ''
                if not href:
                    continue
                basename = href.split('/')[-1]
                label_el = fig.find('label')
                caption_el = fig.find('caption')
                mapping[basename] = {
                    'id': fig.get('id', ''),
                    'label': label_el.get_text(strip=True) if label_el else '',
                    'caption': caption_el.get_text(' ', strip=True) if caption_el else '',
                }

            # add graphics outside of <fig>
            for graphic in soup.find_all('graphic'):
                if graphic.find_parent('fig'):
                    continue
                href = graphic.get('xlink:href') or graphic.get('href') or ''
                if not href:
                    continue
                basename = href.split('/')[-1]
                if basename in mapping:
                    continue
                in_abstract = graphic.find_parent('abstract') is not None
                mapping[basename] = {
                    'id': graphic.get('id') or ('graphical_abstract' if in_abstract else ''),
                    'label': 'Graphical Abstract' if in_abstract else '',
                    'caption': 'Graphical Abstract' if in_abstract else '',
                }

            return mapping
        except Exception as e:
            logging.warning(f"Failed to parse XML captions from {xml_url}: {e}")
            return {}

    def scrape_figure_images(self, pmcid: str) -> List[Dict[str, str]]:
        self._last_page_was_stub = False
        try:
            logging.info(f"Listing S3 figure objects for {pmcid}")
            listing = self._list_s3_article_objects(pmcid)
            if not listing:
                self._last_page_was_stub = True
                logging.info(f"No S3 listing available for {pmcid}")
                return []

            image_keys: List[str] = listing['image_keys']
            xml_key = listing['xml_key']

            caption_map: Dict[str, Dict[str, str]] = {}
            if xml_key:
                caption_map = self._fetch_xml_captions(PMC_OA_S3_BUCKET_URL + xml_key)

            figures = []
            for key in image_keys:
                filename = key.split('/')[-1]
                meta = caption_map.get(filename, {})
                figures.append({
                    'id': meta.get('id') or Path(filename).stem,
                    'caption': (meta.get('caption') or '')[:500],
                    'image_url': PMC_OA_S3_BUCKET_URL + key,
                    'alt_text': meta.get('label', ''),
                })

            logging.info(f"Found {len(figures)} figure images for {pmcid}")
            return figures

        except Exception as e:
            logging.error(f"Failed to enumerate figures for {pmcid}: {e}")
            self._last_page_was_stub = True
            return []

    def _mark_for_retry(self, pmcid: str) -> None:
        clean_id = pmcid.replace('PMC', '')
        retry_path = self.storage.data_dir / "images_retry.txt"
        try:
            existing = set()
            if retry_path.exists():
                existing = {ln.strip() for ln in retry_path.read_text().splitlines() if ln.strip()}
            existing.add(f"PMC{clean_id}")
            retry_path.write_text("\n".join(sorted(existing)) + "\n")
        except Exception as e:
            logging.warning(f"Could not write images_retry.txt: {e}")

    def download_paper_images(self, pmcid: str) -> bool:
        try:
            figures = self.scrape_figure_images(pmcid)
            if not figures:
                logging.info(f"No figures found for PMC{pmcid}")
                if self._last_page_was_stub:
                    self._mark_for_retry(pmcid)
                return True

            paper_dir = self.storage.get_paper_dir(pmcid)
            images_dir = paper_dir / "images"
            images_dir.mkdir(exist_ok=True)

            downloaded_figures = []

            for i, figure in enumerate(figures):
                image_url = figure['image_url']
                figure_id = figure.get('id') or f"figure_{i+1}"

                parsed_url = urlparse(image_url)
                original_filename = Path(parsed_url.path).name

                if '.' in original_filename:
                    filename = f"{figure_id}_{original_filename}"
                else:
                    ext = '.jpg'
                    if 'png' in image_url.lower():
                        ext = '.png'
                    elif 'gif' in image_url.lower():
                        ext = '.gif'
                    elif 'tif' in image_url.lower():
                        ext = '.tif'
                    filename = f"{figure_id}{ext}"

                try:
                    response = self.session.get(image_url, stream=True, timeout=30)
                    response.raise_for_status()

                    image_path = images_dir / filename
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    figure_with_path = figure.copy()
                    figure_with_path['local_image_path'] = f"images/{filename}"
                    downloaded_figures.append(figure_with_path)

                    logging.info(f"Downloaded: {filename}")
                    time.sleep(0.5)  # Be polite to NCBI.

                except Exception as e:
                    logging.warning(f"Failed to download {image_url}: {e}")
                    downloaded_figures.append(figure)

            figures_metadata_path = images_dir / "figures_metadata.json"
            with open(figures_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(downloaded_figures, f, indent=2, ensure_ascii=False)

            logging.info(f"Downloaded {len([f for f in downloaded_figures if 'local_image_path' in f])} images for PMC{pmcid}")
            return True

        except Exception as e:
            logging.error(f"Failed to download images for PMC{pmcid}: {e}")
            return False

    def download_images_for_all_papers(self) -> Dict[str, bool]:
        all_papers = self.storage.get_all_paper_ids()
        results = {}

        logging.info(f"Starting image download for {len(all_papers)} papers...")

        for i, pmcid in enumerate(all_papers, 1):
            logging.info(f"Processing {pmcid} ({i}/{len(all_papers)})")

            paper_dir = self.storage.get_paper_dir(pmcid)
            images_dir = paper_dir / "images"
            if images_dir.exists() and any(images_dir.iterdir()):
                logging.info(f"Images already exist for {pmcid}, skipping")
                results[pmcid] = True
                continue

            success = self.download_paper_images(pmcid)
            results[pmcid] = success

            time.sleep(1)

        success_count = sum(results.values())
        logging.info(f"Image download complete: {success_count}/{len(all_papers)} papers successful")

        return results
