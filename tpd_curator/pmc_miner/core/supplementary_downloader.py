"""Supplementary information downloader for PMC papers."""

import re
import requests
import logging
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup

from tpd_curator.pmc_miner.core.supplementary_archives import (
    SUPPLEMENT_EXTENSIONS,
    convert_legacy_docs,
    extract_zip_archives,
    sha256_file,
)


PMC_OA_S3_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
SCREENABLE_SUPPLEMENT_EXTENSIONS = SUPPLEMENT_EXTENSIONS


@dataclass
class SupplementaryFile:
    """One supplementary file from a PMC OA package."""
    id: str
    filename: str
    caption: str
    file_type: str
    size_bytes: int
    local_path: str
    content_summary: Optional[str] = None


@dataclass
class SupplementaryInfo:
    """All supplementary information for a paper."""
    pmcid: str
    has_supplements: bool
    footnote_references: List[str]
    external_links: List[Dict[str, str]]
    files: List[SupplementaryFile]
    download_status: str
    download_date: str
    total_files: int
    total_size_bytes: int


class SupplementaryDownloader:
    """Downloads supplementary files via the PMC OA tar.gz package service."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PMC-Supplementary-Downloader/1.0 (mailto:research@example.com)'
        })

    def download_supplementary_info(self, pmcid: str) -> SupplementaryInfo:
        if not pmcid.startswith('PMC'):
            pmcid = f'PMC{pmcid}'

        logging.info(f"Starting supplementary download for {pmcid}")

        supp_info = SupplementaryInfo(
            pmcid=pmcid,
            has_supplements=False,
            footnote_references=[],
            external_links=[],
            files=[],
            download_status="starting",
            download_date=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_files=0,
            total_size_bytes=0
        )

        try:
            s3_listing = self._list_s3_article_files(pmcid)
            if not s3_listing:
                supp_info.download_status = "no_oa_package"
                return supp_info

            package_files = self._download_s3_article_files(pmcid, s3_listing)
            if not package_files:
                supp_info.download_status = "download_failed"
                return supp_info

            xml_metadata = self._extract_supplementary_metadata(package_files.get('xml_file'))
            supp_info.footnote_references = xml_metadata['footnote_references']
            supp_info.external_links = xml_metadata['external_links']

            supp_files = self._process_supplementary_files(pmcid, package_files)
            supp_info.files = supp_files
            supp_info.has_supplements = len(supp_files) > 0
            supp_info.total_files = len(supp_files)
            supp_info.total_size_bytes = sum(f.size_bytes for f in supp_files)

            supp_info.download_status = "completed"
            self._save_supplementary_files(pmcid, supp_info)
            logging.info(f"Successfully downloaded {len(supp_files)} supplementary files for {pmcid}")

        except Exception as e:
            logging.error(f"Error downloading supplementary info for {pmcid}: {e}")
            supp_info.download_status = f"error: {str(e)}"

        return supp_info

    def _list_s3_article_files(self, pmcid: str) -> Optional[Dict[str, object]]:
        # change to pmc-oa-opendata S3 bucket
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
            files = [k for k in keys if k.startswith(prefix)]

            logging.info(f"Found {len(files)} S3 objects for {pmcid} (version {latest})")
            return {'version': latest, 'prefix': prefix, 'files': files}

        except Exception as e:
            logging.warning(f"Failed to list S3 files for {pmcid}: {e}")
            return None

    def _download_s3_article_files(
        self, pmcid: str, s3_listing: Dict[str, object]
    ) -> Optional[Dict[str, Path]]:
        try:
            extract_dir = self.output_dir / pmcid / "supplementary_extraction"
            extract_dir.mkdir(parents=True, exist_ok=True)

            extracted_files: Dict[str, object] = {}
            total_bytes = 0

            for key in s3_listing['files']:
                filename = key.split('/')[-1]
                filename_lower = filename.lower()

                is_xml = filename_lower.endswith('.nxml') or filename_lower.endswith('.xml')
                is_supp = bool(re.search(r'-s\d+\.', filename_lower)) or any(
                    pat in filename_lower for pat in ['supp', 'supplement', 'esm', 'moesm', 'si_']
                )
                if is_supp and Path(filename_lower).suffix not in SCREENABLE_SUPPLEMENT_EXTENSIONS:
                    logging.info("Skipping non-screenable supplementary file %s", filename)
                    continue

                if not (is_xml or is_supp):
                    continue

                file_url = PMC_OA_S3_BUCKET_URL + key
                logging.info(f"Downloading {file_url}")

                response = self.session.get(file_url, stream=True, timeout=60)
                if response.status_code != 200:
                    logging.warning(f"Failed to download {key}: HTTP {response.status_code}")
                    continue

                local_path = extract_dir / filename
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                total_bytes += local_path.stat().st_size

                if is_xml:
                    extracted_files['xml_file'] = local_path
                else:
                    extracted_files.setdefault('supplementary_files', []).append(local_path)

            if not extracted_files:
                logging.info(f"No XML/supplementary files in S3 listing for {pmcid}")
                return None

            logging.info(f"Downloaded {total_bytes / 1024:.2f} KB for {pmcid}")
            return extracted_files

        except Exception as e:
            logging.error(f"Failed to download S3 article files for {pmcid}: {e}")
            return None

    def _extract_supplementary_metadata(self, xml_file: Optional[Path]) -> Dict[str, List]:
        metadata = {
            'footnote_references': [],
            'external_links': []
        }

        if not xml_file or not xml_file.exists():
            return metadata

        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'xml')

            fn_groups = soup.find_all('fn-group')
            for fn_group in fn_groups:
                fns = fn_group.find_all('fn')
                for fn in fns:
                    fn_text = fn.get_text()
                    if any(keyword in fn_text.lower() for keyword in ['supplementary', 'esi', 'electronic', 'additional']):
                        metadata['footnote_references'].append(fn_text.strip())

            ext_links = soup.find_all('ext-link')
            for link in ext_links:
                href = link.get('xlink:href', '')
                link_text = link.get_text()
                if any(keyword in (href + link_text).lower() for keyword in ['suppl', 'supplement', 'supporting']):
                    metadata['external_links'].append({
                        'text': link_text,
                        'url': href
                    })

        except Exception as e:
            logging.warning(f"Failed to extract supplementary metadata: {e}")

        return metadata

    def _process_supplementary_files(self, pmcid: str, package_files: Dict[str, Path]) -> List[SupplementaryFile]:
        supp_files = []

        supplementary_files = package_files.get('supplementary_files', [])

        for file_path in supplementary_files:
            try:
                file_info = self._analyze_file(file_path)

                supp_file = SupplementaryFile(
                    id=self._extract_file_id(file_path.name),
                    filename=file_path.name,
                    caption=file_info['caption'],
                    file_type=file_info['file_type'],
                    size_bytes=file_path.stat().st_size,
                    local_path=str(file_path),
                    content_summary=file_info['content_summary']
                )

                supp_files.append(supp_file)

            except Exception as e:
                logging.warning(f"Failed to process supplementary file {file_path}: {e}")

        return supp_files

    def _analyze_file(self, file_path: Path) -> Dict[str, str]:
        file_ext = file_path.suffix.lower()

        file_info = {
            'caption': self._generate_caption(file_path.name),
            'file_type': file_ext[1:] if file_ext else 'unknown',
            'content_summary': None
        }

        try:
            if file_ext == '.pdf':
                file_info['content_summary'] = f"PDF document ({file_path.stat().st_size / 1024:.1f} KB)"

            elif file_ext in ['.xlsx', '.xls']:
                try:
                    import pandas as pd
                    excel_file = pd.ExcelFile(file_path)
                    sheet_count = len(excel_file.sheet_names)
                    file_info['content_summary'] = f"Excel file with {sheet_count} sheets: {', '.join(excel_file.sheet_names[:3])}"
                except ImportError:
                    file_info['content_summary'] = f"Excel file (analysis requires openpyxl)"
                except Exception:
                    file_info['content_summary'] = f"Excel file (unable to analyze)"

            elif file_ext in ['.csv', '.tsv']:
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path, nrows=0, sep='\t' if file_ext == '.tsv' else ',')
                    file_info['content_summary'] = f"CSV file with {len(df.columns)} columns"
                except Exception:
                    file_info['content_summary'] = f"CSV file (unable to analyze)"

            elif file_ext in ['.txt', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    file_info['content_summary'] = f"Text file with {line_count} lines"
                except Exception:
                    file_info['content_summary'] = f"Text file (unable to analyze)"

            elif file_ext == '.zip':
                file_info['content_summary'] = f"ZIP supplementary archive ({file_path.stat().st_size / 1024:.1f} KB)"

            elif file_ext in ['.doc', '.docx']:
                file_info['content_summary'] = f"Word supplementary document ({file_path.stat().st_size / 1024:.1f} KB)"

            else:
                file_info['content_summary'] = f"{file_ext} file ({file_path.stat().st_size / 1024:.1f} KB)"

        except Exception as e:
            file_info['content_summary'] = f"Analysis failed: {str(e)}"

        return file_info

    def _extract_file_id(self, filename: str) -> str:
        import re

        # PMC/Springer 常见文件命名规则:MOESM1、sifile1、ESM、si_N。
        patterns = [
            r'MOESM(\d+)',
            r'sifile(\d+)',
            r'ESM\.(\w+)',
            r'si_(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(0)

        return Path(filename).stem

    def _generate_caption(self, filename: str) -> str:
        filename_lower = filename.lower()

        if 'moesm' in filename_lower:
            if 'moesm1' in filename_lower:
                return "Supplementary Information"
            elif 'moesm2' in filename_lower:
                return "Peer Review File"
            else:
                return f"Supplementary Data"

        elif 'si_' in filename_lower:
            return "Supporting Information"

        elif 'esm' in filename_lower:
            return "Electronic Supplementary Material"

        else:
            return filename

    def _save_supplementary_files(self, pmcid: str, supp_info: SupplementaryInfo):
        paper_dir = self.output_dir / pmcid
        supp_dir = paper_dir / "supplementary"
        supp_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = supp_dir / "supplementary_metadata.json"

        for supp_file in supp_info.files:
            source_path = Path(supp_file.local_path)
            target_path = supp_dir / supp_file.filename

            if source_path.exists() and not target_path.exists():
                try:
                    shutil.copy2(source_path, target_path)
                except Exception as e:
                    logging.warning(f"Failed to copy supplementary file {supp_file.filename}: {e}")
            if target_path.exists():
                supp_file.local_path = str(target_path)

        archive_extractions = extract_zip_archives(supp_dir, max_archive_depth=2)
        document_conversions = convert_legacy_docs(supp_dir)

        metadata = {
            'pmcid': supp_info.pmcid,
            'has_supplements': supp_info.has_supplements,
            'footnote_references': supp_info.footnote_references,
            'external_links': supp_info.external_links,
            'download_status': supp_info.download_status,
            'download_date': supp_info.download_date,
            'total_files': supp_info.total_files,
            'total_size_bytes': supp_info.total_size_bytes,
            'files': [
                {
                    'id': f.id,
                    'filename': f.filename,
                    'caption': f.caption,
                    'file_type': f.file_type,
                    'size_bytes': f.size_bytes,
                    'local_path': f.local_path,
                    'content_summary': f.content_summary,
                    'sha256': sha256_file(Path(f.local_path)) if Path(f.local_path).is_file() else '',
                }
                for f in supp_info.files
            ],
            'archive_extractions': archive_extractions,
            'document_conversions': document_conversions,
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved {len(supp_info.files)} supplementary files to {supp_dir}")

    def cleanup_temporary_files(self, pmcid: str):
        extract_dir = self.output_dir / pmcid / "supplementary_extraction"

        if extract_dir.exists():
            try:
                import shutil
                shutil.rmtree(extract_dir)
                logging.info(f"Cleaned up temporary files for {pmcid}")
            except Exception as e:
                logging.warning(f"Failed to cleanup temporary files for {pmcid}: {e}")
