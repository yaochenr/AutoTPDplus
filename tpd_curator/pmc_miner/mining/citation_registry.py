"""Build a paper-level registry from TPD databases and PMC search results.

The source files in ``data/`` are record-level exports with different citation
representations.  This module normalizes those citations and merges them into
one auditable paper list before any downloads are attempted.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from tpd_curator.pmc_miner.core.client import PMCClient


PROTAC_QUERY = (
    '((PROTAC[Title/Abstract] OR "proteolysis-targeting chimera"[Title/Abstract] '
    'OR "proteolysis targeting chimera"[Title/Abstract] '
    'OR "heterobifunctional degrader"[Title/Abstract] '
    'OR "bifunctional degrader"[Title/Abstract]) '
    'OR ((PROTAC[Full Text] OR "proteolysis-targeting chimera"[Full Text] '
    'OR "heterobifunctional degrader"[Full Text]) '
    'AND (dc50[Full Text] OR dmax[Full Text] OR "maximum degradation"[Full Text] '
    'OR "half-maximal degradation"[Full Text]))) AND "open access"[filter]'
)

MOLECULAR_GLUE_QUERY = (
    '(("molecular glue"[Title/Abstract] OR "molecular glues"[Title/Abstract] '
    'OR "molecular glue degrader"[Title/Abstract]) '
    'OR (("molecular glue"[Full Text] OR "molecular glues"[Full Text]) '
    'AND (dc50[Full Text] OR dmax[Full Text] OR "maximum degradation"[Full Text] '
    'OR "half-maximal degradation"[Full Text]))) AND "open access"[filter]'
)


def normalize_doi(value: object) -> str:
    text = str(value or '').strip().lower()
    if text in {'', 'nan', 'none', 'null'}:
        return ''
    text = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', text)
    text = re.sub(r'^doi:\s*', '', text)
    # DOI fields in source exports can also contain patent titles or complete
    # citations.  Only retain a syntactically valid DOI token.
    match = re.search(r'10\.\d{4,9}/[-._;()/:a-z0-9]+', text, flags=re.IGNORECASE)
    return match.group(0).rstrip(' .;,') if match else ''


def normalize_pmid(value: object) -> str:
    text = str(value or '').strip()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        return ''
    match = re.search(r'(?<!\d)(\d{5,9})(?!\d)', text)
    return match.group(1) if match else ''


def normalize_pmcid(value: object) -> str:
    text = str(value or '').strip().upper()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        return ''
    match = re.search(r'PMC\s*(\d+)', text, flags=re.IGNORECASE)
    return f"PMC{match.group(1)}" if match else ''


def normalize_title(value: object) -> str:
    text = str(value or '').casefold()
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def _year(value: object) -> str:
    match = re.search(r'(?:19|20)\d{2}', str(value or ''))
    return match.group(0) if match else ''


@dataclass
class CitationRecord:
    doi: str = ''
    pmid: str = ''
    pmcid: str = ''
    title: str = ''
    publication_year: str = ''
    journal: str = ''
    citation: str = ''
    modalities: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    source_record_ids: Set[str] = field(default_factory=set)
    database_labels: Set[str] = field(default_factory=set)
    discovery_methods: Set[str] = field(default_factory=set)
    confirmed_relevant: bool = False
    pmc_open_access: Optional[bool] = None
    screening_status: str = ''
    article_type: str = 'unknown'
    screening_method: str = ''
    activity_evidence_level: str = ''
    activity_evidence: List[Dict[str, Any]] = field(default_factory=list)
    screening_reasons: List[str] = field(default_factory=list)
    screened_at: str = ''
    screening_error: str = ''
    download_status: str = ''
    paper_path: str = ''

    def normalized(self) -> "CitationRecord":
        self.doi = normalize_doi(self.doi)
        self.pmid = normalize_pmid(self.pmid)
        self.pmcid = normalize_pmcid(self.pmcid)
        self.title = str(self.title or '').strip()
        self.publication_year = _year(self.publication_year)
        return self

    def merge(self, other: "CitationRecord") -> None:
        for name in (
            'doi', 'pmid', 'pmcid', 'title', 'publication_year', 'journal',
            'citation', 'screening_status', 'article_type', 'screening_method',
            'activity_evidence_level', 'screened_at', 'screening_error',
            'download_status', 'paper_path',
        ):
            if not getattr(self, name) and getattr(other, name):
                setattr(self, name, getattr(other, name))
        self.modalities.update(other.modalities)
        self.sources.update(other.sources)
        self.source_record_ids.update(other.source_record_ids)
        self.database_labels.update(other.database_labels)
        self.discovery_methods.update(other.discovery_methods)
        if other.activity_evidence:
            existing = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in self.activity_evidence}
            for item in other.activity_evidence:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if key not in existing:
                    self.activity_evidence.append(item)
                    existing.add(key)
        for reason in other.screening_reasons:
            if reason not in self.screening_reasons:
                self.screening_reasons.append(reason)
        self.confirmed_relevant = self.confirmed_relevant or other.confirmed_relevant
        if self.pmc_open_access is None and other.pmc_open_access is not None:
            self.pmc_open_access = other.pmc_open_access
        if self.confirmed_relevant:
            self.screening_status = 'accepted_database_seed'

    @property
    def canonical_id(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmcid:
            return f"pmcid:{self.pmcid}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        return f"title:{normalize_title(self.title)}"

    def to_dict(self) -> Dict[str, object]:
        return {
            'canonical_id': self.canonical_id,
            'doi': self.doi,
            'pmid': self.pmid,
            'pmcid': self.pmcid,
            'title': self.title,
            'publication_year': self.publication_year,
            'journal': self.journal,
            'citation': self.citation,
            'modalities': sorted(self.modalities),
            'sources': sorted(self.sources),
            'source_record_ids': sorted(self.source_record_ids),
            'database_labels': sorted(self.database_labels),
            'discovery_methods': sorted(self.discovery_methods),
            'confirmed_relevant': self.confirmed_relevant,
            'pmc_open_access': self.pmc_open_access,
            'screening_status': self.screening_status,
            'article_type': self.article_type,
            'screening_method': self.screening_method,
            'activity_evidence_level': self.activity_evidence_level,
            'activity_evidence': self.activity_evidence,
            'screening_reasons': self.screening_reasons,
            'screened_at': self.screened_at,
            'screening_error': self.screening_error,
            'download_status': self.download_status,
            'paper_path': self.paper_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CitationRecord":
        return cls(
            doi=str(data.get('doi') or ''),
            pmid=str(data.get('pmid') or ''),
            pmcid=str(data.get('pmcid') or ''),
            title=str(data.get('title') or ''),
            publication_year=str(data.get('publication_year') or ''),
            journal=str(data.get('journal') or ''),
            citation=str(data.get('citation') or ''),
            modalities=set(data.get('modalities') or []),
            sources=set(data.get('sources') or []),
            source_record_ids=set(data.get('source_record_ids') or []),
            database_labels=set(data.get('database_labels') or []),
            discovery_methods=set(data.get('discovery_methods') or []),
            confirmed_relevant=bool(data.get('confirmed_relevant')),
            pmc_open_access=data.get('pmc_open_access'),
            screening_status=str(data.get('screening_status') or ''),
            article_type=str(data.get('article_type') or 'unknown'),
            screening_method=str(data.get('screening_method') or ''),
            activity_evidence_level=str(data.get('activity_evidence_level') or ''),
            activity_evidence=list(data.get('activity_evidence') or []),
            screening_reasons=list(data.get('screening_reasons') or []),
            screened_at=str(data.get('screened_at') or ''),
            screening_error=str(data.get('screening_error') or ''),
            download_status=str(data.get('download_status') or ''),
            paper_path=str(data.get('paper_path') or ''),
        )


class CitationRegistry:
    """Merge records whenever any stable article identifier overlaps."""

    def __init__(self) -> None:
        self._records: Dict[int, CitationRecord] = {}
        self._next_id = 1
        self._doi: Dict[str, int] = {}
        self._pmid: Dict[str, int] = {}
        self._pmcid: Dict[str, int] = {}
        self._title: Dict[str, int] = {}

    def _matching_ids(self, record: CitationRecord) -> Set[int]:
        matches: Set[int] = set()
        for value, index in (
            (record.doi, self._doi),
            (record.pmid, self._pmid),
            (record.pmcid, self._pmcid),
            (normalize_title(record.title), self._title),
        ):
            if value and value in index:
                matches.add(index[value])
        return matches

    def _index(self, record_id: int, record: CitationRecord) -> None:
        if record.doi:
            self._doi[record.doi] = record_id
        if record.pmid:
            self._pmid[record.pmid] = record_id
        if record.pmcid:
            self._pmcid[record.pmcid] = record_id
        title_key = normalize_title(record.title)
        if title_key:
            self._title[title_key] = record_id

    def add(self, record: CitationRecord) -> CitationRecord:
        record.normalized()
        if not any((record.doi, record.pmid, record.pmcid, normalize_title(record.title))):
            return record

        matching = sorted(self._matching_ids(record))
        if not matching:
            record_id = self._next_id
            self._next_id += 1
            self._records[record_id] = record
        else:
            record_id = matching[0]
            target = self._records[record_id]
            for duplicate_id in matching[1:]:
                target.merge(self._records.pop(duplicate_id))
                for index in (self._doi, self._pmid, self._pmcid, self._title):
                    for key, value in list(index.items()):
                        if value == duplicate_id:
                            index[key] = record_id
            target.merge(record)
            record = target

        self._index(record_id, record)
        return record

    def has_match(self, record: CitationRecord) -> bool:
        """Whether a resolver result belongs to an existing source record."""
        record.normalized()
        return bool(self._matching_ids(record))

    def records(self) -> List[CitationRecord]:
        return sorted(
            self._records.values(),
            key=lambda item: (
                not item.confirmed_relevant,
                -(int(item.publication_year) if item.publication_year else 0),
                item.canonical_id,
            ),
        )

    def __len__(self) -> int:
        return len(self._records)


def _read_csv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def _load_mgdb(path: Path, registry: CitationRegistry) -> None:
    for row in _read_csv(path):
        reference = str(row.get('Reference') or '').strip()
        registry.add(CitationRecord(
            doi=row.get('DOI', ''),
            pmid=row.get('PMID', ''),
            pmcid=row.get('PMCID', ''),
            title=row.get('Title', ''),
            publication_year=row.get('Publication Year', ''),
            journal=row.get('Journal/Book', ''),
            citation=row.get('Citation', ''),
            modalities={'molecular_glue'},
            sources={'MGDB'},
            source_record_ids={f"MGDB:{reference}"} if reference else set(),
            discovery_methods={'database_seed'},
            confirmed_relevant=True,
            screening_status='accepted_database_seed',
        ))


def _load_mgtbind(path: Path, registry: CitationRegistry) -> None:
    for row in _read_csv(path):
        # The export contains patent rows; only scientific citations are part
        # of the literature registry requested by the user.
        if (row.get('document_type') or '').strip().casefold() != 'scientific literature':
            continue
        record_id = str(row.get('id') or '').strip()
        registry.add(CitationRecord(
            doi=row.get('doi', ''),
            pmid=row.get('pubmed_id', ''),
            title=row.get('title', ''),
            publication_year=row.get('year', ''),
            journal=row.get('journal', ''),
            citation=row.get('content', ''),
            modalities={'molecular_glue'},
            sources={'MGTbind'},
            source_record_ids={f"MGTbind:{record_id}"} if record_id else set(),
            discovery_methods={'database_seed'},
            confirmed_relevant=True,
            screening_status='accepted_database_seed',
        ))


def _load_tack2(path: Path, registry: CitationRegistry) -> None:
    for row in _read_csv(path):
        # TACK2's ``Reference`` column contains DOI strings.  Treating the
        # numeric tail of a DOI as a PMID creates plausible-looking but invalid
        # identifiers (for example ``10.1021/...1c00882`` -> ``00882``).
        doi = normalize_doi(row.get('Reference', ''))
        if not doi:
            continue
        labels = {
            label.strip() for label in (row.get('Database') or '').split(';') if label.strip()
        }
        registry.add(CitationRecord(
            doi=doi,
            modalities={'tpd_unspecified'},
            sources={'TACK2'},
            source_record_ids={f"TACK2:DOI:{doi}"},
            database_labels=labels,
            discovery_methods={'database_seed'},
            confirmed_relevant=True,
            screening_status='accepted_database_seed',
        ))


def _column_name(cell_reference: str) -> str:
    match = re.match(r'([A-Z]+)', cell_reference or '')
    return match.group(1) if match else ''


def _load_protacdb_xlsx(path: Path, registry: CitationRegistry) -> None:
    namespace = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with ZipFile(path) as archive, archive.open('xl/worksheets/sheet1.xml') as handle:
        for _event, row in ET.iterparse(handle, events=('end',)):
            if row.tag != namespace + 'row':
                continue
            if row.attrib.get('r') == '1':
                row.clear()
                continue
            doi = ''
            for cell in row.findall(namespace + 'c'):
                if _column_name(cell.attrib.get('r', '')) != 'BY':
                    continue
                if cell.attrib.get('t') == 'inlineStr':
                    doi = ''.join(
                        node.text or '' for node in cell.findall('.//' + namespace + 't')
                    )
                else:
                    value = cell.find(namespace + 'v')
                    doi = '' if value is None else value.text or ''
                break
            doi = normalize_doi(doi)
            if doi:
                registry.add(CitationRecord(
                    doi=doi,
                    modalities={'protac'},
                    sources={'PROTAC-DB'},
                    source_record_ids={f"PROTAC-DB:DOI:{doi}"},
                    discovery_methods={'database_seed'},
                    confirmed_relevant=True,
                    screening_status='accepted_database_seed',
                ))
            row.clear()


def load_database_registry(data_dir: Path) -> CitationRegistry:
    data_dir = Path(data_dir)
    required = {
        'MGDB_Article_Reference.csv': _load_mgdb,
        'MGTbind_citations.csv': _load_mgtbind,
        'TACK2.csv': _load_tack2,
        'protacdb.xlsx': _load_protacdb_xlsx,
    }
    registry = CitationRegistry()
    for filename, loader in required.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing citation source: {path}")
        loader(path, registry)
    return registry


def enrich_database_identifiers(registry: CitationRegistry, client: PMCClient) -> None:
    records = registry.records()
    pmid_mapping = client.pmids_to_pmcids([record.pmid for record in records if record.pmid])
    for record in list(records):
        pmcid = pmid_mapping.get(record.pmid)
        if pmcid:
            registry.add(CitationRecord(pmid=record.pmid, pmcid=pmcid))

    # DOI-only entries (predominantly PROTAC-DB) are resolved in OR batches.
    unresolved_dois = [
        record.doi for record in registry.records() if record.doi and not record.pmcid
    ]
    for metadata in client.dois_to_pmc_metadata(unresolved_dois):
        resolved = CitationRecord(
            doi=metadata.get('doi', ''),
            pmid=metadata.get('pmid', ''),
            pmcid=metadata.get('pmcid', ''),
            title=metadata.get('title', ''),
            publication_year=metadata.get('pubdate', ''),
            journal=metadata.get('journal', ''),
        )
        # PMC full-text search can occasionally return a neighboring record
        # for a DOI expression.  Resolver-only records must never create new
        # source-less articles in the unified catalog.
        if registry.has_match(resolved):
            registry.add(resolved)

    pmcids = [record.pmcid for record in registry.records() if record.pmcid]
    for metadata in client.get_paper_metadata(pmcids):
        registry.add(CitationRecord(
            doi=metadata.get('doi', ''),
            pmid=metadata.get('pmid', ''),
            pmcid=metadata.get('pmcid', ''),
            title=metadata.get('title', ''),
            publication_year=metadata.get('pubdate', ''),
            journal=metadata.get('journal', ''),
        ))

    oa_ids = set(client.filter_open_access_pmcids(
        [record.pmcid for record in registry.records() if record.pmcid]
    ))
    for record in list(registry.records()):
        if record.pmcid:
            registry.add(CitationRecord(
                pmcid=record.pmcid,
                pmc_open_access=record.pmcid in oa_ids,
            ))


def add_pmc_search_results(
    registry: CitationRegistry,
    client: PMCClient,
    max_results_per_query: Optional[int] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for modality, query in (
        ('protac', PROTAC_QUERY),
        ('molecular_glue', MOLECULAR_GLUE_QUERY),
    ):
        numeric_ids = client.search_query(query, max_results=max_results_per_query)
        counts[modality] = len(numeric_ids)
        for metadata in client.get_paper_metadata(numeric_ids):
            pmcid = metadata.get('pmcid', '')
            registry.add(CitationRecord(
                doi=metadata.get('doi', ''),
                pmid=metadata.get('pmid', ''),
                pmcid=pmcid,
                title=metadata.get('title', ''),
                publication_year=metadata.get('pubdate', ''),
                journal=metadata.get('journal', ''),
                modalities={modality},
                sources={'PMC keyword search'},
                source_record_ids={f"PMC-search:{modality}:{pmcid}"} if pmcid else set(),
                discovery_methods={'pmc_keyword_search'},
                confirmed_relevant=False,
                pmc_open_access=True,
                screening_status='pending_activity_screen',
            ))
    return counts


def write_registry(registry: CitationRegistry, output_dir: Path, summary: Dict) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [record.to_dict() for record in registry.records()]

    json_path = output_dir / 'unified_article_list.json'
    json_path.write_text(
        json.dumps({'summary': summary, 'articles': records}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    csv_path = output_dir / 'unified_article_list.csv'
    fields = [
        'canonical_id', 'doi', 'pmid', 'pmcid', 'title', 'publication_year',
        'journal', 'confirmed_relevant', 'pmc_open_access', 'screening_status',
        'article_type', 'screening_method', 'activity_evidence_level',
        'screening_reasons', 'screened_at', 'screening_error',
        'download_status', 'paper_path', 'modalities', 'sources',
        'database_labels', 'discovery_methods', 'source_record_ids', 'citation',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: (
                    json.dumps(record[key], ensure_ascii=False)
                    if key == 'activity_evidence'
                    else ';'.join(str(item) for item in record[key])
                    if isinstance(record.get(key), list)
                    else record.get(key, '')
                )
                for key in fields
            })

    summary_path = output_dir / 'unified_article_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'json': json_path, 'csv': csv_path, 'summary': summary_path}


def load_registry_json(path: Path) -> tuple[CitationRegistry, Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    registry = CitationRegistry()
    for article in payload.get('articles', []):
        registry.add(CitationRecord.from_dict(article))
    return registry, dict(payload.get('summary', {}))


def download_existing_catalog(
    catalog_json: Path,
    output_dir: Path,
    reuse_dir: Optional[Path] = None,
    download_images: bool = True,
) -> Dict[str, object]:
    registry, summary = load_registry_json(catalog_json)
    download_summary = download_confirmed_articles(
        registry,
        Path(output_dir) / 'papers',
        download_images=download_images,
        reuse_dir=reuse_dir,
    )
    summary['confirmed_downloads'] = download_summary
    paths = write_registry(registry, output_dir, summary)
    return {'summary': summary, 'paths': paths}


def download_confirmed_articles(
    registry: CitationRegistry,
    papers_dir: Path,
    download_images: bool = True,
    reuse_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Download every database-confirmed article available in PMC OA."""
    from tpd_curator.pmc_miner.mining.doi_miner import DOIBasedMiner

    papers_dir = Path(papers_dir)
    miner = DOIBasedMiner(
        output_dir=str(papers_dir),
        paper_type='confirmed_database_articles',
        download_images=download_images,
    )
    statuses: Dict[str, str] = {}
    reusable: Dict[str, Path] = {}
    if reuse_dir:
        reuse_root = Path(reuse_dir)
        if reuse_root.exists():
            for metadata_path in reuse_root.glob('*/PMC*/metadata.json'):
                paper_dir = metadata_path.parent
                if papers_dir in paper_dir.parents:
                    continue
                if (paper_dir / 'full_text.xml').exists() and (paper_dir / 'processed.json').exists():
                    reusable.setdefault(paper_dir.name, paper_dir)
    selected = [
        record for record in registry.records()
        if record.confirmed_relevant and record.pmcid and record.pmc_open_access is True
    ]
    for index, record in enumerate(selected, 1):
        logging.info("Downloading confirmed article %d/%d: %s", index, len(selected), record.pmcid)
        paper_dir = miner.storage.get_paper_dir(record.pmcid)
        if record.pmcid in reusable:
            status = 'reused_local'
            paper_dir = reusable[record.pmcid]
        elif miner.storage.paper_exists(record.pmcid):
            status = 'already_downloaded'
        else:
            status = miner._process_doi(record.doi, record.pmcid)
        record.download_status = status
        if status in {'success', 'already_downloaded', 'reused_local'}:
            record.paper_path = str(paper_dir)
        statuses[record.pmcid] = status

    results = {
        'eligible_confirmed_articles': len(selected),
        'downloaded_or_existing': sum(
            status in {'success', 'already_downloaded', 'reused_local'}
            for status in statuses.values()
        ),
        'reused_local': sum(status == 'reused_local' for status in statuses.values()),
        'newly_downloaded': sum(status == 'success' for status in statuses.values()),
        'statuses': statuses,
    }
    results_path = papers_dir.parent / 'database_download_results.json'
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return results


def build_unified_registry(
    data_dir: Path,
    output_dir: Path,
    resolve_pmc: bool = False,
    include_search: bool = False,
    max_results_per_query: Optional[int] = None,
    download_confirmed: bool = False,
    download_images: bool = True,
    reuse_dir: Optional[Path] = None,
) -> Dict[str, object]:
    registry = load_database_registry(data_dir)
    initial_count = len(registry)
    client = PMCClient()
    if resolve_pmc:
        enrich_database_identifiers(registry, client)
    search_counts: Dict[str, int] = {}
    if include_search:
        search_counts = add_pmc_search_results(registry, client, max_results_per_query)

    download_summary: Dict[str, object] = {}
    if download_confirmed:
        if not resolve_pmc:
            raise ValueError("--download-confirmed requires --resolve-pmc")
        download_summary = download_confirmed_articles(
            registry,
            Path(output_dir) / 'papers',
            download_images=download_images,
            reuse_dir=reuse_dir,
        )

    records = registry.records()
    summary = {
        'database_seed_articles': initial_count,
        'unified_articles': len(records),
        'confirmed_relevant_articles': sum(record.confirmed_relevant for record in records),
        'pmc_keyword_only_articles': sum(
            not record.confirmed_relevant and 'PMC keyword search' in record.sources
            for record in records
        ),
        'articles_with_pmcid': sum(bool(record.pmcid) for record in records),
        'database_and_search_overlap': sum(
            record.confirmed_relevant and 'PMC keyword search' in record.sources
            for record in records
        ),
        'search_result_counts_before_merge': search_counts,
        'confirmed_downloads': download_summary,
        'source_article_counts_after_merge': {
            source: sum(source in record.sources for record in records)
            for source in ('MGDB', 'MGTbind', 'TACK2', 'PROTAC-DB', 'PMC keyword search')
        },
    }
    paths = write_registry(registry, output_dir, summary)
    logging.info("Unified citation registry contains %d articles", len(records))
    return {'registry': registry, 'summary': summary, 'paths': paths}
