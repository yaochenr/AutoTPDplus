"""PMC API client for searching, resolving, and retrieving papers."""

import logging
import re
import time
from typing import Dict, List, Optional

import requests

from tpd_curator.pmc_miner.core.config import (
    NCBI_BASE_URL, PMC_OA_SERVICE_URL, PMC_EFETCH_URL, RATE_LIMIT_DELAY, MAX_RETRIES,
    MOLECULAR_GLUE_KEYWORDS, NCBI_API_KEY, NCBI_EMAIL, NCBI_TOOL,
)


class PMCClient:
    """Client for NCBI E-utilities against PMC."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'{NCBI_TOOL}/0.3 (mailto:{NCBI_EMAIL})'
        })

    @staticmethod
    def _request_identity() -> Dict[str, str]:
        identity = {'tool': NCBI_TOOL, 'email': NCBI_EMAIL}
        if NCBI_API_KEY:
            identity['api_key'] = NCBI_API_KEY
        return identity

    def _rate_limit(self):
        time.sleep(RATE_LIMIT_DELAY)

    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        self._rate_limit()

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logging.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

    def search_query(self, query: str, max_results: Optional[int] = 100) -> List[str]:
        """Run a fully formed PMC query with pagination.

        ``max_results=None`` retrieves every matching identifier.  Returned
        identifiers are the numeric PMC UIDs expected by ESummary.
        """
        page_size = 500
        retstart = 0
        ids: List[str] = []
        total = None

        while total is None or retstart < total:
            requested = page_size
            if max_results is not None:
                remaining = max_results - len(ids)
                if remaining <= 0:
                    break
                requested = min(requested, remaining)

            params = {
                'db': 'pmc',
                'term': query,
                'retstart': retstart,
                'retmax': requested,
                'retmode': 'json',
                'sort': 'pub date',
                **self._request_identity(),
            }
            try:
                response = self._make_request(f"{NCBI_BASE_URL}/esearch.fcgi", params)
                result = response.json().get('esearchresult', {})
                page = result.get('idlist', [])
                total = int(result.get('count', len(page)))
                ids.extend(page)
                if not page:
                    break
                retstart += len(page)
            except Exception as e:
                logging.error(f"Search failed: {e}")
                break

        # Preserve result order while defending against duplicated pages.
        return list(dict.fromkeys(ids))

    def search_papers(self, keywords: List[str], max_results: int = 100) -> List[str]:
        query_parts = []
        for keyword in keywords:
            query_parts.append(f'"{keyword}"[Title/Abstract] OR "{keyword}"[Full Text]')

        search_query = "(" + " OR ".join(query_parts) + ")"

        open_access_filters = [
            '"open access"[filter]',
            '"free full text"[filter]',
            '"loatcfree full text"[sb]'
        ]
        oa_filter = " OR ".join(open_access_filters)
        search_query += f' AND ({oa_filter})'

        pmc_ids = self.search_query(search_query, max_results=max_results)
        logging.info(f"Found {len(pmc_ids)} papers in PMC Open Access subset")
        return pmc_ids

    def get_paper_metadata(self, pmc_ids: List[str]) -> List[Dict]:
        if not pmc_ids:
            return []

        batch_size = 100
        all_metadata = []

        for i in range(0, len(pmc_ids), batch_size):
            batch_ids = pmc_ids[i:i + batch_size]
            # ESummary 只接受不带 PMC 前缀的纯数字 ID。
            numeric_ids = [pid.replace('PMC', '') for pid in batch_ids]
            id_list = ",".join(numeric_ids)

            params = {
                'db': 'pmc',
                'id': id_list,
                'retmode': 'json',
                **self._request_identity(),
            }

            url = f"{NCBI_BASE_URL}/esummary.fcgi"

            try:
                response = self._make_request(url, params)
                data = response.json()

                if 'result' in data:
                    for i, pmc_id in enumerate(batch_ids):
                        numeric_id = numeric_ids[i]
                        if numeric_id in data['result']:
                            metadata = data['result'][numeric_id]
                            article_ids = {
                                item.get('idtype'): item.get('value', '')
                                for item in metadata.get('articleids', [])
                                if item.get('idtype')
                            }
                            all_metadata.append({
                                'pmcid': f"PMC{numeric_id}",
                                'title': metadata.get('title', ''),
                                'authors': metadata.get('authors', []),
                                'journal': metadata.get('fulljournalname', ''),
                                'pubdate': metadata.get('pubdate', ''),
                                'doi': article_ids.get('doi', metadata.get('doi', '')),
                                'pmid': article_ids.get('pmid', metadata.get('pmid', '')),
                            })

            except Exception as e:
                logging.error(f"Failed to get metadata for batch: {e}")

        logging.info(f"Retrieved metadata for {len(all_metadata)} papers")
        return all_metadata

    def get_full_text_xml(self, pmc_id: str) -> Optional[str]:
        if not pmc_id.startswith('PMC'):
            pmc_id = f'PMC{pmc_id}'

        params = {
            'db': 'pmc',
            'id': pmc_id,
            'rettype': 'xml',
            'retmode': 'text',
            **self._request_identity(),
        }

        try:
            response = self._make_request(PMC_EFETCH_URL, params)
            if response.status_code == 200:
                xml_content = response.text

                if xml_content and len(xml_content) > 100:
                    if ('<article' in xml_content or '<pmc-articleset' in xml_content or
                        '<?xml' in xml_content[:100]):

                        if not any(error in xml_content.lower()[:500] for error in
                                 ['error', 'not found', 'unavailable', 'forbidden']):
                            logging.info(f"Successfully retrieved XML for {pmc_id}")
                            return xml_content
                        else:
                            logging.warning(f"{pmc_id}: XML contains error messages")
                    else:
                        logging.warning(f"{pmc_id}: Retrieved content doesn't appear to be valid XML")
                else:
                    logging.warning(f"{pmc_id}: Received empty or very short response")

        except Exception as e:
            logging.error(f"Failed to get full text for {pmc_id}: {e}")

        return None

    def search_and_collect(self, max_papers: int = 50) -> Dict:
        logging.info(f"Starting search for molecular glue papers (max: {max_papers})")

        pmc_ids = self.search_papers(MOLECULAR_GLUE_KEYWORDS, max_papers)

        if not pmc_ids:
            return {"papers": [], "total_found": 0}

        metadata = self.get_paper_metadata(pmc_ids)

        if len(metadata) > max_papers:
            metadata = metadata[:max_papers]

        return {
            "papers": metadata,
            "total_found": len(pmc_ids),
            "collected": len(metadata),
            "search_keywords": MOLECULAR_GLUE_KEYWORDS
        }

    def doi_to_pmcid(self, doi: str) -> Optional[str]:
        try:
            # 先只在 Open Access 子集里查,确保拿到的 PMC ID 有可下载的全文 XML。
            search_query = f'"{doi}"[DOI] AND ("open access"[filter] OR "free full text"[filter])'
            params = {
                'db': 'pmc',
                'term': search_query,
                'retmax': 1,
                'retmode': 'json',
                **self._request_identity(),
            }

            url = f"{NCBI_BASE_URL}/esearch.fcgi"
            response = self._make_request(url, params)
            data = response.json()

            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                pmc_ids = data['esearchresult']['idlist']
                if pmc_ids:
                    pmc_id = f"PMC{pmc_ids[0]}"
                    logging.info(f"Found PMC ID {pmc_id} for DOI {doi} (Open Access)")
                    return pmc_id

            # 兜底:无 OA 过滤再查一次 (结果可能没有全文 XML)。
            search_query = f'"{doi}"[DOI]'
            params = {
                'db': 'pmc',
                'term': search_query,
                'retmax': 1,
                'retmode': 'json',
                **self._request_identity(),
            }

            url = f"{NCBI_BASE_URL}/esearch.fcgi"
            response = self._make_request(url, params)
            data = response.json()

            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                pmc_ids = data['esearchresult']['idlist']
                if pmc_ids:
                    pmc_id = f"PMC{pmc_ids[0]}"
                    logging.info(f"Found PMC ID {pmc_id} for DOI {doi}")
                    return pmc_id
                else:
                    logging.warning(f"No PMC ID found for DOI {doi}")
                    return None
            else:
                logging.warning(f"No search results for DOI {doi}")
                return None

        except Exception as e:
            logging.error(f"Failed to convert DOI {doi} to PMC ID: {e}")
            return None

    def pmids_to_pmcids(self, pmids: List[str]) -> Dict[str, str]:
        """Resolve PubMed IDs to direct PMC article IDs via ELink."""
        clean = list(dict.fromkeys(
            str(pmid).strip() for pmid in pmids if re.fullmatch(r'\d+', str(pmid).strip())
        ))
        mapping: Dict[str, str] = {}
        for start in range(0, len(clean), 100):
            batch = clean[start:start + 100]
            params = [
                ('dbfrom', 'pubmed'), ('db', 'pmc'), ('retmode', 'json'),
                *[('id', pmid) for pmid in batch],
                *self._request_identity().items(),
            ]
            try:
                response = self._make_request(f"{NCBI_BASE_URL}/elink.fcgi", params)
                for linkset in response.json().get('linksets', []):
                    source_ids = linkset.get('ids', [])
                    if not source_ids:
                        continue
                    for db_links in linkset.get('linksetdbs', []):
                        if db_links.get('linkname') != 'pubmed_pmc':
                            continue
                        links = db_links.get('links', [])
                        if links:
                            mapping[source_ids[0]] = f"PMC{links[0]}"
            except Exception as e:
                logging.error(f"PMID to PMCID resolution failed: {e}")
        return mapping

    def dois_to_pmc_metadata(self, dois: List[str]) -> List[Dict]:
        """Resolve DOI batches across all PMC records and return metadata.

        PMCID discovery must not be restricted to the Open Access subset.  OA
        eligibility is classified separately by ``filter_open_access_pmcids``
        after every available PMCID has been resolved.
        """
        clean = list(dict.fromkeys(d.strip().lower() for d in dois if d and d.strip()))
        found_ids: List[str] = []
        for start in range(0, len(clean), 40):
            batch = clean[start:start + 40]
            doi_terms = ' OR '.join(f'"{doi}"[DOI]' for doi in batch)
            query = f'({doi_terms})'
            found_ids.extend(self.search_query(query, max_results=None))
        return self.get_paper_metadata(list(dict.fromkeys(found_ids)))

    def filter_open_access_pmcids(self, pmcids: List[str]) -> List[str]:
        """Return the subset of PMC IDs present in the Open Access collection."""
        numeric = list(dict.fromkeys(
            str(pmcid).upper().replace('PMC', '').strip()
            for pmcid in pmcids
            if re.fullmatch(r'(?:PMC)?\d+', str(pmcid).upper().strip())
        ))
        open_access: List[str] = []
        for start in range(0, len(numeric), 100):
            batch = numeric[start:start + 100]
            uid_terms = ' OR '.join(f'{uid}[UID]' for uid in batch)
            query = f'({uid_terms}) AND "open access"[filter]'
            open_access.extend(self.search_query(query, max_results=None))
        return [f"PMC{uid}" for uid in dict.fromkeys(open_access)]

    @staticmethod
    def validate_full_text_availability(xml_content: str) -> bool:
        if not xml_content:
            return False

        restriction_indicators = [
            "publisher of this article does not allow downloading of the full text",
            "full text in XML form is not available",
            "only the abstract is available"
        ]

        xml_lower = xml_content.lower()
        for indicator in restriction_indicators:
            if indicator in xml_lower:
                return False

        body_indicators = [
            "<body>",
            "<sec",
            "<section",
            "<p>",
            "</body>"
        ]

        for indicator in body_indicators:
            if indicator in xml_lower:
                return True

        return False
