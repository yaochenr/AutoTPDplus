"""JATS XML → ProcessedPaper (sections, tables, figures, supplementary info)."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from tpd_curator.pmc_miner.core.config import MOLECULAR_GLUE_KEYWORDS
from tpd_curator.pmc_miner.core.models import ProcessedPaper


class TextProcessor:
    """Extract structured content from PMC JATS XML."""

    def __init__(self):
        self.section_patterns = {
            'introduction': r'introduction|background',
            'methods': r'methods|methodology|experimental|materials',
            'results': r'results|findings',
            'discussion': r'discussion|conclusion|conclusions'
        }

    def process_xml(self, xml_content: str, pmcid: str) -> Optional[ProcessedPaper]:
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            if not soup.find('article'):
                logging.warning(f"No <article> tag found in XML for PMC{pmcid}")
                return None
            return self._process_with_beautifulsoup(soup, pmcid)
        except Exception as e:
            logging.error(f"Failed to process XML for PMC{pmcid}: {e}")
            return None

    def _process_with_beautifulsoup(self, soup: BeautifulSoup, pmcid: str) -> ProcessedPaper:
        title = self._extract_title_bs(soup)
        abstract = self._extract_abstract_bs(soup)
        article_metadata = self._extract_article_metadata_bs(soup)
        body_text, sections = self._extract_body_bs(soup)
        figures = self._extract_figures_bs(soup)
        tables = self._extract_tables_bs(soup)
        supplementary_info = self._extract_supplementary_info_bs(soup)

        return ProcessedPaper(
            pmcid=pmcid,
            title=title,
            abstract=abstract,
            body_text=body_text,
            references=[],
            figures=figures,
            tables=tables,
            sections=sections,
            xml_metadata=article_metadata,
            supplementary_info=supplementary_info
        )

    def _extract_title_bs(self, soup: BeautifulSoup) -> str:
        title_tag = soup.find('article-title')
        if title_tag:
            return self._clean_text(self._extract_text_with_format(title_tag))

        for tag_name in ['title', 'h1']:
            tag = soup.find(tag_name)
            if tag:
                return self._clean_text(self._extract_text_with_format(tag))

        return ""

    def _extract_abstract_bs(self, soup: BeautifulSoup) -> str:
        abstract_tag = soup.find('abstract')
        if abstract_tag:
            return self._clean_text(self._extract_text_with_format(abstract_tag))

        for selector in ['div.abstract', '.abstract', 'section[title*="abstract" i]']:
            try:
                tag = soup.select_one(selector)
                if tag:
                    return self._clean_text(self._extract_text_with_format(tag))
            except:
                continue

        return ""

    def _extract_article_metadata_bs(self, soup: BeautifulSoup) -> Dict[str, str]:
        metadata = {}

        try:
            article_tag = soup.find('article')
            if article_tag and article_tag.get('article-type'):
                metadata['article_type'] = self._clean_text(article_tag.get('article-type'))

            article_meta = soup.find('article-meta')
            if article_meta:
                doi_tag = article_meta.find('article-id', {'pub-id-type': 'doi'})
                if doi_tag:
                    metadata['doi'] = self._clean_text(doi_tag.get_text())

                pmid_tag = article_meta.find('article-id', {'pub-id-type': 'pmid'})
                if pmid_tag:
                    metadata['pmid'] = self._clean_text(pmid_tag.get_text())

                journal_meta = soup.find('journal-meta')
                if journal_meta:
                    journal_title = journal_meta.find('journal-title')
                    if journal_title:
                        metadata['journal'] = self._clean_text(journal_title.get_text())

                pub_date = article_meta.find('pub-date')
                if pub_date:
                    year = pub_date.find('year')
                    month = pub_date.find('month')
                    day = pub_date.find('day')

                    date_parts = []
                    if year:
                        date_parts.append(year.get_text())
                    if month:
                        date_parts.append(month.get_text())
                    if day:
                        date_parts.append(day.get_text())

                    if date_parts:
                        metadata['pubdate'] = ' '.join(date_parts)

                contrib_group = article_meta.find('contrib-group')
                if contrib_group:
                    authors = []
                    for contrib in contrib_group.find_all('contrib', {'contrib-type': 'author'}):
                        name_tag = contrib.find('name')
                        if name_tag:
                            surname = name_tag.find('surname')
                            given_names = name_tag.find('given-names')
                            if surname and given_names:
                                author_name = f"{given_names.get_text()} {surname.get_text()}"
                                authors.append(author_name.strip())

                    if authors:
                        metadata['authors'] = authors

                subjects = [
                    self._clean_text(subject.get_text())
                    for subject in article_meta.find_all('subject')
                    if self._clean_text(subject.get_text())
                ]
                if subjects:
                    metadata['subjects'] = subjects

        except Exception as e:
            logging.warning(f"Failed to extract article metadata: {e}")

        return metadata

    def _extract_body_bs(self, soup: BeautifulSoup) -> Tuple[str, Dict[str, str]]:
        body_tag = soup.find('body')
        if not body_tag:
            body_tag = soup.find('article') or soup

        sections: Dict[str, str] = {}
        full_text_parts: List[str] = []
        floating_paragraphs: List[str] = []

        def flush_floating():
            if not floating_paragraphs:
                return
            body_text = '\n\n'.join(floating_paragraphs)
            base = 'Main Text'
            title = base
            idx = 2
            while title in sections:
                title = f"{base} ({idx})"
                idx += 1
            sections[title] = body_text
            full_text_parts.append(f"{title}\n{body_text}")
            floating_paragraphs.clear()

        skip_block_tags = {'fig', 'table-wrap', 'supplementary-material',
                           'graphic', 'media', 'inline-graphic'}

        for child in body_tag.children:
            if not hasattr(child, 'name') or not child.name:
                continue

            if child.name in ('sec', 'section'):
                flush_floating()
                section_content = self._extract_section_content_recursive(child)
                if section_content:
                    section_title, section_text = section_content
                    base = section_title or "Unknown Section"
                    title = base
                    idx = 2
                    while title in sections:
                        title = f"{base} ({idx})"
                        idx += 1
                    sections[title] = section_text
                    full_text_parts.append(f"{title}\n{section_text}")
            elif child.name == 'p':
                text = self._clean_text(self._extract_text_with_format(child))
                if text:
                    floating_paragraphs.append(text)
            elif child.name == 'list':
                items = []
                for li in child.find_all('list-item'):
                    item_text = self._clean_text(self._extract_text_with_format(li))
                    if item_text:
                        items.append(f"- {item_text}")
                if items:
                    floating_paragraphs.append('\n'.join(items))
            elif child.name in skip_block_tags:
                continue
            else:
                text = self._clean_text(self._extract_text_with_format(child))
                if text:
                    floating_paragraphs.append(text)

        flush_floating()

        # 极端情况:<body> 里完全没有 <sec> 也没有直接 <p> 子节点,则用递归 <p> 兜底。
        if not sections and not full_text_parts:
            paragraphs = body_tag.find_all('p')
            full_text_parts = [self._clean_text(self._extract_text_with_format(p)) for p in paragraphs]

        full_text = '\n\n'.join(full_text_parts)
        return self._clean_text(full_text), sections

    def _extract_section_content_recursive(self, section_element) -> Optional[Tuple[str, str]]:
        title_tag = section_element.find(['title', 'h1', 'h2', 'h3', 'h4'], recursive=False)
        section_title = self._clean_text(self._extract_text_with_format(title_tag)) if title_tag else "Unknown Section"

        content_parts = []

        direct_content = []
        for child in section_element.children:
            if hasattr(child, 'name'):
                if child.name == 'p':
                    direct_content.append(self._clean_text(self._extract_text_with_format(child)))
                elif child.name in ['div'] and not child.find(['sec', 'section']):
                    div_paras = child.find_all('p')
                    direct_content.extend([self._clean_text(self._extract_text_with_format(p)) for p in div_paras])

        if direct_content:
            content_parts.append('\n\n'.join(direct_content))

        subsections = section_element.find_all(['sec', 'section'], recursive=False)
        for subsection in subsections:
            subsection_content = self._extract_section_content_recursive(subsection)
            if subsection_content:
                sub_title, sub_text = subsection_content
                content_parts.append(f"{sub_title}\n{sub_text}")

        if content_parts:
            full_content = '\n\n'.join(content_parts)
            return section_title, full_content

        return None

    def _extract_figures_bs(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        figures = []
        seen_image_urls = set()

        fig_tags = soup.find_all(['fig', 'figure'])
        for fig in fig_tags:
            fig_id = fig.get('id', '')

            caption_tag = fig.find(['caption', 'figcaption'])
            caption = self._clean_text(self._extract_text_with_format(caption_tag)) if caption_tag else ""

            title_tag = fig.find(['title', 'label'])
            title = self._clean_text(self._extract_text_with_format(title_tag)) if title_tag else ""

            image_url = ""
            graphic_tag = fig.find('graphic')
            if graphic_tag and graphic_tag.get('xlink:href'):
                image_url = graphic_tag.get('xlink:href')

            if caption or title or fig_id:
                figures.append({
                    'id': fig_id,
                    'title': title,
                    'caption': caption,
                    'image_url': image_url
                })
                if image_url:
                    seen_image_urls.add(image_url)
        # Mine <graphic> tag ourside of <fig>
        for graphic in soup.find_all('graphic'):
            if graphic.find_parent(['fig', 'figure']):
                continue
            href = graphic.get('xlink:href') or graphic.get('href') or ''
            if not href or href in seen_image_urls:
                continue
            in_abstract = graphic.find_parent('abstract') is not None
            figures.append({
                'id': graphic.get('id') or ('graphical_abstract' if in_abstract else ''),
                'title': 'Graphical Abstract' if in_abstract else '',
                'caption': 'Graphical Abstract' if in_abstract else '',
                'image_url': href,
            })
            seen_image_urls.add(href)

        return figures

    def _extract_tables_bs(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        tables = []

        # 只遍历 <table-wrap>,避免把嵌套的 <table> 重复计一次。
        table_wrap_tags = soup.find_all('table-wrap')

        if table_wrap_tags:
            for table_wrap in table_wrap_tags:
                table_id = table_wrap.get('id', '')

                caption_tag = table_wrap.find(['caption', 'title'])
                caption = self._clean_text(self._extract_text_with_format(caption_tag)) if caption_tag else ""

                table_tag = table_wrap.find('table')
                content = ""
                if table_tag:
                    rows = table_tag.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        row_text = ' | '.join([self._clean_text(self._extract_text_with_format(cell)) for cell in cells])
                        if row_text.strip():
                            content += row_text + '\n'

                footnotes = []
                table_wrap_foot = table_wrap.find('table-wrap-foot')
                if table_wrap_foot:
                    fn_tags = table_wrap_foot.find_all('fn')
                    for fn in fn_tags:
                        label_tag = fn.find('label')
                        label = self._clean_text(self._extract_text_with_format(label_tag)) if label_tag else ""

                        p_tags = fn.find_all('p')
                        fn_text = ' '.join([self._clean_text(self._extract_text_with_format(p)) for p in p_tags])

                        if fn_text:
                            if label:
                                footnotes.append(f"{label}: {fn_text}")
                            else:
                                footnotes.append(fn_text)

                if footnotes:
                    footnote_text = " | ".join(footnotes)
                    if caption:
                        caption = f"{caption} | {footnote_text}"
                    else:
                        caption = footnote_text

                if caption or content:
                    tables.append({
                        'id': table_id,
                        'caption': caption,
                        'content': content.strip()
                    })
        else:
            standalone_tables = soup.find_all('table')
            for table_tag in standalone_tables:
                table_id = table_tag.get('id', '')

                caption_tag = table_tag.find_previous(['caption', 'title']) or table_tag.find(['caption', 'title'])
                caption = self._clean_text(self._extract_text_with_format(caption_tag)) if caption_tag else ""

                content = ""
                rows = table_tag.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = ' | '.join([self._clean_text(self._extract_text_with_format(cell)) for cell in cells])
                    if row_text.strip():
                        content += row_text + '\n'

                if caption or content:
                    tables.append({
                        'id': table_id,
                        'caption': caption,
                        'content': content.strip()
                    })

        return tables

    def _extract_supplementary_info_bs(self, soup: BeautifulSoup) -> Dict[str, Any]:
        supp_info = {
            'has_supplements': False,
            'footnote_references': [],
            'external_links': [],
            'supplementary_sections': [],
            'supplementary_materials': [],
            'metadata_indicators': {}
        }

        try:
            fn_groups = soup.find_all('fn-group')
            for fn_group in fn_groups:
                fns = fn_group.find_all('fn')
                for fn in fns:
                    fn_text = fn.get_text()
                    if any(keyword in fn_text.lower() for keyword in ['supplementary', 'esi', 'electronic', 'additional']):
                        supp_info['footnote_references'].append(self._clean_text(fn_text))
                        supp_info['has_supplements'] = True

            ext_links = soup.find_all('ext-link')
            for link in ext_links:
                href = link.get('xlink:href', '')
                link_text = link.get_text()
                if any(keyword in (href + link_text).lower() for keyword in ['suppl', 'supplement', 'supporting']):
                    supp_info['external_links'].append({
                        'text': self._clean_text(link_text),
                        'url': href
                    })
                    supp_info['has_supplements'] = True

            all_sections = soup.find_all(['sec', 'section'])
            for sec in all_sections:
                title_tag = sec.find(['title', 'h1', 'h2', 'h3'])
                if title_tag:
                    title = self._clean_text(self._extract_text_with_format(title_tag))
                    if any(keyword in title.lower() for keyword in ['supplementary', 'additional', 'supporting']):
                        content_tags = sec.find_all(['p', 'div'], recursive=True)
                        section_text = '\n\n'.join([self._clean_text(self._extract_text_with_format(tag)) for tag in content_tags])

                        supp_info['supplementary_sections'].append({
                            'title': title,
                            'content': section_text[:500] + '...' if len(section_text) > 500 else section_text
                        })
                        supp_info['has_supplements'] = True

            supp_materials = soup.find_all('supplementary-material')
            for supp in supp_materials:
                supp_id = supp.get('id', '')
                content_type = supp.get('content-type', '')

                label_tag = supp.find('label')
                caption_tag = supp.find('caption')
                label_text = self._clean_text(self._extract_text_with_format(label_tag)) if label_tag else ""
                caption_text = self._clean_text(self._extract_text_with_format(caption_tag)) if caption_tag else ""

                media_elements = supp.find_all('media')
                media_files = []
                for media in media_elements:
                    href = media.get('xlink:href', '')
                    if href:
                        media_files.append({
                            'filename': href,
                            'href': href
                        })

                supp_info['supplementary_materials'].append({
                    'id': supp_id,
                    'content_type': content_type,
                    'label': label_text,
                    'caption': caption_text,
                    'media_files': media_files
                })
                supp_info['has_supplements'] = True

            custom_metas = soup.find_all('custom-meta')
            for meta in custom_metas:
                meta_name_tag = meta.find('meta-name')
                meta_value_tag = meta.find('meta-value')
                if meta_name_tag:
                    meta_name = meta_name_tag.get_text()
                    meta_value = meta_value_tag.get_text() if meta_value_tag else ""

                    if 'supplement' in meta_name.lower():
                        supp_info['metadata_indicators'][meta_name] = meta_value
                        if meta_value.lower() in ['yes', 'true', '1']:
                            supp_info['has_supplements'] = True

        except Exception as e:
            logging.warning(f"Error extracting supplementary info: {e}")

        return supp_info

    def _extract_text_with_format(self, element) -> str:
        if isinstance(element, str):
            return element
        return element.get_text()

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\%\+\=\<\>\'\"±μ°αβγδεζηθικλμνξπρστυφχψω℃℉×÷≤≥≈∼\^\_{}\|‐‑‒–—―]', ' ', text)

        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)

        return text.strip()

    def calculate_relevance_score(self, processed_paper: ProcessedPaper) -> float:
        keywords = MOLECULAR_GLUE_KEYWORDS

        if not keywords:
            return 0.0

        all_text = f"{processed_paper.title} {processed_paper.abstract} {processed_paper.body_text}"
        all_text = all_text.lower()

        total_score = 0.0
        total_words = len(all_text.split())

        for keyword in keywords:
            keyword_lower = keyword.lower()

            title_weight = 3.0
            abstract_weight = 2.0
            body_weight = 1.0

            title_count = processed_paper.title.lower().count(keyword_lower)
            abstract_count = processed_paper.abstract.lower().count(keyword_lower)
            body_count = processed_paper.body_text.lower().count(keyword_lower)

            weighted_score = (title_count * title_weight +
                            abstract_count * abstract_weight +
                            body_count * body_weight)

            total_score += weighted_score

        if total_words > 0:
            return min(total_score / (total_words / 1000), 10.0)

        return total_score
