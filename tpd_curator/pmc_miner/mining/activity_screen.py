"""Auditable quantitative degradation-activity screening for PMC OA papers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup

from tpd_curator.pmc_miner.core.client import PMCClient
from tpd_curator.pmc_miner.core.models import PaperMetadata, ProcessedPaper
from tpd_curator.pmc_miner.core.processor import TextProcessor
from tpd_curator.pmc_miner.core.storage import StorageManager
from tpd_curator.pmc_miner.mining.citation_registry import (
    CitationRecord,
    CitationRegistry,
    load_registry_json,
    write_registry,
)


SCREENING_METHOD = "activity_rules_v6"
ACCEPTED = "accepted_quantitative_activity"
MANUAL = "needs_manual_review"
REJECTED = "rejected_no_quantitative_activity"
ERROR = "screen_error"
TERMINAL_STATUSES = {ACCEPTED, MANUAL, REJECTED, ERROR}

_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_NUMBER = r"(?P<value>[<>~≤≥]?[ ]*\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)"
_CONC_UNIT = r"(?P<unit>pM|nM|[µμu]M|mM|M)"
_TPD_RE = re.compile(
    r"\b(?:PROTACs?|proteolysis[- ]targeting chimeras?|proteolysis targeting chimeras?|"
    r"heterobifunctional degraders?|bifunctional degraders?|target(?:ed)? protein degradation|targeted degradation|"
    r"protein degraders?|protein degradation inducers?|ligand[- ]induced protein degradation|"
    r"molecular glue degraders?|neosubstrates?|cereblon|CRBN|VHL|"
    r"ubiquitin(?:ation)?|proteasom(?:e|al)|E3 ligases?)\b",
    re.IGNORECASE,
)
_TPD_FRONT_RE = re.compile(
    r"\b(?:PROTACs?|proteolysis[- ]targeting chimeras?|proteolysis targeting chimeras?|"
    r"heterobifunctional degraders?|bifunctional degraders?|target(?:ed)? protein degradation|"
    r"targeted degradation|protein degraders?|protein degradation inducers?|"
    r"ligand[- ]induced protein degradation|molecular glue degraders?|neosubstrates?)\b",
    re.IGNORECASE,
)
_DC50_RE = re.compile(
    rf"\bDC[ _-]?50\b[^.;\n]{{0,55}}?{_NUMBER}\s*{_CONC_UNIT}\b",
    re.IGNORECASE,
)
_DMAX_RE = re.compile(
    rf"\bD[ _-]?max\b[^.;\n]{{0,55}}?{_NUMBER}\s*(?P<unit>%)",
    re.IGNORECASE,
)
_DEGRADATION_PERCENT_RE = re.compile(
    rf"(?:degrad(?:ation|ed|es|ing)|reduc(?:tion|ed)\s+(?:in\s+)?(?:target\s+)?protein(?:\s+level)?)"
    rf"[^.;\n%]{{0,35}}?(?:of|by|to|at|was|reached|reaching|is|=|:|approximately|about|around|"
    rf"over|up to|more than|greater than|nearly|almost)\s*{_NUMBER}\s*(?P<unit>%)|"
    rf"(?:degradation|protein reduction)(?:\s+(?:rate|efficiency|level))?\s*(?:of|=|:)?\s*"
    rf"(?P<value_direct>[<>~≤≥]?[ ]*\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)\s*(?P<unit_direct>%)|"
    r"(?P<value_rev>[<>~≤≥]?[ ]*\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)"
    r"\s*(?P<unit_rev>%)\s+(?:target\s+|protein\s+)?degrad(?:ation|ed)",
    re.IGNORECASE,
)
_CONCENTRATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:pM|nM|[µμu]M|mM|M)\b", re.IGNORECASE)
_TIME_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|min(?:ute)?s?|h|hr|hrs|hours?|days?)\b",
    re.IGNORECASE,
)
_WEAK_ACTIVITY_RE = re.compile(
    r"\b(?:dose[- ]dependent|concentration[- ]dependent|time[- ]dependent)\s+(?:target\s+|protein\s+)?"
    r"degrad(?:ation|ation)|western blot|immunoblot|degradation assay|degraded protein",
    re.IGNORECASE,
)
_METRIC_MENTION_RE = re.compile(r"\b(?:DC[ _-]?50|D[ _-]?max)\b", re.IGNORECASE)
_SUPPLEMENT_RE = re.compile(r"\b(?:supplementary|supplemental|supporting information|ESI)\b", re.IGNORECASE)
_NON_TPD_GLUE_RE = re.compile(
    r"(?:molecular glue).{0,100}(?:DNA|RNA|adhesive|hydrogel|polymer|biomaterial|"
    r"receptor dimeri[sz]ation|cell[- ]surface|supramolecular)",
    re.IGNORECASE,
)
_REVIEW_TITLE_RE = re.compile(
    r"\b(?:review|perspective|overview|landscape|bibliometric|scientometric|recent advances|state of the art)\b",
    re.IGNORECASE,
)
_COMPUTATIONAL_TITLE_RE = re.compile(
    r"\b(?:prediction|predictive|machine[- ]learning|deep[- ]learning|QSP model|"
    r"computational|simulations?|models?|in silico|framework)\b",
    re.IGNORECASE,
)
_SECONDARY_NARRATIVE_RE = re.compile(
    r"\b(?:was|were|has been|have been|previously)\s+reported\b|\b(?:prior|previous)\s+(?:work|study|studies)\b",
    re.IGNORECASE,
)
_COMPUTATIONAL_EVIDENCE_RE = re.compile(
    r"\b(?:simulations?|predicted|prediction|training (?:set|data)|test set|dataset|in silico)\b",
    re.IGNORECASE,
)
_PRIMARY_TYPE_RE = re.compile(
    r"(?:research-article|original[- ]research|article|rapid-communication|short-report)",
    re.IGNORECASE,
)
_REVIEW_TYPE_RE = re.compile(r"(?:review|systematic-review|meta-analysis)", re.IGNORECASE)
_COMPOUND_RE = re.compile(
    r"\b(?:compound|PROTAC|degrader|molecular glue|MGD|PROTAC)[- ]?[A-Za-z0-9][A-Za-z0-9_-]*\b|"
    r"\b[A-Z]{1,5}[-–]?\d{1,5}[A-Za-z]?\b"
)
_TARGET_RE = re.compile(
    r"\b(?:target(?:ing)?\s+)?([A-Z][A-Z0-9-]{1,11}|(?:BRD|CDK|BTK|BCL|STAT|KRAS|"
    r"ER|AR|Tau|IKZF|GSPT|SMARCA|WEE|FAK|EGFR|ALK|MEK|MDM|HSP)[A-Za-z0-9-]*)\b"
)


@dataclass
class TextChunk:
    text: str
    section: str
    context_type: str


@dataclass
class ScreeningDecision:
    status: str
    article_type: str
    evidence_level: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    error: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").translate(_SUBSCRIPT_TRANSLATION)).strip()


def infer_article_type(processed: ProcessedPaper) -> str:
    metadata = processed.xml_metadata or {}
    raw_type = str(metadata.get("article_type") or "")
    subjects = " ".join(str(item) for item in metadata.get("subjects", []) or [])
    combined_type = f"{raw_type} {subjects}"
    if _REVIEW_TYPE_RE.search(combined_type) or _REVIEW_TITLE_RE.search(processed.title or ""):
        return "review"
    if _PRIMARY_TYPE_RE.search(raw_type):
        return "primary"
    if raw_type:
        return "other"
    return "unknown"


def iter_text_chunks(processed: ProcessedPaper) -> Iterable[TextChunk]:
    """Yield auditable chunks while deliberately excluding references."""
    if processed.title:
        yield TextChunk(_normalise_text(processed.title), "Title", "title")
    if processed.abstract:
        yield TextChunk(_normalise_text(processed.abstract), "Abstract", "abstract")

    for section, content in (processed.sections or {}).items():
        for part in re.split(r"(?<=[.!?])\s+|\n+", str(content or "")):
            part = _normalise_text(part)
            if part:
                yield TextChunk(part, str(section), "section")

    for table in processed.tables or []:
        caption = _normalise_text(table.get("caption", ""))
        rows = str(table.get("content", "")).splitlines() or [""]
        header = _normalise_text(rows[0]) if rows else ""
        for row in rows:
            text = _normalise_text(" | ".join(item for item in (caption, header, row) if item))
            if text:
                yield TextChunk(text, table.get("id") or caption or "Table", "table")

    for figure in processed.figures or []:
        text = _normalise_text(" ".join(
            str(figure.get(key, "")) for key in ("title", "caption")
        ))
        if text:
            yield TextChunk(text, figure.get("id") or figure.get("title") or "Figure", "figure_caption")


def _article_tpd_context(processed: ProcessedPaper) -> bool:
    front = _normalise_text(f"{processed.title} {processed.abstract}")
    if _TPD_RE.search(front):
        return True
    return any(_TPD_RE.search(chunk.text) for chunk in iter_text_chunks(processed))


def _entities(text: str) -> Tuple[str, str]:
    compound_match = _COMPOUND_RE.search(text)
    target_match = _TARGET_RE.search(text)
    return (
        compound_match.group(0) if compound_match else "",
        target_match.group(1) if target_match else "",
    )


def _evidence_from_match(
    metric: str,
    match: re.Match,
    chunk: TextChunk,
    origin: str,
) -> Dict[str, Any]:
    compound, target = _entities(chunk.text)
    value = (
        match.groupdict().get("value")
        or match.groupdict().get("value_direct")
        or match.groupdict().get("value_rev")
        or ""
    )
    unit = (
        match.groupdict().get("unit")
        or match.groupdict().get("unit_direct")
        or match.groupdict().get("unit_rev")
        or ""
    )
    return {
        "metric": metric,
        "value": value.replace(" ", ""),
        "unit": unit.replace("uM", "µM"),
        "compound": compound,
        "target": target,
        "section": chunk.section,
        "context_type": chunk.context_type,
        "evidence_origin": origin,
        "snippet": chunk.text[:500],
    }


def classify_processed_paper(processed: ProcessedPaper, include_reviews: bool = True) -> ScreeningDecision:
    article_type = infer_article_type(processed)
    chunks = list(iter_text_chunks(processed))
    tpd_context = _article_tpd_context(processed)
    front = _normalise_text(f"{processed.title} {processed.abstract}")
    front_tpd_context = bool(_TPD_FRONT_RE.search(front))

    if not tpd_context:
        reason = "non_tpd_molecular_glue_context" if _NON_TPD_GLUE_RE.search(front) else "no_tpd_context"
        return ScreeningDecision(REJECTED, article_type, "none", reasons=[reason])

    evidence: List[Dict[str, Any]] = []
    unparsed_metric = False
    weak_activity = False
    supplement_hint = False
    for chunk in chunks:
        text = chunk.text
        local_tpd_context = bool(_TPD_RE.search(text))
        chunk_has_tpd = local_tpd_context or front_tpd_context
        if not chunk_has_tpd:
            continue

        matched_spans: set[Tuple[int, int, str]] = set()
        for metric, pattern in (("DC50", _DC50_RE), ("Dmax", _DMAX_RE)):
            for match in pattern.finditer(text):
                matched_spans.add((match.start(), match.end(), metric))
                origin = "secondary_summary" if article_type == "review" else "primary"
                evidence.append(_evidence_from_match(metric, match, chunk, origin))

        metric_mentions = list(_METRIC_MENTION_RE.finditer(text))
        if metric_mentions and not matched_spans:
            unparsed_metric = True

        for match in _DEGRADATION_PERCENT_RE.finditer(text):
            if not (_CONCENTRATION_RE.search(text) or _TIME_RE.search(text)):
                continue
            compound, target = _entities(text)
            if not (
                local_tpd_context
                or re.search(r"\b(?:target|protein)\b", text, re.IGNORECASE)
                or (compound and target)
            ):
                continue
            origin = "secondary_summary" if article_type == "review" else "primary"
            evidence.append(_evidence_from_match("degradation_percent", match, chunk, origin))

        if _WEAK_ACTIVITY_RE.search(text):
            weak_activity = True
        if _SUPPLEMENT_RE.search(text) and (
            _METRIC_MENTION_RE.search(text) or _WEAK_ACTIVITY_RE.search(text)
        ):
            supplement_hint = True

    # De-duplicate identical metric/value/location hits.
    unique: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for item in evidence:
        key = (
            item["metric"], item["value"], item["unit"],
            item["section"], item["snippet"],
        )
        unique.setdefault(key, item)
    evidence = list(unique.values())

    if article_type == "review":
        table_evidence = [item for item in evidence if item["context_type"] == "table"]
        if include_reviews and table_evidence:
            return ScreeningDecision(
                ACCEPTED, article_type, "secondary_summary", table_evidence,
                ["review_table_contains_quantitative_degradation_activity"],
            )
        if evidence:
            return ScreeningDecision(
                MANUAL, article_type, "possible_quantitative", evidence,
                ["review_quantitative_evidence_not_in_structured_table"],
            )
        if unparsed_metric or weak_activity or supplement_hint:
            reasons = ["review_requires_manual_confirmation"]
            if supplement_hint:
                reasons.append("activity_may_be_in_supplement")
            return ScreeningDecision(MANUAL, article_type, "weak", reasons=reasons)
        return ScreeningDecision(REJECTED, article_type, "none", reasons=["review_without_structured_quantitative_data"])

    experimental_evidence: List[Dict[str, Any]] = []
    contextual_evidence: List[Dict[str, Any]] = []
    for item in evidence:
        section = item["section"].lower()
        snippet = item["snippet"]
        non_result_section = (
            item["context_type"] == "section"
            and not re.search(r"\bresults?|findings?\b", section)
            and re.search(r"\b(?:introduction|background|methods?|materials?|discussion|conclusions?)\b", section)
        )
        secondary_narrative = bool(_SECONDARY_NARRATIVE_RE.search(snippet))
        if non_result_section or secondary_narrative:
            contextual_evidence.append(item)
        else:
            experimental_evidence.append(item)

    if (
        _COMPUTATIONAL_TITLE_RE.search(processed.title or "")
        or (
            experimental_evidence
            and all(_COMPUTATIONAL_EVIDENCE_RE.search(item["snippet"]) for item in experimental_evidence)
        )
    ) and evidence:
        return ScreeningDecision(
            MANUAL, article_type, "possible_quantitative", evidence,
            ["computational_or_prediction_article_requires_experiment_confirmation"],
        )
    if experimental_evidence:
        return ScreeningDecision(
            ACCEPTED, article_type, "strong", experimental_evidence,
            ["quantitative_degradation_activity_detected"],
        )
    if contextual_evidence:
        return ScreeningDecision(
            MANUAL, article_type, "possible_quantitative", contextual_evidence,
            ["quantitative_evidence_only_in_background_methods_or_secondary_narrative"],
        )
    if unparsed_metric or weak_activity or supplement_hint:
        reasons: List[str] = []
        if unparsed_metric:
            reasons.append("activity_metric_without_parseable_value")
        if weak_activity:
            reasons.append("qualitative_or_image_based_degradation_activity")
        if supplement_hint:
            reasons.append("activity_may_be_in_supplement")
        return ScreeningDecision(MANUAL, article_type, "weak", reasons=reasons)
    if article_type in {"unknown", "other"} and "molecular glue" in front.lower():
        return ScreeningDecision(MANUAL, article_type, "ambiguous", reasons=["ambiguous_molecular_glue_or_article_type"])
    return ScreeningDecision(REJECTED, article_type, "none", reasons=["no_quantitative_degradation_activity"])


def _extract_supplement_chunks(paper_dir: Path) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    supp_dir = paper_dir / "supplementary"
    if not supp_dir.exists():
        return chunks
    ignored_metadata = {
        "supplementary_metadata.json",
        "extraction_manifest.json",
        "document_normalization_manifest.json",
    }
    for path in sorted(supp_dir.rglob("*")):
        if not path.is_file() or path.name in ignored_metadata:
            continue
        suffix = path.suffix.lower()
        relative_path = path.relative_to(supp_dir).as_posix()
        try:
            texts: List[str] = []
            if suffix in {".txt", ".csv", ".tsv", ".md"}:
                texts = [path.read_text(encoding="utf-8", errors="ignore")]
            elif suffix == ".docx":
                with zipfile.ZipFile(path) as archive:
                    xml = archive.read("word/document.xml")
                texts = [BeautifulSoup(xml, "xml").get_text(" ")]
            elif suffix in {".xlsx", ".xls"}:
                try:
                    import pandas as pd
                    sheets = pd.read_excel(path, sheet_name=None, header=None, nrows=5000)
                    texts = [frame.to_csv(index=False, header=False) for frame in sheets.values()]
                except Exception as exc:
                    logging.warning("Could not parse spreadsheet supplement %s: %s", path, exc)
            elif suffix == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                    texts = ["\n".join(
                        page.extract_text() or "" for page in reader.pages[:75]
                    )]
                except Exception as exc:
                    logging.warning("Could not parse PDF supplement %s: %s", path, exc)
            for text in texts:
                for part in re.split(r"(?<=[.!?])\s+|\n+", text):
                    part = _normalise_text(part)
                    if part:
                        chunks.append(TextChunk(part, relative_path, "supplement"))
        except Exception as exc:
            logging.warning("Could not read supplement %s: %s", path, exc)
    return chunks


def _screen_supplements(processed: ProcessedPaper, paper_dir: Path) -> Optional[ScreeningDecision]:
    supplement_chunks = _extract_supplement_chunks(paper_dir)
    if not supplement_chunks:
        return None
    if not _TPD_FRONT_RE.search(_normalise_text(f"{processed.title} {processed.abstract}")):
        return None
    synthetic = ProcessedPaper(
        pmcid=processed.pmcid,
        title=processed.title,
        abstract=processed.abstract,
        sections={f"{chunk.section}:{index}": chunk.text for index, chunk in enumerate(supplement_chunks)},
        xml_metadata=processed.xml_metadata,
    )
    decision = classify_processed_paper(synthetic, include_reviews=True)
    if decision.status == ACCEPTED:
        for item in decision.evidence:
            item["context_type"] = "supplement"
            item["section"] = item.get("section") or "supplement"
        decision.reasons = ["quantitative_degradation_activity_in_supplement"]
        return decision
    return None


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _screen_record(
    record: CitationRecord,
    client: PMCClient,
    processor: TextProcessor,
    cache: StorageManager,
    include_reviews: bool,
    screen_supplements: bool = True,
) -> ScreeningDecision:
    try:
        processed = cache.load_processed_paper(record.pmcid)
        if processed is None:
            xml = cache.load_paper_xml(record.pmcid) or client.get_full_text_xml(record.pmcid)
            if not xml:
                raise RuntimeError("full_text_xml_unavailable")
            if not client.validate_full_text_availability(xml):
                raise RuntimeError("full_text_xml_not_available")
            processed = processor.process_xml(xml, record.pmcid)
            if processed is None:
                raise RuntimeError("jats_xml_parse_failed")
            metadata = PaperMetadata(
                pmcid=record.pmcid,
                pmid=record.pmid,
                doi=record.doi,
                title=record.title or processed.title,
                journal=record.journal,
                pubdate=record.publication_year,
                abstract=processed.abstract,
            )
            cache.save_paper_xml(record.pmcid, xml)
            cache.save_paper_metadata(record.pmcid, metadata)
            cache.save_processed_paper(record.pmcid, processed, download_supplements=False)

        decision = classify_processed_paper(processed, include_reviews=include_reviews)
        if screen_supplements and decision.status == MANUAL and "activity_may_be_in_supplement" in decision.reasons:
            downloader = None
            if not cache.has_supplementary_files(record.pmcid):
                from tpd_curator.pmc_miner.core.supplementary_downloader import SupplementaryDownloader
                downloader = SupplementaryDownloader(cache.data_dir)
                downloader.download_supplementary_info(record.pmcid)
            supplement_decision = _screen_supplements(processed, cache.get_paper_dir(record.pmcid))
            if downloader is not None:
                downloader.cleanup_temporary_files(record.pmcid)
            if supplement_decision:
                decision = supplement_decision
        return decision
    except Exception as exc:
        logging.exception("Screening failed for %s", record.pmcid)
        return ScreeningDecision(ERROR, "unknown", "none", reasons=["screening_failed"], error=str(exc))


def _apply_decision(record: CitationRecord, decision: ScreeningDecision) -> None:
    record.screening_status = decision.status
    record.article_type = decision.article_type
    record.screening_method = SCREENING_METHOD
    record.activity_evidence_level = decision.evidence_level
    record.activity_evidence = decision.evidence
    record.screening_reasons = decision.reasons
    record.screened_at = _utc_now()
    record.screening_error = decision.error


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _write_screening_outputs(
    registry: CitationRegistry,
    output_dir: Path,
    run_summary: Dict[str, Any],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    screened = [
        record for record in registry.records()
        if not record.confirmed_relevant and record.screening_status in TERMINAL_STATUSES
    ]
    fields = [
        "pmcid", "doi", "pmid", "title", "publication_year", "journal",
        "modalities", "article_type", "screening_status", "activity_evidence_level",
        "screening_reasons", "evidence_count", "screened_at", "screening_error",
        "download_status", "paper_path",
    ]
    paths: Dict[str, Path] = {}
    for status, filename in (
        (ACCEPTED, "accepted_articles.csv"),
        (MANUAL, "manual_review_queue.csv"),
        (REJECTED, "rejected_articles.csv"),
        (ERROR, "screen_errors.csv"),
    ):
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in screened:
                if record.screening_status != status:
                    continue
                row = record.to_dict()
                row["evidence_count"] = len(record.activity_evidence)
                writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
        paths[status] = path

    evidence_path = output_dir / "screening_evidence.json"
    _atomic_json(evidence_path, {
        "screening_method": SCREENING_METHOD,
        "articles": [record.to_dict() for record in screened],
    })
    summary_path = output_dir / "screening_summary.json"
    _atomic_json(summary_path, run_summary)
    paths["evidence"] = evidence_path
    paths["summary"] = summary_path
    return paths


def _write_validation_sample(
    registry: CitationRegistry,
    output_dir: Path,
    limit: int = 100,
    accepted_only: bool = False,
) -> Path:
    records = [
        record for record in registry.records()
        if not record.confirmed_relevant and record.screening_status in TERMINAL_STATUSES
        and (not accepted_only or record.screening_status == ACCEPTED)
    ]
    strata: Dict[Tuple[str, str, str], List[CitationRecord]] = defaultdict(list)
    for record in records:
        modality = "+".join(sorted(record.modalities)) or "unknown"
        year_band = "2024+" if (record.publication_year and int(record.publication_year) >= 2024) else "<=2023"
        strata[(modality, year_band, record.article_type)].append(record)
    for values in strata.values():
        values.sort(key=lambda item: hashlib.sha256(item.pmcid.encode()).hexdigest())

    sample: List[CitationRecord] = []
    while len(sample) < min(limit, len(records)) and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(sample) < limit:
                sample.append(strata[key].pop(0))

    filename = "accepted_validation_sample_100.csv" if accepted_only else "validation_sample_100.csv"
    path = output_dir / filename
    fields = [
        "pmcid", "doi", "title", "publication_year", "modalities", "article_type",
        "screening_status", "activity_evidence_level", "screening_reasons",
        "human_label", "human_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sample:
            data = record.to_dict()
            writer.writerow({key: _csv_value(data.get(key, "")) for key in fields})
    return path


def quarantine_invalid_tack2_directories(catalog_root: Path) -> List[str]:
    papers_dir = catalog_root / "papers"
    quarantine = catalog_root / "quarantine" / "invalid_tack2_resolution"
    moved: List[str] = []
    for pmcid in ("PMC6834731", "PMC6957805", "PMC7297535"):
        source = papers_dir / pmcid
        target = quarantine / pmcid
        if source.exists() and not target.exists():
            quarantine.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(pmcid)
    return moved


def _find_reusable(reuse_dir: Optional[Path], papers_dir: Path) -> Dict[str, Path]:
    reusable: Dict[str, Path] = {}
    if not reuse_dir or not reuse_dir.exists():
        return reusable
    for metadata_path in reuse_dir.glob("*/PMC*/metadata.json"):
        paper_dir = metadata_path.parent
        if papers_dir in paper_dir.parents:
            continue
        if (paper_dir / "full_text.xml").exists() and (paper_dir / "processed.json").exists():
            reusable.setdefault(paper_dir.name, paper_dir)
    return reusable


def download_accepted_articles(
    records: Sequence[CitationRecord],
    papers_dir: Path,
    reuse_dir: Optional[Path],
    download_images: bool,
) -> Dict[str, Any]:
    from tpd_curator.pmc_miner.mining.doi_miner import DOIBasedMiner

    miner = DOIBasedMiner(
        output_dir=str(papers_dir),
        paper_type="accepted_activity_articles",
        download_images=download_images,
    )
    reusable = _find_reusable(reuse_dir, papers_dir)
    statuses: Dict[str, str] = {}
    for index, record in enumerate(records, 1):
        logging.info("Downloading accepted article %d/%d: %s", index, len(records), record.pmcid)
        paper_dir = miner.storage.get_paper_dir(record.pmcid)
        if record.pmcid in reusable:
            status = "reused_local"
            paper_dir = reusable[record.pmcid]
        elif miner.storage.paper_exists(record.pmcid):
            status = "already_downloaded"
        else:
            status = miner._process_doi(record.doi, record.pmcid)
        record.download_status = status
        if status in {"success", "already_downloaded", "reused_local"}:
            record.paper_path = str(paper_dir)
        statuses[record.pmcid] = status
    return {
        "eligible_accepted_articles": len(records),
        "downloaded_or_existing": sum(value in {"success", "already_downloaded", "reused_local"} for value in statuses.values()),
        "newly_downloaded": sum(value == "success" for value in statuses.values()),
        "reused_local": sum(value == "reused_local" for value in statuses.values()),
        "statuses": statuses,
    }


def screen_catalog(
    catalog_json: Path,
    output_dir: Path,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    include_reviews: bool = False,
    download_accepted: bool = False,
    reuse_dir: Optional[Path] = None,
    download_images: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    catalog_json = Path(catalog_json)
    output_dir = Path(output_dir)
    registry, catalog_summary = load_registry_json(catalog_json)
    candidates = [
        record for record in registry.records()
        if not record.confirmed_relevant
        and record.pmc_open_access is True
        and record.pmcid
        and (record.screening_status == "pending_activity_screen" or record.screening_status in TERMINAL_STATUSES)
        and (year_from is None or (record.publication_year and int(record.publication_year) >= year_from))
        and (year_to is None or (record.publication_year and int(record.publication_year) <= year_to))
    ]
    if not resume:
        candidates = [record for record in candidates if record.screening_status == "pending_activity_screen"]

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = StorageManager(output_dir / "screening_cache")
    client = PMCClient()
    processor = TextProcessor()
    processed_now = 0
    skipped_resume = 0
    for index, record in enumerate(candidates, 1):
        if resume and record.screening_status in TERMINAL_STATUSES and record.screening_method == SCREENING_METHOD:
            skipped_resume += 1
            continue
        logging.info("Screening %d/%d: %s", index, len(candidates), record.pmcid)
        if (
            record.screening_method
            and record.screening_method != SCREENING_METHOD
            and record.screening_status == ACCEPTED
            and "quantitative_degradation_activity_in_supplement" in record.screening_reasons
        ):
            record.screening_method = SCREENING_METHOD
            record.screened_at = _utc_now()
            processed_now += 1
            continue
        decision = _screen_record(
            record, client, processor, cache, include_reviews,
            screen_supplements=not bool(record.screening_method and record.screening_method != SCREENING_METHOD),
        )
        _apply_decision(record, decision)
        processed_now += 1
        if processed_now % 10 == 0:
            checkpoint_summary = dict(catalog_summary)
            checkpoint_summary["activity_screening_checkpoint"] = {
                "method": SCREENING_METHOD,
                "processed_now": processed_now,
                "updated_at": _utc_now(),
            }
            write_registry(registry, catalog_json.parent, checkpoint_summary)

    selected_after = [
        record for record in registry.records()
        if not record.confirmed_relevant
        and record.pmc_open_access is True
        and record.pmcid
        and (year_from is None or (record.publication_year and int(record.publication_year) >= year_from))
        and (year_to is None or (record.publication_year and int(record.publication_year) <= year_to))
        and record.screening_status in TERMINAL_STATUSES
    ]
    selected_counts = Counter(record.screening_status for record in selected_after)

    quarantine_moved = quarantine_invalid_tack2_directories(catalog_json.parent)
    download_summary: Dict[str, Any] = {}
    if download_accepted:
        accepted = [record for record in selected_after if record.screening_status == ACCEPTED]
        download_summary = download_accepted_articles(
            accepted,
            catalog_json.parent / "papers",
            reuse_dir,
            download_images,
        )

    cumulative = [
        record for record in registry.records()
        if not record.confirmed_relevant and record.screening_status in TERMINAL_STATUSES
    ]
    cumulative_counts = Counter(record.screening_status for record in cumulative)
    run_summary = {
        "screening_method": SCREENING_METHOD,
        "started_year_from": year_from,
        "started_year_to": year_to,
        "include_reviews": include_reviews,
        "selected_articles": len(selected_after),
        "processed_now": processed_now,
        "skipped_resume": skipped_resume,
        "selected_status_counts": dict(selected_counts),
        "selected_reconciles": sum(selected_counts.values()) == len(selected_after),
        "cumulative_screened_articles": len(cumulative),
        "cumulative_status_counts": dict(cumulative_counts),
        "cumulative_unique_pmcids": len({record.pmcid for record in cumulative}),
        "quarantined_invalid_tack2_directories": quarantine_moved,
        "accepted_downloads": download_summary,
        "completed_at": _utc_now(),
    }
    catalog_summary["activity_screening"] = run_summary
    write_registry(registry, catalog_json.parent, catalog_summary)
    paths = _write_screening_outputs(registry, output_dir, run_summary)
    paths["validation_sample"] = _write_validation_sample(registry, output_dir)
    paths["accepted_validation_sample"] = _write_validation_sample(
        registry, output_dir, accepted_only=True
    )
    return {"summary": run_summary, "paths": paths}
