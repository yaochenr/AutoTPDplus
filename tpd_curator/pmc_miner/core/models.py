"""Data models for PMC paper storage and processing."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class PaperMetadata:
    """Bibliographic metadata for a PMC paper."""
    pmcid: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    authors: List[str] = None
    journal: str = ""
    pubdate: str = ""
    abstract: str = ""
    keywords: List[str] = None
    download_date: Optional[str] = None
    relevance_score: Optional[float] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []
        if self.download_date is None:
            self.download_date = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperMetadata':
        return cls(**data)


@dataclass
class ProcessedPaper:
    """Structured output of TextProcessor: sections, tables, figures, supp info."""
    pmcid: str
    title: str = ""
    abstract: str = ""
    body_text: str = ""
    references: List[str] = None
    figures: List[Dict[str, str]] = None
    tables: List[Dict[str, str]] = None
    sections: Dict[str, str] = None
    xml_metadata: Dict[str, str] = None
    supplementary_info: Optional[Dict[str, Any]] = None
    extraction_date: Optional[str] = None

    def __post_init__(self):
        if self.references is None:
            self.references = []
        if self.figures is None:
            self.figures = []
        if self.tables is None:
            self.tables = []
        if self.sections is None:
            self.sections = {}
        if self.xml_metadata is None:
            self.xml_metadata = {}
        if self.supplementary_info is None:
            self.supplementary_info = {}
        if self.extraction_date is None:
            self.extraction_date = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedPaper':
        return cls(**data)


@dataclass
class PaperRecord:
    """Metadata + processed content + raw XML for one paper."""
    metadata: PaperMetadata
    processed: Optional[ProcessedPaper] = None
    raw_xml: Optional[str] = None
    status: str = "pending"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "processed": self.processed.to_dict() if self.processed else None,
            "raw_xml": self.raw_xml,
            "status": self.status,
            "error_message": self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperRecord':
        metadata = PaperMetadata.from_dict(data["metadata"])
        processed = ProcessedPaper.from_dict(data["processed"]) if data.get("processed") else None

        return cls(
            metadata=metadata,
            processed=processed,
            raw_xml=data.get("raw_xml"),
            status=data.get("status", "pending"),
            error_message=data.get("error_message")
        )


class PaperInventory:

    def __init__(self):
        self.papers: Dict[str, PaperRecord] = {}
        self.search_metadata: Dict[str, Any] = {}

    def add_paper(self, paper: PaperRecord):
        self.papers[paper.metadata.pmcid] = paper

    def get_paper(self, pmcid: str) -> Optional[PaperRecord]:
        return self.papers.get(pmcid)

    def update_status(self, pmcid: str, status: str, error_message: str = None):
        if pmcid in self.papers:
            self.papers[pmcid].status = status
            if error_message:
                self.papers[pmcid].error_message = error_message

    def get_papers_by_status(self, status: str) -> List[PaperRecord]:
        return [paper for paper in self.papers.values() if paper.status == status]

    def get_summary(self) -> Dict[str, Any]:
        status_counts = {}
        for paper in self.papers.values():
            status_counts[paper.status] = status_counts.get(paper.status, 0) + 1

        return {
            "total_papers": len(self.papers),
            "status_breakdown": status_counts,
            "search_metadata": self.search_metadata
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "papers": {pmcid: paper.to_dict() for pmcid, paper in self.papers.items()},
            "search_metadata": self.search_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperInventory':
        inventory = cls()
        inventory.search_metadata = data.get("search_metadata", {})

        for pmcid, paper_data in data.get("papers", {}).items():
            paper = PaperRecord.from_dict(paper_data)
            inventory.papers[pmcid] = paper

        return inventory

    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'PaperInventory':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            return cls()
