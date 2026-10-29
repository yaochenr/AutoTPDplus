"""Core PMC mining primitives: API client, XML processor, storage, models, config."""

from tpd_curator.pmc_miner.core.client import PMCClient
from tpd_curator.pmc_miner.core.config import (
    MAX_RETRIES,
    MOLECULAR_GLUE_KEYWORDS,
    POST_FILTER_KEYWORDS,
    RATE_LIMIT_DELAY,
)
from tpd_curator.pmc_miner.core.models import (
    PaperInventory,
    PaperMetadata,
    PaperRecord,
    ProcessedPaper,
)
from tpd_curator.pmc_miner.core.processor import TextProcessor
from tpd_curator.pmc_miner.core.storage import StorageManager
from tpd_curator.pmc_miner.core.supplementary_downloader import (
    SupplementaryDownloader,
    SupplementaryFile,
    SupplementaryInfo,
)

__all__ = [
    "PMCClient",
    "TextProcessor",
    "StorageManager",
    "SupplementaryDownloader",
    "SupplementaryFile",
    "SupplementaryInfo",
    "PaperMetadata",
    "ProcessedPaper",
    "PaperRecord",
    "PaperInventory",
    "MOLECULAR_GLUE_KEYWORDS",
    "POST_FILTER_KEYWORDS",
    "RATE_LIMIT_DELAY",
    "MAX_RETRIES",
]
