"""pmc_miner — mine full-text papers, tables, figures, and supplementary files from PMC."""

from tpd_curator.pmc_miner.core.client import PMCClient
from tpd_curator.pmc_miner.core.models import (
    PaperInventory,
    PaperMetadata,
    PaperRecord,
    ProcessedPaper,
)
from tpd_curator.pmc_miner.core.processor import TextProcessor
from tpd_curator.pmc_miner.core.storage import StorageManager
from tpd_curator.pmc_miner.core.supplementary_downloader import SupplementaryDownloader
from tpd_curator.pmc_miner.mining.doi_miner import DOIBasedMiner
from tpd_curator.pmc_miner.mining.search_miner import PostFilter, SearchBasedMiner
from tpd_curator.pmc_miner.utils.image_downloader import PMCImageDownloader

__version__ = "0.2.0"

__all__ = [
    "SearchBasedMiner",
    "DOIBasedMiner",
    "PostFilter",
    "PMCClient",
    "TextProcessor",
    "StorageManager",
    "SupplementaryDownloader",
    "PMCImageDownloader",
    "PaperMetadata",
    "ProcessedPaper",
    "PaperRecord",
    "PaperInventory",
    "__version__",
]
