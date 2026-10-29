"""Configuration settings for PMC paper mining."""

import os

# Initial search keywords
MOLECULAR_GLUE_KEYWORDS = [
    "molecular glue",
    "molecular glues",
    "molecular glue degrader",
    "molecular glue degradation",
]

# Post-filter keywords: papers must contain at least one of these in abstract or full text to be considered relevant
POST_FILTER_KEYWORDS = [
    "degrader",
    "degradation",
    "proteasome",
    "TPD",
    "targeted protein degradation",
    "protein degradation",
]

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_SERVICE_URL = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
PMC_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Rate limit: 3 req/s 
RATE_LIMIT_DELAY = 0.34
MAX_RETRIES = 3

BATCH_SIZE = 100
MAX_PAPERS = 1000

# NCBI asks automated clients to identify themselves with a real contact
# address.  Keep credentials/configuration outside the repository while
# retaining conservative defaults for backwards compatibility.
NCBI_TOOL = os.environ.get("NCBI_TOOL", "PMC-TPD-Miner")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "research@example.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
