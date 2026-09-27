from dotenv import load_dotenv

load_dotenv()

# --- Research Node ---
MAX_SEARCH_ITERATIONS = 3
MIN_SOURCES = 5
QUERIES_PER_ITERATION = 3
RESULTS_PER_QUERY = 5
SNIPPET_CHARS = 500
# Each research round also runs its first query restricted to these domains. Open web
# search mostly returns blogs and vendor pages, and authority is part of the confidence.
OFFICIAL_DOMAINS = ["europa.eu"]

# --- Reviewer Node ---
MAX_REVIEWS = 2

# --- LLM ---
MODEL_NAME = "claude-sonnet-5"
MAX_TOKENS = 16000
# Effort per step: writing search queries is simple; extraction and review need judgment
EFFORT = {
    "queries": "low",
    "supervisor": "low",
    "analyst": "medium",
    "writer": "medium",
    "reviewer": "medium",
    "baseline": "medium",
}
# USD per 1M tokens (Claude Sonnet 5 list price)
PRICE_PER_MTOK = {"input": 2.0, "output": 10.0}

# --- Confidence ---
# confidence = authority × recency × corroboration, computed in src/confidence.py

# Source authority weights
SOURCE_AUTHORITY_WEIGHTS = {
    "official_eu": 1.0,
    "eu_guidance": 0.85,
    "national_authority": 0.7,
    "legal_analysis": 0.5,
    "industry": 0.3,
    "news_blog": 0.15,
}

# Recency multiplier by source age: (maximum age in days, multiplier), checked in order
RECENCY_MULTIPLIERS = [
    (90, 1.0),
    (182, 0.8),
    (365, 0.5),
    (None, 0.2),
]
# Sources without a publication date may be current, but nothing shows it
UNDATED_MULTIPLIER = 0.8

# Corroboration by number of distinct domains supporting a claim (3 or more: 1.4)
CORROBORATION = {1: 1.0, 2: 1.2}
CORROBORATION_MAX = 1.4

# --- Source classification ---
# Matched against the end of the host name ("europa.eu" covers "ec.europa.eu").
# Checked in order; the first match wins. Unmatched hosts containing "law" or
# "legal" count as legal_analysis, and everything else as news_blog.
DOMAIN_TO_SOURCE_TYPE = [
    # Official legal text
    ("eur-lex.europa.eu", "official_eu"),
    ("op.europa.eu", "official_eu"),
    # EU institutions and agencies: Commission, AI Office, Parliament, Council, EDPS, ENISA
    ("europa.eu", "eu_guidance"),
    # National authorities and governments
    ("datatilsynet.no", "national_authority"),
    ("nkom.no", "national_authority"),
    ("regjeringen.no", "national_authority"),
    ("digdir.no", "national_authority"),
    ("cnil.fr", "national_authority"),
    ("gouv.fr", "national_authority"),
    ("bund.de", "national_authority"),
    ("bundesnetzagentur.de", "national_authority"),
    ("aepd.es", "national_authority"),
    ("gob.es", "national_authority"),
    ("garanteprivacy.it", "national_authority"),
    ("autoriteitpersoonsgegevens.nl", "national_authority"),
    ("rijksoverheid.nl", "national_authority"),
    ("imy.se", "national_authority"),
    ("ico.org.uk", "national_authority"),
    ("gov.uk", "national_authority"),
    ("gov", "national_authority"),  # US federal agencies
    # Legal analysis and policy research
    ("artificialintelligenceact.eu", "legal_analysis"),  # Future of Life Institute, unofficial
    ("iapp.org", "legal_analysis"),
    ("oecd.ai", "legal_analysis"),
    ("oecd.org", "legal_analysis"),
    ("brookings.edu", "legal_analysis"),
    ("adalovelaceinstitute.org", "legal_analysis"),
    ("cset.georgetown.edu", "legal_analysis"),
    ("gibsondunn.com", "legal_analysis"),
    ("dlapiper.com", "legal_analysis"),
    ("cliffordchance.com", "legal_analysis"),
    ("linklaters.com", "legal_analysis"),
    ("freshfields.com", "legal_analysis"),
    ("aoshearman.com", "legal_analysis"),
    ("hoganlovells.com", "legal_analysis"),
    ("whitecase.com", "legal_analysis"),
    ("bakermckenzie.com", "legal_analysis"),
    ("lw.com", "legal_analysis"),
    ("skadden.com", "legal_analysis"),
    ("cov.com", "legal_analysis"),
    ("insideglobaltech.com", "legal_analysis"),  # Covington
    ("mayerbrown.com", "legal_analysis"),
    ("orrick.com", "legal_analysis"),
    ("wsgr.com", "legal_analysis"),
    ("nortonrosefulbright.com", "legal_analysis"),
    ("osborneclarke.com", "legal_analysis"),
    ("taylorwessing.com", "legal_analysis"),
    ("fieldfisher.com", "legal_analysis"),
    ("twobirds.com", "legal_analysis"),
    ("loyensloeff.com", "legal_analysis"),
    ("wiersholm.no", "legal_analysis"),
    ("thommessen.no", "legal_analysis"),
    ("bahr.no", "legal_analysis"),
    ("schjodt.no", "legal_analysis"),
    # Industry, vendors and consultancies
    ("pwc.com", "industry"),
    ("deloitte.com", "industry"),
    ("kpmg.com", "industry"),
    ("ey.com", "industry"),
    ("accenture.com", "industry"),
    ("mckinsey.com", "industry"),
    ("bcg.com", "industry"),
    ("ibm.com", "industry"),
    ("microsoft.com", "industry"),
    ("aws.amazon.com", "industry"),
    ("holisticai.com", "industry"),
    ("credo.ai", "industry"),
    ("securiti.ai", "industry"),
    ("onetrust.com", "industry"),
    ("trail-ml.com", "industry"),
    ("modulos.ai", "industry"),
]
