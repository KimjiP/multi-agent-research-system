from dotenv import load_dotenv

load_dotenv()

# --- Research Node ---
MAX_SEARCH_ITERATIONS = 3
MIN_SOURCES = 5
QUERIES_PER_ITERATION = 3

# --- Reviewer Node ---
MAX_REVIEWS = 2

# --- LLM ---
MODEL_NAME = "claude-sonnet-4-20250514"
TEMPERATURE = 0.2

# --- Source Authority Weights ---
# Used by the Analyst to compute confidence scores
# confidence = authority_weight × recency_multiplier × corroboration_factor
SOURCE_AUTHORITY_WEIGHTS = {
    "official_eu": 1.0,
    "eu_guidance": 0.85,
    "national_authority": 0.7,
    "legal_analysis": 0.5,
    "industry": 0.3,
    "news_blog": 0.15,
}

# --- Recency Multipliers ---
RECENCY_MULTIPLIERS = {
    "0_90_days": 1.0,
    "3_6_months": 0.8,
    "6_12_months": 0.5,
    "12_plus_months": 0.2,
}

# --- Domain-to-Source-Type Mapping ---
DOMAIN_TO_SOURCE_TYPE = {
    "eur-lex.europa.eu": "official_eu",
    "artificialintelligenceact.eu": "official_eu",
    "digital-strategy.ec.europa.eu": "eu_guidance",
    "ec.europa.eu": "eu_guidance",
    "ai-regulation.com": "legal_analysis",
}
