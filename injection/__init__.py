"""Prompt-injection test of the V2 pipeline: planted sources in a fixed source list (see TEST_PLAN.md)."""

import os

# No LangSmith tracing for these runs, so the planted texts stay on this machine.
# Set before src.config loads .env, which does not override variables already set.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
