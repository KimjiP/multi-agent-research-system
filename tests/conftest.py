import os

# No LangSmith tracing in the tests, so no test sends anything over the network.
# Set before src.config loads .env, which does not override variables already set.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
