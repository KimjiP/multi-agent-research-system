# The Streamlit app, for Google Cloud Run. Cloud Build builds this image from the
# source (gcloud run deploy --source .), so no local Docker is needed.
# The API keys are not in the image: Cloud Run passes them in from Secret Manager.
FROM python:3.14.7-slim

# uv, at the same version as in development and in the GitHub workflows
COPY --from=ghcr.io/astral-sh/uv:0.10.1 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# The locked dependencies first, so a code change does not install them again
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev
# Files keep their permissions from the source folder; make them readable for every user
RUN chmod -R a+rX /app

# The app runs as a user without root rights
RUN useradd --create-home app
USER app

ENV PATH="/app/.venv/bin:$PATH"
# Cloud Run sets PORT; exec makes Streamlit receive Cloud Run's stop signal
CMD exec streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 \
    --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
