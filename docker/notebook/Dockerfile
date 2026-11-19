FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Configure Poetry
ENV POETRY_VIRTUALENVS_CREATE=false 

RUN pip install --no-cache-dir poetry

# 3. Copy dependency files (cache-friendly)
COPY docker/notebook/pyproject.toml docker/notebook/poetry.lock* ./

RUN poetry install --no-root --no-cache

# 4. Copy source code
COPY notebook ./notebook