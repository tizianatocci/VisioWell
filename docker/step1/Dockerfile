FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Install Python 3.10 and essentials
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip curl git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# 2. Install Poetry using python3.10 explicitly
#RUN curl -sSL https://install.python-poetry.org | python3 -
RUN pip install --no-cache-dir poetry
# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

RUN poetry self add poetry-plugin-export

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false
# Install Python dependencies (package-mode = false -> no local package install)

# Copy dependency files first (cache-friendly: dependencies only reinstall
# if pyproject.toml / poetry.lock change, not on every source code edit)
COPY docker/step1/pyproject.toml docker/step1/poetry.lock ./

# Install PyTorch + torchvision CUDA
RUN pip install --no-cache-dir torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130

# install poetry
#RUN poetry install --no-root --no-cache
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt && \
    sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/torchaudio/d' requirements.txt && \
    sed -i '/nvidia-/d' requirements.txt && \
    sed -i '/triton/d' requirements.txt && \
    sed -i '/cuda/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Install CuPy CUDA 13.x (only for step0)
#RUN poetry run pip install cupy-cuda13x==13.6.0

# Copy all source code
COPY src ./src

# No CMD needed - Nextflow will call the scripts directly