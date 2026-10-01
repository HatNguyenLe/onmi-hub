# syntax=docker/dockerfile:1
FROM python:3.12-slim

# CPU wheels by default; override for GPU, e.g. --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMNI_DATA_DIR=/data \
    HF_HOME=/data/hf-cache

WORKDIR /app

RUN pip install --index-url ${TORCH_INDEX_URL} torch

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

RUN useradd --create-home --uid 1000 omni \
    && mkdir -p /data \
    && chown -R omni:omni /data
USER omni

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/settings', timeout=4)" || exit 1

WORKDIR /app/backend
# Single worker: the model cache and fine-tune queue live in-process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
