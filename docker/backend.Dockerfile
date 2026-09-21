FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    HUGGINGFACE_HUB_CACHE=/data/hf-cache/hub \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git wget ca-certificates ffmpeg \
        build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# world_engine  (commit b3f1e725) requires torch==2.11.0
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install "world_engine @ git+https://github.com/Overworldai/world_engine.git@b3f1e725b222679a517632918cc78bba0c9fa433"

COPY backend /app
WORKDIR /app
EXPOSE 8791

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8791/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8791", "--ws-ping-interval", "20", "--ws-ping-timeout", "60"]
