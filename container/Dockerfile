# syntax=docker/dockerfile:1
# Supply a reviewed linux/amd64 Python 3.11 base by sha256 digest.
# CUDA 12.8 user libraries are supplied by the frozen torch wheel lock.
ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:061c190f3d5815b341a18dfe8a45771f8ba68c15a607ea56997fde3009ac4eec
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG GIT_SHA
ARG BUILD_CREATED
ARG MODEL_CONTRACT_SHA
ARG LOCK_SHA
LABEL org.opencontainers.image.created=$BUILD_CREATED \
      org.opencontainers.image.revision=$GIT_SHA \
      research.model-contract-sha256=$MODEL_CONTRACT_SHA \
      research.lock-sha256=$LOCK_SHA \
      research.base-image=$BASE_IMAGE
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/huggingface
WORKDIR /app
COPY pyproject.toml README.md ./
COPY requirements/model-cu128.lock requirements/model-cu128.lock
COPY src src
COPY configs configs
COPY scripts scripts
COPY container/verify_image.py /usr/local/bin/verify_image.py
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends iperf3 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W > /usr/local/share/os-packages.tsv \
 && pip install --no-cache-dir pip==25.1.1 setuptools==78.1.0 wheel==0.45.1 \
 && command -v iperf3 \
 && python -c 'import sys; assert sys.version_info[:2] == (3,11)' \
 && python /usr/local/bin/verify_image.py build "$BASE_IMAGE" "$GIT_SHA" "$MODEL_CONTRACT_SHA" "$LOCK_SHA" \
 && pip install --no-cache-dir --no-build-isolation --report /usr/local/share/model-install-report.json -r requirements/model-cu128.lock \
 && pip freeze --all > /usr/local/share/python-packages.txt \
 && pip check \
 && python /usr/local/bin/verify_image.py installed \
 && useradd --uid 10001 --create-home researcher
USER 10001:10001
ENTRYPOINT ["adaptive-vlm"]
CMD ["--help"]
