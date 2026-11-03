# syntax=docker/dockerfile:1
# Supply a reviewed linux/amd64 Python 3.11 base by sha256 digest.
# CUDA 12.8 user libraries are supplied by the frozen torch wheel lock.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG GIT_SHA
ARG MODEL_CONTRACT_SHA
ARG LOCK_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA \
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
RUN command -v iperf3 \
 && python -c 'import sys; assert sys.version_info[:2] == (3,11)' \
 && python /usr/local/bin/verify_image.py build "$BASE_IMAGE" "$GIT_SHA" "$MODEL_CONTRACT_SHA" "$LOCK_SHA" \
 && pip install --no-cache-dir --no-build-isolation -r requirements/model-cu128.lock \
 && pip check \
 && python /usr/local/bin/verify_image.py installed \
 && useradd --uid 10001 --create-home researcher
USER 10001:10001
ENTRYPOINT ["adaptive-vlm"]
CMD ["--help"]
