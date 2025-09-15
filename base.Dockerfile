FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
LABEL authors="Marcel Gohsen"

SHELL ["/bin/bash", "-c"]

RUN set -x \
    && apt update \
    && apt install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock README.md /app/
WORKDIR /app/

RUN --mount=type=cache,target=/root/.cache set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install poetry \
    && python3 -m poetry config virtualenvs.create false \
    && python3 -m poetry --no-root --no-interaction --no-ansi install \
    && python3 -m pip install --no-build-isolation flash-attn

COPY . /app/

RUN --mount=type=cache,target=/root/.cache python3 -m poetry --no-interaction --no-ansi install