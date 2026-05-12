FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash tmux \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest

WORKDIR /app
CMD ["tail", "-f", "/dev/null"]
