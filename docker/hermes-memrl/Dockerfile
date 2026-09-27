# Bakes the MemRL memory provider (docker/hermes-memrl/memrl/) into the
# pinned upstream Hermes image as a bundled plugin at
# /opt/hermes/plugins/memory/memrl, so `memory.provider: memrl` in a Hermes
# config.yaml activates it. See docker/hermes-memrl/README.md.
#
# Per-task Hermes containers run on a task-private network, so everything the
# plugin needs at runtime is pre-baked and the Hugging Face hub is forced
# offline:
#
#  1. CPU-only torch + sentence-transformers in Hermes's own venv. The base
#     image installs no numpy, so it is added at the version Hermes's own
#     pyproject.toml pins; every package is an addition, none is replaced.
#  2. The BAAI/bge-small-en-v1.5 embedding model, downloaded into a fixed,
#     world-readable HF_HOME at build time.
#
# The plugin's unit tests run at build time against the real
# agent.memory_provider, then a loader smoke check confirms Hermes can
# discover the plugin and reports it available.
#
# Build from the repo root:
#   docker build -f docker/hermes-memrl/Dockerfile -t <tag> .
FROM docker.io/nousresearch/hermes-agent:v2026.5.29.2

USER root
ENV HF_HOME=/opt/hermes-models/hf
ARG TORCH_VERSION=2.9.1
ARG SENTENCE_TRANSFORMERS_VERSION=5.1.2
ARG NUMPY_VERSION=2.4.3
RUN uv pip install --no-cache-dir --python /opt/hermes/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && uv pip install --no-cache-dir --python /opt/hermes/.venv/bin/python \
        "sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION}" "numpy==${NUMPY_VERSION}" \
    && /opt/hermes/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')" \
    && chmod -R a+rX /opt/hermes-models \
    && chown -R hermes:hermes /opt/hermes/.venv
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY docker/hermes-memrl/ /tmp/memrl-src/
RUN set -eu; \
    uv pip install --no-cache-dir --python /opt/hermes/.venv/bin/python --target /tmp/pytest-deps pytest; \
    cd /opt/hermes; \
    PYTHONPATH=/opt/hermes:/tmp/pytest-deps /opt/hermes/.venv/bin/python -m pytest -q -p no:cacheprovider /tmp/memrl-src/tests; \
    cp -r /tmp/memrl-src/memrl /opt/hermes/plugins/memory/memrl; \
    chmod -R a+rX /opt/hermes/plugins/memory/memrl; \
    /opt/hermes/.venv/bin/python -c "from plugins.memory import load_memory_provider; p = load_memory_provider('memrl'); assert p is not None and p.is_available(), p.unavailable_reason() if p else 'not found'; print('memrl provider available')"; \
    rm -rf /tmp/memrl-src /tmp/pytest-deps
