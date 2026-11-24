# ClaimGuard demo image (CPU).
#
#   docker build -t claimguard .
#   docker run -p 7860:7860 claimguard                  -> http://localhost:7860
#
# By default the container runs the offline config: rule-based baseline on a small
# sample, no download, starts in a few seconds on any machine.
#
# Full system (my fine-tuned verifier + the SciFact index), mount the two folders and
# switch the config:
#   docker run -p 7860:7860 -e CLAIMGUARD_CONFIG=configs/default.yaml \
#     -v "$(pwd)/models:/app/models" -v "$(pwd)/data/processed:/app/data/processed" claimguard
# The retrieval models are downloaded from the Hugging Face hub at the first start.
# If something is missing, the app falls back to the offline baseline and says so.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLAIMGUARD_CONFIG=configs/offline.yaml \
    PORT=7860

WORKDIR /app

# CPU-only torch first: the default wheel ships CUDA libraries and is several GB bigger
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# dependencies before the code, so Docker can cache this layer when only the code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY data/sample ./data/sample
COPY data/french_bank ./data/french_bank
COPY data/scripts ./data/scripts
COPY app.py ./

# the toy corpus is always available (offline fallback + examples)
RUN python data/scripts/preprocess.py --raw data/sample --out data/processed/sample

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860', timeout=4)"
CMD ["python", "app.py"]
