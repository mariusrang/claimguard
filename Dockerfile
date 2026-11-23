# ClaimGuard demo image (CPU).
#
#   docker build -t claimguard .
#   docker run -p 7860:7860 claimguard                  -> http://localhost:7860
#
# By default the container starts with the config given in CLAIMGUARD_CONFIG.
# The neural models are downloaded from the Hugging Face hub at the first start
# (mount a volume on /root/.cache/huggingface to keep them between runs).
# If they cannot be loaded, the app falls back to the offline rule-based baseline,
# so the container always starts.
#
# To run my fine-tuned verifier, mount the models folder:
#   docker run -p 7860:7860 -v $(pwd)/models:/app/models -v $(pwd)/data/processed:/app/data/processed claimguard

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLAIMGUARD_CONFIG=configs/default.yaml \
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
CMD ["python", "app.py"]
