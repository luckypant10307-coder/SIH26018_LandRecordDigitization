# Hosted demo image for the Intelligent Land Record Digitization and
# Validation System. SIH 2026 | PS 26018.
#
# Python 3.12, not 3.13+: PaddlePaddle caps at 3.12, and pinning the newest
# interpreter buys nothing here while breaking optional paths.
FROM python:3.12-slim

# --------------------------------------------------------------------------
# System packages
# --------------------------------------------------------------------------
# Tesseract's LANGUAGE PACKS are the reason this image cannot be a plain
# buildpack deploy: pip cannot install them, and without them a Devanagari
# khatauni produces confident nonsense rather than text. Packs are listed
# explicitly rather than via tesseract-ocr-all (~1 GB of scripts this project
# does not read).
#
# hin+eng are load-bearing; the rest cover the scripts ocr_engine.py advertises.
# libgl1/libglib2.0-0 are OpenCV's runtime shared libraries - the headless
# wheel still links against them, and omitting them fails at import, not build.
RUN apt-get update && apt-get install --no-install-recommends -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        tesseract-ocr-ben \
        tesseract-ocr-guj \
        tesseract-ocr-kan \
        tesseract-ocr-mal \
        tesseract-ocr-mar \
        tesseract-ocr-ori \
        tesseract-ocr-pan \
        tesseract-ocr-tam \
        tesseract-ocr-tel \
        tesseract-ocr-asm \
        tesseract-ocr-nep \
        tesseract-ocr-san \
        tesseract-ocr-urd \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------------------
# Python dependencies
# --------------------------------------------------------------------------
# Copied and installed BEFORE the source so an edit to a .py file does not
# invalidate this layer and re-download 200 MB of wheels on every deploy.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
COPY . .

# .env is excluded by .dockerignore and must never be baked into an image
# layer - anyone who can pull the image can read a baked secret. Supply
# BHASHINI_* through the platform's own environment/secret settings instead.
#
# Without them the system still runs: Indic place names simply do not resolve
# against the Latin administrative master, and --check reports that plainly.

# The platform assigns the real port at runtime; run.py reads $PORT. 8000 is
# only the local default, and EXPOSE is documentation, not a binding.
ENV PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    AUTO_SEED=1
EXPOSE 8000

# Storage holds the SQLite database, uploads and per-document working files.
# On a free tier this filesystem is EPHEMERAL: every redeploy starts empty,
# which is why AUTO_SEED exists. Anything a judge must not lose has to be
# exported, or the instance given a real volume.
RUN mkdir -p storage/uploads storage/work

# Fails the build rather than the demo if a language pack or an import is
# missing. --check exits non-zero only on a broken interpreter, so grep the
# thing that actually matters.
RUN python run.py --check | tee /tmp/check.txt \
    && grep -q "Tesseract OCR         : yes" /tmp/check.txt \
    && grep -q "LGD admin master      : loaded" /tmp/check.txt \
    && grep -q "Native PDF text layer : yes" /tmp/check.txt

CMD ["python", "run.py"]
