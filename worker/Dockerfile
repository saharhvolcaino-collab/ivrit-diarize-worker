# Hebrew call transcription + speaker attribution, one image.
#
# Deliberately lean: no NeMo, no pyannote, no gated models, no tokens. The
# diarization chain here (ECAPA windows -> purified clustering -> per-word
# refinement -> conversational splitting) was validated on real recordings
# before being promoted to GPU, and every model in it is ungated.
#
# Build from the REPO ROOT (the pipeline/ package is copied in):
#   docker build -f worker/Dockerfile -t ivrit-diarize-worker .

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ctranslate2 >= 4.6.3 carries no cuDNN dependency; the base image's torch
# provides CUDA/cuBLAS. numpy<2.5 keeps numba importable (unconstrained, the
# resolver backtracks numba to a 2021 release that cannot import).
RUN pip install \
        "faster-whisper==1.2.1" \
        "ctranslate2==4.8.1" \
        "speechbrain>=1.0" \
        "silero-vad" \
        "scikit-learn" \
        "numpy<2.5" \
        "huggingface-hub" \
        "runpod"

# Fail the build, not the first paid request, if the dependency graph broke.
RUN python -c "import ctranslate2, faster_whisper, speechbrain, sklearn, numpy, torch; \
print('ct2', ctranslate2.__version__, '| torch', torch.__version__, \
      '| numpy', numpy.__version__)"

WORKDIR /app
COPY pipeline/ /app/pipeline/
COPY worker/handler.py /app/handler.py

# ---- bake weights ----------------------------------------------------------
# RunPod bills from worker start; multi-GB downloads during cold start are
# time paid for on every scale-up. Everything is fetched at build time.

# Whisper (CTranslate2 snapshot into the HF cache the runtime will read).
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ivrit-ai/whisper-large-v3-ct2')"

# ECAPA speaker encoder - loading it through our own module also materialises
# the copied savedir layout the runtime expects.
RUN python -c "\
from pipeline.diarize import _get_encoder; _get_encoder(); import silero_vad; silero_vad.load_silero_vad(); print('ecapa baked')"

# Offline from here on: no HF round-trips, rate limits or outages at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    ECAPA_DEVICE=cuda

CMD ["python", "-u", "/app/handler.py"]
