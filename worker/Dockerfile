# syntax=docker/dockerfile:1.7
# Hebrew call transcription + speaker attribution, one image.
#
# Four diarizers ship together so the transcript can pick a winner instead of
# us picking one on reputation: our own ECAPA window chain, NeMo's Sortformer,
# NeMo's telephonic MSDD, and pyannote's community pipeline. The first three
# are ungated. pyannote is not - its weights need a Hugging Face token at build
# time - so it is baked best-effort and reports itself missing if the token was
# absent, rather than taking the image down with it.
#
# The image is large and the build runs on a hosted runner with a finite disk,
# so caches are purged in the layer that creates them rather than at the end
# (a later `rm` cannot shrink an earlier layer).
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

# NeMo's ASR extra carries Sortformer and the telephonic MSDD diarizer. numpy
# stays capped below 2.5 because numba - a transitive dependency through
# librosa - cannot import above it, and pip resolves that combination happily
# and then fails at runtime.
#
# The base image is a `runtime` tag and ships no compiler, but several of
# NeMo's transitive dependencies have no wheel for this interpreter and build
# from source. The toolchain is installed and removed inside its own layer so
# ~250 MB of gcc does not ride along to every worker.
#
# The pin file below is the point of this whole section. NeMo depends on torch,
# and the torch already in this image carries a local version tag (+cu128) that
# pip does not match against a plain requirement - so it happily "satisfies"
# the dependency by downloading torch again from PyPI, which drags in the whole
# nvidia-cu12 wheel set: gigabytes of download, more unpacked, on top of a CUDA
# stack that was already here. Pinning the installed versions without their
# local segment makes pip keep what is here. Cheap insurance, not a fix for an
# observed failure - this layer has never been watched succeeding.
RUN python -c 'import importlib.metadata as m; ns=("torch","torchaudio","torchvision"); p={}; [p.setdefault(d.metadata["Name"], d.version.split("+")[0]) for d in m.distributions() if d.metadata["Name"] in ns]; print("\n".join(k+"=="+v for k,v in p.items()))' > /tmp/torch-pins.txt \
    && cat /tmp/torch-pins.txt

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install -c /tmp/torch-pins.txt "numpy<2.5" "nemo_toolkit[asr]==2.7.3" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# If pip replaced torch anyway, every downstream layer would run on a CPU build
# and the failure would only surface as a slow, wrong first request.
RUN python -c "import torch; assert torch.version.cuda, 'torch lost its CUDA build during the NeMo install'; print('torch', torch.__version__)"

# Fail the build, not the first paid request, if the dependency graph broke.
# Sortformer is required; the import is unguarded on purpose. MSDD is checked
# separately below because it is allowed to be missing.
RUN python -c "import ctranslate2, faster_whisper, speechbrain, sklearn, numpy, numba, torch; \
from nemo.collections.asr.models import SortformerEncLabelModel; \
print('ct2', ctranslate2.__version__, '| torch', torch.__version__, \
      '| numpy', numpy.__version__, '| numba', numba.__version__)"

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

# Sortformer: ungated, CC-BY-4.0, one .nemo file. Trained on Fisher / NIST SRE
# / CALLHOME - 8 kHz telephone upsampled to 16 kHz, which is our exact signal.
# The HF cache copy is deleted in the same layer; keeping both would carry a
# second copy of the weights in the published image for no benefit.
RUN mkdir -p /opt/nemo_models /opt/nemo_cfg && python -c "\
import shutil; from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='nvidia/diar_streaming_sortformer_4spk-v2', \
                    filename='diar_streaming_sortformer_4spk-v2.nemo'); \
shutil.copy(p, '/opt/nemo_models/sortformer.nemo'); print('sortformer baked')" \
    && rm -rf /root/.cache/huggingface/hub/models--nvidia--diar_streaming_sortformer_4spk-v2

# Post-processing thresholds tuned on CALLHOME (8 kHz telephone), and NVIDIA's
# telephonic inference preset - five window scales rather than one, so short
# backchannels survive.
ADD https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/speaker_tasks/diarization/conf/post_processing/diar_streaming_sortformer_4spk-v2_callhome-part1.yaml /opt/nemo_cfg/callhome_pp.yaml
ADD https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml /opt/nemo_cfg/diar_infer_telephonic.yaml

# MSDD's own weights, so the first request does not pay for the download.
#
# Deliberately non-fatal. NVIDIA is retiring the clustering diarization stack
# in favour of Sortformer, so these checkpoints may disappear from NGC without
# notice. MSDD is a comparison arm, not the product - losing it should cost us
# the arm, not the whole image. The marker file tells the runtime which is the
# case, so a missing engine is reported honestly instead of failing per request.
RUN python -c "\
import nemo.collections.asr as na; \
na.models.EncDecSpeakerLabelModel.from_pretrained('titanet_large'); \
na.models.EncDecClassificationModel.from_pretrained('vad_multilingual_marblenet'); \
na.models.msdd_models.EncDecDiarLabelModel.from_pretrained('diar_msdd_telephonic'); \
print('msdd baked')" && touch /opt/nemo_models/msdd.ok \
    || echo 'MSDD unavailable at build time - engine will report itself missing'

# pyannote: segmentation first, embeddings only afterwards. Included because
# that ordering targets the failure we measured - on this audio the gap between
# same-speaker and different-speaker embedding distance is 0.13 against a
# within-speaker spread of 0.72, so anything that decides turn boundaries
# without consulting an embedding is worth a measurement.
RUN pip install -c /tmp/torch-pins.txt "pyannote.audio>=4.0"

# The token is mounted, never copied: a build-arg would be readable in the
# image history forever. Best-effort for the same reason MSDD is - a gated
# download that fails should cost this engine, not the whole build.
RUN --mount=type=secret,id=hf_token \
    mkdir -p /opt/pyannote && \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null)" python -c "\
import os; \
from pyannote.audio import Pipeline; \
t = os.environ.get('HF_TOKEN') or None; \
p = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=t); \
assert p is not None, 'gated: conditions not accepted for this token'; \
print('pyannote baked')" && touch /opt/pyannote/ready.ok \
    || echo 'pyannote unavailable at build time - engine will report itself missing'

# Offline from here on: no HF round-trips, rate limits or outages at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    ECAPA_DEVICE=cuda \
    SORTFORMER_PATH=/opt/nemo_models/sortformer.nemo \
    SORTFORMER_PP=/opt/nemo_cfg/callhome_pp.yaml \
    MSDD_CONFIG=/opt/nemo_cfg/diar_infer_telephonic.yaml \
    MSDD_MARKER=/opt/nemo_models/msdd.ok \
    PYANNOTE_MARKER=/opt/pyannote/ready.ok \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

CMD ["python", "-u", "/app/handler.py"]
