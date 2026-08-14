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

# torch 2.8, not 2.7. pyannote.audio 4 requires torch>=2.8.0 while the 2.7.1
# base made the two diarizer stacks unsolvable together - pip said so out
# loud rather than quietly installing a broken pair, which is what the single
# combined resolution below is for. 2.8.0 satisfies pyannote and clears NeMo's
# torch>=2.6.0 with room to spare; cuda12.8 matches the previous base, so the
# runtime CUDA story is unchanged.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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

# NeMo and pyannote install together, in one resolution, on purpose. They share
# lightning, omegaconf, speechbrain and transformers, so installing them in
# separate steps lets the second one silently move a shared dependency out from
# under the first. Resolving them together makes pip refuse an incompatible
# combination instead of producing one - a build that fails is far cheaper than
# an image that builds and then breaks on the first paid request. It has
# already earned that: the pair was genuinely unsolvable on torch 2.7 and pip
# refused, rather than leaving us to discover it on the GPU.
#
# lightning is pinned rather than left to the resolver because the two stacks
# leave it exactly one legal value - NeMo's asr extra caps it at <=2.4.0 and
# pyannote floors it at >=2.4. Naming 2.4.0 turns a corridor one version wide
# into a stated fact, so the next conflict here fails loudly instead of
# looking like bad luck.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install -c /tmp/torch-pins.txt \
        "numpy<2.5" \
        "lightning==2.4.0" \
        "nemo_toolkit[asr]==2.7.3" \
        "pyannote.audio>=4.0" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# One check, after every install, covering everything the worker imports. It
# used to run before pyannote arrived, which meant it verified a dependency
# graph that no longer existed by the time the image was published.
#
# torch.version.cuda is checked because pip replacing torch with a CPU wheel
# would otherwise surface only as a slow, wrong first request on the GPU.
RUN python -c "\
import ctranslate2, faster_whisper, speechbrain, sklearn, numpy, numba, torch; \
from nemo.collections.asr.models import SortformerEncLabelModel; \
from pyannote.audio import Pipeline; \
assert torch.version.cuda, 'torch lost its CUDA build during the installs'; \
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

# The turbo model, as a second opinion for word-level verification. Same
# Hebrew training data, distilled architecture, so it makes different mistakes
# - agreement between the two is the strongest confidence signal available
# here, and disagreement locates the errors worth reviewing.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ivrit-ai/whisper-large-v3-turbo-ct2'); print('turbo baked as verifier')"

# ECAPA speaker encoder - loading it through our own module also materialises
# the copied savedir layout the runtime expects.
RUN python -c "\
from pipeline.diarize import _get_encoder; _get_encoder(); import silero_vad; silero_vad.load_silero_vad(); print('ecapa baked')"

# The Hebrew CTC aligner is no longer baked: measured on this telephony
# audio it lost to VAD snapping (it was trained on clean 16 kHz broadcast
# speech), the feature ships off by default, and its 1.2 GB download was
# build-timeout budget spent on a losing arm. RunPod terminates builds at
# 1800 s; a cold-cache build with every bake in place ran past it.

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

# Post-processing thresholds tuned on CALLHOME (8 kHz telephone).
ADD https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/speaker_tasks/diarization/conf/post_processing/diar_streaming_sortformer_4spk-v2_callhome-part1.yaml /opt/nemo_cfg/callhome_pp.yaml

# MSDD is gone from the image. It never once produced a result here (every
# request failed on a NeMo config key), NVIDIA is retiring the clustering
# stack it belongs to, and its three NGC downloads were the slowest part of
# a build that RunPod now terminates at 1800 s. The capabilities probe keys
# on the marker file, so without it the engine reports itself missing and
# clients see an honest absence, not an error.

# pyannote: segmentation first, embeddings only afterwards. Included because
# that ordering targets the failure we measured - on this audio the gap between
# same-speaker and different-speaker embedding distance is 0.13 against a
# within-speaker spread of 0.72, so anything that decides turn boundaries
# without consulting an embedding is worth a measurement.
#
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
print('pyannote community-1 baked')" && touch /opt/pyannote/community1.ok \
    || echo 'pyannote community-1 unavailable - engine will report itself missing'


# Offline from here on: no HF round-trips, rate limits or outages at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    ECAPA_DEVICE=cuda \
    SORTFORMER_PATH=/opt/nemo_models/sortformer.nemo \
    SORTFORMER_PP=/opt/nemo_cfg/callhome_pp.yaml \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

CMD ["python", "-u", "/app/handler.py"]
