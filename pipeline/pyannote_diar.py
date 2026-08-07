"""pyannote's community pipeline, as a third opinion on who is speaking.

Worth running here for a specific reason rather than reputation. Our own chain
embeds 1.5-second windows and clusters them, and on this audio that is a dead
end: measured on a real call, the mean embedding distance between two segments
of the *same* speaker is 0.72, while the gap between same-speaker and
different-speaker pairs is only 0.13. The identity is buried under the noise,
so no clustering scheme can recover it.

pyannote is built the other way round. A segmentation network first decides,
at fine resolution and end to end, where the speaker changes inside a sliding
window - a local decision that never consults a speaker embedding. Embeddings
enter only afterwards, to stitch those local decisions into globally consistent
labels. The measurement above indicts the second stage, not the first, and it
is the first stage that decides whether a question and its answer end up in one
turn or two.

Sortformer answers the same objection differently: no embeddings at all. They
fail differently, which is the point of running both.

`speaker-diarization-community-1` is CC-BY-4.0 but gated - Hugging Face wants
contact details accepted before it will serve the weights - so unlike every
other model here the build needs a token. The bake is best-effort for that
reason: a missing token should cost this engine, not the image.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Only the current generation. pyannote renamed rather than renumbered, so
# "3.1" reads as the higher version while community-1 is the one that came
# after and replaced it - their own model card marks 3.1 legacy and puts the
# improvement in speaker counting and assignment, which is our failure. The
# legacy pipeline was briefly wired up here to settle that by measurement
# rather than assertion; it is not worth the disk or the build step.
VARIANTS = {
    "pyannote": ("pyannote/speaker-diarization-community-1",
                 "/opt/pyannote/community1.ok"),
}

_pipelines: dict[str, object] = {}


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Result:
    """Deliberately the same shape as nemo_diar.NemoResult.

    The worker runs every engine through one code path, so a new engine that
    matches this contract needs no special handling anywhere downstream.
    """
    turns: list[Turn]
    speakers: list[str]
    engine: str
    seconds: float = 0.0
    dropped_speech_sec: float = 0.0
    notes: list[str] = field(default_factory=list)


def _fetch_at_runtime(model_id: str) -> None:
    """Pull the gated weights on first use, in a throwaway interpreter.

    Exists for build environments that cannot hold a secret - RunPod's GitHub
    integration builds the image on their infrastructure with no way to mount
    a token, so the bake step there cannot fetch gated weights. The download
    moves to the first cold start instead, where the token arrives as an
    ordinary endpoint environment variable.

    The subprocess is the clean part: this image pins HF_HUB_OFFLINE=1 so the
    runtime never depends on Hugging Face being up, and huggingface_hub reads
    that flag once at import. A child interpreter with the flag overridden can
    download into the shared cache without this process ever going online -
    after it exits, the load below proceeds offline from the cache it filled.
    """
    import subprocess
    import sys

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"{model_id} is not in this image and no HF_TOKEN is set on the "
            "endpoint, so it cannot be fetched. Add the token as an endpoint "
            "environment variable, or bake the weights at build time.")
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from huggingface_hub import snapshot_download; "
         "snapshot_download(repo_id=sys.argv[1], token=sys.argv[2])",
         model_id, token],
        check=True,
        env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
    )


def _cached(model_id: str) -> bool:
    """True if the weights are already in the local Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache
    return try_to_load_from_cache(model_id, "config.yaml") is not None


def _load(engine: str = "pyannote"):
    if engine in _pipelines:
        return _pipelines[engine]

    model_id, marker = VARIANTS[engine]
    if not Path(marker).exists() and not _cached(model_id):
        _fetch_at_runtime(model_id)

    import torch
    from pyannote.audio import Pipeline

    # HF_HUB_OFFLINE is set in the image, so this resolves from the baked cache
    # and never reaches the network - the token is a build-time concern only.
    p = Pipeline.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if p is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained returned None for {model_id} - the usual "
            "cause is unaccepted gating conditions rather than a bad path.")
    if torch.cuda.is_available():
        p.to(torch.device("cuda"))
    _pipelines[engine] = p
    return p


def _as_waveform(audio_path: str | Path) -> dict:
    """Hand pyannote the samples, not a filename.

    Its default reader goes through torchcodec, which needs a matching FFmpeg
    shared library and silently is not there on some platforms - the failure
    surfaces as a decode error at inference time, long after the build passed.
    The worker has already produced 16 kHz mono PCM by this point, so reading
    it ourselves removes that dependency entirely and costs one file read.
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return {"waveform": torch.from_numpy(np.ascontiguousarray(mono))[None, :],
            "sample_rate": int(sr)}


def _iter_turns(output):
    """Read turns out of whichever result shape this pyannote version returns.

    community-1 returns an object carrying several views of the result;
    3.x returned a bare Annotation. Preferring the exclusive view is not a
    compatibility detail but a correctness one: words are assigned to exactly
    one speaker downstream, so overlapping regions would otherwise be counted
    for both and the loser's words would be silently reattributed.
    """
    ann = None
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        ann = getattr(output, attr, None)
        if ann is not None:
            break
    if ann is None:
        ann = output  # a bare Annotation, as 3.x returned

    if hasattr(ann, "itertracks"):
        for segment, _track, speaker in ann.itertracks(yield_label=True):
            yield segment.start, segment.end, speaker
    else:
        for segment, speaker in ann:
            yield segment.start, segment.end, speaker


def diarize(audio_path: str | Path, num_speakers: int | None = 2,
            engine: str = "pyannote") -> Result:
    """num_speakers=None asks the pipeline to count the speakers itself.

    pyannote estimates the count from the audio when it is not given one -
    clustering the segmentation output and choosing the number of clusters
    rather than being handed it. Passing 2 overrides that judgement, which is
    right when a call is known to be two-party and wrong the moment a
    supervisor joins or a third line is conferenced in.
    """
    t0 = time.time()
    pipeline = _load(engine)

    kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
    output = pipeline(_as_waveform(audio_path), **kwargs)

    raw = [Turn(round(s, 3), round(e, 3), spk)
           for s, e, spk in _iter_turns(output) if e > s]
    raw.sort(key=lambda t: t.start)

    # Relabel by first appearance so every engine names speakers the same way
    # and the transcripts can be diffed against each other.
    rename: dict[str, str] = {}
    for t in raw:
        if t.speaker not in rename:
            rename[t.speaker] = f"SPEAKER_{len(rename):02d}"
        t.speaker = rename[t.speaker]

    notes = []
    if num_speakers is None:
        held = {}
        for t in raw:
            held[t.speaker] = held.get(t.speaker, 0.0) + (t.end - t.start)
        notes.append(
            f"counted {len(rename)} speaker(s) from the audio; "
            + ", ".join(f"{k}={v:.0f}s" for k, v in sorted(held.items())))
    elif len(rename) > num_speakers:
        extra = sorted(rename.values())[num_speakers:]
        held = sum(t.end - t.start for t in raw if t.speaker in extra)
        notes.append(
            f"found {len(rename)} speakers where {num_speakers} were expected; "
            f"the surplus labels hold {held:.0f}s of speech.")

    return Result(
        turns=raw,
        speakers=sorted(set(rename.values())),
        engine=engine,
        seconds=round(time.time() - t0, 2),
        notes=notes,
    )
