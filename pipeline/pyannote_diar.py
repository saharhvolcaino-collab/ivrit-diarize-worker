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

MODEL_ID = os.environ.get("PYANNOTE_MODEL",
                          "pyannote/speaker-diarization-community-1")
MARKER = os.environ.get("PYANNOTE_MARKER", "/opt/pyannote/ready.ok")

_pipeline = None


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


def _load():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not Path(MARKER).exists():
        raise RuntimeError(
            f"{MODEL_ID} was not baked into this image. It is gated: accept the "
            "conditions on its Hugging Face page and give the build an HF_TOKEN.")

    import torch
    from pyannote.audio import Pipeline

    # HF_HUB_OFFLINE is set in the image, so this resolves from the baked cache
    # and never reaches the network - the token is a build-time concern only.
    p = Pipeline.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if p is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained returned None for {MODEL_ID} - the usual "
            "cause is unaccepted gating conditions rather than a bad path.")
    if torch.cuda.is_available():
        p.to(torch.device("cuda"))
    _pipeline = p
    return p


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


def diarize(audio_path: str | Path, num_speakers: int = 2) -> Result:
    t0 = time.time()
    pipeline = _load()

    # 16 kHz mono is what the worker already writes, and what the pipeline
    # wants; passing the path lets pyannote do its own IO rather than us
    # guessing at its tensor layout.
    output = pipeline(str(audio_path), num_speakers=num_speakers)

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
    if len(rename) > num_speakers:
        extra = sorted(rename.values())[num_speakers:]
        held = sum(t.end - t.start for t in raw if t.speaker in extra)
        notes.append(
            f"found {len(rename)} speakers where {num_speakers} were expected; "
            f"the surplus labels hold {held:.0f}s of speech.")

    return Result(
        turns=raw,
        speakers=sorted(set(rename.values())),
        engine="pyannote",
        seconds=round(time.time() - t0, 2),
        notes=notes,
    )
