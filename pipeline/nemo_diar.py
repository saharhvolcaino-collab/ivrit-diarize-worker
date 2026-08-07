"""Speaker diarization by neural frame-level segmentation, not sliding windows.

Our own diarizer embeds 1.5-second windows and clusters them. When a question
and its answer fall inside one window - which on a call-centre recording is
most of the interesting moments - that window's embedding is a blend of two
voices, and no amount of clustering can pull them apart afterwards. Silhouette
separation of 0.13-0.18 on a male/female call is what that looks like.

Both models here replace the window with a network that resolves turns at
frame resolution and was trained on telephone speech:

  Sortformer    end-to-end, 80 ms frames, trained on Fisher / NIST SRE /
                CALLHOME - 8 kHz telephone, the same signal we feed it.
                Under the one published cross-model comparison run at a
                single protocol, it scores 10.1 DER on 2-speaker CALLHOME
                against ~19.9 for the pyannote-class systems.
  MSDD          NVIDIA's telephonic preset, trained on ~1500 hours of Fisher.
                Embeds at five window scales at once (1.5 / 1.25 / 1.0 /
                0.75 / 0.5 s) rather than one, so a half-second backchannel
                is caught by the short scale instead of being averaged away.
                Unlike Sortformer it accepts a speaker count.

They fail differently, so we run both and let the transcript decide.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
# Sortformer emits one frame per 80 ms.
FRAME_SEC = 0.08

_sortformer = None
_msdd_cfg_path = os.environ.get(
    "MSDD_CONFIG", "/opt/nemo_cfg/diar_infer_telephonic.yaml")


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class NemoResult:
    turns: list[Turn]
    speakers: list[str]
    engine: str
    seconds: float = 0.0
    # Sortformer only: how much speech landed on tracks we discarded.
    dropped_speech_sec: float = 0.0
    notes: list[str] = field(default_factory=list)


def _to_wav16k(path: str | Path) -> str:
    """NeMo wants 16 kHz mono PCM on disk; upsampling 8 kHz is expected here.

    Sortformer's training data was itself 8 kHz telephone upsampled to 16 kHz,
    so this is the signal shape it learned on, not a compromise.
    """
    fd, out = tempfile.mkstemp(suffix=".nemo16k.wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
         "-af", "aresample=resampler=soxr:precision=28:dither_method=none:osr=16000",
         "-ac", "1", "-c:a", "pcm_s16le", out],
        check=True, capture_output=True,
    )
    return out


# ------------------------------------------------------------------ sortformer


def _load_sortformer():
    global _sortformer
    if _sortformer is None:
        from nemo.collections.asr.models import SortformerEncLabelModel
        path = os.environ.get("SORTFORMER_PATH", "/opt/nemo_models/sortformer.nemo")
        m = (SortformerEncLabelModel.restore_from(path, map_location="cuda", strict=False)
             if Path(path).exists() else
             SortformerEncLabelModel.from_pretrained(
                 "nvidia/diar_streaming_sortformer_4spk-v2", map_location="cuda"))
        m.eval()
        # The "very high latency" preset - effectively offline. It is both the
        # most accurate configuration and the cheapest (RTF 0.002 against 0.093
        # for the low-latency one), because long chunks amortise the encoder.
        mod = m.sortformer_modules
        mod.chunk_len = 340
        mod.chunk_right_context = 40
        mod.fifo_len = 40
        mod.spkcache_update_period = 300
        mod.spkcache_len = 188
        if hasattr(mod, "_check_streaming_parameters"):
            mod._check_streaming_parameters()
        _sortformer = m
    return _sortformer


def _active_tracks(probs: np.ndarray, min_sec: float = 2.0
                   ) -> tuple[np.ndarray, float, list[str]]:
    """Let the model say how many people it heard, instead of being told.

    Sortformer decides speaker count for itself - four output tracks, each
    either carrying a voice or staying near zero. Forcing two throws away that
    judgement, and on a call where a supervisor joins or a third party is
    conferenced in, the discarded track is a real person whose words then get
    attributed to someone else.

    So the only filter here is duration: a track holding less than min_sec of
    speech across the whole recording is treated as leakage rather than a
    participant. Two seconds is deliberately low - a backchannel speaker who
    only says "yes, correct" twice is still a speaker, and the count is
    reported rather than hidden, so an implausible answer is visible instead
    of silently applied.
    """
    active = (probs > 0.5).sum(axis=0)
    keep = np.where(active * FRAME_SEC >= min_sec)[0]
    if keep.size == 0:                       # nothing cleared the bar
        keep = np.array([int(np.argmax(active))])
    dropped = sorted(set(range(probs.shape[1])) - set(keep.tolist()))
    dropped_sec = float(active[dropped].sum() * FRAME_SEC) if dropped else 0.0

    notes = [f"counted {keep.size} speaker(s) from the audio; "
             + ", ".join(f"track {int(t)}={active[t] * FRAME_SEC:.0f}s"
                         for t in keep)]
    return probs[:, np.sort(keep)], dropped_sec, notes


def _collapse_tracks(probs: np.ndarray, n: int) -> tuple[np.ndarray, float, list[str]]:
    """Reduce Sortformer's four output tracks to the n speakers we expect.

    The checkpoint's output layer is four wide and cannot be narrowed without
    retraining - there is no num_speakers argument, and the `max_num_of_spks`
    field that looks like one is never read on the inference path. So the
    selection happens here: keep the n tracks that are active longest, in
    arrival order so track 0 stays whoever spoke first.

    Speech on discarded tracks is reported rather than silently dropped. On a
    clean two-party call those tracks should be near-empty; when they are not,
    that is information about the audio - crosstalk, a third party, a failure -
    and hiding it would turn a visible problem into an invisible one.
    """
    active = (probs > 0.5).sum(axis=0)
    order = np.argsort(-active)
    keep = np.sort(order[:n])
    dropped = sorted(set(range(probs.shape[1])) - set(keep.tolist()))
    dropped_sec = float(active[dropped].sum() * FRAME_SEC) if dropped else 0.0

    notes = []
    if dropped_sec > 5.0:
        notes.append(
            f"{dropped_sec:.0f}s of speech landed on discarded tracks - the audio "
            "may contain crosstalk or more than the expected number of speakers."
        )
    return probs[:, keep], dropped_sec, notes


def _probs_to_turns(probs2: np.ndarray, min_dur: float = 0.20) -> list[Turn]:
    voiced = probs2.max(axis=1) > 0.5
    labels = np.where(voiced, probs2.argmax(axis=1), -1)

    turns: list[Turn] = []
    cur, start = None, 0
    for i, lab in enumerate(np.append(labels, -2)):
        if lab != cur:
            if cur is not None and cur >= 0 and (i - start) * FRAME_SEC >= min_dur:
                turns.append(Turn(round(start * FRAME_SEC, 3),
                                  round(i * FRAME_SEC, 3),
                                  f"SPEAKER_{int(cur):02d}"))
            cur, start = lab, i
    return turns


def diarize_sortformer(audio_path: str | Path,
                       num_speakers: int | None = 2) -> NemoResult:
    """num_speakers=None lets Sortformer report the count it actually heard."""
    import time
    import torch

    t0 = time.time()
    wav = _to_wav16k(audio_path)
    try:
        model = _load_sortformer()
        pp = os.environ.get("SORTFORMER_PP", "/opt/nemo_cfg/callhome_pp.yaml")
        with torch.no_grad():
            _, tensors = model.diarize(
                audio=[wav], batch_size=1, include_tensor_outputs=True,
                postprocessing_yaml=pp if Path(pp).exists() else None,
                verbose=False,
            )
        probs = tensors[0].squeeze(0).float().cpu().numpy()
        probs2, dropped, notes = (
            _active_tracks(probs) if num_speakers is None
            else _collapse_tracks(probs, num_speakers))
        turns = _probs_to_turns(probs2)
        return NemoResult(
            turns=turns,
            speakers=sorted({t.speaker for t in turns}),
            engine="sortformer",
            seconds=round(time.time() - t0, 2),
            dropped_speech_sec=round(dropped, 2),
            notes=notes,
        )
    finally:
        Path(wav).unlink(missing_ok=True)


# ----------------------------------------------------------------------- msdd


def diarize_msdd(audio_path: str | Path,
                 num_speakers: int | None = 2) -> NemoResult:
    """NVIDIA's telephonic clustering diarizer, via its manifest-driven API.

    Unlike Sortformer this one takes the speaker count directly, which removes
    the largest single error source in clustering-based diarization.
    """
    import time

    # The image builds MSDD on a best-effort basis: NVIDIA is retiring the
    # clustering stack, and its checkpoints can vanish from NGC between builds.
    # Say so plainly rather than surfacing a from_pretrained stack trace, and
    # never reach for the network here - the runtime is offline by design.
    if not Path(os.environ.get("MSDD_MARKER", "/opt/nemo_models/msdd.ok")).exists():
        raise RuntimeError(
            "MSDD weights were not baked into this image (NVIDIA is retiring "
            "the clustering diarizer). Use sortformer or ecapa.")

    from omegaconf import OmegaConf
    from nemo.collections.asr.models import NeuralDiarizer

    t0 = time.time()
    wav = _to_wav16k(audio_path)
    workdir = tempfile.mkdtemp(prefix="msdd_")
    try:
        manifest = Path(workdir) / "manifest.json"
        manifest.write_text(json.dumps({
            "audio_filepath": wav,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            # Giving the count turns speaker counting from a guess into a
            # constraint - the single biggest lever a clustering diarizer has.
            "num_speakers": num_speakers,   # null tells NeMo to estimate
            "rttm_filepath": None,
            "uem_filepath": None,
        }) + "\n", encoding="utf-8")

        cfg = OmegaConf.load(_msdd_cfg_path)
        cfg.diarizer.manifest_filepath = str(manifest)
        cfg.diarizer.out_dir = workdir
        # Told a number, MSDD treats it as truth; told nothing, it estimates
        # and needs an upper bound to search under.
        cfg.diarizer.clustering.parameters.oracle_num_speakers = (
            num_speakers is not None)
        cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers or 8

        NeuralDiarizer(cfg=cfg).diarize()

        rttm = next(Path(workdir).rglob("*.rttm"), None)
        if rttm is None:
            return NemoResult([], [], "msdd", round(time.time() - t0, 2),
                              notes=["MSDD produced no RTTM output."])

        turns: list[Turn] = []
        for line in rttm.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                start, dur = float(p[3]), float(p[4])
                turns.append(Turn(round(start, 3), round(start + dur, 3), p[7]))
        turns.sort(key=lambda t: t.start)

        # Relabel in order of first appearance so both engines agree on naming.
        rename: dict[str, str] = {}
        for t in turns:
            if t.speaker not in rename:
                rename[t.speaker] = f"SPEAKER_{len(rename):02d}"
            t.speaker = rename[t.speaker]

        return NemoResult(
            turns=turns,
            speakers=sorted(set(rename.values())),
            engine="msdd",
            seconds=round(time.time() - t0, 2),
        )
    finally:
        Path(wav).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------ word assignment


def assign_words(words: list[dict], turns: list[Turn]) -> list[dict]:
    """Give each word the speaker whose turn covers most of it."""
    out: list[dict] = []
    for w in words:
        overlap: dict[str, float] = {}
        for t in turns:
            lo, hi = max(w["start"], t.start), min(w["end"], t.end)
            if hi > lo:
                overlap[t.speaker] = overlap.get(t.speaker, 0.0) + (hi - lo)
        if overlap:
            best, amount = max(overlap.items(), key=lambda kv: kv[1])
            span = max(w["end"] - w["start"], 1e-6)
            out.append({**w, "speaker": best,
                        "speaker_confidence": round(min(amount / span, 1.0), 3)})
        else:
            mid = (w["start"] + w["end"]) / 2
            nearest = min(
                turns,
                key=lambda t: 0.0 if t.start <= mid <= t.end
                else min(abs(mid - t.end), abs(t.start - mid)),
                default=None,
            )
            out.append({**w, "speaker": nearest.speaker if nearest else None,
                        "speaker_confidence": 0.0})
    return out
