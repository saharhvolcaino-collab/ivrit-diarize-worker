"""Deeper look at a recording: clipping, quantisation, silence, turn structure.

probe.py answers "what is this file". This answers "how damaged is it, and does
it actually look like a two-person conversation". Both questions need answering
before we can predict how the pipeline will behave, and neither is visible from
the container metadata.
"""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

TARGET_SR = 16000
FRAME_MS = 20


@dataclass
class Analysis:
    path: str
    duration_sec: float
    clipped_sample_pct: float
    clipped_runs: int
    distinct_levels: int
    effective_bits: float
    speech_pct: float
    silence_pct: float
    digital_silence_pct: float
    est_snr_db: float
    speech_segments: int
    median_segment_sec: float
    long_gaps: int
    energy_bimodality: float
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode to float32 mono at 16 kHz using soxr, without touching gain."""
    tmp = path.with_suffix(".analyze.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-af", f"aresample=resampler=soxr:precision=28:osr={TARGET_SR}",
             "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True,
        )
        with wave.open(str(tmp), "rb") as wf:
            sr = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, sr
    finally:
        tmp.unlink(missing_ok=True)


def _raw_levels(path: Path) -> tuple[int, float]:
    """Count distinct sample values in the ORIGINAL file.

    Decoding to 16-bit hides the source quantisation, so we decode to the
    native-ish depth and count unique values. An 8-bit source can only produce
    256 levels no matter what container it sits in, and that ceiling tells us
    how much of the quiet end of the signal survived.
    """
    tmp = path.with_suffix(".levels.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True,
        )
        with wave.open(str(tmp), "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        distinct = int(np.unique(data).size)
        bits = float(np.log2(distinct)) if distinct > 1 else 0.0
        return distinct, bits
    finally:
        tmp.unlink(missing_ok=True)


def _clipping(x: np.ndarray) -> tuple[float, int]:
    """Fraction of samples at full scale, and how many separate runs they form.

    A handful of isolated full-scale samples is meaningless. Long consecutive
    runs mean the waveform was flat-topped, which is real distortion.
    """
    at_rail = np.abs(x) >= 0.999
    pct = 100.0 * float(at_rail.mean())
    # Count runs of >=3 consecutive railed samples.
    if not at_rail.any():
        return 0.0, 0
    idx = np.flatnonzero(np.diff(at_rail.astype(np.int8)) != 0) + 1
    runs = np.split(at_rail, idx)
    long_runs = sum(1 for r in runs if r[0] and r.size >= 3)
    return pct, int(long_runs)


def _frames(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(sr * FRAME_MS / 1000)
    usable = (x.size // n) * n
    return x[:usable].reshape(-1, n)


def _analyse_activity(x: np.ndarray, sr: int) -> dict:
    """Energy-based speech/silence split and turn-structure estimate.

    This is not a VAD and is not meant to be - it is a sanity check that the
    file contains alternating speech of the kind a two-party call produces,
    so we find out now rather than after paying for GPU time.
    """
    frames = _frames(x, sr)
    if frames.size == 0:
        return {}
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)

    # Digital silence - frames that are exactly zero - is not a noise floor, it
    # is the absence of a signal. Including it makes SNR look spectacular for
    # files that simply mute their gaps, so measure the floor only over frames
    # that carry something.
    non_silent = db[db > -80.0]
    if non_silent.size < 10:
        non_silent = db
    floor = float(np.percentile(non_silent, 10))
    peak = float(np.percentile(non_silent, 90))
    snr = peak - floor
    digital_silence_pct = 100.0 * float((db <= -80.0).mean())
    # Threshold sits above the floor but below the speech level.
    thresh = floor + max(6.0, min(0.4 * snr, 18.0))
    active = db > thresh

    speech_pct = 100.0 * float(active.mean())

    # Merge across short gaps so one pause does not split a sentence in two.
    merged = active.copy()
    gap_frames = int(300 / FRAME_MS)
    i = 0
    while i < merged.size:
        if not merged[i]:
            j = i
            while j < merged.size and not merged[j]:
                j += 1
            if 0 < i and j < merged.size and (j - i) <= gap_frames:
                merged[i:j] = True
            i = j
        else:
            i += 1

    edges = np.diff(merged.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if merged[0]:
        starts = np.r_[0, starts]
    if merged[-1]:
        ends = np.r_[ends, merged.size]
    lengths = (ends - starts) * FRAME_MS / 1000.0

    # Long gaps often mark hold time or dead air.
    gaps = []
    for a, b in zip(ends[:-1], starts[1:]):
        gaps.append((b - a) * FRAME_MS / 1000.0)
    long_gaps = sum(1 for g in gaps if g > 3.0)

    # How separable the active frames are into two loudness modes. In a mono
    # call where one party is on a handset and the other down a phone line,
    # the two speakers usually sit at different levels, which shows up as a
    # bimodal energy histogram. Weak evidence, but free.
    active_db = db[active]
    bimodality = 0.0
    if active_db.size > 50:
        mid = float(np.median(active_db))
        lo, hi = active_db[active_db <= mid], active_db[active_db > mid]
        if lo.size and hi.size:
            spread = float(np.std(active_db))
            if spread > 0:
                bimodality = float(abs(hi.mean() - lo.mean()) / spread)

    return {
        "speech_pct": round(speech_pct, 1),
        "silence_pct": round(100.0 - speech_pct, 1),
        "digital_silence_pct": round(digital_silence_pct, 1),
        "est_snr_db": round(snr, 1),
        "speech_segments": int(starts.size),
        "median_segment_sec": round(float(np.median(lengths)), 2) if lengths.size else 0.0,
        "long_gaps": long_gaps,
        "energy_bimodality": round(bimodality, 2),
    }


def analyse(path: str | Path) -> Analysis:
    path = Path(path)
    x, sr = _decode(path)
    duration = x.size / sr

    clip_pct, clip_runs = _clipping(x)
    distinct, bits = _raw_levels(path)
    activity = _analyse_activity(x, sr)

    a = Analysis(
        path=str(path),
        duration_sec=round(duration, 1),
        clipped_sample_pct=round(clip_pct, 4),
        clipped_runs=clip_runs,
        distinct_levels=distinct,
        effective_bits=round(bits, 1),
        speech_pct=activity.get("speech_pct", 0.0),
        silence_pct=activity.get("silence_pct", 0.0),
        digital_silence_pct=activity.get("digital_silence_pct", 0.0),
        est_snr_db=activity.get("est_snr_db", 0.0),
        speech_segments=activity.get("speech_segments", 0),
        median_segment_sec=activity.get("median_segment_sec", 0.0),
        long_gaps=activity.get("long_gaps", 0),
        energy_bimodality=activity.get("energy_bimodality", 0.0),
    )

    if a.clipped_runs == 0 and a.clipped_sample_pct < 0.01:
        a.notes.append(
            "Peak touches full scale but there are no sustained clipped runs - "
            "this is a normalised file, not a distorted one. No action needed."
        )
    elif a.clipped_runs > 0:
        a.notes.append(
            f"{a.clipped_runs} sustained clipped runs ({a.clipped_sample_pct:.3f}% of "
            "samples). Real flat-topping - unrecoverable, and it will hurt speaker "
            "embeddings more than transcription."
        )

    if a.effective_bits <= 8.5:
        a.notes.append(
            f"Only {a.distinct_levels} distinct sample values (~{a.effective_bits:.1f} "
            "bits). Quiet speech is buried in quantisation noise; expect the worst "
            "errors on hesitant or trailing-off words."
        )

    if a.est_snr_db < 20:
        a.notes.append(
            f"Estimated SNR ~{a.est_snr_db} dB is low. Both transcription and speaker "
            "separation will degrade."
        )

    if a.silence_pct > 40:
        a.notes.append(
            f"{a.silence_pct}% of the file is non-speech. VAD is essential - Whisper "
            "hallucinates on silence at very high rates, so this is the single most "
            "important guard for this file."
        )

    if a.energy_bimodality >= 1.2:
        a.notes.append(
            f"Active-frame energy is clearly bimodal (separation {a.energy_bimodality}). "
            "The two parties sit at different levels, which usually helps diarization."
        )
    elif a.energy_bimodality < 0.8:
        a.notes.append(
            f"Active-frame energy is close to unimodal (separation {a.energy_bimodality}). "
            "The two parties are at similar levels, so diarization gets no free help "
            "from loudness and must rely purely on voice timbre."
        )

    return a


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Assess recording damage and turn structure.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    results = [analyse(a) for a in args.audio]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        return 0

    for r in results:
        print(f"\n=== {Path(r.path).name} ===")
        print(f"  duration           {r.duration_sec:.1f} s")
        print(f"  quantisation       {r.distinct_levels} distinct levels (~{r.effective_bits:.1f} bits)")
        print(f"  clipping           {r.clipped_sample_pct:.4f}% of samples, {r.clipped_runs} sustained runs")
        print(f"  speech / silence   {r.speech_pct}% / {r.silence_pct}%")
        print(f"  estimated SNR      {r.est_snr_db} dB  (digital silence {r.digital_silence_pct}%)")
        print(f"  speech segments    {r.speech_segments} (median {r.median_segment_sec}s), "
              f"{r.long_gaps} gaps over 3s")
        print(f"  energy bimodality  {r.energy_bimodality}")
        for n in r.notes:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
