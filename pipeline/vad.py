"""Stage 2b - cut the silence out before it ever reaches the model.

Call-centre audio is mostly not speech. Hold time, typing, transfers, dead
air - our own measurements put one sample file at 60.5% non-speech. That is
the normal condition for this kind of recording, not a defect in one file.

Sending that silence to Whisper is actively harmful. Whisper is generative: it
must emit tokens for every window it is given, so when a window contains
nothing it falls back to the highest-prior sequence in its training data. For
an ivrit.ai model that means Knesset boilerplate. Worse, its own no_speech_prob
guard reported 0.000 on every hallucinated segment we produced - the built-in
detector simply does not fire on this audio, which is why the VAD has to be
external.

Cutting silence therefore does three things at once: it removes the
hallucination trigger, it shrinks the upload, and it cuts GPU time roughly in
proportion to how much silence there was.

The cost of trimming is that every timestamp the model returns refers to the
trimmed timeline. TimelineMap exists to put them back.
"""

from __future__ import annotations

import json
import subprocess
import wave
from bisect import bisect_right
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# Defaults tuned for two-party telephone calls rather than clean speech.
# faster-whisper ships min_silence_duration_ms=2000, which is chosen to keep
# segments near Whisper's 30 s window; we want tighter turn boundaries because
# a diarizer runs downstream, so we cut much shorter.
DEFAULTS = dict(
    # 0.4 rather than the usual 0.5, chosen by sweeping both sample recordings
    # and scoring on signals hallucination cannot inflate - speaker count,
    # repeated lines, and turns stretched over silence. Word count is a trap
    # here: a threshold that admits silence scores well on it precisely because
    # the model invents text to fill that silence. At 0.3 the sparse recording
    # came back with six speakers and a line repeated six times; at 0.4, two
    # speakers and neither artefact, with the highest word confidence of any
    # setting tried.
    threshold=0.4,
    min_speech_duration_ms=100,   # drop sub-100 ms line-noise blips
    min_silence_duration_ms=500,  # tighter than the library default
    speech_pad_ms=250,            # below ~200 ms you clip plosive onsets
    max_speech_duration_s=25.0,   # stay inside Whisper's 30 s window
)


@dataclass
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TimelineMap:
    """Translates times on the trimmed audio back to the original recording.

    The trimmed file is a concatenation of speech spans, so a point in it maps
    back by adding up the durations of everything that came before.
    """

    spans: list[Span]

    def __post_init__(self) -> None:
        self._cum: list[float] = []
        total = 0.0
        for s in self.spans:
            self._cum.append(total)
            total += s.duration
        self.trimmed_duration = total

    def to_original(self, t: float, *, edge_tolerance: float = 0.15) -> float:
        """Map a time on the trimmed clock back to the original recording.

        Concatenation makes this mapping brittle at span boundaries: a
        timestamp that lands a few milliseconds before the end of a span gets
        attributed to that span rather than the next one, and since the two are
        separated by however much silence we cut, a 10 ms error can surface as
        a minute-long one. That is not hypothetical - it is exactly how a line
        spoken at 5:34 came back labelled 4:35.

        So when a time falls within `edge_tolerance` of a span's end and
        another span follows, treat it as the start of the next span. No word
        begins in the last 150 ms of a speech region and finishes in the one
        after a silence, so the next span is the honest answer.
        """
        if not self.spans:
            return t

        idx = bisect_right(self._cum, t) - 1
        idx = max(0, min(idx, len(self.spans) - 1))
        span = self.spans[idx]
        offset = t - self._cum[idx]

        remaining = span.duration - offset
        if remaining < edge_tolerance and idx + 1 < len(self.spans):
            return self.spans[idx + 1].start

        return span.start + min(max(offset, 0.0), span.duration)

    def to_dict(self) -> dict:
        return {
            "trimmed_duration": round(self.trimmed_duration, 3),
            "spans": [{"start": round(s.start, 3), "end": round(s.end, 3)} for s in self.spans],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimelineMap":
        return cls(spans=[Span(**s) for s in d["spans"]])


@dataclass
class VadResult:
    source: str
    output: str
    map_path: str
    original_duration: float
    speech_duration: float
    removed_duration: float
    speech_ratio: float
    span_count: int
    longest_gap: float
    notes: list[str] = field(default_factory=list)


def _load_16k_mono(path: Path) -> np.ndarray:
    tmp = path.with_suffix(".vadtmp.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-af", f"aresample=resampler=soxr:precision=28:osr={SAMPLE_RATE}",
             "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True,
        )
        with wave.open(str(tmp), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        tmp.unlink(missing_ok=True)


def detect_speech(path: str | Path, **overrides) -> tuple[list[Span], float]:
    """Return speech spans in the ORIGINAL timeline, plus total duration."""
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch

    # Multi-threaded float reductions are not bit-stable, and a probability
    # that lands a hair's breadth from the threshold then crosses it on one run
    # and not the next. One span more or less of trim changes the audio the
    # ASR sees, which changed transcripts between supposedly identical runs.
    # Single-threaded inference makes the whole pipeline reproducible; the VAD
    # is cheap enough not to care about the speed.
    torch.set_num_threads(1)

    params = {**DEFAULTS, **overrides}
    audio = _load_16k_mono(Path(path))
    total_duration = len(audio) / SAMPLE_RATE

    model = load_silero_vad()
    stamps = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=params["threshold"],
        min_speech_duration_ms=params["min_speech_duration_ms"],
        min_silence_duration_ms=params["min_silence_duration_ms"],
        speech_pad_ms=params["speech_pad_ms"],
        max_speech_duration_s=params["max_speech_duration_s"],
        return_seconds=True,
    )

    spans = [Span(start=float(s["start"]), end=float(s["end"])) for s in stamps]
    return spans, total_duration


def trim(
    source: str | Path,
    out_dir: str | Path = "out",
    *,
    make_opus: bool = True,
    **overrides,
) -> VadResult:
    """Write a speech-only copy of the recording plus its timeline map."""
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spans, total = detect_speech(source, **overrides)

    if not spans:
        raise RuntimeError(
            f"No speech detected in {source.name}. Either the file is silent or the "
            "VAD threshold is too high for this recording."
        )

    audio = _load_16k_mono(source)
    pieces = [
        audio[int(s.start * SAMPLE_RATE): int(s.end * SAMPLE_RATE)]
        for s in spans
    ]
    trimmed = np.concatenate(pieces)

    stem = source.stem.replace(".16k", "")
    wav_out = out_dir / f"{stem}.speech.wav"
    with wave.open(str(wav_out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((np.clip(trimmed, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    tmap = TimelineMap(spans=spans)
    map_path = out_dir / f"{stem}.timeline.json"
    map_path.write_text(json.dumps(tmap.to_dict(), indent=2), encoding="utf-8")

    output_path = wav_out
    if make_opus:
        opus_out = out_dir / f"{stem}.speech.opus"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_out),
             "-c:a", "libopus", "-b:a", "24k", "-vbr", "on",
             "-application", "voip", "-ac", "1", str(opus_out)],
            check=True, capture_output=True,
        )
        output_path = opus_out

    speech = sum(s.duration for s in spans)
    gaps = [b.start - a.end for a, b in zip(spans[:-1], spans[1:])]
    longest_gap = max(gaps) if gaps else 0.0

    result = VadResult(
        source=str(source),
        output=str(output_path),
        map_path=str(map_path),
        original_duration=round(total, 2),
        speech_duration=round(speech, 2),
        removed_duration=round(total - speech, 2),
        speech_ratio=round(speech / total, 4) if total else 0.0,
        span_count=len(spans),
        longest_gap=round(longest_gap, 2),
    )

    if result.speech_ratio < 0.5:
        result.notes.append(
            f"Only {result.speech_ratio:.0%} of this recording is speech. Trimming "
            f"removes {result.removed_duration:.0f}s, which cuts GPU cost by roughly "
            "the same proportion and removes the main hallucination trigger."
        )
    if longest_gap > 30:
        result.notes.append(
            f"Longest silent gap is {longest_gap:.0f}s - likely hold time. Untrimmed, "
            "this is exactly the kind of stretch Whisper fills with invented text."
        )
    if result.speech_ratio > 0.95:
        result.notes.append(
            "Almost the whole file is speech, so trimming buys little here. Harmless "
            "to run anyway - it keeps one code path for every recording."
        )

    return result


def remap_transcript(parsed: dict, tmap: TimelineMap, *, split_gap: float = 2.0) -> dict:
    """Put a transcript back on the original clock, re-cutting turns as needed.

    Words are mapped individually rather than deriving a turn's span from its
    endpoints. On the trimmed clock a single turn can run straight through a
    boundary, but on the original clock the same turn may straddle a minute of
    hold music - so a turn whose words jump more than `split_gap` seconds gets
    split there. Without this a turn's duration counts silence we deliberately
    removed, which is how speaker shares ended up summing to more than the
    total speech in the file.
    """
    out = json.loads(json.dumps(parsed))  # deep copy
    new_turns: list[dict] = []

    for turn in out.get("turns", []):
        words = turn.get("words") or []

        if not words:
            start = tmap.to_original(turn["start"])
            new_turns.append({
                **turn,
                "start": round(start, 2),
                "end": round(max(start, tmap.to_original(turn["end"])), 2),
            })
            continue

        for w in words:
            w["start"] = round(tmap.to_original(w["start"]), 3)
            w["end"] = round(max(w["start"], tmap.to_original(w["end"])), 3)

        chunk: list[dict] = [words[0]]
        chunks: list[list[dict]] = []
        for prev, cur in zip(words, words[1:]):
            if cur["start"] - prev["end"] > split_gap:
                chunks.append(chunk)
                chunk = [cur]
            else:
                chunk.append(cur)
        chunks.append(chunk)

        for group in chunks:
            text = " ".join(w["word"] for w in group if w.get("word")).strip()
            new_turns.append({
                "speaker": turn["speaker"],
                "start": round(group[0]["start"], 2),
                "end": round(group[-1]["end"], 2),
                "text": text,
                "words": group,
            })

    # Splitting can interleave turns out of order relative to the original clock.
    new_turns.sort(key=lambda t: t["start"])

    # A split may leave two adjacent fragments from the same speaker; rejoin
    # those that are genuinely contiguous so the transcript still reads well.
    merged: list[dict] = []
    for turn in new_turns:
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and turn["start"] - merged[-1]["end"] <= split_gap
        ):
            prev = merged[-1]
            prev["end"] = max(prev["end"], turn["end"])
            prev["text"] = f"{prev['text']} {turn['text']}".strip()
            prev["words"].extend(turn["words"])
        else:
            merged.append(turn)

    out["turns"] = merged
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Remove non-speech before transcription.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out")
    p.add_argument("--threshold", type=float, default=DEFAULTS["threshold"],
                   help="Speech probability cutoff. Raise toward 0.6 for noisy lines.")
    p.add_argument("--pad", type=int, default=DEFAULTS["speech_pad_ms"],
                   help="Milliseconds kept either side of speech")
    p.add_argument("--no-opus", action="store_true")
    args = p.parse_args(argv)

    for item in args.audio:
        r = trim(
            item, args.out,
            make_opus=not args.no_opus,
            threshold=args.threshold,
            speech_pad_ms=args.pad,
        )
        saved = 100 * (1 - r.speech_ratio)
        print(f"\n=== {Path(r.source).name} ===")
        print(f"  original     {r.original_duration:7.1f}s")
        print(f"  speech       {r.speech_duration:7.1f}s  in {r.span_count} spans")
        print(f"  removed      {r.removed_duration:7.1f}s  ({saved:.0f}% of the file)")
        print(f"  longest gap  {r.longest_gap:7.1f}s")
        for n in r.notes:
            print(f"  - {n}")
        print(f"  -> {r.output}")
        print(f"  -> {r.map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
