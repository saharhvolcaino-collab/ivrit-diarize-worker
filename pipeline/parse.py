"""Turn a raw RunPod response into something a person can read and a script can score.

The worker returns a deeply nested stream: the job output holds a list of
batches, each batch a list of events, each event either a progress tick or a
payload of segments. Roughly two thirds of it is progress noise. This flattens
that into segments, merges consecutive segments from the same speaker into
turns, and writes three views:

  .txt   a readable transcript, one turn per speaker
  .json  structured turns and words, for scoring
  .srt   subtitles, for spot-checking timing against the audio

Speaker labels stay as SPEAKER_00 / SPEAKER_01 here. Mapping those to real
names is a separate stage - it needs voiceprints, and it deserves to be
measured on its own rather than hidden inside this conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Word:
    word: str
    start: float
    end: float
    speaker: str | None
    probability: float | None


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transcript:
    source: str
    turns: list[Turn]
    speakers: list[str]
    duration_sec: float
    word_count: int
    mean_word_confidence: float
    low_confidence_words: int
    unattributed_words: int

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_sec": self.duration_sec,
            "speakers": self.speakers,
            "word_count": self.word_count,
            "mean_word_confidence": self.mean_word_confidence,
            "low_confidence_words": self.low_confidence_words,
            "unattributed_words": self.unattributed_words,
            "turns": [
                {
                    "speaker": t.speaker,
                    "start": round(t.start, 2),
                    "end": round(t.end, 2),
                    "text": t.text,
                    "words": [asdict(w) for w in t.words],
                }
                for t in self.turns
            ],
        }


def extract_segments(raw: dict) -> list[dict]:
    """Pull segment payloads out of the batched event stream."""
    output = raw.get("output")
    if output is None:
        raise ValueError("No 'output' in response - the job may have failed")

    # Non-streaming responses wrap everything in {"result": [...]}.
    containers = output if isinstance(output, list) else [output]
    segments: list[dict] = []

    for container in containers:
        result = container.get("result") if isinstance(container, dict) else container
        if result is None:
            continue
        for batch in result:
            events = batch if isinstance(batch, list) else [batch]
            for event in events:
                if not isinstance(event, dict) or event.get("type") != "segments":
                    continue
                data = event.get("data")
                if isinstance(data, list):
                    segments.extend(d for d in data if isinstance(d, dict))
                elif isinstance(data, dict):
                    segments.append(data)

    return segments


def _segment_speaker(seg: dict) -> str:
    """Pick one speaker for a segment.

    The worker reports a list, because a segment can straddle a speaker change.
    When it does, we attribute the segment to whoever holds the most words in
    it - the same max-overlap rule the rest of the field uses, applied at word
    granularity rather than by duration.
    """
    words = seg.get("words") or []
    tally: dict[str, int] = {}
    for w in words:
        spk = w.get("speaker")
        if spk:
            tally[spk] = tally.get(spk, 0) + 1
    if tally:
        return max(tally.items(), key=lambda kv: kv[1])[0]

    speakers = seg.get("speakers") or []
    return speakers[0] if speakers else "UNKNOWN"


def build_turns(segments: list[dict]) -> list[Turn]:
    """Merge consecutive same-speaker segments into readable turns.

    Whisper segments break on prosody and punctuation, not on speaker changes,
    so an unmerged transcript reads as a stutter of one-line fragments. Merging
    by speaker is what makes it look like a conversation.
    """
    turns: list[Turn] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)

        words = [
            Word(
                word=(w.get("word") or "").strip(),
                start=float(w.get("start") or 0.0),
                end=float(w.get("end") or 0.0),
                speaker=w.get("speaker"),
                probability=w.get("probability"),
            )
            for w in (seg.get("words") or [])
        ]

        # Zero-length segments carrying no words are decoder artefacts.
        if not text and not words:
            continue

        speaker = _segment_speaker(seg)

        if turns and turns[-1].speaker == speaker:
            prev = turns[-1]
            prev.end = max(prev.end, end)
            prev.text = f"{prev.text} {text}".strip()
            prev.words.extend(words)
        else:
            turns.append(Turn(speaker=speaker, start=start, end=end, text=text, words=words))

    return turns


def parse(raw_path: str | Path) -> Transcript:
    raw_path = Path(raw_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    segments = extract_segments(raw)
    turns = build_turns(segments)

    all_words = [w for t in turns for w in t.words]
    confidences = [w.probability for w in all_words if w.probability is not None]

    return Transcript(
        source=raw_path.name,
        turns=turns,
        speakers=sorted({t.speaker for t in turns}),
        duration_sec=round(max((t.end for t in turns), default=0.0), 2),
        word_count=len(all_words),
        mean_word_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        low_confidence_words=sum(1 for c in confidences if c < 0.5),
        unattributed_words=sum(1 for w in all_words if not w.speaker),
    )


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _srt_clock(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_text(t: Transcript, dest: str | Path, label_map: dict[str, str] | None = None) -> None:
    label_map = label_map or {}
    lines = [
        f"# {t.source}",
        f"# {t.duration_sec:.0f}s · {len(t.turns)} turns · {t.word_count} words · "
        f"{len(t.speakers)} speakers",
        "",
    ]
    for turn in t.turns:
        name = label_map.get(turn.speaker, turn.speaker)
        lines.append(f"[{_clock(turn.start)}] {name}:")
        lines.append(turn.text)
        lines.append("")
    Path(dest).write_text("\n".join(lines), encoding="utf-8")


def write_srt(t: Transcript, dest: str | Path, label_map: dict[str, str] | None = None) -> None:
    label_map = label_map or {}
    blocks = []
    for i, turn in enumerate(t.turns, 1):
        name = label_map.get(turn.speaker, turn.speaker)
        blocks.append(
            f"{i}\n{_srt_clock(turn.start)} --> {_srt_clock(turn.end)}\n{name}: {turn.text}\n"
        )
    Path(dest).write_text("\n".join(blocks), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Convert a RunPod response into a readable transcript.")
    p.add_argument("raw", nargs="+", help="Raw .runpod.json file(s)")
    p.add_argument("--out", default="out")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in args.raw:
        t = parse(item)
        stem = Path(item).name.replace(".runpod.json", "")

        write_text(t, out_dir / f"{stem}.txt")
        write_srt(t, out_dir / f"{stem}.srt")
        (out_dir / f"{stem}.parsed.json").write_text(
            json.dumps(t.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        share = {}
        for turn in t.turns:
            share[turn.speaker] = share.get(turn.speaker, 0.0) + turn.duration

        print(f"\n=== {stem} ===")
        print(f"  {t.duration_sec:.0f}s · {len(t.turns)} turns · {t.word_count} words")
        print(f"  speakers: {', '.join(t.speakers)}")
        for spk, secs in sorted(share.items(), key=lambda kv: -kv[1]):
            pct = 100 * secs / sum(share.values()) if share else 0
            print(f"    {spk:12} {secs:6.1f}s  ({pct:4.1f}% of speech)")
        print(f"  mean word confidence: {t.mean_word_confidence:.3f}")
        print(f"  words below 0.5 confidence: {t.low_confidence_words} "
              f"({100 * t.low_confidence_words / max(t.word_count, 1):.1f}%)")
        if t.unattributed_words:
            print(f"  [!] {t.unattributed_words} words carry no speaker label")
        print(f"  -> {out_dir / (stem + '.txt')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
