"""Run the same audio through several diarization configurations and compare.

There is no published Hebrew DER for any diarizer, and the benchmarks that do
exist are English telephone speech scored under protocols that do not match
each other. Picking a configuration from those numbers would be guessing. This
runs the candidates over the same trimmed audio and reports what each produced,
so the choice is made on our own recordings.

What to look for, in order of importance:

  speakers      Two-party calls must yield exactly two. More means the diarizer
                is splitting one person; fewer means it merged them.
  turns         Too few suggests missed speaker changes - the failure that put
                one party's question inside the other party's sentence.
  share         A 50/50 split is not automatically right, but a 95/5 split on a
                conversation usually means one side was swallowed.
  switch_rate   Turns per minute of speech. Real conversation sits in a band;
                far below it means missed boundaries, far above means jitter.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from pipeline import parse as parse_mod
from pipeline import runpod_client as rp
from pipeline import vad as vad_mod


# The worker's default engine is "ivrit", which accepts only min/max_speakers.
# Its "pyannote" engine accepts num_speakers, a hard constraint rather than a
# bound. Both are worth measuring: the constraint helps only if the diarizer's
# error is speaker counting rather than boundary placement.
CONFIGS: dict[str, dict | None] = {
    "default": None,
    "ivrit-2spk": {"engine": "ivrit", "min_speakers": 2, "max_speakers": 2},
    "pyannote-2spk": {"engine": "pyannote", "num_speakers": 2},
    "pyannote-free": {"engine": "pyannote"},
}


def _metrics(t: parse_mod.Transcript) -> dict:
    share: dict[str, float] = {}
    for turn in t.turns:
        share[turn.speaker] = share.get(turn.speaker, 0.0) + turn.duration
    total = sum(share.values()) or 1.0
    speech_min = total / 60.0

    return {
        "speakers": len(t.speakers),
        "turns": len(t.turns),
        "words": t.word_count,
        "confidence": t.mean_word_confidence,
        "low_conf_pct": round(100 * t.low_confidence_words / max(t.word_count, 1), 1),
        "share": {k: round(100 * v / total, 1) for k, v in sorted(share.items())},
        "switch_rate": round(len(t.turns) / speech_min, 1) if speech_min else 0.0,
    }


def run(audio: str | Path, out_dir: str | Path = "out/compare") -> dict:
    audio = Path(audio)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    creds = rp.load_credentials()

    # Trim once and reuse, so every configuration sees identical input and any
    # difference is attributable to the diarizer rather than to the audio.
    trimmed = vad_mod.trim(audio, out_dir)
    timeline = vad_mod.TimelineMap.from_dict(
        json.loads(Path(trimmed.map_path).read_text(encoding="utf-8"))
    )
    print(f"\n{audio.name}")
    print(f"  speech {trimmed.speech_duration:.0f}s of {trimmed.original_duration:.0f}s "
          f"({trimmed.speech_ratio:.0%}), {trimmed.span_count} spans\n")

    results: dict[str, dict] = {}

    for name, args in CONFIGS.items():
        print(f"  [{name}] ", end="", flush=True)
        started = time.time()
        try:
            state = rp.transcribe(
                trimmed.output, creds, diarize=True, verbose=False, diarization_args=args
            )
        except rp.RunPodError as exc:
            print(f"FAILED - {exc}")
            results[name] = {"error": str(exc)}
            continue

        raw = out_dir / f"{audio.stem}.{name}.json"
        raw.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        t = parse_mod.parse(raw)
        data = vad_mod.remap_transcript(t.to_dict(), timeline)
        turns = [
            parse_mod.Turn(
                speaker=x["speaker"], start=x["start"], end=x["end"], text=x["text"],
                words=[parse_mod.Word(**w) for w in x["words"]],
            )
            for x in data["turns"]
        ]
        final = parse_mod.Transcript(
            source=f"{audio.name} [{name}]", turns=turns, speakers=data["speakers"],
            duration_sec=trimmed.original_duration, word_count=data["word_count"],
            mean_word_confidence=data["mean_word_confidence"],
            low_confidence_words=data["low_confidence_words"],
            unattributed_words=data["unattributed_words"],
        )
        parse_mod.write_text(final, out_dir / f"{audio.stem}.{name}.txt")

        m = _metrics(final)
        m["gpu_sec"] = round((state.get("executionTime") or 0) / 1000, 1)
        m["wall_sec"] = round(time.time() - started, 1)
        results[name] = m
        print(f"{m['speakers']} speakers, {m['turns']} turns, "
              f"share {list(m['share'].values())}, {m['gpu_sec']}s GPU")

    return results


def report(all_results: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
    for audio, results in all_results.items():
        print(f"\n{audio}")
        print(f"  {'config':16} {'spk':>4} {'turns':>6} {'words':>6} {'conf':>6} "
              f"{'sw/min':>7}  share")
        print("  " + "-" * 72)
        for name, m in results.items():
            if "error" in m:
                print(f"  {name:16} FAILED: {m['error'][:44]}")
                continue
            share = " / ".join(f"{v:.0f}%" for v in m["share"].values())
            print(f"  {name:16} {m['speakers']:4d} {m['turns']:6d} {m['words']:6d} "
                  f"{m['confidence']:6.3f} {m['switch_rate']:7.1f}  {share}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Compare diarization configurations.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out/compare")
    args = p.parse_args(argv)

    all_results = {}
    for item in args.audio:
        all_results[Path(item).name] = run(item, args.out)

    report(all_results)
    Path(args.out, "summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
