"""Find the VAD threshold that keeps the most speech without letting silence back in.

Trimming silence was the single largest quality win in this pipeline, but it
cuts both ways: too aggressive and real speech disappears from the transcript
with no error and no trace. The only honest way to choose the threshold is to
vary it and watch what happens to the transcript.

Read the output as a trade-off, not a leaderboard:

  speech_ratio  how much audio survives. Higher keeps more, including silence.
  words         recall. If a lower threshold adds words, it was cutting speech.
  conf          precision. If added words arrive with falling confidence, the
                threshold is now admitting noise the model then invents over.
  words/speech  the tell. Real speech runs about 2-3 words per second. A
                configuration well under that is paying GPU time for silence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline import parse as parse_mod
from pipeline import runpod_client as rp
from pipeline import vad as vad_mod

THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def sweep(audio: str | Path, out_dir: str | Path = "out/sweep") -> list[dict]:
    audio = Path(audio)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    creds = rp.load_credentials()

    rows: list[dict] = []
    print(f"\n{audio.name}")

    for th in THRESHOLDS:
        stage = out_dir / f"th{th}"
        stage.mkdir(parents=True, exist_ok=True)

        trimmed = vad_mod.trim(audio, stage, threshold=th)
        timeline = vad_mod.TimelineMap.from_dict(
            json.loads(Path(trimmed.map_path).read_text(encoding="utf-8"))
        )

        try:
            state = rp.transcribe(trimmed.output, creds, diarize=True, verbose=False)
        except rp.RunPodError as exc:
            print(f"  threshold {th}: FAILED - {exc}")
            continue

        raw = stage / f"{audio.stem}.json"
        raw.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        t = parse_mod.parse(raw)
        data = vad_mod.remap_transcript(t.to_dict(), timeline)

        words = data["word_count"]
        speech = trimmed.speech_duration
        confs = [
            w["probability"]
            for turn in data["turns"] for w in turn["words"]
            if w.get("probability") is not None
        ]
        row = {
            "threshold": th,
            "speech_sec": round(speech, 1),
            "speech_ratio": round(trimmed.speech_ratio, 3),
            "spans": trimmed.span_count,
            "words": words,
            "speakers": len(data["speakers"]),
            "confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "low_conf": sum(1 for c in confs if c < 0.5),
            "words_per_speech_sec": round(words / speech, 2) if speech else 0.0,
            "gpu_sec": round((state.get("executionTime") or 0) / 1000, 1),
        }
        rows.append(row)
        print(f"  threshold {th}: speech {speech:5.0f}s ({trimmed.speech_ratio:.0%}), "
              f"{words:4d} words, conf {row['confidence']:.3f}, "
              f"{row['words_per_speech_sec']:.2f} w/s, {row['gpu_sec']:.1f}s GPU")

    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Sweep the VAD threshold.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out/sweep")
    args = p.parse_args(argv)

    all_rows = {}
    for item in args.audio:
        all_rows[Path(item).name] = sweep(item, args.out)

    print(f"\n{'=' * 76}\nVAD THRESHOLD SWEEP\n{'=' * 76}")
    for name, rows in all_rows.items():
        print(f"\n{name}")
        print(f"  {'thr':>5} {'speech':>8} {'ratio':>7} {'words':>6} {'conf':>7} "
              f"{'w/sec':>7} {'lowconf':>8}")
        print("  " + "-" * 62)
        for r in rows:
            print(f"  {r['threshold']:5.1f} {r['speech_sec']:7.0f}s {r['speech_ratio']:7.1%} "
                  f"{r['words']:6d} {r['confidence']:7.3f} "
                  f"{r['words_per_speech_sec']:7.2f} {r['low_conf']:8d}")

        best = max(rows, key=lambda r: r["words"]) if rows else None
        if best:
            print(f"  most words at threshold {best['threshold']} "
                  f"({best['words']} words, {best['confidence']:.3f} confidence)")

    Path(args.out, "sweep.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
