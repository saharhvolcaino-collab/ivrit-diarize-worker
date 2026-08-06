"""Measure how much the pipeline's output varies between identical runs.

This exists because a single run misled me. A VAD threshold sweep picked 0.4 as
the winner on speaker count and confidence; the very next run of that same
configuration returned a different speaker count. Whisper's temperature
fallback resamples when its decoder fails a quality check, so identical input
does not guarantee identical output - and the diarizer then inherits whatever
segmentation the ASR produced.

Any comparison between configurations is meaningless until we know how big that
noise is. If run-to-run spread is larger than the gap between two settings,
the gap is not evidence.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import parse as parse_mod
from pipeline import quality as quality_mod
from pipeline import runpod_client as rp
from pipeline import vad as vad_mod


def repeat(audio: str | Path, n: int, threshold: float, out_dir: Path) -> list[dict]:
    audio = Path(audio)
    out_dir.mkdir(parents=True, exist_ok=True)
    creds = rp.load_credentials()

    # Trim once. Silero is deterministic, so reusing one trimmed file isolates
    # the variance to the parts that are actually stochastic.
    trimmed = vad_mod.trim(audio, out_dir, threshold=threshold)
    timeline = vad_mod.TimelineMap.from_dict(
        json.loads(Path(trimmed.map_path).read_text(encoding="utf-8"))
    )

    rows = []
    for i in range(n):
        try:
            state = rp.transcribe(trimmed.output, creds, diarize=True, verbose=False)
        except rp.RunPodError as exc:
            print(f"    run {i + 1}: FAILED {exc}")
            continue

        raw = out_dir / f"{audio.stem}.run{i}.json"
        raw.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        t = parse_mod.parse(raw)
        data = vad_mod.remap_transcript(t.to_dict(), timeline)
        q = quality_mod.assess(data, expected_speakers=2)

        rows.append({
            "run": i + 1,
            "speakers": q.speakers,
            "turns": q.turns,
            "words": q.words,
            "confidence": q.mean_confidence,
            "repeated": q.repeated_lines,
            "stretched": q.stretched_turns,
            "verdict": q.verdict,
        })
        print(f"    run {i + 1}: {q.speakers} spk, {q.turns} turns, {q.words} words, "
              f"conf {q.mean_confidence:.3f}, {q.verdict}")

    return rows


def spread(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    if isinstance(vals[0], float):
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return f"{statistics.mean(vals):.3f}  [{lo:.3f}-{hi:.3f}]  sd {sd:.3f}"
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.1f}  [{lo}-{hi}]  sd {sd:.1f}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Quantify run-to-run variance.")
    p.add_argument("audio", nargs="+")
    p.add_argument("-n", type=int, default=5, help="Runs per file")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--out", default="out/repeat")
    args = p.parse_args(argv)

    all_rows = {}
    for item in args.audio:
        print(f"\n{Path(item).name}  (threshold {args.threshold}, {args.n} runs)")
        all_rows[Path(item).name] = repeat(
            item, args.n, args.threshold, Path(args.out) / Path(item).stem
        )

    print(f"\n{'=' * 74}\nRUN-TO-RUN VARIANCE\n{'=' * 74}")
    for name, rows in all_rows.items():
        if not rows:
            continue
        print(f"\n{name}")
        for key in ("speakers", "turns", "words", "confidence"):
            print(f"  {key:12} {spread(rows, key)}")
        verdicts = Counter(r["verdict"] for r in rows)
        print(f"  {'verdict':12} {dict(verdicts)}")

        spk = [r["speakers"] for r in rows]
        if len(set(spk)) > 1:
            print(f"  [!] Speaker count is not stable across identical runs "
                  f"({sorted(set(spk))}). Differences smaller than this between "
                  "configurations are noise, not signal.")

    Path(args.out, "variance.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
