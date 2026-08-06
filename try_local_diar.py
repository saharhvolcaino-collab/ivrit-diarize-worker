"""A/B the worker's diarization against local window-based diarization.

Same words, same audio, same VAD trim. The only thing that changes is where
the speaker labels come from, so any difference is attributable to the
diarizer rather than to transcription.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import diarize as diar_mod
from pipeline import parse as parse_mod
from pipeline import quality as quality_mod
from pipeline import runpod_client as rp
from pipeline import vad as vad_mod


def _turns_from_words(words: list[dict], split_gap: float = 1.0) -> list[dict]:
    """Group words into turns, cutting wherever the speaker changes."""
    if not words:
        return []
    turns = []
    cur = {"speaker": words[0]["speaker"], "start": words[0]["start"],
           "end": words[0]["end"], "words": [words[0]]}
    for w in words[1:]:
        if w["speaker"] == cur["speaker"] and (w["start"] - cur["end"]) <= split_gap:
            cur["end"] = w["end"]
            cur["words"].append(w)
        else:
            turns.append(cur)
            cur = {"speaker": w["speaker"], "start": w["start"],
                   "end": w["end"], "words": [w]}
    turns.append(cur)
    for t in turns:
        t["text"] = " ".join(x["word"] for x in t["words"]).strip()
    return turns


def compare(audio: str | Path, out_dir: Path) -> dict:
    audio = Path(audio)
    out_dir.mkdir(parents=True, exist_ok=True)
    creds = rp.load_credentials()

    trimmed = vad_mod.trim(audio, out_dir)
    timeline = vad_mod.TimelineMap.from_dict(
        json.loads(Path(trimmed.map_path).read_text(encoding="utf-8"))
    )
    print(f"\n{audio.name}  ({trimmed.speech_ratio:.0%} speech)")

    state = rp.transcribe(trimmed.output, creds, diarize=True, verbose=False)
    raw = out_dir / f"{audio.stem}.raw.json"
    raw.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- what the worker produced -----------------------------------------
    worker = parse_mod.parse(raw)
    worker_data = vad_mod.remap_transcript(worker.to_dict(), timeline)
    q_worker = quality_mod.assess(worker_data, expected_speakers=2)

    # --- same words, local speaker labels ---------------------------------
    # Diarize the trimmed audio so both sides see identical input, then remap.
    local = diar_mod.diarize(trimmed.output, num_speakers=2)
    flat_words = [
        {k: w[k] for k in ("word", "start", "end", "probability")}
        for t in worker.to_dict()["turns"] for w in t["words"]
    ]
    relabelled = diar_mod.assign_words(flat_words, local.turns)
    local_data = {"turns": _turns_from_words(relabelled)}
    local_data = vad_mod.remap_transcript(local_data, timeline)
    local_data["word_count"] = len(relabelled)
    local_data["mean_word_confidence"] = worker_data["mean_word_confidence"]
    local_data["low_confidence_words"] = worker_data["low_confidence_words"]
    local_data["unattributed_words"] = 0
    local_data["speakers"] = sorted({t["speaker"] for t in local_data["turns"]})
    q_local = quality_mod.assess(local_data, expected_speakers=2)

    for label, q, data in (("worker", q_worker, worker_data),
                           ("local ", q_local, local_data)):
        share = {}
        for t in data["turns"]:
            share[t["speaker"]] = share.get(t["speaker"], 0.0) + (t["end"] - t["start"])
        total = sum(share.values()) or 1.0
        pct = " / ".join(f"{100 * v / total:.0f}%" for v in sorted(share.values(), reverse=True))
        print(f"  {label}: {q.speakers} spk, {q.turns} turns, "
              f"buried={q.buried_answers}, share {pct}, {q.verdict}")

    print(f"  voice separation (silhouette): {local.separation}"
          f"   windows: {local.window_count}")
    for n in local.notes:
        print(f"  [!] {n}")

    # Write the local version out so the text can be read side by side.
    word_fields = ("word", "start", "end", "speaker", "probability")
    turns = [
        parse_mod.Turn(
            speaker=t["speaker"], start=t["start"], end=t["end"], text=t["text"],
            words=[parse_mod.Word(**{k: w.get(k) for k in word_fields})
                   for w in t["words"]],
        )
        for t in local_data["turns"]
    ]
    t_obj = parse_mod.Transcript(
        source=f"{audio.name} [local diarization]", turns=turns,
        speakers=local_data["speakers"], duration_sec=trimmed.original_duration,
        word_count=local_data["word_count"],
        mean_word_confidence=local_data["mean_word_confidence"],
        low_confidence_words=local_data["low_confidence_words"], unattributed_words=0,
    )
    parse_mod.write_text(t_obj, out_dir / f"{audio.stem}.local.txt")

    return {
        "worker": {"speakers": q_worker.speakers, "turns": q_worker.turns,
                   "buried": q_worker.buried_answers,
                   "dominant": q_worker.dominant_speaker_pct, "verdict": q_worker.verdict},
        "local": {"speakers": q_local.speakers, "turns": q_local.turns,
                  "buried": q_local.buried_answers,
                  "dominant": q_local.dominant_speaker_pct, "verdict": q_local.verdict,
                  "separation": local.separation},
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Compare worker vs local diarization.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out/localdiar")
    args = p.parse_args(argv)

    results = {}
    for item in args.audio:
        results[Path(item).name] = compare(item, Path(args.out))

    print(f"\n{'=' * 72}\nWORKER vs LOCAL DIARIZATION\n{'=' * 72}")
    print(f"  {'file':22} {'source':8} {'spk':>4} {'turns':>6} {'buried':>7} "
          f"{'dominant':>9}  verdict")
    print("  " + "-" * 68)
    for name, r in results.items():
        for src in ("worker", "local"):
            d = r[src]
            print(f"  {name[:22]:22} {src:8} {d['speakers']:4d} {d['turns']:6d} "
                  f"{d['buried']:7d} {d['dominant']:8.1f}%  {d['verdict']}")

    Path(args.out, "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
