"""End-to-end: a recording in, a speaker-attributed Hebrew transcript out.

    python run.py path/to/call.wav

The stages, and why each one earns its place:

  probe       Formats vary across the recording estate - GSM at 8 kHz in one
              file, 8-bit PCM at 11 kHz in another - so decisions are made per
              file rather than assumed.
  vad         Call audio is mostly not speech. One of our samples is 14%
              speech; the silence made Whisper invent five minutes of Knesset
              boilerplate. Cutting it first removes the hallucination trigger
              and cuts GPU cost by the same proportion.
  transcribe  ivrit.ai's Hebrew-tuned Whisper on RunPod, with diarization.
  remap       Timestamps come back on the trimmed clock; they get put back on
              the original recording's clock so they line up with the audio.

Run with --no-vad to reproduce the untrimmed behaviour for comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Hebrew in a Windows console defaults to a codepage that cannot represent it,
# which turns a progress line into a crash. Force UTF-8 on the streams we own.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pipeline import probe as probe_mod
from pipeline import vad as vad_mod
from pipeline import preprocess as prep_mod
from pipeline import quality as quality_mod
from pipeline import runpod_client as rp
from pipeline import parse as parse_mod
from pipeline import speakers as speakers_mod
from pipeline import diarize as diar_mod


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def process(
    source: str | Path,
    out_dir: str | Path = "out",
    *,
    use_vad: bool = True,
    diarize: bool = True,
    vad_threshold: float | None = None,
    expected_speakers: int | None = 2,
    creds: rp.Credentials | None = None,
) -> dict:
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    started = time.time()
    if vad_threshold is None:
        vad_threshold = vad_mod.DEFAULTS["threshold"]

    print(f"\n{'=' * 60}\n{source.name}\n{'=' * 60}")

    # --- inspect -----------------------------------------------------------
    prof = probe_mod.probe(source)
    print(f"[probe]  {prof.codec} · {prof.declared_sample_rate} Hz · "
          f"{prof.channels}ch · {_fmt(prof.duration_sec)}")
    if prof.is_narrowband:
        print(f"         narrowband - real content stops at "
              f"{prof.effective_bandwidth_hz:.0f} Hz")
    if prof.channel_relation and prof.channel_relation.verdict == "true-stereo":
        print("         [!] channels differ - one speaker per channel would give "
              "exact attribution. Worth splitting instead of diarizing.")

    # --- normalise ---------------------------------------------------------
    prep = prep_mod.preprocess(source, out_dir, make_opus=False)
    print(f"[prep]   -> 16 kHz mono"
          + (f", +{prep.gain_applied_db} dB" if prep.gain_applied_db else ""))

    # --- trim --------------------------------------------------------------
    timeline = None
    if use_vad:
        v = vad_mod.trim(prep.output, out_dir, threshold=vad_threshold)
        timeline = vad_mod.TimelineMap.from_dict(
            json.loads(Path(v.map_path).read_text(encoding="utf-8"))
        )
        upload = v.output
        print(f"[vad]    speech {v.speech_duration:.0f}s of {v.original_duration:.0f}s "
              f"({v.speech_ratio:.0%}) in {v.span_count} spans")
        if v.speech_ratio < 0.5:
            print(f"         removed {v.removed_duration:.0f}s of silence - "
                  f"this is what stops the model inventing text")
    else:
        p2 = prep_mod.preprocess(source, out_dir, make_opus=True)
        upload = p2.opus_path
        print("[vad]    skipped")

    # --- transcribe --------------------------------------------------------
    creds = creds or rp.load_credentials()
    state = rp.transcribe(upload, creds, diarize=diarize, verbose=False)
    exec_s = (state.get("executionTime") or 0) / 1000
    delay_s = (state.get("delayTime") or 0) / 1000
    print(f"[gpu]    {exec_s:.1f}s execution, {delay_s:.1f}s queue+coldstart")

    raw_path = out_dir / f"{stem}.raw.json"
    raw_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- assemble ----------------------------------------------------------
    # Two candidate attributions are built from the same words and the quality
    # checks arbitrate. This exists because neither source wins on every file:
    # the worker's segment-level labels merged both parties on one call
    # (96%/4%, answers buried inside the asker's turns), while local
    # window-based diarization fixed that call but drifted on another. The
    # failure modes are opposite, and they are distinguishable automatically.
    transcript = parse_mod.parse(raw_path)
    base = transcript.to_dict()

    def _package(turns_list: list[dict]) -> tuple[dict, quality_mod.QualityReport]:
        """Remap to the original clock, fold artefact speakers, and score."""
        data = {"turns": [dict(t) for t in turns_list]}
        if timeline is not None:
            data = vad_mod.remap_transcript(data, timeline)
        if diarize and expected_speakers:
            data["turns"], _col = speakers_mod.collapse_to_expected(
                data["turns"], expected=expected_speakers
            )
        data["speakers"] = sorted({t["speaker"] for t in data["turns"]})
        report = quality_mod.assess(
            data, expected_speakers=expected_speakers if diarize else None
        )
        return data, report

    flat = [
        {"word": w["word"], "start": w["start"], "end": w["end"],
         "speaker": w.get("speaker"), "probability": w.get("probability")}
        for t in base["turns"] for w in t["words"]
    ]

    candidates: dict[str, tuple[dict, quality_mod.QualityReport]] = {}

    if diarize and flat and all(w["speaker"] for w in flat):
        # Candidate 1 - the worker's labels, with mid-sentence speaker flips
        # voted away sentence by sentence.
        worker_words = diar_mod.realign_by_punctuation(flat)
        candidates["worker"] = _package(diar_mod.turns_from_words(worker_words))

        # Candidate 2 - ignore the worker's labels and re-diarize by voice in
        # fixed sliding windows, at a resolution Whisper's segmentation cannot
        # blur. CPU-only. Skipped when the two voices are so similar that the
        # clustering would be arbitrary anyway.
        try:
            local = diar_mod.diarize(upload, num_speakers=expected_speakers or 2)
            if local.separation >= 0.10:
                local_words = diar_mod.assign_words(flat, local.turns)
                # Short answers lose the window vote to their neighbours, so
                # each word is re-judged against the two voice signatures and
                # flipped only on decisive evidence.
                flips = 0
                if local.centroids is not None:
                    local_words, flips = diar_mod.refine_word_speakers(
                        upload, local_words, local.centroids
                    )
                local_words = diar_mod.realign_by_punctuation(local_words)
                # The conversational prior runs last: text after a mid-turn
                # question goes to the other party unless the voice clearly
                # vetoes. This is the fix for the question-and-answer pairs
                # that survive everything above.
                handed = 0
                if local.centroids is not None:
                    local_words, handed = diar_mod.split_buried_answers(
                        upload, local_words, local.centroids
                    )
                local_words = [
                    {k: w.get(k)
                     for k in ("word", "start", "end", "speaker", "probability")}
                    for w in local_words
                ]
                candidates["local"] = _package(diar_mod.turns_from_words(local_words))
                print(f"[diar]   local voice pass: {local.window_count} windows, "
                      f"separation {local.separation}, {flips} words re-judged, "
                      f"{handed} answers handed to the other party")
            else:
                print(f"[diar]   local pass skipped - voices inseparable "
                      f"(silhouette {local.separation})")
        except Exception as exc:
            print(f"[diar]   local pass failed ({type(exc).__name__}: {exc}) - "
                  "keeping worker labels")
    else:
        candidates["worker"] = _package(base["turns"])

    # Buried answers decide first: they are the direct evidence of words on the
    # wrong speaker, which is the failure being arbitrated. Problem count only
    # breaks ties - it also counts artefacts like repeated short lines, which
    # finer turn-splitting produces innocently, and letting it lead once made
    # the arbiter prefer four buried exchanges over one. A final tie keeps the
    # worker, whose boundaries align with the ASR segmentation.
    def _rank(item):
        name, (_, rep) = item
        return (rep.buried_answers, len(rep.problems), 0 if name == "worker" else 1)

    chosen_name, (data, chosen_report) = min(candidates.items(), key=_rank)
    if len(candidates) > 1:
        other = next(k for k in candidates if k != chosen_name)
        orep = candidates[other][1]
        print(f"[spk]    chose {chosen_name} labels "
              f"({len(chosen_report.problems)} issue(s), "
              f"{chosen_report.buried_answers} buried) over {other} "
              f"({len(orep.problems)} issue(s), {orep.buried_answers} buried)")

    # Whatever labels survived arbitration and collapse, present them in order
    # of first appearance - SPEAKER_10 leaking out of an eleven-way clustering
    # accident is noise the reader should never see.
    rename: dict[str, str] = {}
    for t in data["turns"]:
        if t["speaker"] not in rename:
            rename[t["speaker"]] = f"SPEAKER_{len(rename):02d}"
        t["speaker"] = rename[t["speaker"]]
        for w in t.get("words", []):
            if w.get("speaker"):
                w["speaker"] = rename.get(w["speaker"], t["speaker"])
    data["speakers"] = sorted(set(rename.values()))

    all_words = [w for t in data["turns"] for w in t.get("words", [])]
    confs = [w["probability"] for w in all_words if w.get("probability") is not None]
    word_fields = ("word", "start", "end", "speaker", "probability")
    turns = [
        parse_mod.Turn(
            speaker=t["speaker"], start=t["start"], end=t["end"], text=t["text"],
            words=[parse_mod.Word(**{k: w.get(k) for k in word_fields})
                   for w in t["words"]],
        )
        for t in data["turns"]
    ]
    final = parse_mod.Transcript(
        source=source.name,
        turns=turns,
        speakers=data["speakers"],
        duration_sec=prof.duration_sec,
        word_count=len(all_words),
        mean_word_confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
        low_confidence_words=sum(1 for c in confs if c < 0.5),
        unattributed_words=sum(1 for w in all_words if not w.get("speaker")),
    )

    parse_mod.write_text(final, out_dir / f"{stem}.txt")
    parse_mod.write_srt(final, out_dir / f"{stem}.srt")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(final.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- report ------------------------------------------------------------
    share: dict[str, float] = {}
    for t in final.turns:
        share[t.speaker] = share.get(t.speaker, 0.0) + t.duration
    total_speech = sum(share.values()) or 1.0

    print(f"[out]    {len(final.turns)} turns · {final.word_count} words · "
          f"{len(final.speakers)} speakers")
    for spk, secs in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"           {spk:12} {secs:6.1f}s ({100 * secs / total_speech:4.1f}%)")
    result = final.to_dict()

    # Judge the transcript on its shape, not on the model's own confidence -
    # the decoder reported 0.000 no-speech probability on every hallucinated
    # segment it produced, so its self-assessment is not usable here.
    report = quality_mod.assess(
        result, expected_speakers=expected_speakers if diarize else None
    )
    print(quality_mod.format_report(report))
    result["quality"] = report.to_dict()
    (out_dir / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"         {time.time() - started:.1f}s total -> {out_dir / (stem + '.txt')}")

    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Transcribe a Hebrew call with speaker separation."
    )
    p.add_argument("audio", nargs="+", help="Recording(s) to process")
    p.add_argument("--out", default="out")
    p.add_argument("--no-vad", action="store_true",
                   help="Send the full recording, silence included (for comparison)")
    p.add_argument("--no-diarize", action="store_true", help="Transcript only")
    # Read the default from one place. Hard-coding it here silently overrode the
    # tuned value and made two different settings look like run-to-run noise.
    p.add_argument("--vad-threshold", type=float,
                   default=vad_mod.DEFAULTS["threshold"],
                   help="Raise toward 0.6 on noisy lines, lower if speech is dropped")
    args = p.parse_args(argv)

    try:
        creds = rp.load_credentials()
    except rp.RunPodError as exc:
        print(f"ERROR: {exc}")
        return 1

    failures = 0
    for item in args.audio:
        try:
            process(
                item, args.out,
                use_vad=not args.no_vad,
                diarize=not args.no_diarize,
                vad_threshold=args.vad_threshold,
                creds=creds,
            )
        except Exception as exc:
            print(f"\nERROR on {item}: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
