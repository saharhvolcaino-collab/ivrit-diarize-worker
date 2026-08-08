"""Standalone probe of OpenAI's 4o speech models - NOT wired into the pipeline.

    python openai_asr.py <recording...>                 transcribe + diarize
    python openai_asr.py <recording...> --mode diarize  diarization only

Two models, because OpenAI splits the job in two:

  gpt-4o-transcribe          text only, no timestamps, no speakers. Reads
                             token logprobs, so we still get a confidence
                             number to compare against Whisper's.
  gpt-4o-transcribe-diarize  same family, returns speaker-labelled segments
                             (A/B/C...) with start/end times - transcription
                             and diarization in one call, which is exactly
                             the combination this repo builds from parts.

Deliberately external: it reuses preprocess (same 16 kHz mono, same gain) and
parse (same .txt/.srt/.html renderers) so the outputs land in the same shape
as gpu.py's and can be read side by side, but nothing in the worker or the
consensus machinery knows this script exists. If the numbers justify it, the
integration is a separate decision.

Talks to the REST API with `requests` directly - the local openai package is
a broken mixed-version install, and a multipart POST needs no SDK.

Needs OPENAI_API_KEY in .env or the environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import parse as parse_mod
from pipeline import preprocess as prep_mod

API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL_TEXT = "gpt-4o-transcribe"
MODEL_DIAR = "gpt-4o-transcribe-diarize"

# The 4o transcribe models cap input length around 25 minutes. Chunk under
# that with margin; each chunk's timestamps are shifted back by its offset.
# Speaker letters are NOT stable across chunks - a multi-chunk run prints a
# warning, because A in chunk 2 need not be A from chunk 1.
CHUNK_SEC = 1200.0
USD_PER_MIN = 0.006  # both 4o transcribe models, audio input


def _api_key() -> str:
    import os
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    raise SystemExit("No OPENAI_API_KEY in environment or .env")


def _ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:400]}")


def _post(key: str, model: str, audio: Path, language: str,
          diarized: bool, prompt: str | None = None) -> dict:
    """One multipart call to /v1/audio/transcriptions. Raises with the API's
    own error text - OpenAI's messages name the exact rejected parameter,
    which beats anything we could guess here."""
    data: list[tuple[str, str]] = [
        ("model", model),
        ("language", language),
        ("temperature", "0"),
    ]
    if diarized:
        data += [("response_format", "diarized_json"),
                 ("chunking_strategy", "auto")]
    else:
        # logprobs ride along free and give a Whisper-comparable confidence.
        data += [("response_format", "json"), ("include[]", "logprobs")]
        if prompt:
            data.append(("prompt", prompt))

    with audio.open("rb") as fh:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}"},
            data=data,
            files={"file": (audio.name, fh, "audio/mpeg")},
            timeout=600,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"{model} HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _chunks(wav_16k: Path, out_dir: Path, stem: str, duration: float,
            tempo: float) -> list[tuple[Path, float]]:
    """MP3 chunks ready for upload, each with its offset on the ORIGINAL clock.

    MP3 64 kbps mono because the endpoint accepts it, it is transparent for
    narrowband speech, and it keeps an hour-long call far below the 25 MB cap.
    The .opus transport copy the pipeline already makes is not on OpenAI's
    accepted-format list, so it cannot be reused here.
    """
    filters = []
    if tempo != 1.0:
        filters += ["-af", f"atempo={tempo}"]
    stretched = duration / tempo if tempo else duration

    pieces: list[tuple[Path, float]] = []
    if stretched <= CHUNK_SEC:
        mp3 = out_dir / f"{stem}.4o.mp3"
        _ffmpeg(["-i", str(wav_16k), *filters,
                 "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", str(mp3)])
        return [(mp3, 0.0)]

    n = int(stretched // CHUNK_SEC) + 1
    for i in range(n):
        mp3 = out_dir / f"{stem}.4o.{i:02d}.mp3"
        _ffmpeg(["-ss", f"{i * CHUNK_SEC:.3f}", "-t", f"{CHUNK_SEC:.3f}",
                 "-i", str(wav_16k), *filters,
                 "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", str(mp3)])
        pieces.append((mp3, i * CHUNK_SEC * tempo))
    return pieces


def _mean_token_prob(raw: dict) -> float | None:
    import math
    lps = [t.get("logprob") for t in raw.get("logprobs") or []
           if t.get("logprob") is not None]
    return round(sum(math.exp(lp) for lp in lps) / len(lps), 4) if lps else None


def process(source: Path, key: str, out_dir: Path, mode: str,
            language: str, tempo: float, prompt: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    started = time.time()
    print(f"\n=== {source.name} ===")

    # Same 16 kHz mono, same gain decision as every other run in this repo -
    # otherwise a difference in results could be the preprocessing, not the model.
    prep = prep_mod.preprocess(source, out_dir, make_opus=False)
    pieces = _chunks(Path(prep.output), out_dir, stem, prep.duration_sec, tempo)
    if len(pieces) > 1:
        print(f"[!] {len(pieces)} chunks - speaker letters are per-chunk and may "
              f"not agree across the boundary")
    est = prep.duration_sec / 60 * USD_PER_MIN * (2 if mode == "both" else 1)
    print(f"[send]   {prep.duration_sec / 60:.1f} min · tempo {tempo} · "
          f"~${est:.3f} per model pass")

    # --- plain transcription -------------------------------------------------
    if mode in ("both", "transcribe"):
        t0 = time.time()
        texts, probs = [], []
        raws = []
        for mp3, _off in pieces:
            raw = _post(key, MODEL_TEXT, mp3, language, diarized=False, prompt=prompt)
            raws.append(raw)
            texts.append((raw.get("text") or "").strip())
            p = _mean_token_prob(raw)
            if p is not None:
                probs.append(p)
        text = "\n".join(t for t in texts if t)
        (out_dir / f"{stem}.transcribe.json").write_text(
            json.dumps(raws if len(raws) > 1 else raws[0],
                       ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"{stem}.transcribe.txt").write_text(
            parse_mod.rtl(text) + "\n", encoding="utf-8")
        conf = f" · token confidence {sum(probs) / len(probs):.3f}" if probs else ""
        print(f"[text]   {len(text.split())} words in {time.time() - t0:.1f}s{conf}"
              f"  -> {stem}.transcribe.txt")

    # --- transcription + diarization ------------------------------------------
    if mode in ("both", "diarize"):
        t0 = time.time()
        segments: list[dict] = []
        raws = []
        for mp3, off in pieces:
            raw = _post(key, MODEL_DIAR, mp3, language, diarized=True)
            raws.append(raw)
            for seg in raw.get("segments") or []:
                segments.append({
                    "speaker": f"SPEAKER_{seg.get('speaker', '?')}",
                    # tempo stretches the clock the model sees; multiply back.
                    "start": (float(seg.get("start") or 0.0)) * tempo + off,
                    "end": (float(seg.get("end") or 0.0)) * tempo + off,
                    "text": (seg.get("text") or "").strip(),
                })
        (out_dir / f"{stem}.diarize.json").write_text(
            json.dumps(raws if len(raws) > 1 else raws[0],
                       ensure_ascii=False, indent=2), encoding="utf-8")

        # Merge consecutive same-speaker segments into turns, exactly as
        # parse.build_turns does for Whisper segments.
        turns: list[parse_mod.Turn] = []
        for seg in segments:
            if not seg["text"]:
                continue
            if turns and turns[-1].speaker == seg["speaker"]:
                turns[-1].end = max(turns[-1].end, seg["end"])
                turns[-1].text = f"{turns[-1].text} {seg['text']}".strip()
            else:
                turns.append(parse_mod.Turn(
                    speaker=seg["speaker"], start=seg["start"],
                    end=seg["end"], text=seg["text"], words=[]))

        word_count = sum(len(t.text.split()) for t in turns)
        transcript = parse_mod.Transcript(
            source=f"{source.name} [{MODEL_DIAR}]",
            turns=turns,
            speakers=sorted({t.speaker for t in turns}),
            duration_sec=prep.duration_sec,
            word_count=word_count,
            mean_word_confidence=0.0,   # the diarize endpoint returns no logprobs
            low_confidence_words=0,
            unattributed_words=0,
        )
        parse_mod.write_text(transcript, out_dir / f"{stem}.txt")
        parse_mod.write_srt(transcript, out_dir / f"{stem}.srt")
        parse_mod.write_html(transcript, out_dir / f"{stem}.html")

        share: dict[str, float] = {}
        for t in turns:
            share[t.speaker] = share.get(t.speaker, 0.0) + (t.end - t.start)
        total = sum(share.values()) or 1.0
        print(f"[diar]   {len(turns)} turns · {word_count} words · "
              f"{len(transcript.speakers)} speakers · "
              + " / ".join(f"{100 * v / total:.0f}%"
                           for v in sorted(share.values(), reverse=True))
              + f" · {time.time() - t0:.1f}s  -> {stem}.txt / .html")

    print(f"[done]   {time.time() - started:.1f}s wall")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Probe OpenAI gpt-4o-transcribe(-diarize) on recordings - "
                    "standalone, no pipeline changes.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out/openai")
    p.add_argument("--mode", default="both",
                   choices=["both", "transcribe", "diarize"])
    p.add_argument("--language", default="he")
    p.add_argument("--tempo", type=float, default=1.0,
                   help="slow the audio first, as the worker does at 0.85 - "
                        "that win was measured on Whisper and may not transfer, "
                        "so the default here is untouched audio")
    p.add_argument("--prompt", default=None,
                   help="hotwords/context for the transcribe model "
                        "(the diarize model does not take a prompt)")
    args = p.parse_args(argv)

    key = _api_key()
    out_dir = Path(args.out)

    failures = 0
    for item in args.audio:
        path = Path(item)
        files = ([f for f in sorted(path.iterdir())
                  if f.suffix.lower() in {".wav", ".wmv", ".mp3", ".opus", ".ogg",
                                          ".m4a", ".flac", ".aac", ".wma"}
                  and not f.name.endswith((".16k.wav", ".speech.wav"))]
                 if path.is_dir() else [path])
        for f in files:
            try:
                process(f, key, out_dir, args.mode, args.language,
                        args.tempo, args.prompt)
            except Exception as exc:
                print(f"  ERROR {f.name}: {exc}")
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
