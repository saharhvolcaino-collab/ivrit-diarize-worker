"""Does HOW we present audio to the STT engine change what it hears?

    python benchmark_seg.py <recording.wav> --variants A,B,C,D,E,F
    python benchmark_seg.py <recording.wav> --variants G          (needs
        worker >= v0.13, which passes initial_prompt through)

One engine, one recording, one decode configuration. The only thing that
varies is segmentation: where the audio is cut, how much context each window
carries, whether windows overlap, and (G) whether the previous window's tail
is offered as decoding context. Everything the sceptical reading demands is
held fixed - same model, same tempo, same beam, same internal VAD, no
hotwords, no lexicon, no post-editing.

Windows are always cut inside VAD-detected pauses, never inside speech. A
"cut" therefore costs no phonemes; what it costs is *context* - the words on
the far side of the cut that the decoder no longer sees. The variants trade
that context against window size in different ways:

  A  the whole file in one request - what production sends today
  B  pause-packed windows targeting 20-30 s
  C  as B, plus 3 s of overlap into each next window; the overlap zone is
     deduplicated at the midpoint, so every kept word sits >=1.5 s from its
     window's edge and was decoded with context on both sides
  D  pause-packed windows targeting 8-15 s   (short: little context)
  E  pause-packed windows targeting 25-45 s  (long: much context)
  F  adaptive - boundaries at the LONGEST pause in range, on the theory
     that the deepest silence is the safest place to lose context
  G  as F, plus the last ~25 words of the previous window's transcript as
     initial_prompt - context carried across the cut in text form. Only
     words the engine itself produced are passed; nothing external.

Scoring happens outside this script, against the baseline's own weak spots,
because no reference transcript exists. This script's job is to produce
per-variant transcripts that are honestly comparable: same clock, same
format, plus per-variant call counts and wall time.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import runpod_client as rp
from pipeline import vad as vad_mod

API = "https://api.runpod.ai/v2"
SR = 16000
PAD = 0.25          # keep this much silence on each side of a window's speech
OVERLAP = 3.0       # C: how far each window reaches into the previous one


# --------------------------------------------------------------- segmentation

def pack_windows(spans, lo: float, hi: float, adaptive: bool = False):
    """Pack VAD spans into windows of lo..hi seconds, cutting only in gaps.

    Greedy: extend the current window span by span; once its duration passes
    `lo`, the next gap is a legal cut point. Plain packing cuts at the FIRST
    legal gap; adaptive scans every legal gap up to `hi` and cuts at the
    LONGEST one, trading a little window-size regularity for boundaries in
    the deepest silence available.
    """
    wins: list[tuple[float, float]] = []
    i, n = 0, len(spans)
    while i < n:
        start = spans[i].start
        j = i
        best_j, best_gap = None, -1.0
        while j < n and spans[j].end - start <= hi:
            if spans[j].end - start >= lo and j + 1 < n:
                gap = spans[j + 1].start - spans[j].end
                if adaptive:
                    if gap > best_gap:
                        best_gap, best_j = gap, j
                else:
                    best_j = j
                    break
            j += 1
        if best_j is None:
            # Either the tail of the file, or one stretch longer than hi:
            # take everything up to the last span that fits (at least one).
            best_j = i
            while best_j + 1 < n and spans[best_j + 1].end - start <= hi:
                best_j += 1
        wins.append((start, spans[best_j].end))
        i = best_j + 1
    return wins


def build_variant(spans, variant: str):
    if variant == "B":
        return pack_windows(spans, 20, 30), 0.0
    if variant == "C":
        return pack_windows(spans, 20, 30), OVERLAP
    if variant == "D":
        return pack_windows(spans, 8, 15), 0.0
    if variant == "E":
        return pack_windows(spans, 25, 45), 0.0
    if variant in ("F", "G"):
        return pack_windows(spans, 15, 40, adaptive=True), 0.0
    raise ValueError(variant)


# ------------------------------------------------------------------ transport

def _opus_bytes(audio: np.ndarray) -> bytes:
    """Encode a clip exactly the way production transport does.

    Raw 16-bit WAV of the whole-file baseline is ~36 MB base64 against
    RunPod's 10 MiB request cap - the first run died on it. OPUS at 24 kbps
    is what the existing clients ship, fits every variant with room to
    spare, and keeps the transport identical across variants so codec loss
    cannot masquerade as a segmentation effect.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "pipe:1"],
        input=audio.astype(np.float32).tobytes(),
        capture_output=True, check=True)
    return proc.stdout


def _submit(creds, endpoint: str, audio: np.ndarray, prompt: str | None = None):
    payload = {"input": {
        "audio_base64": base64.b64encode(_opus_bytes(audio)).decode("ascii"),
        "audio_format": "opus",
        "language": "he",
        "diarize": False,
    }}
    if prompt:
        payload["input"]["initial_prompt"] = prompt
    job = rp._request(f"{API}/{endpoint}/run", creds.api_key, payload)
    return job["id"]


def _poll(creds, endpoint: str, job_id: str) -> dict:
    state = rp.poll(rp.Credentials(api_key=creds.api_key, endpoint_id=endpoint),
                    job_id, timeout_sec=2400)
    out = state.get("output") or {}
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


# -------------------------------------------------------------------- merging

def merge_windows(results, wins, overlap: float):
    """Stitch per-window words onto the recording's clock.

    With overlap, both windows decoded the shared zone; the cut for keeping
    words is the zone's midpoint, so whichever copy survives was decoded at
    least half the overlap away from its window's edge - the whole point of
    paying for the second decode.
    """
    words: list[dict] = []
    for k, ((w_start, w_end), res) in enumerate(zip(wins, results)):
        lo_keep = -1e9
        hi_keep = 1e9
        if overlap > 0:
            if k > 0:
                lo_keep = (wins[k - 1][1] - (wins[k][0])) / 2 + wins[k][0] \
                    if wins[k - 1][1] > wins[k][0] else w_start
            if k + 1 < len(wins) and wins[k + 1][0] < w_end:
                hi_keep = (wins[k + 1][0] + w_end) / 2
        clip_start = res.get("_clip_start", w_start)
        for w in res.get("words") or []:
            t = float(w["start"]) + clip_start
            if lo_keep <= t < hi_keep:
                words.append({**w, "start": round(t, 2),
                              "end": round(float(w["end"]) + clip_start, 2)})
    words.sort(key=lambda w: w["start"])
    return words


# ----------------------------------------------------------------------- main

def main(argv=None) -> int:
    import gpu as gpu_mod
    import soundfile as sf

    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--variants", default="A,B,C,D,E,F")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--out", default="out/seg_bench")
    args = p.parse_args(argv)

    creds = rp.load_credentials()
    endpoint = gpu_mod._endpoint_id(args.endpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One local 16 kHz decode; every variant slices this same array, so no
    # variant can benefit from a different resample path.
    src = Path(args.audio)
    with tempfile.TemporaryDirectory() as td:
        wav16 = Path(td) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-af", "volume=-1dB,aresample=resampler=soxr:precision=28:dither_method=none:osr=16000",
             "-ac", "1", "-c:a", "pcm_s16le", str(wav16)],
            check=True, capture_output=True)
        audio, _ = sf.read(str(wav16), dtype="float32")
        spans, _total = vad_mod.detect_speech(wav16, threshold=0.4)
    total = len(audio) / SR
    print(f"{src.name}: {total:.0f}s, {len(spans)} speech spans")

    summary = {}
    for variant in [v.strip().upper() for v in args.variants.split(",")]:
        t0 = time.time()
        if variant == "A":
            wins, overlap = [(0.0, total)], 0.0
        else:
            wins, overlap = build_variant(spans, variant)
        clips = []
        for k, (s, e) in enumerate(wins):
            lo = max(0.0, s - PAD)
            if overlap and k > 0:
                lo = max(0.0, wins[k - 1][1] - overlap)
            hi = min(total, e + PAD)
            clips.append((lo, audio[int(lo * SR):int(hi * SR)]))

        results: list[dict] = [None] * len(wins)
        if variant == "G":
            # Sequential by construction: each window's prompt is the tail of
            # the merged text so far. This is the price of carried context.
            tail: list[str] = []
            for k, (lo, clip) in enumerate(clips):
                prompt = " ".join(tail[-25:]) if tail else None
                res = _poll(creds, endpoint, _submit(creds, endpoint, clip, prompt))
                res["_clip_start"] = lo
                results[k] = res
                tail.extend(w["word"] for w in (res.get("words") or []))
        else:
            def run_one(k_lo_clip):
                k, (lo, clip) = k_lo_clip
                res = _poll(creds, endpoint, _submit(creds, endpoint, clip))
                res["_clip_start"] = lo
                return k, res
            with ThreadPoolExecutor(max_workers=3) as ex:
                for k, res in ex.map(run_one, enumerate(clips)):
                    results[k] = res

        words = merge_windows(results, wins, overlap)
        elapsed = time.time() - t0
        lens = [e - s for s, e in wins]
        summary[variant] = {
            "calls": len(wins), "seconds": round(elapsed, 1),
            "words": len(words),
            "win_len_median": round(float(np.median(lens)), 1),
            "win_len_min": round(min(lens), 1), "win_len_max": round(max(lens), 1),
        }
        print(f"  {variant}: {len(wins):3} calls, windows "
              f"{summary[variant]['win_len_min']}-{summary[variant]['win_len_max']}s "
              f"(median {summary[variant]['win_len_median']}), "
              f"{len(words)} words, {elapsed:.0f}s")

        (out_dir / f"{variant}.json").write_text(
            json.dumps({"windows": wins, "summary": summary[variant],
                        "words": words}, ensure_ascii=False), encoding="utf-8")
        # A plain timed transcript per variant, for side-by-side reading.
        lines, cur, cur_t = [], [], None
        for w in words:
            if cur and (w["start"] - cur_end > 1.0):
                lines.append(f"[{cur_t:7.1f}]  " + " ".join(cur))
                cur, cur_t = [], None
            if cur_t is None:
                cur_t = w["start"]
            cur.append(w["word"])
            cur_end = w["end"]
        if cur:
            lines.append(f"[{cur_t:7.1f}]  " + " ".join(cur))
        (out_dir / f"{variant}.txt").write_text(
            "\n".join(lines), encoding="utf-8")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
