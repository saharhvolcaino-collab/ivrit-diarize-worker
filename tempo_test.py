"""Does slowing the audio down improve Hebrew transcription?

    python tempo_test.py <recording> --rates 1.0,0.85,0.75,1.25

The idea is worth testing because, unlike denoising, time-stretching removes
nothing: every spectral cue survives, the speech simply occupies more frames.
Whisper's mel hops every 10 ms, so a 60 ms phoneme is described by 6 frames at
natural rate and 8 at 0.75x. On narrowband audio where the spectral evidence
is thin, more temporal evidence per phoneme is a real hypothesis.

The argument against is equally real: Whisper was trained on speech at natural
rate, and stretching pushes the input off that distribution - the same
mechanism that made every enhancement attempt here fail.

So it is measured rather than argued. `atempo` is WSOLA: it changes duration
without touching pitch, which matters because a resampling-based slowdown
would shift the whole spectrum down and shrink the band we already lack.

1.25x is included deliberately as a control. If slowing helps because of frame
density, speeding up must hurt for the same reason. If both help, or both
hurt, the mechanism is not what we think and the result means nothing.

Judged on two-model agreement, which needs no reference transcript: where the
full model and the turbo model produce the same word, that word is probably
right. The baseline on this estate is 73.3%.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import runpod_client as rp

API = "https://api.runpod.ai/v2"


def stretch(src: Path, rate: float, out_dir: Path) -> Path:
    """Change duration by `rate` without changing pitch."""
    dst = out_dir / f"{src.stem}.t{rate:.2f}.opus"
    if rate == 1.0:
        chain = "aresample=resampler=soxr:precision=28:dither_method=none:osr=16000"
    else:
        # atempo is limited to 0.5-2.0 per instance; chain if we ever go past.
        chain = (f"atempo={rate},"
                 "aresample=resampler=soxr:precision=28:dither_method=none:osr=16000")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-af", chain, "-ac", "1", "-c:a", "libopus", "-b:a", "24k", str(dst)],
        check=True, capture_output=True)
    return dst


def run(path: Path, endpoint: str, creds) -> dict:
    blob = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = {"input": {
        "audio_base64": blob, "audio_format": "opus", "language": "he",
        "diarize": True, "num_speakers": "auto",
        "engines": ["ecapa", "sortformer"], "verify": True,
    }}
    job = rp._request(f"{API}/{endpoint}/run", creds.api_key, payload)
    state = rp.poll(rp.Credentials(api_key=creds.api_key, endpoint_id=endpoint),
                    job["id"], timeout_sec=2400)
    out = state.get("output") or {}
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def main(argv: list[str] | None = None) -> int:
    import gpu as gpu_mod

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio")
    p.add_argument("--rates", default="1.0,0.85,0.75,1.25")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--out", default="out/tempo")
    args = p.parse_args(argv)

    creds = rp.load_credentials()
    endpoint = gpu_mod._endpoint_id(args.endpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.audio)

    rows = []
    for rate in [float(r) for r in args.rates.split(",")]:
        print(f"\n=== tempo {rate:.2f}x ===")
        clip = stretch(src, rate, out_dir)
        t0 = time.time()
        try:
            res = run(clip, endpoint, creds)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        (out_dir / f"{src.stem}.t{rate:.2f}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

        q = res.get("quality", {})
        v = (res.get("meta") or {}).get("verification") or {}
        alt = (res.get("alternates") or {}).get("sortformer", {}).get("quality", {})
        rows.append({
            "rate": rate,
            "agree": v.get("agreement_pct", 0.0),
            "words": q.get("words", 0),
            "contested": v.get("contested", 0),
            "buried_ecapa": q.get("buried_answers", 0),
            "buried_sf": alt.get("buried_answers", 0),
            "verdict": q.get("verdict", "?"),
            "sec": round(time.time() - t0),
        })
        print(f"  agreement {rows[-1]['agree']}%   words {rows[-1]['words']}   "
              f"buried {rows[-1]['buried_ecapa']}/{rows[-1]['buried_sf']}   "
              f"{rows[-1]['sec']}s")

    print(f"\n{'='*68}\n{'rate':>6} {'agreement':>10} {'words':>7} "
          f"{'contested':>10} {'buried e/sf':>12}  verdict")
    print("=" * 68)
    for r in rows:
        print(f"{r['rate']:6.2f} {r['agree']:9.1f}% {r['words']:7} "
              f"{r['contested']:10} {r['buried_ecapa']:5}/{r['buried_sf']:<5}  "
              f"{r['verdict']}")

    base = next((r for r in rows if r["rate"] == 1.0), None)
    if base:
        print(f"\nagainst 1.00x ({base['agree']}% agreement):")
        for r in rows:
            if r["rate"] == 1.0:
                continue
            d = r["agree"] - base["agree"]
            print(f"  {r['rate']:.2f}x  {d:+.1f} points  "
                  f"{'better' if d > 0.5 else 'worse' if d < -0.5 else 'no change'}")
    Path(out_dir, "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
