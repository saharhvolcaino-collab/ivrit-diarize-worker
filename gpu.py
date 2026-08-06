"""Client for OUR worker - the full pipeline runs on the GPU, this stays thin.

    python gpu.py <recording...>              transcribe + diarize on RunPod
    python gpu.py <folder> --batch            validation round + report

Everything the local pipeline did in stages - VAD, diarization, refinement,
splitting, quality scoring - happens inside one GPU call now. What remains
client-side is exactly what must stay close to the files: format probing,
compressing for transport, and rendering results.

Uses RUNPOD_ENDPOINT_V2 from .env (falls back to --endpoint). The old
endpoint keeps serving through run.py until this one is proven.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import parse as parse_mod
from pipeline import preprocess as prep_mod
from pipeline import quality as quality_mod
from pipeline import runpod_client as rp

API = "https://api.runpod.ai/v2"


def _endpoint_id(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("RUNPOD_ENDPOINT_V2="):
            val = line.split("=", 1)[1].strip()
            if val:
                return val
    raise SystemExit(
        "No RUNPOD_ENDPOINT_V2 in .env - add the new endpoint's ID or pass --endpoint"
    )


def process(source: Path, endpoint: str, creds: rp.Credentials, out_dir: Path,
            num_speakers: int = 2, hotwords: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"\n=== {source.name} ===")

    # OPUS keeps a ten-minute call around 1.5 MB - far under the 10 MB payload
    # cap, so audio travels inline and no object storage is needed.
    prep = prep_mod.preprocess(source, out_dir, make_opus=True)
    blob = base64.b64encode(Path(prep.opus_path).read_bytes()).decode("ascii")
    print(f"[send]   {len(blob) / 1024:.0f} KB")

    payload = {"input": {
        "audio_base64": blob,
        "audio_format": "opus",
        "language": "he",
        "diarize": True,
        "num_speakers": num_speakers,
        **({"hotwords": hotwords} if hotwords else {}),
    }}

    job = rp._request(f"{API}/{endpoint}/run", creds.api_key, payload)
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"no job id: {json.dumps(job)[:300]}")

    state = rp.poll(
        rp.Credentials(api_key=creds.api_key, endpoint_id=endpoint), job_id,
        on_update=lambda s, st: print(f"[job]    {s.lower()}"),
    )
    out = state.get("output") or {}
    if "error" in out:
        raise RuntimeError(f"worker error: {out['error']}\n{out.get('traceback', '')[-500:]}")

    meta = out.get("meta", {})
    print(f"[gpu]    asr {meta.get('asr_sec', '?')}s · "
          f"diar {out.get('diarization', {}).get('seconds', '?')}s · "
          f"total {meta.get('total_sec', '?')}s on GPU "
          f"({(state.get('delayTime') or 0) / 1000:.1f}s queue)")

    # Render. The worker returns finished turns and word-level speakers on the
    # original clock - no remapping, no arbitration, just presentation.
    stem = source.stem
    (out_dir / f"{stem}.gpu.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    turns = [
        parse_mod.Turn(speaker=t["speaker"], start=t["start"], end=t["end"],
                       text=t["text"], words=[])
        for t in out.get("turns", [])
    ]
    words = out.get("words", [])
    confs = [w["probability"] for w in words if w.get("probability") is not None]
    transcript = parse_mod.Transcript(
        source=source.name, turns=turns,
        speakers=out.get("speakers", []),
        duration_sec=meta.get("duration_sec", 0.0),
        word_count=len(words),
        mean_word_confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
        low_confidence_words=sum(1 for c in confs if c < 0.5),
        unattributed_words=sum(1 for w in words if not w.get("speaker")),
    )
    parse_mod.write_text(transcript, out_dir / f"{stem}.txt")
    parse_mod.write_srt(transcript, out_dir / f"{stem}.srt")
    # The view meant for human reading - a browser renders mixed
    # Hebrew/English correctly, which no plain-text editor guarantees.
    parse_mod.write_html(transcript, out_dir / f"{stem}.html",
                         quality=out.get("quality"))

    q = out.get("quality", {})
    share: dict[str, float] = {}
    for t in out.get("turns", []):
        share[t["speaker"]] = share.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    total = sum(share.values()) or 1.0
    print(f"[out]    {len(turns)} turns · {len(words)} words · "
          f"{len(transcript.speakers)} speakers · "
          + " / ".join(f"{100 * v / total:.0f}%" for v in
                       sorted(share.values(), reverse=True)))
    print(f"[q]      {q.get('verdict', '?').upper()} · buried {q.get('buried_answers', '?')} "
          f"· confidence {q.get('mean_confidence', 0):.3f}")
    for p in q.get("problems", []):
        print(f"         [!] {p}")
    print(f"[done]   {time.time() - started:.1f}s wall -> {out_dir / (stem + '.txt')}")

    result = dict(out)
    result["_file"] = source.name
    return result


AUDIO_EXTS = {".wav", ".wmv", ".mp3", ".opus", ".ogg", ".m4a", ".flac", ".aac", ".wma"}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run the GPU worker on recordings.")
    p.add_argument("paths", nargs="+")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--out", default="out/gpu")
    p.add_argument("--num-speakers", type=int, default=2)
    p.add_argument("--hotwords", default=None)
    args = p.parse_args(argv)

    creds = rp.load_credentials()
    endpoint = _endpoint_id(args.endpoint)
    out_dir = Path(args.out)

    files: list[Path] = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            files += [f for f in sorted(path.iterdir())
                      if f.suffix.lower() in AUDIO_EXTS
                      and not f.name.endswith((".16k.wav", ".speech.wav"))]
        else:
            files.append(path)

    failures = 0
    for f in files:
        try:
            process(f, endpoint, creds, out_dir,
                    num_speakers=args.num_speakers, hotwords=args.hotwords)
        except Exception as exc:
            print(f"  ERROR {f.name}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
