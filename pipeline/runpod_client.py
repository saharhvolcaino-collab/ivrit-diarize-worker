"""Talk to a RunPod serverless endpoint running the ivrit.ai worker.

The payload shape here is not obvious and the repo contains two stale examples
that the current handler rejects, so it is worth stating what is actually
required. From `infer.py` on the worker side:

  - `model` is mandatory; the handler errors without it, and only the three
    models baked into the image will load (it opens them with
    local_files_only=True).
  - `transcribe_args` is mandatory and must carry either `blob` (base64) or
    `url`.
  - `diarize: true` routes through the stable-whisper path, which cannot
    stream - so streaming and diarization are mutually exclusive.

One thing to know about this worker: decoding parameters do not survive the
trip. `ivrit-py`'s `audio.py:1069` accepts **kwargs and forwards none of them,
so beam_size, vad_filter and initial_prompt are transmitted and then silently
dropped. We send them anyway to keep the request self-documenting, but nothing
here should be read as "we tuned the decoder" - that needs a forked worker.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API_BASE = "https://api.runpod.ai/v2"
DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
# RunPod rejects anything larger on /run. Base64 inflates by ~4/3, so the real
# ceiling on raw audio is around 7.5 MB.
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024
POLL_INTERVAL_SEC = 3.0


class RunPodError(RuntimeError):
    pass


@dataclass
class Credentials:
    api_key: str
    endpoint_id: str


def load_credentials(env_path: str | Path = ".env") -> Credentials:
    """Read credentials from .env, falling back to the process environment."""
    values: dict[str, str] = {}
    path = Path(env_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")

    api_key = values.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY", "")
    endpoint_id = values.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_ENDPOINT_ID", "")

    if not api_key:
        raise RunPodError(
            "RUNPOD_API_KEY is empty. Generate a key from the endpoint's Quick start "
            "panel and paste it into .env"
        )
    if not endpoint_id:
        raise RunPodError("RUNPOD_ENDPOINT_ID is empty in .env")
    return Credentials(api_key=api_key, endpoint_id=endpoint_id)


def _request(url: str, api_key: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 401:
            raise RunPodError("401 Unauthorized - the API key is wrong or lacks write access")
        if exc.code == 404:
            raise RunPodError(f"404 Not Found - check the endpoint ID. Server said: {body}")
        raise RunPodError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RunPodError(f"Network error reaching RunPod: {exc.reason}") from exc


def health(creds: Credentials) -> dict:
    """Worker and job counters. `throttled` above zero means no GPU capacity."""
    return _request(f"{API_BASE}/{creds.endpoint_id}/health", creds.api_key, timeout=30)


def build_payload(
    audio_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str = "he",
    diarize: bool = True,
    word_timestamps: bool = True,
    diarization_args: dict | None = None,
) -> dict:
    """Assemble the request body, sending the audio inline as base64.

    Inline transport works because OPUS at 24 kbps puts a ten-minute call in
    about 1.5 MB. Anything approaching an hour needs a presigned URL instead,
    and this raises rather than letting RunPod reject the request opaquely.
    """
    audio_path = Path(audio_path)
    raw = audio_path.read_bytes()
    blob = base64.b64encode(raw).decode("ascii")

    if len(blob) > MAX_PAYLOAD_BYTES:
        raise RunPodError(
            f"{audio_path.name} is {len(blob) / 1024 / 1024:.1f} MB once base64-encoded, "
            f"over RunPod's {MAX_PAYLOAD_BYTES / 1024 / 1024:.0f} MB cap on /run. "
            "Upload it and pass a presigned URL instead."
        )

    transcribe_args: dict = {
        "blob": blob,
        "language": language,
        "diarize": diarize,
        "output_options": {
            "word_timestamps": word_timestamps,
            "extra_data": True,
        },
    }

    # The worker defaults to its own "ivrit" diarization engine, which accepts
    # only min/max_speakers. Its "pyannote" engine accepts num_speakers, which
    # is a hard constraint rather than a bound - worth using when the speaker
    # count is known, since guessing it is the largest single error source in
    # diarization.
    if diarize and diarization_args:
        transcribe_args["diarization_args"] = diarization_args

    return {
        "input": {
            # Diarization only exists on the stable-whisper path.
            "engine": "stable-whisper" if diarize else "faster-whisper",
            "model": model,
            # stable-whisper needs the whole file, so streaming is off.
            "streaming": False,
            "transcribe_args": transcribe_args,
        }
    }


def submit(creds: Credentials, payload: dict) -> str:
    result = _request(f"{API_BASE}/{creds.endpoint_id}/run", creds.api_key, payload)
    job_id = result.get("id")
    if not job_id:
        raise RunPodError(f"No job id in response: {json.dumps(result)[:400]}")
    return job_id


def poll(
    creds: Credentials,
    job_id: str,
    *,
    timeout_sec: int = 1800,
    on_update=None,
) -> dict:
    """Wait for a job, reporting status changes as they happen."""
    started = time.time()
    last_status = None

    while True:
        state = _request(f"{API_BASE}/{creds.endpoint_id}/status/{job_id}", creds.api_key, timeout=30)
        status = state.get("status")

        if status != last_status and on_update:
            on_update(status, state)
            last_status = status

        if status == "COMPLETED":
            return state
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            detail = json.dumps(state.get("output") or state)[:600]
            raise RunPodError(f"Job {status}: {detail}")

        if time.time() - started > timeout_sec:
            raise RunPodError(
                f"Gave up waiting after {timeout_sec}s (last status: {status}). "
                "The job is still running on RunPod - check the Requests tab."
            )
        time.sleep(POLL_INTERVAL_SEC)


def transcribe(
    audio_path: str | Path,
    creds: Credentials | None = None,
    *,
    model: str = DEFAULT_MODEL,
    diarize: bool = True,
    verbose: bool = True,
    diarization_args: dict | None = None,
) -> dict:
    """Submit one file and return the completed job payload."""
    creds = creds or load_credentials()
    audio_path = Path(audio_path)

    payload = build_payload(
        audio_path, model=model, diarize=diarize, diarization_args=diarization_args
    )
    size_mb = len(payload["input"]["transcribe_args"]["blob"]) / 1024 / 1024

    if verbose:
        print(f"  sending {audio_path.name} ({size_mb:.1f} MB base64, diarize={diarize})")

    job_id = submit(creds, payload)
    if verbose:
        print(f"  job {job_id}")

    def report(status, state):
        if not verbose:
            return
        if status == "IN_QUEUE":
            print("  queued...")
        elif status in ("IN_PROGRESS", "RUNNING"):
            delay = state.get("delayTime")
            extra = f" (cold start + queue: {delay / 1000:.1f}s)" if delay else ""
            print(f"  running{extra}")

    started = time.time()
    state = poll(creds, job_id, on_update=report)

    if verbose:
        exec_ms = state.get("executionTime") or 0
        print(f"  done in {time.time() - started:.1f}s wall, {exec_ms / 1000:.1f}s on GPU")

    return state


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Send audio to the ivrit.ai RunPod endpoint.")
    p.add_argument("audio", nargs="*", help="Audio file(s) - use the OPUS copies from out/")
    p.add_argument("--check", action="store_true", help="Only check endpoint health")
    p.add_argument("--no-diarize", action="store_true", help="Transcribe without speaker labels")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="out", help="Where to write raw JSON responses")
    args = p.parse_args(argv)

    try:
        creds = load_credentials()
    except RunPodError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"endpoint {creds.endpoint_id}")

    try:
        h = health(creds)
    except RunPodError as exc:
        print(f"ERROR: {exc}")
        return 1

    workers = h.get("workers", {})
    jobs = h.get("jobs", {})
    print(f"  workers: {workers}")
    print(f"  jobs:    {jobs}")

    if workers.get("throttled"):
        print("  [!] Workers are throttled - RunPod has no capacity for your GPU "
              "selection. Add more GPU types in the endpoint configuration.")

    if args.check or not args.audio:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for item in args.audio:
        print(f"\n{Path(item).name}")
        try:
            state = transcribe(item, creds, model=args.model, diarize=not args.no_diarize)
        except RunPodError as exc:
            print(f"  ERROR: {exc}")
            failures += 1
            continue

        dest = out_dir / f"{Path(item).stem}.runpod.json"
        dest.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  raw response -> {dest}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
