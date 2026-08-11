"""Nine inversions of one 1.1-second island, all against ivrit-ai.

    python flag_experiments.py --endpoint <id>

The research object: seconds 7.6-8.9 of Rec722_11-18-44 - the phrase every
ivrit-ai configuration hears as "מה שאת אומרת" and a listening LLM hears,
six runs out of six, as "מה שלומך". Measured profile: an island between two
hard silence gates, 16 dB quieter than the other party, ~4 effective bits,
its first word down at 2-3 bits.

Each experiment changes exactly one thing about how that island reaches the
engine. The question each asks is stated inline; between them they cover
level, tempo, isolation, repetition, decoder context and search width.
Everything runs ASR-only (diarize off) on 12-second-or-less clips, so the
whole battery costs agorot and minutes.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import runpod_client as rp

API = "https://api.runpod.ai/v2"
SR = 16000
CLIP = "out/flag_lab/flag_12s_16k.wav"
ISLAND = (7.55, 9.00)          # the target phrase, gate to gate
GAIN_TO = -18.0                # bring the island to the loud party's level


def _wav_b64(x: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _island(x: np.ndarray, pad: float = 0.2) -> np.ndarray:
    lo, hi = ISLAND
    return x[int((lo - pad) * SR):int((hi + pad) * SR)]


def _gained(seg: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
    g = 10 ** (GAIN_TO / 20) / rms
    return np.clip(seg * g, -0.999, 0.999)


def _slow(seg: np.ndarray, rate: float) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-af", f"atempo={rate}", "-f", "f32le", "pipe:1"],
        input=seg.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def build_experiments(x: np.ndarray):
    isl = _island(x)
    gap = np.zeros(int(0.4 * SR), dtype=np.float32)
    return [
        # What does the engine hear with everything as-is? (control)
        ("A_baseline_12s",      x,                          {"tempo": 1.0}),
        # Does the production slowdown change this phrase?
        ("B_slow085_server",    x,                          {"tempo": 0.85}),
        # Is LEVEL the killer? The island raised to the loud party's volume.
        ("C_island_gained_in_context",
         np.concatenate([x[:int(ISLAND[0]*SR)], _gained(x[int(ISLAND[0]*SR):int(ISLAND[1]*SR)]), x[int(ISLAND[1]*SR):]]),
         {"tempo": 1.0}),
        # Is CONTEXT the killer? The island alone, no surrounding speech to
        # anchor the decoder's language prior.
        ("D_island_alone",      isl,                        {"tempo": 1.0, "vad_filter": False}),
        # Island alone, amplified: level and isolation together.
        ("E_island_alone_gained", _gained(isl),             {"tempo": 1.0, "vad_filter": False}),
        # More frames per phoneme on just the weak audio.
        ("F_island_slow075",    _slow(_gained(isl), 0.75),  {"tempo": 1.0, "vad_filter": False}),
        # Three passes over the same evidence in one decode.
        ("G_island_x3",         np.concatenate([_gained(isl), gap, _gained(isl), gap, _gained(isl)]),
         {"tempo": 1.0, "vad_filter": False}),
        # The decoder handed the real conversational context as text - the
        # same trick that let the listening LLM solve it.
        ("H_island_with_prompt", _gained(isl),
         {"tempo": 1.0, "vad_filter": False,
          "initial_prompt": "שלום, מדבר תומר. שלום תומר, מדברת מכפר בלום, מה נשמע?"}),
        # Wider beam search over the strongest signal version.
        ("I_island_beam10",     _gained(isl),
         {"tempo": 1.0, "vad_filter": False, "beam_size": 10}),
    ]


def main(argv=None) -> int:
    import argparse
    import gpu as gpu_mod

    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=None)
    args = p.parse_args(argv)
    creds = rp.load_credentials()
    endpoint = gpu_mod._endpoint_id(args.endpoint)

    x, sr = sf.read(CLIP, dtype="float32")
    assert sr == SR

    results = []
    for name, audio, opts in build_experiments(x):
        payload = {"input": {
            "audio_base64": _wav_b64(audio), "audio_format": "wav",
            "language": "he", "diarize": False, **opts}}
        try:
            job = rp._request(f"{API}/{endpoint}/run", creds.api_key, payload)
            state = rp.poll(rp.Credentials(api_key=creds.api_key,
                                           endpoint_id=endpoint),
                            job["id"], timeout_sec=600)
            out = state.get("output") or {}
            text = " ".join(w["word"] for w in out.get("words") or [])
        except Exception as exc:
            text = f"ERROR {type(exc).__name__}: {exc}"
        results.append((name, text))
        print(f"{name:28} {text[:110]}")

    Path("out/flag_lab/experiments.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
