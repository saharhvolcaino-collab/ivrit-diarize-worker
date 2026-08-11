"""Studio session on the 1.4-second island - every trick on the console.

    python studio_lab.py --endpoint <id>

The nine presentation experiments proved the phrase's consonant skeleton IS
in the audio (decoded alone, the engine hears "שומעים" - the same שומ core
as "שלומך") and that whole-island gain, tempo, prompt and beam width do not
recover the missing phonemes. This session goes below that level, at the
signal itself, the way a mixing engineer would rescue a bad location take:

  syllable-level gain riding   the phrase's own envelope swings 20 dB in
                               1.4s; each 50 ms slice is ridden to target,
                               so the weak first word stops hiding
  consonant shelf              the ל/ך cues live at the top of our band;
                               a +9..+12 dB shelf above 2 kHz relights them
  harmonic exciter             classic studio brightener: controlled
                               distortion regenerates high harmonics FROM
                               the real signal - deterministic, not an AI
                               guess, unlike bandwidth "extension"
  varispeed                    tape-style slowdown WITHOUT pitch hold:
                               shifts the whole spectrum down, so cues at
                               our 3.85 kHz ceiling land mid-band where the
                               engine's resolution is best
  floor subtraction            we measured the quantisation floor exactly
                               (-53 dBFS white); spectral subtraction with
                               that known profile is surgery, not guessing
  stacked combos               the winners, together

Each variant goes to ivrit-ai (ASR only, no VAD, tempo pinned 1.0) and the
verdict is one line: did it say the phrase. The full board also goes to disk
as wavs so a human can listen to what the engine heard.
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
ISLAND = (7.55, 9.00)
OUT = Path("out/flag_lab/studio")


# ------------------------------------------------------------------ DSP tools

def syllable_agc(x: np.ndarray, target_db: float = -20.0,
                 frame_ms: float = 50.0, max_gain_db: float = 24.0) -> np.ndarray:
    """Ride the gain per 50 ms slice - a fader hand, not a compressor.

    The island's envelope swings from -44 to -23 dBFS inside 1.4 seconds;
    one gain number leaves the weak word weak. Per-slice riding flattens the
    envelope while a 3-frame smoothing keeps the gain curve from zipping.
    """
    n = max(1, int(SR * frame_ms / 1000))
    frames = len(x) // n
    gains = np.ones(frames + 1, dtype=np.float32)
    for i in range(frames):
        seg = x[i * n:(i + 1) * n]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        db = 20 * np.log10(rms)
        want = np.clip(target_db - db, 0.0, max_gain_db)   # boost only
        gains[i] = 10 ** (want / 20)
    k = np.array([0.25, 0.5, 0.25])
    gains[:frames] = np.convolve(gains[:frames], k, mode="same")
    env = np.repeat(gains[:frames], n)
    env = np.pad(env, (0, len(x) - len(env)), mode="edge")
    return np.clip(x * env, -0.999, 0.999)


def shelf(x: np.ndarray, hz: float, gain_db: float) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-af", f"highshelf=f={hz}:g={gain_db}", "-f", "f32le", "pipe:1"],
        input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def exciter(x: np.ndarray, amount: float = 0.35) -> np.ndarray:
    """Aural exciter: harmonics manufactured from the signal itself.

    tanh soft-clipping generates odd harmonics of whatever is really there;
    high-passing keeps only the new brightness and mixes it under the dry
    signal. Nothing is imagined - a harmonic exists only if its fundamental
    does. This is the honest cousin of neural bandwidth extension.
    """
    from scipy.signal import butter, sosfilt
    drive = np.tanh(4.0 * x)
    sos = butter(4, 2000 / (SR / 2), btype="highpass", output="sos")
    bright = sosfilt(sos, drive).astype(np.float32)
    return np.clip(x + amount * bright, -0.999, 0.999)


def varispeed(x: np.ndarray, rate: float) -> np.ndarray:
    """Tape slowdown: duration AND pitch change together.

    atempo preserves pitch; varispeed deliberately does not. At rate 0.8 the
    whole spectrum shifts down x0.8, so cues crammed against our 3.85 kHz
    ceiling land near 3.1 kHz - the meat of the engine's band - and every
    phoneme lasts 25% longer. The voice sounds deeper; the engine has never
    cared about that half as much as about seeing the cues at all.
    """
    from scipy.signal import resample_poly
    up = int(round(SR / rate))
    y = resample_poly(x, up, SR).astype(np.float32)
    return y


def floor_subtract(x: np.ndarray, floor_db: float = -53.0,
                   alpha: float = 2.0) -> np.ndarray:
    """Spectral subtraction of the measured quantisation floor.

    Generic denoisers failed here because they guessed at noise this file
    does not have. This subtracts the one noise we measured precisely: a
    flat floor at -53 dBFS. Over-subtraction (alpha 2) with a spectral gate
    floor keeps musical noise down on a clip this short.
    """
    n_fft, hop = 512, 128
    from numpy.fft import irfft, rfft
    win = np.hanning(n_fft).astype(np.float32)
    floor_amp = 10 ** (floor_db / 20) * np.sqrt(n_fft)
    frames = 1 + max(0, (len(x) - n_fft)) // hop
    y = np.zeros(len(x) + n_fft, dtype=np.float32)
    norm = np.zeros_like(y)
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = rfft(seg * win)
        mag = np.abs(spec)
        cleaned = np.maximum(mag - alpha * floor_amp, 0.1 * mag)
        out = irfft(cleaned * np.exp(1j * np.angle(spec)), n_fft)
        y[i * hop:i * hop + n_fft] += (out * win).astype(np.float32)
        norm[i * hop:i * hop + n_fft] += win ** 2
    y = y[:len(x)] / np.maximum(norm[:len(x)], 1e-6)
    return y.astype(np.float32)


# ---------------------------------------------------------------- the session

def build_board(x: np.ndarray):
    lo, hi = ISLAND
    isl = x[int((lo - 0.2) * SR):int((hi + 0.2) * SR)].copy()

    agc = syllable_agc(isl)
    board = [
        ("S1_syllable_agc",        agc),
        ("S2_shelf9",              shelf(isl, 2000, 9)),
        ("S3_agc_shelf9",          shelf(agc, 2000, 9)),
        ("S4_exciter",             exciter(isl)),
        ("S5_agc_exciter",         exciter(agc)),
        ("S6_varispeed85",         varispeed(isl, 0.85)),
        ("S7_varispeed75",         varispeed(isl, 0.75)),
        ("S8_agc_varispeed85",     varispeed(agc, 0.85)),
        ("S9_floorsub",            floor_subtract(isl)),
        ("S10_floorsub_agc",       syllable_agc(floor_subtract(isl))),
        ("S11_full_stack",         varispeed(exciter(shelf(syllable_agc(isl), 2000, 9)), 0.9)),
        ("S12_agc_shelf12",        shelf(agc, 1800, 12)),
    ]
    return board


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
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for name, audio in build_board(x):
        sf.write(str(OUT / f"{name}.wav"), audio, SR, subtype="PCM_16")
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        payload = {"input": {
            "audio_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "audio_format": "wav", "language": "he",
            "diarize": False, "vad_filter": False, "tempo": 1.0}}
        try:
            job = rp._request(f"{API}/{endpoint}/run", creds.api_key, payload)
            state = rp.poll(rp.Credentials(api_key=creds.api_key,
                                           endpoint_id=endpoint),
                            job["id"], timeout_sec=600)
            text = " ".join(w["word"] for w in
                            (state.get("output") or {}).get("words") or [])
        except Exception as exc:
            text = f"ERROR {type(exc).__name__}: {exc}"
        results.append((name, text))
        print(f"{name:22} {text[:100]}")

    (OUT / "board.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
