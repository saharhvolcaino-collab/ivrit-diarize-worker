"""Turn clean Hebrew speech into OUR telephone channel, for training data.

The whole fine-tuning plan rests on one asset: the ability to manufacture
unlimited (degraded audio, correct transcript) pairs. Clean transcribed
Hebrew exists in quantity; recordings from our channel with transcripts do
not. So the channel itself is reproduced here, and every parameter comes
from measurement of a real call (the full DSP audit of Rec722_11-18-44),
not from a textbook telephone simulation:

    passband        ~120 Hz high-pass, content ceiling ~3850 Hz with a
                    gentle ~30 dB/oct roll-off (NOT a brickwall at 3400 -
                    the measured spectrum meets the noise floor at 3850)
    sample rate     11025 Hz on the wire
    quantisation    8-bit unsigned linear, 256 levels. The measured error
                    is white and signal-uncorrelated because an ambient
                    floor of ~0.8 LSB acts as natural dither - so a floor
                    like it is added BEFORE quantising. Skipping that would
                    produce correlated harmonic distortion the real channel
                    does not have, and the model would learn to fight an
                    artefact that never occurs.
    silence gating  the recorder zeroes anything at ~1 LSB: 45% of a real
                    file is exact digital zero, in runs from 6 ms to 4 s,
                    including 1,738 closures INSIDE utterances. Trained
                    without this, the model meets hard-gated speech for the
                    first time in production.
    speaker levels  per-speaker gains drawn from the measured spread -
                    median -18 dBFS with a quiet-party tail down to -33
                    (17.5 dB between loudest and quietest cluster). The
                    quiet tail is exactly where production WER concentrates,
                    so training must contain it.
    reverberation   median RT60 ~0.4 s (p90 1.4): synthetic exponential-
                    decay RIR, applied with probability, because the far
                    parties on a speakerphone carry it and the near party
                    does not.

Usage (single file or a directory tree):

    python training/degrade.py clean.wav out.wav [--seed 7]
    python training/degrade.py clean_dir/ out_dir/ --jobs 4

Output is 16 kHz PCM16 (post-channel, resampled the same way production
resamples), ready to feed a Whisper training pipeline directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIRE_SR = 11025
OUT_SR = 16000

# Measured on the real estate; every constant traceable to the DSP audit.
HP_HZ = 120.0
LP_HZ = 3700.0            # -3 dB point that puts the noise-floor meet at ~3850
AMBIENT_LSB = 0.84        # the natural-dither floor, in 8-bit LSBs
GATE_LSB = 1.5            # recorder squelch threshold
GATE_MIN_MS = 6.0         # shortest zero-run seen that is clearly the gate
LEVEL_MEDIAN_DB = -18.0
LEVEL_SPREAD_DB = 6.5     # lognormal-ish spread; tail reaches ~-33 dBFS
QUIET_PARTY_P = 0.25      # chance a file is a "quiet party" recording
QUIET_EXTRA_DB = -12.0
REVERB_P = 0.5
RT60_CHOICES = (0.25, 0.4, 0.8, 1.4)


def _butter(x: np.ndarray, sr: int, hz: float, kind: str, order: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(order, hz / (sr / 2), btype=kind, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _reverb(x: np.ndarray, sr: int, rt60: float, rng) -> np.ndarray:
    """Exponential-decay noise RIR - the standard cheap stand-in for a room.

    Good enough here because the goal is not to model one room but to expose
    training to the FAMILY of smearing the speakerphone parties carry.
    """
    n = int(sr * min(rt60, 1.5))
    if n < 8:
        return x
    t = np.arange(n) / sr
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.91 * t / rt60)
    rir[0] = 1.0                      # direct path dominates
    rir /= np.abs(rir).sum() ** 0.5
    from scipy.signal import fftconvolve
    wet = fftconvolve(x, rir)[: x.size].astype(np.float32)
    # 50/50 direct-to-reverberant keeps it a telephone room, not a cathedral.
    return 0.7 * x + 0.5 * wet


def degrade(audio: np.ndarray, sr: int, seed: int | None = None) -> np.ndarray:
    """Clean float32 mono at any rate -> our channel, returned at 16 kHz."""
    from scipy.signal import resample_poly

    rng = np.random.default_rng(seed)
    x = audio.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Onto the wire rate first: everything the channel does happens at 11025.
    if sr != WIRE_SR:
        from math import gcd
        g = gcd(WIRE_SR, sr)
        x = resample_poly(x, WIRE_SR // g, sr // g).astype(np.float32)

    # Room, for the far parties.
    if rng.random() < REVERB_P:
        x = _reverb(x, WIRE_SR, float(rng.choice(RT60_CHOICES)), rng)

    # Passband.
    x = _butter(x, WIRE_SR, HP_HZ, "highpass", order=2)
    x = _butter(x, WIRE_SR, LP_HZ, "lowpass", order=6)

    # Recording level, drawn from the measured distribution.
    target_db = LEVEL_MEDIAN_DB + float(rng.standard_normal()) * LEVEL_SPREAD_DB
    if rng.random() < QUIET_PARTY_P:
        target_db += QUIET_EXTRA_DB
    target_db = float(np.clip(target_db, -34.0, -12.0))
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    x = x * (10 ** (target_db / 20) / rms)

    # The ambient floor that self-dithers the quantiser (see module docstring).
    lsb = 1.0 / 128.0
    x = x + rng.standard_normal(x.size).astype(np.float32) * (AMBIENT_LSB * lsb / 3)

    # 8-bit unsigned linear: clip, quantise to 256 levels, back to float.
    x = np.clip(x, -1.0, 1.0)
    x = np.round((x + 1.0) * 127.5)
    x = (x / 127.5 - 1.0).astype(np.float32)

    # Squelch: frames whose level sits at the gate threshold go to exact zero,
    # exactly as the recorder does - including mid-utterance closures.
    frame = max(1, int(WIRE_SR * GATE_MIN_MS / 1000))
    n_frames = x.size // frame
    if n_frames:
        f = x[: n_frames * frame].reshape(n_frames, frame)
        quiet = np.sqrt(np.mean(np.square(f), axis=1)) < (GATE_LSB * lsb)
        f[quiet] = 0.0
        x[: n_frames * frame] = f.reshape(-1)

    # Off the wire: the same resample production performs.
    from math import gcd
    g = gcd(OUT_SR, WIRE_SR)
    y = resample_poly(x, OUT_SR // g, WIRE_SR // g).astype(np.float32)
    peak = float(np.abs(y).max())
    if peak > 0.999:
        y *= 0.999 / peak
    return y


def _one(src: Path, dst: Path, seed: int | None) -> str:
    import soundfile as sf
    audio, sr = sf.read(str(src), dtype="float32")
    y = degrade(audio, sr, seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, OUT_SR, subtype="PCM_16")
    return f"{src.name}: {len(audio)/sr:.1f}s -> degraded"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if src.is_file():
        print(_one(src, dst, args.seed))
        return 0

    files = sorted(src.rglob("*.wav")) + sorted(src.rglob("*.flac")) \
        + sorted(src.rglob("*.mp3"))
    print(f"{len(files)} files")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_one, f, dst / f.relative_to(src).with_suffix(".wav"),
                          None if args.seed is None else args.seed + i)
                for i, f in enumerate(files)]
        for k, fu in enumerate(futs):
            fu.result()
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(files)}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
