"""The studio master: every recording gets the full treatment before ASR.

Born in out/flag_lab/studio (twelve treatments auditioned on the flagship
island) and promoted to a standard pre-pass after a measured head-to-head
on a full call through the complete pipeline:

                       raw     studio
    flagship phrase    wrong   RIGHT ("בסדר, מה שלומך")
    buried answers     15      12
    rescue adoptions   26      29
    model agreement    63.5%   63.9%

The chain, in mixing order - every stage manufactured from the signal
itself, nothing imagined:

  1. floor_subtract   spectral subtraction of the measured quantisation
                      floor (-53 dBFS on this telephony chain). Generic
                      denoisers guessed at noise this channel does not
                      have; this removes the one noise we measured.
  2. syllable_agc     a fader hand riding gain per 50 ms slice - weak
                      syllables stand up (boost only, up to +24 dB),
                      strong ones are left alone. This is the "give every
                      spoken part its life" stage.
  3. high shelf       +9 dB above 2 kHz: consonant cues live against this
                      channel's 3.85 kHz ceiling; the shelf lights them.
  4. exciter          tanh harmonics of what is really there, high-passed
                      and mixed under the dry signal - presence without
                      invention. A harmonic exists only if its fundamental
                      does.

Deliberately NOT here: varispeed. Blanket slowdown broke content it did
not help (measured: "מה תומר"); it lives in the referee's per-region probe
ladder instead, where it is applied only to disputed clips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def floor_subtract(x: np.ndarray, floor_db: float = -53.0,
                   alpha: float = 2.0) -> np.ndarray:
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


def syllable_agc(x: np.ndarray, sr: int, target_db: float = -20.0,
                 frame_ms: float = 50.0, max_gain_db: float = 24.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    frames = len(x) // n
    if frames < 1:
        return x
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


def high_shelf(x: np.ndarray, sr: int, hz: float = 2000.0,
               gain_db: float = 9.0) -> np.ndarray:
    """RBJ audio-EQ-cookbook high shelf biquad - no external tools."""
    from scipy.signal import lfilter
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * hz / sr
    cs, sn = np.cos(w0), np.sin(w0)
    s = 1.0
    alpha = sn / 2 * np.sqrt((a_ + 1 / a_) * (1 / s - 1) + 2)
    two_sqrt_a_alpha = 2 * np.sqrt(a_) * alpha
    b0 = a_ * ((a_ + 1) + (a_ - 1) * cs + two_sqrt_a_alpha)
    b1 = -2 * a_ * ((a_ - 1) + (a_ + 1) * cs)
    b2 = a_ * ((a_ + 1) + (a_ - 1) * cs - two_sqrt_a_alpha)
    a0 = (a_ + 1) - (a_ - 1) * cs + two_sqrt_a_alpha
    a1 = 2 * ((a_ - 1) - (a_ + 1) * cs)
    a2 = (a_ + 1) - (a_ - 1) * cs - two_sqrt_a_alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, x).astype(np.float32)


def exciter(x: np.ndarray, sr: int, amount: float = 0.35) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    drive = np.tanh(4.0 * x)
    sos = butter(4, min(2000 / (sr / 2), 0.95), btype="highpass", output="sos")
    bright = sosfilt(sos, drive).astype(np.float32)
    return np.clip(x + amount * bright, -0.999, 0.999)


def master(x: np.ndarray, sr: int) -> np.ndarray:
    y = floor_subtract(x)
    y = syllable_agc(y, sr)
    y = high_shelf(y, sr)
    y = exciter(y, sr)
    peak = float(np.abs(y).max() or 0.0)
    if peak > 0.98:
        y = y * (0.98 / peak)
    return y.astype(np.float32)


def master_to(audio: Path, out_dir: Path) -> Path:
    """Master `audio` into out_dir/<stem>.studio.wav and return the path."""
    import soundfile as sf
    x, sr = sf.read(str(audio), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dst = Path(out_dir) / f"{Path(audio).stem}.studio.wav"
    sf.write(str(dst), master(x, sr), sr)
    return dst
