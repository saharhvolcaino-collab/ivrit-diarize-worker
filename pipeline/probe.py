"""Stage 0 - inspect a recording before we spend money on GPU time.

Telephone recordings lie about themselves constantly: a file can be stereo with
both speakers mixed identically into both channels, or claim 16 kHz while
carrying nothing above 4 kHz because it was upsampled from a narrowband codec.
Both cases change which pipeline we should run, and neither is visible from the
container format alone. So we measure instead of trusting the header.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path


class ProbeError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise ProbeError(
            f"{tool} not found on PATH. Install ffmpeg (it ships both ffmpeg and ffprobe)."
        )
    return path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ProbeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    return proc.stdout


@dataclass
class ChannelRelation:
    """How the channels of a stereo file relate to each other.

    correlation ~1.0 means the channels carry the same signal, so the file is
    "fake stereo" - dual-mono - and splitting it gains nothing. Values closer to
    0 mean the channels are genuinely different, which for a call recording
    usually means one speaker per channel.
    """

    correlation: float
    rms_db_per_channel: list[float]
    verdict: str  # "dual-mono" | "true-stereo" | "n/a"


@dataclass
class AudioProfile:
    path: str
    container: str
    codec: str
    duration_sec: float
    declared_sample_rate: int
    channels: int
    bit_rate: int | None
    # Measured, not declared.
    effective_bandwidth_hz: float | None = None
    is_narrowband: bool | None = None
    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    channel_relation: ChannelRelation | None = None
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


def _ffprobe_streams(path: Path) -> dict:
    out = _run([
        _require("ffprobe"), "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate,duration",
        "-show_entries", "format=format_name,duration,bit_rate",
        "-of", "json", str(path),
    ])
    data = json.loads(out)
    if not data.get("streams"):
        raise ProbeError(f"No audio stream found in {path}")
    return data


def _volume_stats(path: Path) -> tuple[float | None, float | None]:
    """Read mean/max volume from ffmpeg's volumedetect filter."""
    proc = subprocess.run(
        [_require("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    mean = peak = None
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            mean = float(line.split("mean_volume:")[1].split("dB")[0].strip())
        elif "max_volume:" in line:
            peak = float(line.split("max_volume:")[1].split("dB")[0].strip())
    return mean, peak


def _measure_bandwidth(path: Path, declared_sr: int) -> float | None:
    """Estimate the highest frequency that actually carries energy.

    We decode a mono 16 kHz copy, run a coarse DFT over a handful of windows
    spread through the file, and find the top bin holding a meaningful share of
    the spectrum's energy. A file that says 16 kHz but stops at ~3.4 kHz was
    upsampled from a G.711 telephone stream, and no amount of resampling will
    put the missing octave back.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    tmp = path.with_suffix(".probe.wav")
    try:
        _run([
            _require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp),
        ])
        with wave.open(str(tmp), "rb") as wf:
            sr = wf.getframerate()
            total = wf.getnframes()
            if total == 0:
                return None
            window = min(1 << 15, total)
            # Sample a few places rather than the whole file - silence at the
            # start of a call would otherwise dominate the estimate.
            starts = [int(total * f) for f in (0.15, 0.35, 0.55, 0.75)]
            spectra = []
            for start in starts:
                if start + window > total:
                    continue
                wf.setpos(start)
                raw = np.frombuffer(wf.readframes(window), dtype=np.int16).astype(np.float64)
                if raw.size < window or not np.any(raw):
                    continue
                spec = np.abs(np.fft.rfft(raw * np.hanning(raw.size)))
                spectra.append(spec)
            if not spectra:
                return None
            avg = np.mean(spectra, axis=0)

        power = avg ** 2
        total_power = power.sum()
        if total_power <= 0:
            return None
        # Highest bin below which 99% of the energy lives.
        cumulative = np.cumsum(power) / total_power
        idx = int(np.searchsorted(cumulative, 0.99))
        freqs = np.fft.rfftfreq(window, d=1.0 / sr)
        return float(freqs[min(idx, len(freqs) - 1)])
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def _channel_relation(path: Path, channels: int) -> ChannelRelation | None:
    """Decide whether a stereo file is genuinely two-channel.

    This is the single highest-value check in this module. If a call really has
    the agent on left and the customer on right, speaker attribution becomes
    exact and free, and every diarization model becomes unnecessary.
    """
    if channels < 2:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    tmp = path.with_suffix(".probe2.wav")
    try:
        _run([
            _require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path), "-ac", "2", "-ar", "8000", "-c:a", "pcm_s16le", str(tmp),
        ])
        with wave.open(str(tmp), "rb") as wf:
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if frames.size < 4:
            return None
        stereo = frames.reshape(-1, 2).astype(np.float64)
        left, right = stereo[:, 0], stereo[:, 1]

        def rms_db(x):
            r = math.sqrt(float(np.mean(x ** 2))) if x.size else 0.0
            return round(20 * math.log10(r / 32768.0), 2) if r > 0 else -120.0

        if np.std(left) < 1e-9 or np.std(right) < 1e-9:
            corr = 1.0 if np.allclose(left, right) else 0.0
        else:
            corr = float(np.corrcoef(left, right)[0, 1])

        verdict = "dual-mono" if corr > 0.98 else "true-stereo"
        return ChannelRelation(
            correlation=round(corr, 4),
            rms_db_per_channel=[rms_db(left), rms_db(right)],
            verdict=verdict,
        )
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def probe(path: str | Path) -> AudioProfile:
    path = Path(path)
    if not path.exists():
        raise ProbeError(f"File not found: {path}")

    data = _ffprobe_streams(path)
    stream = data["streams"][0]
    fmt = data.get("format", {})

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    declared_sr = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)
    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")

    profile = AudioProfile(
        path=str(path),
        container=fmt.get("format_name", "unknown"),
        codec=stream.get("codec_name", "unknown"),
        duration_sec=round(duration, 2),
        declared_sample_rate=declared_sr,
        channels=channels,
        bit_rate=int(bit_rate) if bit_rate else None,
    )

    profile.mean_volume_db, profile.max_volume_db = _volume_stats(path)
    bandwidth = _measure_bandwidth(path, declared_sr)
    if bandwidth is not None:
        profile.effective_bandwidth_hz = round(bandwidth, 1)
        profile.is_narrowband = bandwidth < 4200
    profile.channel_relation = _channel_relation(path, channels)

    _assess(profile)
    return profile


def _assess(p: AudioProfile) -> None:
    """Turn measurements into decisions about how to run the pipeline."""

    if p.channel_relation and p.channel_relation.verdict == "true-stereo":
        p.recommendations.append(
            "Channels are genuinely different (correlation "
            f"{p.channel_relation.correlation}). Split them and transcribe each side "
            "separately - this gives exact speaker attribution and makes the "
            "diarization model unnecessary."
        )
    elif p.channel_relation and p.channel_relation.verdict == "dual-mono":
        p.warnings.append(
            f"File reports {p.channels} channels but they are identical "
            f"(correlation {p.channel_relation.correlation}). This is mixed mono in a "
            "stereo wrapper - splitting gains nothing. Diarization is required."
        )

    if p.channels == 1:
        p.recommendations.append(
            "Single mixed channel - speaker separation must come from a "
            "diarization model. Pass num_speakers=2 so the model never has to "
            "guess the speaker count."
        )

    if p.is_narrowband:
        p.warnings.append(
            f"Effective bandwidth is only ~{p.effective_bandwidth_hz:.0f} Hz. This is "
            "narrowband telephone audio regardless of the declared "
            f"{p.declared_sample_rate} Hz. Whisper and the diarization models were "
            "trained on wideband speech, so expect degraded accuracy. Resampling "
            "up cannot restore the missing content."
        )

    if p.declared_sample_rate and p.declared_sample_rate < 16000:
        p.recommendations.append(
            f"Declared rate {p.declared_sample_rate} Hz is below the 16 kHz the models "
            "expect - resample before inference."
        )

    if p.mean_volume_db is not None and p.mean_volume_db < -30:
        p.warnings.append(
            f"Mean volume {p.mean_volume_db} dB is very low. Apply loudness "
            "normalisation before inference."
        )

    if p.max_volume_db is not None and p.max_volume_db >= -0.1:
        p.warnings.append(
            f"Peak volume {p.max_volume_db} dB indicates clipping. Clipped speech "
            "hurts both transcription and speaker embeddings."
        )

    if p.duration_sec > 1800:
        p.recommendations.append(
            f"Duration {p.duration_sec / 60:.1f} min - raise the RunPod endpoint "
            "executionTimeout above its 600 s default or the job will be killed."
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a recording before transcription.")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to inspect")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args(argv)

    profiles = []
    for item in args.audio:
        try:
            profiles.append(probe(item))
        except ProbeError as exc:
            print(f"ERROR {item}: {exc}")
            return 1

    if args.json:
        print(json.dumps([asdict(p) for p in profiles], indent=2, ensure_ascii=False))
        return 0

    for p in profiles:
        print(f"\n=== {p.path} ===")
        print(f"  codec            {p.codec} in {p.container}")
        print(f"  duration         {p.duration_sec:.1f} s ({p.duration_sec / 60:.1f} min)")
        print(f"  declared rate    {p.declared_sample_rate} Hz, {p.channels} channel(s)")
        if p.effective_bandwidth_hz is not None:
            tag = "NARROWBAND" if p.is_narrowband else "wideband"
            print(f"  measured content up to {p.effective_bandwidth_hz:.0f} Hz  [{tag}]")
        if p.mean_volume_db is not None:
            print(f"  loudness         mean {p.mean_volume_db} dB, peak {p.max_volume_db} dB")
        if p.channel_relation:
            cr = p.channel_relation
            print(f"  channels         {cr.verdict} (correlation {cr.correlation}, "
                  f"RMS {cr.rms_db_per_channel} dB)")
        for w in p.warnings:
            print(f"  [!] {w}")
        for r in p.recommendations:
            print(f"  [>] {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
