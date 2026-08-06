"""Stage 1 - convert a recording into what the models expect, and nothing more.

The deliberate design choice here is restraint. Three independent 2025-26
studies found that neural speech enhancement makes Whisper's error rate WORSE,
not better - in one, enhanced audio lost to raw noisy audio in all 40 tested
configurations. The mechanism is that Whisper learned its noise robustness from
680k hours of unprocessed web audio, so "cleaning" the signal moves it out of
the distribution the model knows.

The same argument rules out bandwidth extension for our 8 kHz telephone audio:
every model that invents the missing octave is generative, and inventing the
wrong fricative is worse than having none.

So this stage resamples, downmixes, and stops. The only judgement call it makes
is gain, and even that is pure linear scaling.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

TARGET_SR = 16000
# Leave a sliver of headroom so downstream float conversion cannot overshoot.
PEAK_TARGET_DB = -1.0
# Below this, raising gain amplifies quantisation noise more than it helps.
MIN_USEFUL_GAIN_DB = 0.5
MAX_GAIN_DB = 20.0


class PreprocessError(RuntimeError):
    pass


@dataclass
class PreprocessResult:
    source: str
    output: str
    source_sample_rate: int
    output_sample_rate: int
    gain_applied_db: float
    duration_sec: float
    output_bytes: int
    opus_path: str | None
    opus_bytes: int | None
    actions: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise PreprocessError("ffmpeg not found on PATH")
    return exe


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreprocessError(f"ffmpeg failed: {proc.stderr.strip()[:600]}")
    return proc


def _measure_peak_db(path: Path) -> float:
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in proc.stderr.splitlines():
        if "max_volume:" in line:
            return float(line.split("max_volume:")[1].split("dB")[0].strip())
    return 0.0


def _probe_rate_and_duration(path: Path) -> tuple[int, float]:
    exe = shutil.which("ffprobe")
    if not exe:
        raise PreprocessError("ffprobe not found on PATH")
    proc = subprocess.run(
        [exe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    data = json.loads(proc.stdout or "{}")
    sr = int(data.get("streams", [{}])[0].get("sample_rate") or 0)
    dur = float(data.get("format", {}).get("duration") or 0.0)
    return sr, dur


def preprocess(
    source: str | Path,
    out_dir: str | Path = "out",
    make_opus: bool = True,
    normalise: bool = True,
) -> PreprocessResult:
    """Produce a 16 kHz mono WAV for inference, plus an OPUS copy for transport.

    The WAV is what the models consume. The OPUS copy exists because RunPod caps
    request payloads at 10 MB, and OPUS at 32 kbps carries an hour of speech in
    about 14 MB - small enough that short calls can go inline instead of via
    object storage. Longer files still need a presigned URL.
    """
    source = Path(source)
    if not source.exists():
        raise PreprocessError(f"File not found: {source}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    wav_out = out_dir / f"{stem}.16k.wav"

    src_rate, duration = _probe_rate_and_duration(source)
    actions: list[str] = []

    # soxr at high precision costs nothing and avoids the resampler being the
    # weakest link. Whisper's own reference loader uses ffmpeg's default swr;
    # no published study isolates the difference, so treat this as cheap
    # insurance rather than a measured win.
    # No dither. Dither is deliberately random noise, which makes every run of
    # the pipeline produce a slightly different file - VAD boundaries jittered
    # by a span between runs and everything downstream shifted with them. For
    # speech analysis the theoretical quantisation benefit is irrelevant;
    # reproducibility is not.
    filters = [f"aresample=resampler=soxr:precision=28:dither_method=none:osr={TARGET_SR}"]
    actions.append(f"resample {src_rate} -> {TARGET_SR} Hz (soxr, no dither)")

    gain_db = 0.0
    if normalise:
        peak = _measure_peak_db(source)
        candidate = PEAK_TARGET_DB - peak
        if candidate > MIN_USEFUL_GAIN_DB:
            gain_db = min(candidate, MAX_GAIN_DB)
            filters.append(f"volume={gain_db:.2f}dB")
            actions.append(
                f"peak normalise: {peak:.1f} dB -> {PEAK_TARGET_DB} dB "
                f"(+{gain_db:.2f} dB linear gain)"
            )
        else:
            actions.append(
                f"peak already at {peak:.1f} dB - no gain applied "
                "(raising it would only clip)"
            )

    _run([
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-af", ",".join(filters),
        "-ac", "1", "-c:a", "pcm_s16le", "-ar", str(TARGET_SR),
        str(wav_out),
    ])
    actions.append("downmix to mono, 16-bit PCM")

    opus_path = None
    opus_bytes = None
    if make_opus:
        opus_out = out_dir / f"{stem}.opus"
        # 24 kbps mono is transparent for narrowband speech - the source here is
        # already band-limited well below what the codec preserves at this rate.
        _run([
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(wav_out),
            "-c:a", "libopus", "-b:a", "24k", "-vbr", "on",
            "-application", "voip", "-ac", "1",
            str(opus_out),
        ])
        opus_path = str(opus_out)
        opus_bytes = opus_out.stat().st_size
        actions.append(f"OPUS 24 kbps copy for transport ({opus_bytes / 1024:.0f} KB)")

    return PreprocessResult(
        source=str(source),
        output=str(wav_out),
        source_sample_rate=src_rate,
        output_sample_rate=TARGET_SR,
        gain_applied_db=round(gain_db, 2),
        duration_sec=round(duration, 2),
        output_bytes=wav_out.stat().st_size,
        opus_path=opus_path,
        opus_bytes=opus_bytes,
        actions=actions,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Prepare a recording for transcription (resample only - no enhancement)."
    )
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out", help="Output directory")
    p.add_argument("--no-opus", action="store_true", help="Skip the OPUS transport copy")
    p.add_argument("--no-normalise", action="store_true", help="Skip peak normalisation")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    results = []
    for item in args.audio:
        try:
            results.append(preprocess(
                item, args.out,
                make_opus=not args.no_opus,
                normalise=not args.no_normalise,
            ))
        except PreprocessError as exc:
            print(f"ERROR {item}: {exc}")
            return 1

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        return 0

    for r in results:
        print(f"\n=== {Path(r.source).name} ===")
        for a in r.actions:
            print(f"  - {a}")
        print(f"  -> {r.output}  ({r.output_bytes / 1024 / 1024:.2f} MB)")
        if r.opus_path:
            inline_ok = (r.opus_bytes * 4 / 3) < 10 * 1024 * 1024
            verdict = "fits inline in a RunPod /run payload" if inline_ok else \
                      "too large for an inline payload - use a presigned URL"
            print(f"  -> {r.opus_path}  ({r.opus_bytes / 1024:.0f} KB, {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
