"""Bring every speech segment to a common level before any model sees it.

The recordings arrive as 8-bit PCM at 11025 Hz. Eight bits fixes the noise
floor for the whole file at -52.9 dBFS, and it does not move when a talker
does. Measured on one real call, per speech segment:

    loudest decile   -16.5 dBFS   SNR 36.4 dB   ~6.1 effective bits
    median           -21.1 dBFS   SNR 31.8 dB   ~5.3 effective bits
    quietest decile  -36.7 dBFS   SNR 16.2 dB   ~2.7 effective bits

The quietest span in that call touches 7 of the 256 available levels. The
loudest touches 217. Both were fed to models trained on 16-bit audio.

That 20 dB spread is the problem this module addresses, and it is worth being
exact about what it can and cannot do.

It does NOT restore information. The bits were destroyed at capture and no
gain undoes that; a segment holding 7 distinct levels still holds 7 after
scaling. What it fixes is the mismatch: a model handed speech at -50 dBFS is
being shown something unlike anything in its training data, and neural
networks degrade badly off-distribution even when the underlying signal would
have been sufficient. Presenting speech at a familiar level is distribution
matching, and it helps even though the signal itself did not improve.

For speaker attribution the argument is sharper. ECAPA compares segments to
each other, so a systematic 20 dB offset between the two parties is the
loudest single difference in the data - louder than the difference between
their voices. It invites the embedding to encode the channel rather than the
person, which is exactly the failure measured here: same-speaker distance
0.72 against a same-versus-different gap of only 0.13. Removing the offset
does not add voice information, but it stops the recording level from
drowning what little there is.

Deliberately not noise reduction, which was tested on this estate and made
things worse: no spectral processing, no filtering, nothing invented. Each
segment is multiplied by one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# -20 dBFS RMS is the conventional level for speech corpora and roughly where
# these recordings already sit at their median, so the median segment barely
# moves and only the outliers are pulled in.
TARGET_DBFS = -20.0

# A ceiling on gain. A segment 30 dB down is mostly quantisation noise, and
# amplifying it to target would present the model with loud noise wearing the
# shape of speech - worse than quiet speech, because it looks confident.
MAX_GAIN_DB = 18.0
MAX_CUT_DB = 12.0

# Below this a "segment" is a click or a breath, not speech worth levelling.
MIN_SEGMENT_SEC = 0.20

# Crossfade across segment joins. A gain step at a boundary is a click, and a
# click is a broadband transient that every one of these models reads as an
# event. 30 ms is inaudible and far shorter than any phoneme.
FADE_SEC = 0.030


@dataclass
class LevelReport:
    audio: np.ndarray
    adjusted: int
    skipped: int
    gain_db: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def spread_before(self) -> float:
        return self._spread(self._before)

    _before: list[float] = field(default_factory=list)

    @staticmethod
    def _spread(vals: list[float]) -> float:
        if not vals:
            return 0.0
        return float(np.percentile(vals, 90) - np.percentile(vals, 10))


def _db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 20.0 * float(np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def level_by_speaker(
    audio: np.ndarray,
    turns: list[dict],
    sample_rate: int,
    *,
    target_dbfs: float = TARGET_DBFS,
    max_gain_db: float = MAX_GAIN_DB,
    max_cut_db: float = MAX_CUT_DB,
) -> LevelReport:
    """Give each speaker one gain, so the parties match and nothing pumps.

    Levelling per VAD span was the wrong unit and measurably hurt. A VAD span
    is a stretch of speech, not a turn: it routinely contains both parties, so
    normalising it equalises nothing between them. Worse, spans carry 250 ms
    of padding at each end, and lifting near-silence by up to 18 dB puts a
    burst of amplified quantisation noise exactly where the diarizer is
    deciding turn boundaries.

    A speaker is the right unit. The measured problem was a 20 dB offset
    *between the parties*, and one gain per speaker removes exactly that while
    leaving the dynamics inside a speaker's own speech untouched - a person
    getting louder when they are annoyed is information, not a defect.

    Turns come from a first diarization pass, which is unreliable at boundaries
    - that is the whole problem here. It does not need to be reliable: the gain
    is the median over all of a speaker's turns, so a handful of misattributed
    seconds moves it very little.
    """
    if not turns:
        return LevelReport(audio, 0, 0, notes=["no turns; audio untouched"])

    by_speaker: dict[str, list[float]] = {}
    for t in turns:
        i0 = max(0, int(float(t["start"]) * sample_rate))
        i1 = min(audio.size, int(float(t["end"]) * sample_rate))
        if i1 - i0 < int(MIN_SEGMENT_SEC * sample_rate):
            continue
        by_speaker.setdefault(t["speaker"], []).append(_db(audio[i0:i1]))

    if len(by_speaker) < 2:
        return LevelReport(audio, 0, 0,
                           notes=["only one speaker found; nothing to balance"])

    medians = {spk: float(np.median(v)) for spk, v in by_speaker.items()}
    gain_of = {spk: float(np.clip(target_dbfs - m, -max_cut_db, max_gain_db))
               for spk, m in medians.items()}

    out = audio.astype(np.float32, copy=True)
    fade = max(1, int(FADE_SEC * sample_rate))
    adjusted = 0
    for t in turns:
        g_db = gain_of.get(t["speaker"], 0.0)
        if abs(g_db) < 0.5:
            continue
        i0 = max(0, int(float(t["start"]) * sample_rate))
        i1 = min(out.size, int(float(t["end"]) * sample_rate))
        if i1 <= i0:
            continue
        gain = 10.0 ** (g_db / 20.0)
        window = np.full(i1 - i0, gain, dtype=np.float32)
        edge = min(fade, (i1 - i0) // 2)
        if edge > 0:
            ramp = np.linspace(1.0, gain, edge, dtype=np.float32)
            window[:edge] = ramp
            window[-edge:] = ramp[::-1]
        out[i0:i1] *= window
        adjusted += 1

    peak = float(np.abs(out).max())
    if peak > 0.999:
        out *= 0.999 / peak

    before = list(medians.values())
    after = [m + gain_of[s] for s, m in medians.items()]
    rep = LevelReport(out, adjusted, 0, list(gain_of.values()))
    rep._before = before
    rep.notes.append(
        "per-speaker gain: "
        + ", ".join(f"{s}={medians[s]:.1f}->{medians[s]+gain_of[s]:.1f} dBFS"
                    for s in sorted(medians))
        + f"; between-speaker spread {max(before)-min(before):.1f} dB -> "
          f"{max(after)-min(after):.1f} dB")
    return rep


def level_speech(
    audio: np.ndarray,
    spans: list[tuple[float, float]],
    sample_rate: int,
    *,
    target_dbfs: float = TARGET_DBFS,
    max_gain_db: float = MAX_GAIN_DB,
    max_cut_db: float = MAX_CUT_DB,
) -> LevelReport:
    """Scale each speech span toward `target_dbfs`, leaving silence alone.

    Gaps between spans keep their original samples: they carry no speech, and
    amplifying them would raise the noise floor into the range where the VAD
    and the diarizer start finding things that are not there.
    """
    if not spans:
        return LevelReport(audio, 0, 0, notes=["no speech spans; audio untouched"])

    out = audio.astype(np.float32, copy=True)
    n = out.size
    fade = max(1, int(FADE_SEC * sample_rate))

    before: list[float] = []
    gains: list[float] = []
    adjusted = skipped = 0

    for start, end in spans:
        i0, i1 = max(0, int(start * sample_rate)), min(n, int(end * sample_rate))
        if i1 - i0 < int(MIN_SEGMENT_SEC * sample_rate):
            skipped += 1
            continue

        seg_db = _db(audio[i0:i1])
        before.append(seg_db)
        gain_db = float(np.clip(target_dbfs - seg_db, -max_cut_db, max_gain_db))
        # Recorded for every measured span, including the ones left alone -
        # otherwise `before` and `gains` describe different sets of spans and
        # the reported spread is nonsense. It read 21.9 -> 29.6 dB on a file
        # the processing had in fact pulled together.
        gains.append(gain_db if abs(gain_db) >= 0.5 else 0.0)
        if abs(gain_db) < 0.5:
            continue

        gain = 10.0 ** (gain_db / 20.0)
        # Ramp the gain in and out so the joins with untouched silence are
        # smooth rather than stepped.
        window = np.full(i1 - i0, gain, dtype=np.float32)
        edge = min(fade, (i1 - i0) // 2)
        if edge > 0:
            ramp = np.linspace(1.0, gain, edge, dtype=np.float32)
            window[:edge] = ramp
            window[-edge:] = ramp[::-1]
        out[i0:i1] *= window
        adjusted += 1

    peak = float(np.abs(out).max())
    if peak > 0.999:
        trim = 20 * np.log10(peak / 0.999)
        out *= 0.999 / peak
        gains = [g - trim for g in gains]

    rep = LevelReport(out, adjusted, skipped, gains)
    rep._before = before
    after = [b + g for b, g in zip(before, gains)] if gains else before
    rep.notes.append(
        f"levelled {adjusted} speech span(s) to {target_dbfs:.0f} dBFS; "
        f"level spread {rep._spread(before):.1f} dB -> {rep._spread(after):.1f} dB"
    )
    if skipped:
        rep.notes.append(f"{skipped} span(s) shorter than "
                         f"{MIN_SEGMENT_SEC}s left untouched")
    return rep
