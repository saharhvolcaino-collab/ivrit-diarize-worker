"""Pull word timestamps back onto the speech they actually name.

Whisper derives word boundaries from decoder cross-attention, not from the
waveform, and the alignment has no notion of whether a moment contains sound.
Across a pause it stretches: the word before a gap keeps running, or the word
after one starts early, until the two meet somewhere in the silence.

On this recording estate that is not a rounding error. The telephony chain
suppresses comfort noise, so gaps between turns are digital silence - exactly
0.0 samples - and they are frequent. One measured case, the opening of a
conference call:

    words as Whisper reported them        what the audio contains
    2.18-2.90  "מדבר תומר,"               2.40-3.00  speech
    3.00-4.26  "שלום"                     3.00-4.00  silence, -99 dBFS
    4.37-4.85  "תומר,"                    4.10-4.90  speech

"שלום" is spoken at 4.1, not 3.0. The reported gap between the two turns is
0.10s; the real one is 1.1s. Two consequences, both of which we were living
with:

  - Turn splitting keys off gaps of a second or more, so a plain speaker
    change was invisible and both parties landed in one turn.
  - Words are attributed by time overlap against the diarizer's segments, so
    a second of silence was voting on who spoke.

Snapping costs one intersection per word against the VAD's speech spans, and
it happens before any diarizer sees the words, so every engine benefits from
it equally.
"""

from __future__ import annotations

from bisect import bisect_left


def snap_words_to_speech(
    words: list[dict],
    spans: list[tuple[float, float]],
    *,
    max_shift: float = 2.0,
) -> tuple[list[dict], int]:
    """Clip each word to the speech it overlaps, and report how many moved.

    A word overlapping speech is trimmed to that overlap - silence padding on
    either side is removed and nothing is invented. A word lying entirely in
    silence is moved to the nearest speech edge, but only if that edge is
    within max_shift seconds; beyond that the timestamp is too far gone to
    repair by guessing, and leaving it visibly wrong is better than moving it
    somewhere confidently wrong.

    `spans` must be sorted and non-overlapping, which is what a VAD returns.
    """
    if not spans or not words:
        return words, 0

    starts = [s for s, _ in spans]
    out: list[dict] = []
    moved = 0

    for w in words:
        ws, we = float(w["start"]), float(w["end"])

        # The last span starting at or before this word, and its neighbour -
        # the only two that can overlap it once spans are disjoint and sorted.
        i = max(0, bisect_left(starts, ws) - 1)
        best: tuple[float, float] | None = None
        best_overlap = 0.0
        for s, e in spans[i:i + 3]:
            if s >= we:
                break
            overlap = min(we, e) - max(ws, s)
            if overlap > best_overlap:
                best_overlap, best = overlap, (max(ws, s), min(we, e))

        if best is not None:
            new_s, new_e = best
        else:
            # Entirely inside a gap. Snap to whichever speech edge is closer.
            mid = (ws + we) / 2
            near = min(spans, key=lambda sp: min(abs(mid - sp[0]), abs(mid - sp[1])))
            dist = min(abs(mid - near[0]), abs(mid - near[1]))
            if dist > max_shift:
                out.append(w)
                continue
            width = max(we - ws, 0.05)
            if abs(mid - near[0]) <= abs(mid - near[1]):
                new_s = near[0]
                new_e = min(near[1], near[0] + width)
            else:
                new_e = near[1]
                new_s = max(near[0], near[1] - width)

        if new_e <= new_s:                       # degenerate after clipping
            out.append(w)
            continue
        if abs(new_s - ws) > 0.02 or abs(new_e - we) > 0.02:
            moved += 1
        out.append({**w, "start": round(new_s, 3), "end": round(new_e, 3)})

    return out, moved
