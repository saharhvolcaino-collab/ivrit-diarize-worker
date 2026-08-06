"""Fold away spurious speakers when the real count is known.

Diarizers on this audio produce a recurring artefact: two solid speakers plus a
third holding a second or two of speech. On a call we know has two parties,
that third label is fragments of one of the real two - a moment of crosstalk, a
short backchannel, a boundary the clusterer placed badly.

Tuning the VAD threshold to make it disappear does not work. On our two sample
recordings the thresholds that give exactly two speakers are different for each
file, so any single setting breaks one of them. That is a sign the knob is the
wrong one: the artefact comes from the diarizer, and it should be fixed where
it happens.

Since the speaker count is known ahead of time for call-centre audio, the
minor labels can be reassigned to whichever real speaker they sit next to. This
is a heuristic, so it refuses to run when the extra speaker is large enough to
be real - hiding a genuine third party, or a diarizer that has split someone
down the middle, would be worse than leaving the artefact visible.
"""

from __future__ import annotations

from dataclasses import dataclass

# Above this share of total speech, an "extra" speaker is not an artefact.
# Merging it would hide either a real third party or a diarizer failure that
# should be surfaced rather than smoothed over.
MAX_ARTEFACT_SHARE = 0.15


@dataclass
class CollapseReport:
    applied: bool
    reason: str
    merged: dict[str, str]          # dropped label -> label it was folded into
    merged_seconds: float
    speakers_before: int
    speakers_after: int


def collapse_to_expected(
    turns: list[dict],
    expected: int = 2,
    max_artefact_share: float = MAX_ARTEFACT_SHARE,
) -> tuple[list[dict], CollapseReport]:
    """Reassign minor speaker labels to the nearest major one.

    Each turn belonging to a dropped label goes to whichever kept speaker is
    closest in time - the neighbour it interrupts is almost always the person
    who was actually talking.
    """
    if not turns:
        return turns, CollapseReport(False, "no turns", {}, 0.0, 0, 0)

    share: dict[str, float] = {}
    for t in turns:
        share[t["speaker"]] = share.get(t["speaker"], 0.0) + (t["end"] - t["start"])

    before = len(share)
    if before <= expected:
        return turns, CollapseReport(
            False, f"{before} speakers, nothing to collapse", {}, 0.0, before, before
        )

    total = sum(share.values()) or 1.0
    ranked = sorted(share.items(), key=lambda kv: -kv[1])
    keep = {name for name, _ in ranked[:expected]}
    drop = [(name, secs) for name, secs in ranked[expected:]]

    biggest_drop_share = max(secs / total for _, secs in drop)
    if biggest_drop_share > max_artefact_share:
        return turns, CollapseReport(
            applied=False,
            reason=(
                f"extra speaker holds {biggest_drop_share:.0%} of speech, above the "
                f"{max_artefact_share:.0%} artefact ceiling - this looks like a real "
                "speaker or a genuine diarization failure, so it is left alone"
            ),
            merged={}, merged_seconds=0.0,
            speakers_before=before, speakers_after=before,
        )

    kept_indices = [i for i, t in enumerate(turns) if t["speaker"] in keep]
    merged: dict[str, str] = {}
    merged_seconds = 0.0
    out = [dict(t) for t in turns]

    for i, turn in enumerate(out):
        if turn["speaker"] in keep:
            continue

        mid = (turn["start"] + turn["end"]) / 2
        nearest, best = None, None
        for j in kept_indices:
            other = out[j]
            gap = 0.0 if other["start"] <= mid <= other["end"] else min(
                abs(mid - other["end"]), abs(other["start"] - mid)
            )
            if best is None or gap < best:
                best, nearest = gap, other["speaker"]

        if nearest:
            merged[turn["speaker"]] = nearest
            merged_seconds += turn["end"] - turn["start"]
            turn["speaker"] = nearest
            for w in turn.get("words", []):
                w["speaker"] = nearest

    # Merging can leave adjacent turns with the same speaker; rejoin them so the
    # transcript still reads as continuous speech.
    joined: list[dict] = []
    for turn in out:
        if joined and joined[-1]["speaker"] == turn["speaker"] and \
                turn["start"] - joined[-1]["end"] <= 1.0:
            prev = joined[-1]
            prev["end"] = max(prev["end"], turn["end"])
            prev["text"] = f"{prev['text']} {turn['text']}".strip()
            prev.setdefault("words", []).extend(turn.get("words", []))
        else:
            joined.append(turn)

    return joined, CollapseReport(
        applied=True,
        reason=f"folded {len(merged)} minor label(s) into the {expected} main speakers",
        merged=merged,
        merged_seconds=round(merged_seconds, 2),
        speakers_before=before,
        speakers_after=len({t["speaker"] for t in joined}),
    )
