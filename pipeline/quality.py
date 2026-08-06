"""Score a transcript for the failures this audio actually produces.

Whisper does not fail by returning nothing. It fails by returning fluent,
confident text that was never spoken - on our sparse recording it filled five
minutes of hold time with Knesset boilerplate, at 0.000 reported no-speech
probability. Anything that reads the model's own confidence will miss that.

These checks look at the shape of the output instead, and they are chosen so
that hallucination makes them worse rather than better:

  repeated lines     a phrase emitted verbatim several times is the signature
                     of the decoder conditioning on its own invention
  stretched turns    speech runs 2-3 words per second; a turn far below that
                     was narrated over silence
  excess speakers    a two-party call producing five means segments with no
                     real voice got scattered across whatever the diarizer had
  buried answers     a question mark with substantial text after it, inside one
                     turn, is a reply that got absorbed into the asker's turn -
                     the clearest evidence that two people were merged into one
  low confidence     weak on its own, but corroborating alongside the others

The buried-answer check exists because the others missed a real failure. One
call came back as a clean-looking two-speaker transcript in which almost every
exchange - "what is your name?" followed by "Mendy" - sat inside a single
speaker's turn. Speaker count was right, confidence was high, and the transcript
was still wrong about who said what.

Word count is deliberately absent as a quality signal - it rewards exactly the
behaviour we are trying to detect.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict, field

# Conversational Hebrew sits near 2-3 words/sec. Below this a turn is mostly
# silence with text laid over it.
MIN_WORDS_PER_SEC = 0.8
MIN_TURN_SEC_TO_JUDGE = 2.0
LOW_CONFIDENCE = 0.5

# Words after a mid-turn question mark before we call it a buried reply. A
# couple of words can be the speaker qualifying their own question; a full
# clause is somebody answering.
MIN_WORDS_AFTER_QUESTION = 3


@dataclass
class QualityReport:
    turns: int
    words: int
    speakers: int
    expected_speakers: int | None
    mean_confidence: float
    low_confidence_words: int
    low_confidence_pct: float
    repeated_lines: int
    repeated_examples: list[str]
    stretched_turns: int
    stretched_examples: list[str]
    buried_answers: int
    buried_examples: list[str]
    dominant_speaker_pct: float
    words_per_speech_sec: float
    verdict: str = "unknown"
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def assess(transcript: dict, expected_speakers: int | None = 2) -> QualityReport:
    turns = transcript.get("turns", [])
    words = [w for t in turns for w in t.get("words", [])]
    confidences = [w["probability"] for w in words if w.get("probability") is not None]

    # Only multi-word lines count as suspicious repetition. Backchannels -
    # "כן.", "לא.", "אוקיי." - repeat constantly in real conversation; the
    # hallucination signature is a whole PHRASE coming back verbatim.
    texts = [
        t["text"].strip() for t in turns
        if t.get("text", "").strip() and len(t["text"].split()) >= 3
    ]
    counts = Counter(texts)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    repeated_examples = [t for t, c in counts.most_common(3) if c > 1]

    stretched: list[str] = []
    for t in turns:
        duration = t["end"] - t["start"]
        n = len(t.get("words", []))
        if duration >= MIN_TURN_SEC_TO_JUDGE and n / max(duration, 1e-6) < MIN_WORDS_PER_SEC:
            # Reading a phone number aloud is naturally slow - digit-heavy
            # turns are dictation, not silence narrated over.
            text = t.get("text", "")
            digits = sum(ch.isdigit() for ch in text)
            if digits >= max(4, len(text.replace(" ", "")) // 3):
                continue
            stretched.append(f"{text[:40]} ({n} words over {duration:.0f}s)")

    # A question followed by a full clause, inside one turn, is the other party
    # answering - absorbed because the diarizer never placed a boundary there.
    buried: list[str] = []
    for t in turns:
        text = t.get("text", "")
        for match in re.finditer(r"\?", text):
            tail = text[match.end():].strip()
            if len(tail.split()) >= MIN_WORDS_AFTER_QUESTION:
                q_start = max(0, match.start() - 30)
                buried.append(f"{text[q_start:match.end()].strip()} >> {tail[:30]}")
                break

    speech = sum(t["end"] - t["start"] for t in turns) or 1.0
    low = sum(1 for c in confidences if c < LOW_CONFIDENCE)

    share: dict[str, float] = {}
    for t in turns:
        share[t["speaker"]] = share.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    dominant = 100 * max(share.values()) / sum(share.values()) if share else 0.0

    report = QualityReport(
        turns=len(turns),
        words=len(words),
        speakers=len({t["speaker"] for t in turns}),
        expected_speakers=expected_speakers,
        mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        low_confidence_words=low,
        low_confidence_pct=round(100 * low / max(len(words), 1), 1),
        repeated_lines=repeated,
        repeated_examples=repeated_examples[:3],
        stretched_turns=len(stretched),
        stretched_examples=stretched[:3],
        buried_answers=len(buried),
        buried_examples=buried[:3],
        dominant_speaker_pct=round(dominant, 1),
        words_per_speech_sec=round(len(words) / speech, 2),
    )

    if expected_speakers and report.speakers > expected_speakers:
        report.problems.append(
            f"{report.speakers} speakers on a {expected_speakers}-party call. Segments "
            "with no real voice tend to get scattered across extra speakers, so this "
            "usually points at invented text rather than at the diarizer alone."
        )
    if expected_speakers and report.speakers < expected_speakers:
        report.problems.append(
            f"Only {report.speakers} speaker(s) found where {expected_speakers} were "
            "expected - one party is being merged into the other."
        )
    if repeated:
        report.problems.append(
            f"{repeated} repeated line(s), e.g. \"{repeated_examples[0][:50]}\". Verbatim "
            "repetition is the decoder conditioning on its own output; lower the VAD "
            "threshold's tolerance for silence or verify condition_on_previous_text is off."
        )
    if stretched:
        report.problems.append(
            f"{len(stretched)} turn(s) far below conversational speed, e.g. "
            f"\"{stretched[0]}\". These spans are mostly silence with text laid over them."
        )
    # Judge buried answers by rate, not count. Thirty-four buried exchanges in
    # a 345-turn call is the same 10% failure as one in ten - but an absolute
    # threshold branded every long call unreliable while short calls with the
    # same defect rate passed.
    buried_rate = len(buried) / max(len(turns), 1)
    if buried:
        severity = "widespread" if buried_rate >= 0.10 else "isolated"
        report.problems.append(
            f"{len(buried)} of {len(turns)} turns ({100 * buried_rate:.0f}%) contain "
            f"a question and its answer ({severity}), e.g. \"{buried[0][:70]}\". "
            "The diarizer merged the parties at these points."
        )
    # Imbalance alone proves nothing: one of our samples runs 88/12 and was
    # confirmed correct by ear. It only becomes evidence when several exchanges
    # are also buried, so a single buried answer must not trigger this.
    if len(buried) >= 3 and report.dominant_speaker_pct > 85:
        report.problems.append(
            f"One speaker holds {report.dominant_speaker_pct}% of the talk time. "
            "Together with the buried answers above, this points at both parties "
            "being collapsed into one label rather than a genuinely one-sided call."
        )
    if report.low_confidence_pct > 10:
        report.problems.append(
            f"{report.low_confidence_pct}% of words below {LOW_CONFIDENCE} confidence."
        )

    # Special shapes first: an empty or near-empty result is its own category
    # (a 0.24 s butt-dial is not "unreliable", it is an empty recording), and a
    # genuinely one-sided recording (voicemail, IVR message) should say so
    # rather than scream about a missing second speaker.
    if report.words < 5:
        report.verdict = "empty"
        report.problems = ["Recording contains little or no speech - nothing to judge."]
        return report

    if (expected_speakers and report.speakers == 1 and report.buried_answers == 0
            and report.words < 120):
        report.verdict = "one-sided"
        report.problems = [
            "Short single-voice recording - likely a voicemail or message rather "
            "than a failed two-party call. Verify by ear if it matters."
        ]
        return report

    # Rate-aware verdict: isolated artefacts make a file worth a glance,
    # widespread ones make it untrustworthy.
    severe = (
        (buried_rate >= 0.10 and len(buried) >= 3)
        or (expected_speakers and report.speakers > expected_speakers)
        or (len(buried) >= 3 and report.dominant_speaker_pct > 85)
        or report.repeated_lines >= 3
    )
    report.verdict = "clean" if not report.problems else (
        "unreliable" if severe else "suspect"
    )
    return report


def format_report(r: QualityReport) -> str:
    lines = [
        f"  quality: {r.verdict.upper()}",
        f"    {r.turns} turns · {r.words} words · {r.speakers} speakers · "
        f"{r.words_per_speech_sec} words/sec",
        f"    confidence {r.mean_confidence:.3f}, {r.low_confidence_words} weak words "
        f"({r.low_confidence_pct}%)",
    ]
    for p in r.problems:
        lines.append(f"    [!] {p}")
    return "\n".join(lines)
