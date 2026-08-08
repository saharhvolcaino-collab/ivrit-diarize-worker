"""Two models, one audio: agreement marks what is certain, and what is not.

Running the same decode twice teaches nothing here. Decoding is pinned -
temperature 0, fixed beam - precisely so a file transcribes the same way
every time, and two identical passes differ only by GPU noise.

Two *different* models is another matter. ivrit-ai publish a full
whisper-large-v3 and a turbo distillation, trained on the same Hebrew data
but with different capacity, so they fail differently. Where they produce the
same word, that word is about as certain as this pipeline can make it. Where
they diverge, something is wrong at that spot - and knowing precisely where
the transcript is unreliable is worth more than a slightly better guess,
because it can be reviewed, re-run, or shown to a human.

This is the classic ROVER idea reduced to two systems: align the outputs, keep
the agreements, and arbitrate the rest. Arbitration is by the decoder's own
word probability, which is a weak signal - hallucinated segments have been
observed reporting 0.000 no-speech probability on this estate, so confidence
is not trusted to *find* errors. It is only used to break a tie once the
disagreement has already been located by two models differing, which is a
much narrower job.

Nothing is invented: every output word came from one of the two passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Hebrew final forms, so "שלום." and "שלום" compare equal, and so do the two
# spellings a model may pick for the same spoken word.
_PUNCT = re.compile(r"[^\w֐-׿]+")


@dataclass
class ConsensusReport:
    words: list[dict]
    agreed: int
    disagreed: int
    only_a: int
    only_b: int
    disagreements: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def agreement_pct(self) -> float:
        total = self.agreed + self.disagreed + self.only_a + self.only_b
        return round(100.0 * self.agreed / total, 1) if total else 0.0


def _norm(word: str) -> str:
    return _PUNCT.sub("", (word or "").strip()).lower()


def _mean_prob(words: list[dict]) -> float:
    vals = [float(w.get("probability") or 0.0) for w in words]
    return sum(vals) / len(vals) if vals else 0.0


def merge(a: list[dict], b: list[dict], *,
          label_a: str = "primary", label_b: str = "secondary",
          arbitrate: str = "primary") -> ConsensusReport:
    """Align two word streams and mark every word as agreed or contested.

    `arbitrate` decides what a contested span becomes in the output:

      "primary"     keep the stronger model's reading (default)
      "probability" keep whichever side the decoder was more confident about

    Probability is not the default, and a test is the reason. On a
    reconstruction of a real error from this estate - "נסגרו" misheard as
    "נפגרו" - the wrong word carried the higher confidence (0.80 against
    0.55), so arbitrating on it would have installed the error and marked it
    resolved. Whisper's confidence has already been caught reporting 0.000
    no-speech probability across five minutes of pure hallucination here; it
    is not evidence.

    So both readings are always recorded in `disagreements`, whichever one is
    placed in the text. Locating the uncertain spans is the reliable half of
    this technique; resolving them needs a reader that understands the
    sentence, which is what the correction layer is for.
    """
    if not a:
        return ConsensusReport(b, 0, 0, 0, len(b),
                               notes=[f"{label_a} produced nothing; kept {label_b}"])
    if not b:
        return ConsensusReport(a, 0, 0, len(a), 0,
                               notes=[f"{label_b} produced nothing; kept {label_a}"])

    ta = [_norm(w.get("word", "")) for w in a]
    tb = [_norm(w.get("word", "")) for w in b]
    sm = SequenceMatcher(a=ta, b=tb, autojunk=False)

    out: list[dict] = []
    agreed = disagreed = only_a = only_b = 0
    diffs: list[dict] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out.append({**a[i1 + k], "agreement": True})
            agreed += i2 - i1
            continue

        seg_a, seg_b = a[i1:i2], b[j1:j2]
        # Timestamps always come from the primary pass where it has any, so a
        # tie broken in favour of the other model cannot drag the timeline
        # somewhere the diarizer will not recognise.
        if tag == "replace":
            pick_b = (arbitrate == "probability"
                      and _mean_prob(seg_b) > _mean_prob(seg_a) + 0.05)
            chosen = seg_b if pick_b else seg_a
            disagreed += max(len(seg_a), len(seg_b))
        elif tag == "delete":                       # only the primary has it
            chosen, pick_b = seg_a, False
            only_a += len(seg_a)
        else:                                       # insert - only the second
            chosen, pick_b = seg_b, True
            only_b += len(seg_b)

        chosen_out = [{**w, "agreement": False,
                       "source": label_b if pick_b else label_a}
                      for w in chosen]
        out.extend(chosen_out)

        ref = seg_a or seg_b
        diffs.append({
            "start": round(float(ref[0]["start"]), 2),
            "end": round(float(ref[-1]["end"]), 2),
            label_a: " ".join(w.get("word", "") for w in seg_a),
            label_b: " ".join(w.get("word", "") for w in seg_b),
            "kept": label_b if pick_b else label_a,
            "prob_a": round(_mean_prob(seg_a), 3),
            "prob_b": round(_mean_prob(seg_b), 3),
            # Working state for a later vote, stripped before the response
            # leaves the worker. References to the emitted dicts themselves,
            # because merge() sorts the output at the end - an index recorded
            # here would silently point at the wrong words afterwards, while
            # object identity survives any reordering.
            "_words_a": seg_a,
            "_words_b": seg_b,
            "_out_words": chosen_out,
        })

    out.sort(key=lambda w: float(w.get("start") or 0.0))
    rep = ConsensusReport(out, agreed, disagreed, only_a, only_b, diffs)
    rep.notes.append(
        f"{label_a} and {label_b} agree on {rep.agreement_pct}% of words "
        f"({agreed} agreed, {disagreed} contested, "
        f"{only_a} only in {label_a}, {only_b} only in {label_b})")
    return rep
