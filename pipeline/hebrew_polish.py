"""Grammar-only Hebrew polish for finished transcripts. No lexicon.

Three small tools, all operating on the transcript AFTER every recognition
decision has been made - nothing here feeds back into ASR, so the no-word-
banks constraint is untouched. Each rule below is a closed grammatical fact
of Hebrew (function words, inflection morphology), not a vocabulary list.

  strip_marks      remove directional formatting characters (LRM/RLM,
                   embeddings, isolates, BOM). Editors and players render
                   them as stray glyphs inside RTL text.
  question_marks   sentences opening with a strong interrogative (האם, למה,
                   מדוע...) end with "?" - Whisper reliably drops it. Only
                   the unambiguous openers: sentence-initial האם is a
                   question in Hebrew, full stop. The ambiguous ones (מה,
                   מי, איך...) are left alone; wrong punctuation is worse
                   than missing punctuation.
  gender_signals   Hebrew inflects first-person present by gender, so a
                   speaker saying "אני מדברת" is unambiguously a woman.
                   That is a transcript-internal fact a diarizer cannot
                   see - so a turn carrying a feminine self-marker filed
                   under a speaker who never uses one, while another
                   speaker does, smells like a boundary misattribution.
                   Diagnostic ONLY: it flags, it never rewrites.
"""

from __future__ import annotations

import re

_DIR_MARKS = re.compile("[‎‏‪-‮⁦-⁩﻿]")

# Closed class: interrogatives that open a direct question and nothing else
# when sentence-initial. Optionally prefixed by a conjunction (ו/אז/אבל).
_STRONG_Q = ("האם", "למה", "מדוע", "כיצד", "מתי", "איפה", "לאן",
             "מאיפה", "מניין", "היכן", "וכי")
_LEAD_CONJ = ("אז", "אבל", "ו")

_SENT_SPLIT = re.compile(r"([.!?]+\s*|$)")

# First-person anchor, with the single-letter clitics Hebrew allows on it.
_ANI = ("אני", "ואני", "שאני", "כשאני")
# Adverbs that legally sit between אני and its inflected predicate.
_SKIP = ("לא", "ממש", "מאוד", "כבר", "עדיין", "קצת", "רק", "גם", "פשוט")


def strip_marks(text: str) -> str:
    return _DIR_MARKS.sub("", text or "")


def _is_strong_opener(sentence: str) -> bool:
    toks = sentence.split()
    if toks and toks[0] in _LEAD_CONJ:
        toks = toks[1:]
    if not toks:
        return False
    head = toks[0]
    if head in _STRONG_Q:
        return True
    # One-letter ו prefix fused onto the opener: ולמה, והאם...
    return head.startswith("ו") and head[1:] in _STRONG_Q


def question_marks(text: str) -> str:
    """Fix the terminal punctuation of unambiguous direct questions."""
    parts = _SENT_SPLIT.split(text or "")
    out = []
    # parts alternates [sentence, delimiter, sentence, delimiter, ...]
    for i in range(0, len(parts) - 1, 2):
        sent, delim = parts[i], parts[i + 1]
        if sent.strip() and _is_strong_opener(sent.strip()) \
                and "?" not in delim:
            delim = "?" + delim.lstrip(".!") if delim.strip() else "? "
        out.append(sent)
        out.append(delim)
    out.extend(parts[len(parts) - 1 if len(parts) % 2 else len(parts):])
    return "".join(out).rstrip()


def _feminine_self_marker(text: str) -> str | None:
    """A first-person feminine present-tense form, or None.

    The pattern: אני + (adverbs) + participle ending in ת- (מדברת, אומרת,
    יודעת...). Masculine present participles never take final ת, so a hit
    is one-directional evidence. Excluded to stay lexicon-free and safe:
    infinitives (ל- prefix), plural/abstract ות- endings, and short tokens.
    """
    toks = re.sub(r"[^\w֐-׿ ]+", " ", text or "").split()
    for i, tok in enumerate(toks):
        if tok not in _ANI:
            continue
        for nxt in toks[i + 1:i + 4]:
            if nxt in _SKIP:
                continue
            if (len(nxt) >= 4 and nxt.endswith("ת")
                    and not nxt.endswith("ות")
                    and not nxt.startswith("ל")):
                return nxt
            break
    return None


def gender_signals(turns: list) -> dict:
    """Per-speaker feminine self-marker counts + misattribution suspects.

    `turns` is any sequence with .speaker/.start/.text (or dict keys).
    Returns {"speakers": {name: count}, "suspects": [(start, speaker, word)]}
    where a suspect is a feminine-marked turn filed under a speaker with no
    other feminine evidence while some other speaker has at least two.
    """
    def _get(t, k):
        return t[k] if isinstance(t, dict) else getattr(t, k)

    hits: list[tuple[float, str, str]] = []
    counts: dict[str, int] = {}
    for t in turns:
        w = _feminine_self_marker(_get(t, "text"))
        if w:
            spk = _get(t, "speaker")
            counts[spk] = counts.get(spk, 0) + 1
            hits.append((float(_get(t, "start")), spk, w))

    strong = {s for s, c in counts.items() if c >= 2}
    suspects = [(ts, spk, w) for ts, spk, w in hits
                if counts.get(spk, 0) == 1 and strong - {spk}]
    return {"speakers": counts, "suspects": suspects}
