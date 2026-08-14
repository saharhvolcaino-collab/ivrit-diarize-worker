"""The presentation pass: from decoder word-soup to a readable protocol.

Competitor output (screenshot, 2026-08-14) reads like meeting minutes:
full sentences, punctuation, clean turns. That fluency is an audio-LLM
signature - and the same ear this pipeline already trusts for disputed
regions can polish the WHOLE call for agorot: audio in at $1/1M tokens
prices a 14-minute call around 3 cents.

Design, all constraints inherited from the campaign's measurements:
  - the draft turns anchor the LLM to the pipeline's structure (speaker
    labels and timing survive untouched); the audio keeps it honest
  - transcribe-only prompt: fix misheard words, punctuate, never add
    content, never translate. Conversation-internal context only.
  - per-turn guards: a polished turn is adopted only if it stays Hebrew
    and within a sane length ratio of the draft; anything else keeps the
    draft. A failed window keeps all its drafts - polish can only be a
    no-op, never a loss.
  - temperature 0, MINIMAL thinking, same retry/backoff taxonomy as the
    rescue lane.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline import rescue as _r

WINDOW_SEC = 150.0
MAX_RATIO = 2.0
MIN_RATIO = 0.45

_HEB = re.compile(r"[֐-׿]")


def _prompt(draft_lines: list[str]) -> str:
    return "\n".join([
        "Listen to the attached Hebrew call-centre phone recording.",
        "",
        "Below is a draft transcript of this clip, split into numbered "
        "speaker turns, produced by an ASR system. It contains recognition "
        "errors and has poor punctuation.",
        "",
        "Your task: return the corrected transcript for EACH numbered turn.",
        "- Fix only what the audio supports: misheard words, missing or "
        "wrong punctuation.",
        "- Keep every turn's boundaries exactly - do not move words between "
        "turns, do not merge or split turns.",
        "- Do NOT add anything that is not spoken. Do NOT translate. Keep "
        "names exactly as they sound.",
        "- If a turn is already correct, return it unchanged.",
        "",
        "Draft:",
        *draft_lines,
        "",
        'Return ONLY JSON: {"turns": {"<number>": "<corrected text>", ...}}',
    ])


def polish(turns: list, wav_path: str, api_key: str, *,
           model: str = _r.DEFAULT_MODEL) -> dict:
    """Polish turn texts in place. Returns stats.

    `turns` is a list of objects with .start/.end/.text (parse.Turn).
    """
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32")
    stats = {"windows": 0, "turns": len(turns), "polished": 0,
             "kept_draft": 0, "errors": 0,
             "prompt_tokens": 0, "output_tokens": 0}

    # Windows of consecutive turns, cut on turn boundaries near WINDOW_SEC.
    windows: list[list[int]] = [[]]
    win_start = turns[0].start if turns else 0.0
    for i, t in enumerate(turns):
        if t.end - win_start > WINDOW_SEC and windows[-1]:
            windows.append([])
            win_start = t.start
        windows[-1].append(i)

    def _fetch(idxs):
        lo = max(0.0, turns[idxs[0]].start - 0.3)
        hi = turns[idxs[-1]].end + 0.3
        draft = [f"{i}. {turns[i].text}" for i in idxs]
        try:
            return _r._ask_gemini(api_key, model, _prompt(draft),
                                  _r._clip_wav_b64(audio, sr, lo, hi))
        except Exception as exc:
            return exc

    # Windows are independent until adoption - fetch concurrently, adopt
    # serially. Four in flight keeps a 14-minute call under ~15 seconds.
    from concurrent.futures import ThreadPoolExecutor
    live = [w for w in windows if w]
    with ThreadPoolExecutor(max_workers=4) as ex:
        answers = list(ex.map(_fetch, live))

    for idxs, ans in zip(live, answers):
        stats["windows"] += 1
        if isinstance(ans, Exception):
            stats["errors"] += 1
            stats["kept_draft"] += len(idxs)
            continue
        stats["prompt_tokens"] += ans.get("prompt_tokens", 0)
        stats["output_tokens"] += ans.get("output_tokens", 0)
        fixed = (ans.get("_raw") or {}).get("turns")
        if not isinstance(fixed, dict):
            stats["errors"] += 1
            stats["kept_draft"] += len(idxs)
            continue

        for i in idxs:
            new = str(fixed.get(str(i), "")).strip()
            old = turns[i].text
            old_len = max(len(re.sub(r"\s+", "", old)), 1)
            new_len = len(re.sub(r"\s+", "", new))
            if (new and _HEB.search(new)
                    and MIN_RATIO <= new_len / old_len <= MAX_RATIO):
                if new != old:
                    turns[i].text = new
                    stats["polished"] += 1
                else:
                    stats["kept_draft"] += 1
            else:
                stats["kept_draft"] += 1
    return stats
