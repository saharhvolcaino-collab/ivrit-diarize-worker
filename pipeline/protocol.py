"""The protocol lane: one full-call pass through the listening LLM.

Measured (2026-08-14, five-run matrix on the flagship call, all numbers
from usageMetadata): the flagship phrase decoded correctly in 5/5 runs of
the WHOLE call, at every thinking level - a robustness no clip-level path
achieved. The winning setting is thinkingLevel LOW at temperature 0:
matched HIGH on every hard check (flagship phrase, כפר בלום, the digit
readout, full 14-minute coverage) at half the latency (21.6s) and 73% of
the cost (~$0.0021/audio-minute). HIGH's advantage did not survive its
own repeat run, so it was luck, not the thinking budget.

Known limits, why the ASR skeleton stays the anchor:
  - not reproducible run to run even at temp 0 (86% character similarity,
    one run 10% shorter) - semantically stable, not byte-stable
  - no word-level timestamps
  - real-world proper nouns garble without external grounding (קיבוץ came
    out K-Flow / קק"ל / קייק across runs; AI Studio's clean version had
    Google-Search grounding on)

The audio ships as 32 kbps mono MP3 (~10x smaller than WAV, duration
verified preserved in the experiments) because the inline API caps
request size well under a full-call WAV.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.request

from pipeline import rescue as _r

DEFAULT_THINKING = os.environ.get("PROTOCOL_THINKING", "LOW")

_PROMPT = "\n".join([
    "תמלל את שיחת הטלפון המצורפת - שיחת מוקד שירות בעברית - במלואה, מתחילתה ועד סופה.",
    "",
    "כללים:",
    "- תמלול נאמן בלבד: אל תוסיף דבר שלא נאמר, אל תתרגם, אל תשפר ניסוחים.",
    "- זהה את הדוברים. כאשר שם של דובר עולה מתוך השיחה עצמה - השתמש בשם "
    "כתווית (למשל 'תומר:'); אחרת השתמש בתפקיד ('נציג:', 'לקוחה:').",
    "- חלק את התמלול לבלוקים עם חותמת זמן בתחילת כל בלוק בפורמט mm:ss.",
    "- מספרים וספרות: בדיוק כפי שנאמרו.",
    "- פיסוק מלא ותקין.",
    "",
    "החזר טקסט בלבד, ללא הערות וללא JSON.",
])


def _to_mp3(wav_path: str) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "call.mp3")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", wav_path, "-ac", "1", "-ar", "16000", "-b:a", "32k", dst],
            check=True)
        with open(dst, "rb") as f:
            return f.read()


def _ask_text(api_key: str, model: str, prompt: str, mp3_b64: str,
              thinking: str) -> dict:
    """Plain-text generateContent with the rescue lane's error taxonomy."""
    import base64  # noqa: F401  (b64 passed in, kept for symmetry)
    import urllib.error

    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}",
                data=json.dumps({
                    "contents": [{"parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "audio/mp3",
                                         "data": mp3_b64}},
                    ]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "thinkingConfig": {"thinkingLevel": thinking},
                    },
                }).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            text = "".join(p.get("text", "")
                           for p in d["candidates"][0]["content"]["parts"])
            usage = d.get("usageMetadata", {})
            return {"text": _r._DIR_MARKS.sub("", text).strip(),
                    "prompt_tokens": int(usage.get("promptTokenCount", 0)),
                    "output_tokens": int(usage.get("candidatesTokenCount", 0)),
                    "thought_tokens": int(usage.get("thoughtsTokenCount", 0))}
        except urllib.error.HTTPError as exc:
            body = _r._http_body(exc)
            if exc.code in (402, 404):
                raise _r._ModelExhausted(f"HTTP {exc.code}: {body[:160]}")
            if exc.code != 429:
                raise
            low = body.lower()
            if "perday" in low or "prepay" in low or "credit" in low:
                raise _r._ModelExhausted(f"quota: {body[:160]}")
            last = exc
            time.sleep(_r._retry_wait(exc, body, attempt))
        except (TimeoutError, OSError) as exc:
            last = exc
            time.sleep(4.0 * (attempt + 1))
    raise last


def transcribe_full_call(wav_path: str, api_key: str, *,
                         model: str = _r.DEFAULT_MODEL,
                         thinking: str = DEFAULT_THINKING) -> dict:
    """One pass over the whole call. Returns {text, tokens, seconds}."""
    import base64

    started = time.time()
    mp3_b64 = base64.b64encode(_to_mp3(wav_path)).decode("ascii")

    chain = [model] + [m for m in _r.FALLBACK_MODELS if m != model]
    last: Exception | None = None
    for m in chain:
        try:
            out = _ask_text(api_key, m, _PROMPT, mp3_b64, thinking)
            # Shave any politeness preamble: the protocol starts at the
            # first timestamped line.
            lines = out["text"].splitlines()
            first = next((i for i, ln in enumerate(lines)
                          if __import__("re").match(r"^\d{1,2}:\d{2}", ln.strip())),
                         0)
            out["text"] = "\n".join(lines[first:]).strip()
            out["model"] = m
            out["thinking"] = thinking
            out["seconds"] = round(time.time() - started, 1)
            return out
        except _r._ModelExhausted as exc:
            last = exc
    raise last or RuntimeError("no protocol model available")
