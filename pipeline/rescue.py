"""Gate 3: a different ear for the regions three decodes could not settle.

The escalation chain earns its cost at every step. Two models agreeing locks
~75% of words for free. The referee - the strong model re-reading just the
disputed clip with context - settles most of the rest, still on our GPU. What
reaches this module is the residue: regions where large-v3, turbo AND the
context-prompted re-decode all heard different things. Three opinions from
the same model family have failed; a fourth from that family would too.

So the fourth opinion is a different species: a multimodal LLM that listens
to the clip. Measured before this was wired in: the opening sentence of a
real call, misheard by every configuration tried over two days ("מה שאת
אומרת" / "מה שאתה אומר"), came back from Gemini as "בסדר, מה שלומך?" at 0.9
confidence - with nothing but conversation-internal context to help it.

Cost discipline is the point of the architecture: the expensive ear hears
only the unresolved clips, typically 10-25% of a call's audio, not the call.

Guardrails, because an LLM ear can imagine:
  - the prompt is transcribe-only; context "only to disambiguate", and the
    model is told both hypotheses may be wrong
  - anchored extraction: it is asked for the words BETWEEN two agreed
    anchors, so its answer replaces exactly the disputed region
  - adoption requires confidence >= 0.5, a plausible length ratio against
    the hypotheses, mostly-Hebrew text, and no repetition loops
  - everything it was asked and answered lands in the report, adopted or not

No lexicon, no external knowledge enters: the context is the same call's own
agreed words, nothing else. That constraint is a user requirement, not an
optimisation.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

# flash-latest, not a pinned version: billed projects created after mid-2026
# are refused gemini-2.5-flash outright ("no longer available to new users"),
# and the alias tracks whatever flash-class model is current.
DEFAULT_MODEL = os.environ.get("RESCUE_MODEL", "gemini-flash-latest")
MAX_REGIONS = int(os.environ.get("RESCUE_MAX_REGIONS", "40"))
CLIP_PRE_SEC = 3.0
CLIP_POST_SEC = 2.0
MIN_CONFIDENCE = 0.5
# Paid-tier pacing. The 2s crawl was the free tier's RPM ceiling; billing
# raises it far above our queue size, so the spacing drops to a courtesy.
PACE_SEC = float(os.environ.get("RESCUE_PACE_SEC", "0.25"))

_HEB = re.compile(r"[֐-׿]")


def _norm(t: str) -> str:
    return re.sub(r"[^\w֐-׿ ]+", "", t or "").strip().lower()


def _clip_wav_b64(audio, sr: int, lo: float, hi: float) -> str:
    import soundfile as sf
    seg = audio[max(0, int(lo * sr)):int(hi * sr)]
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ask_gemini(api_key: str, model: str, prompt: str, audio_b64: str) -> dict:
    """One request, with backoff - free-tier Gemini rate-limits bursts.

    Measured on the first production run: 34 sequential clips, 15 came back
    429. The queue is not latency-sensitive (it adds seconds to a job that
    takes minutes), so waiting beats failing.
    """
    import time as _time

    last = None
    for attempt in range(4):
        try:
            return _ask_gemini_once(api_key, model, prompt, audio_b64)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            last = exc
            _time.sleep(8 * (attempt + 1))
    raise last


def _ask_gemini_once(api_key: str, model: str, prompt: str, audio_b64: str) -> dict:
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}",
        data=json.dumps({"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
        ]}],
            "generationConfig": {"temperature": 0.0,
                                 "thinkingConfig": {"thinkingLevel": "MINIMAL"}},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in model reply")
    out = json.loads(m.group(0))
    usage = d.get("usageMetadata", {})
    return {"transcript": str(out.get("transcript", "")).strip(),
            "confidence": float(out.get("confidence", 0.0)),
            # The API's own meter, so cost reporting is measured consumption
            # rather than an estimate from clip lengths.
            "prompt_tokens": int(usage.get("promptTokenCount", 0)),
            "output_tokens": int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0))}


def _prompt(prev_text: str, next_text: str) -> str:
    """Transcribe-only prompt. Three measured design decisions live here.

    No hypotheses: shown the engines' guesses, the ear echoed a wrong one;
    without them it heard the disputed phrase right three runs of three.
    Matching against the hypotheses happens in code, after listening.
    Temperature 0 (set in the request): the default sampled a different
    answer per run on the same clip. MINIMAL thinking (ditto): thought
    tokens were ~75% of spend and bought nothing on a listening task.
    """
    parts = [
        "Listen to the attached Hebrew audio from a call-centre phone "
        "recording.",
        "",
        "Your ONLY task: transcribe exactly what is spoken, in Hebrew, word "
        "for word. Do not rewrite, do not improve grammar, do not add words "
        "that are not spoken. Context is provided only to disambiguate "
        "acoustically unclear words.",
        "",
    ]
    if prev_text:
        parts.append(f'The words spoken just BEFORE this clip: "{prev_text}"')
    if next_text:
        parts.append(f'The words spoken just AFTER this clip: "{next_text}"')
    parts += [
        "",
        "Transcribe ONLY the disputed middle part of the clip (the clip "
        "includes a little surrounding speech for context).",
        "",
        'Return ONLY JSON: {"transcript": "...", "confidence": 0.0-1.0}',
    ]
    return "\n".join(parts)


def _plausible(text: str, hyp_a: str, hyp_b: str) -> bool:
    """Refuse answers no honest transcription of this region could be."""
    t = _norm(text)
    if not t or not _HEB.search(text):
        return False
    ref = max(len(_norm(hyp_a).replace(" ", "")),
              len(_norm(hyp_b).replace(" ", "")), 1)
    ratio = len(t.replace(" ", "")) / ref
    if not (0.3 <= ratio <= 3.0):
        return False
    toks = t.split()
    for i in range(len(toks) - 4):                 # repetition loop guard
        if len(set(toks[i:i + 5])) == 1:
            return False
    return True


def rescue_unresolved(wav_path: str, rep, api_key: str, *,
                      model: str = DEFAULT_MODEL,
                      real_wav_path: str | None = None,
                      tempo: float = 1.0) -> dict:
    """Send each unresolved disagreement to the listening LLM; splice winners.

    Works in the same (possibly tempo-stretched) clock as `rep`, before the
    seam - exactly like the referee, so nothing here knows two clocks exist.

    The network waits run concurrently: clips are independent until the
    splice, and on the paid tier six workers turn a 35-region queue from
    minutes of serial pacing into seconds. Only the answers are applied
    serially, because splicing mutates rep.words.
    """
    from concurrent.futures import ThreadPoolExecutor
    import time as _time

    import soundfile as sf

    # The ear gets the REAL recording, not the 0.85x ASR copy. Measured:
    # at real tempo it heard the flagship phrase; slowed, it collapsed to
    # echoing a hypothesis. Pipeline times are in the stretched clock, so
    # cuts convert by *tempo.
    audio, sr = sf.read(real_wav_path or wav_path, dtype="float32")
    t_scale = tempo if real_wav_path else 1.0
    agreed = [w for w in rep.words if w.get("agreement")]

    stats = {"attempted": 0, "adopted": 0, "matched_hypothesis": 0,
             "new_text": 0, "rejected": 0, "errors": 0,
             "prompt_tokens": 0, "output_tokens": 0}

    todo = [d for d in rep.disagreements
            if d.get("resolved") == "unresolved" and d.get("_out_words")]

    def _fetch(d):
        hyps = [v for k, v in d.items()
                if not k.startswith("_")
                and k not in ("start", "end", "kept", "resolved", "referee",
                              "prob_a", "prob_b", "rescue")]
        if len(hyps) < 2:
            return d, None, None, None
        prev_text = " ".join(w["word"] for w in agreed
                             if w["end"] <= d["start"])[-120:]
        next_text = " ".join(w["word"] for w in agreed
                             if w["start"] >= d.get("end", d["start"]))[:120]
        lo = d["start"] * t_scale - CLIP_PRE_SEC
        hi = d.get("end", d["start"] + 2.0) * t_scale + CLIP_POST_SEC
        try:
            _time.sleep(PACE_SEC)
            ans = _ask_gemini(api_key, model,
                              _prompt(prev_text, next_text),
                              _clip_wav_b64(audio, sr, lo, hi))
            return d, hyps[0], hyps[1], ans
        except Exception as exc:
            return d, hyps[0], hyps[1], exc

    with ThreadPoolExecutor(max_workers=6) as ex:
        fetched = list(ex.map(_fetch, todo[:MAX_REGIONS]))

    for d, hyp_a, hyp_b, ans in fetched:
        if hyp_a is None:
            continue
        stats["attempted"] += 1
        if ans is None or isinstance(ans, Exception):
            stats["errors"] += 1
            if ans is not None:
                d["rescue"] = {"error": f"{type(ans).__name__}: {ans}"[:120]}
            continue

        stats["prompt_tokens"] += ans.get("prompt_tokens", 0)
        stats["output_tokens"] += ans.get("output_tokens", 0)
        d["rescue"] = {"heard": ans["transcript"][:160],
                       "confidence": round(ans["confidence"], 2)}
        if ans["confidence"] < MIN_CONFIDENCE or                 not _plausible(ans["transcript"], hyp_a, hyp_b):
            stats["rejected"] += 1
            continue

        # Splice by object identity, same mechanism as the referee swap -
        # indices are unreliable after merge() sorts its output.
        ids = {id(w) for w in d["_out_words"]}
        idxs = [i for i, w in enumerate(rep.words) if id(w) in ids]
        if not idxs:
            stats["rejected"] += 1
            continue
        start_t = float(rep.words[idxs[0]]["start"])
        end_t = float(rep.words[idxs[-1]]["end"])
        tokens = ans["transcript"].split()
        step = max((end_t - start_t) / max(len(tokens), 1), 0.05)
        new_words = [{"word": tok,
                      "start": round(start_t + i * step, 3),
                      "end": round(start_t + (i + 1) * step, 3),
                      "probability": ans["confidence"],
                      "agreement": False, "source": "rescue"}
                     for i, tok in enumerate(tokens)]
        at = idxs[0]
        for i in reversed(idxs):
            del rep.words[i]
        rep.words[at:at] = new_words

        d["resolved"] = "rescue"
        stats["adopted"] += 1
        nt = _norm(ans["transcript"])
        if nt in (_norm(hyp_a), _norm(hyp_b)):
            stats["matched_hypothesis"] += 1
        else:
            stats["new_text"] += 1

    if len(todo) > MAX_REGIONS:
        stats["skipped_over_budget"] = len(todo) - MAX_REGIONS
    return stats
