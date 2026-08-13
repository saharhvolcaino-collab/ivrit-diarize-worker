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
# gemini-3-flash-preview: measured on the flagship clip as byte-identical
# across three runs at temperature 0 (flash-latest still wobbled), heard the
# target phrase every time, accepts MINIMAL thinking, 59 output tokens per
# clip. On Vertex AI the same family ships as "gemini-3-flash" if the rescue
# path ever moves behind the user's direct Vertex billing.
DEFAULT_MODEL = os.environ.get("RESCUE_MODEL", "gemini-3-flash-preview")
# If the pinned model dies mid-run (404 on a retired preview, 402, a daily
# quota), the queue moves to the next model instead of erroring 40 clips.
# gemini-2.5-flash vanishing for new billed projects is exactly this event.
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "RESCUE_FALLBACK", "gemini-flash-latest").split(",") if m.strip()]
# 40 was measured too tight: the conference call left 41 unresolved regions
# outside the budget. At ~400 prompt tokens a clip the marginal cost of a
# region is a hundredth of an agora - the budget exists against runaway
# calls, not to save money on real ones.
MAX_REGIONS = int(os.environ.get("RESCUE_MAX_REGIONS", "80"))
CLIP_PRE_SEC = 3.0
CLIP_POST_SEC = 2.0
MIN_CONFIDENCE = 0.5
# Paid-tier pacing. The 2s crawl was the free tier's RPM ceiling; billing
# raises it far above our queue size, so the spacing drops to a courtesy.
PACE_SEC = float(os.environ.get("RESCUE_PACE_SEC", "0.25"))

_HEB = re.compile(r"[֐-׿]")
# Directional formatting characters (LRM/RLM, embeddings, isolates, BOM).
# Players and plain-text viewers render them as stray glyphs in RTL text,
# and LLM output sometimes carries them invisibly.
_DIR_MARKS = re.compile("[‎‏‪-‮⁦-⁩﻿]")


def _norm(t: str) -> str:
    return re.sub(r"[^\w֐-׿ ]+", "", t or "").strip().lower()


class _ModelExhausted(Exception):
    """This model is out for the whole run - retrying it is pure loss.

    404: the model was retired (previews do vanish mid-month). 402 or a
    prepay/credit message: billing is empty. A per-DAY quota 429: no wait
    inside one job will refill it. All three mean: stop asking this model,
    try the next one in the chain.
    """


def _http_body(exc) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:800]
    except Exception:
        return ""


def _retry_wait(exc, body: str, attempt: int) -> float:
    """Wait what the server asked for, not a blind guess - capped at 180s."""
    hdr = (exc.headers or {}).get("Retry-After")
    if hdr:
        try:
            return min(float(hdr), 180.0)
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    if m:
        return min(float(m.group(1)), 180.0)
    return 8.0 * (attempt + 1)


def _clip_wav_b64(audio, sr: int, lo: float, hi: float) -> str:
    import soundfile as sf
    seg = audio[max(0, int(lo * sr)):int(hi * sr)]
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ask_gemini(api_key: str, model: str, prompt: str, audio_b64: str) -> dict:
    """One request, with a taxonomy instead of blind retries.

    Measured on the first production run: 34 sequential clips, 15 came back
    429. The queue is not latency-sensitive (it adds seconds to a job that
    takes minutes), so waiting beats failing - but only for errors a wait
    can fix. Permanent ones (model retired, billing empty, daily quota)
    raise _ModelExhausted so the caller fails over instead of stalling
    every clip for the full backoff ladder.
    """
    import time as _time

    last = None
    for attempt in range(4):
        try:
            return _ask_gemini_once(api_key, model, prompt, audio_b64)
        except urllib.error.HTTPError as exc:
            body = _http_body(exc)
            if exc.code in (402, 404):
                raise _ModelExhausted(f"HTTP {exc.code}: {body[:160]}")
            if exc.code != 429:
                raise
            low = body.lower()
            if "perday" in low or "prepay" in low or "credit" in low:
                raise _ModelExhausted(f"quota: {body[:160]}")
            last = exc
            _time.sleep(_retry_wait(exc, body, attempt))
        except (TimeoutError, urllib.error.URLError) as exc:
            # A dropped read is transient; the clip is not at fault.
            # Measured: exactly one timeout in 80 concurrent fetches.
            last = exc
            _time.sleep(4.0 * (attempt + 1))
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
    return {"transcript": _DIR_MARKS.sub(
                "", str(out.get("transcript", ""))).strip(),
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


def _anchor_trim(text: str, prev_text: str, next_text: str) -> str:
    """Cut an over-long answer down to the disputed middle.

    The clip deliberately carries ~5s of surrounding speech for context, and
    the ear often transcribes all of it - a correct answer that then fails
    the length guard. Measured on the validation sweep: 25-29 rejections per
    call, most of them this. But the surplus words are exactly the AGREED
    words around the region, which we hold verbatim: strip the longest
    transcript prefix that matches the tail of the preceding agreed text,
    and the longest suffix matching the head of the following agreed text.
    Matching is on normalised tokens, and only whole-sequence overlaps are
    removed - a genuine repeated word inside the disputed region does not
    match a multi-token anchor and survives.
    """
    toks = text.split()
    if not toks:
        return text
    tok_n = [_norm(w) for w in toks]
    prev_n = [_norm(w) for w in prev_text.split()]
    cut = 0
    for k in range(1, min(len(prev_n), len(tok_n)) + 1):
        if tok_n[:k] == prev_n[-k:]:
            cut = k
    toks, tok_n = toks[cut:], tok_n[cut:]
    next_n = [_norm(w) for w in next_text.split()]
    cut = 0
    for k in range(1, min(len(next_n), len(tok_n)) + 1):
        if tok_n[-k:] == next_n[:k]:
            cut = k
    if cut:
        toks = toks[:-cut]
    return " ".join(toks)


def _fuzzy_anchor_trim(text: str, prev_text: str, next_text: str) -> str:
    """Anchor-trim for when the anchors themselves were misheard.

    Measured failure (flagship phrase, v0.24 run): Gemini heard the whole
    clip RIGHT - including the caller identity every engine had garbled -
    at 0.98 confidence, and the answer was rejected because the engines'
    "agreed" context words did not match Gemini's rendering of them token
    for token. Exact matching punishes the ear for being better than the
    anchor.

    So: fuzzy. Slide a window over the answer's tokens looking for the
    best SequenceMatcher ratio against the anchor (the last few agreed
    words before the region / the first few after). A window scoring
    >= 0.75 is the context; cut through it. The plausibility and
    confidence gates still stand behind this - a bad cut dies there.
    """
    from difflib import SequenceMatcher

    def _best_window(tok_n: list[str], anchor: list[str]) -> tuple:
        a = " ".join(anchor)
        best_r, best_span = 0.0, None
        for size in {max(1, len(anchor) - 1), len(anchor), len(anchor) + 1}:
            for i in range(0, len(tok_n) - size + 1):
                r = SequenceMatcher(a=a, b=" ".join(tok_n[i:i + size]),
                                    autojunk=False).ratio()
                if r > best_r:
                    best_r, best_span = r, (i, i + size)
        return best_r, best_span

    toks = text.split()
    tok_n = [_norm(w) for w in toks]
    prev_n = [_norm(w) for w in prev_text.split()][-4:]
    if prev_n and len(tok_n) > 1:
        r, span = _best_window(tok_n, prev_n)
        if r >= 0.75 and span and span[1] < len(toks):
            toks = toks[span[1]:]
            tok_n = tok_n[span[1]:]
    next_n = [_norm(w) for w in next_text.split()][:4]
    if next_n and len(tok_n) > 1:
        r, span = _best_window(tok_n, next_n)
        if r >= 0.75 and span and span[0] > 0:
            toks = toks[:span[0]]
    return " ".join(toks)


def rescue_unresolved(wav_path: str, rep, api_key: str, *,
                      model: str = DEFAULT_MODEL,
                      real_wav_path: str | None = None,
                      tempo: float = 1.0,
                      max_regions: int | None = None) -> dict:
    """Send each unresolved disagreement to the listening LLM; splice winners.

    Works in the same (possibly tempo-stretched) clock as `rep`, before the
    seam - exactly like the referee, so nothing here knows two clocks exist.

    The network waits run concurrently: clips are independent until the
    splice, and on the paid tier six workers turn a 35-region queue from
    minutes of serial pacing into seconds. Only the answers are applied
    serially, because splicing mutates rep.words.
    """
    import threading
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
             "new_text": 0, "trimmed": 0, "rejected": 0, "errors": 0,
             "prompt_tokens": 0, "output_tokens": 0}

    todo = [d for d in rep.disagreements
            if d.get("resolved") == "unresolved" and d.get("_out_words")]

    # Failover chain, shared by every worker thread: a model that raised
    # _ModelExhausted is dead for the rest of the run, and every clip after
    # that goes straight to the next model instead of rediscovering it.
    chain = [model] + [m for m in FALLBACK_MODELS if m != model]
    dead: set[str] = set()
    dead_lock = threading.Lock()

    def _ask_chain(prompt: str, clip_b64: str) -> dict:
        last: Exception | None = None
        for m in chain:
            with dead_lock:
                if m in dead:
                    continue
            try:
                return _ask_gemini(api_key, m, prompt, clip_b64)
            except _ModelExhausted as exc:
                with dead_lock:
                    dead.add(m)
                last = exc
        raise last or RuntimeError("no rescue model available")

    def _fetch(d):
        hyps = [v for k, v in d.items()
                if not k.startswith("_")
                and k not in ("start", "end", "kept", "resolved", "referee",
                              "prob_a", "prob_b", "rescue")]
        if len(hyps) < 2:
            return d, None, None, None, "", ""
        prev_text = " ".join(w["word"] for w in agreed
                             if w["end"] <= d["start"])[-120:]
        next_text = " ".join(w["word"] for w in agreed
                             if w["start"] >= d.get("end", d["start"]))[:120]
        lo = d["start"] * t_scale - CLIP_PRE_SEC
        hi = d.get("end", d["start"] + 2.0) * t_scale + CLIP_POST_SEC
        try:
            _time.sleep(PACE_SEC)
            ans = _ask_chain(_prompt(prev_text, next_text),
                             _clip_wav_b64(audio, sr, lo, hi))
            return d, hyps[0], hyps[1], ans, prev_text, next_text
        except Exception as exc:
            return d, hyps[0], hyps[1], exc, prev_text, next_text

    limit = int(max_regions) if max_regions else MAX_REGIONS
    with ThreadPoolExecutor(max_workers=6) as ex:
        fetched = list(ex.map(_fetch, todo[:limit]))

    for d, hyp_a, hyp_b, ans, prev_text, next_text in fetched:
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
        if ans["confidence"] < MIN_CONFIDENCE:
            stats["rejected"] += 1
            continue
        if not _plausible(ans["transcript"], hyp_a, hyp_b):
            # Before rejecting, try shaving echoed context off the answer:
            # a transcript of the WHOLE clip is not a wrong answer, just an
            # untrimmed one. Exact token overlap first; if the anchors were
            # themselves misheard by the engines, the fuzzy pass finds them
            # by similarity instead.
            trimmed = _anchor_trim(ans["transcript"], prev_text, next_text)
            if trimmed == ans["transcript"] or \
                    not _plausible(trimmed, hyp_a, hyp_b):
                trimmed = _fuzzy_anchor_trim(ans["transcript"],
                                             prev_text, next_text)
            if trimmed != ans["transcript"] and \
                    _plausible(trimmed, hyp_a, hyp_b):
                ans["transcript"] = trimmed
                d["rescue"]["trimmed"] = True
                stats["trimmed"] += 1
            else:
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

    if len(todo) > limit:
        stats["skipped_over_budget"] = len(todo) - limit
    if dead:
        stats["models_exhausted"] = sorted(dead)
    return stats
