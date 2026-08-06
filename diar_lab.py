"""Iterate on speaker attribution locally, on CPU, without touching the GPU.

    python diar_lab.py out/strong_a/Rec722_83_11-18-44_08-06-2026.gpu.json
    python diar_lab.py <cached.gpu.json> --variants baseline,short,very-short

Transcription is held fixed: the words come from a cached worker response, so
every difference between variants is the diarizer and nothing else. That is the
same discipline compare_diar.py enforces on the GPU, but here a variant costs
minutes of laptop time instead of a rebuild, a redeploy and a paid request.

Two things are measured.

`buried` counts turns holding both a question and its answer - the failure the
transcripts actually exhibit, and the one a silhouette score will not show you.

`--diagnose` asks a prior question: is the information even in the audio? For
each buried answer it embeds the words before the question mark and the words
after it separately, and reports the cosine distance between them. If the two
halves of a merged turn embed identically, no clustering scheme can separate
them and no better diarizer will either - the answer would have to come from
somewhere other than voice. Worth knowing before spending another build on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import diarize as diar
from pipeline import quality
from pipeline import vad as vad_mod

# Torch defaults to half the cores here. The worker runs this on a GPU where
# that is irrelevant, but the lab is CPU-bound and sweeps are the whole point.
try:
    import os as _os
    import torch as _torch
    _torch.set_num_threads(_os.cpu_count() or 4)
except Exception:
    pass

# Window geometry is the parameter under test, so it is set per variant. These
# are module-level constants in the diarizer because production has one answer;
# the lab is where that answer gets chosen.
VARIANTS: dict[str, dict] = {
    "baseline": {"window": 1.5, "hop": 0.25},
    # A window that spans a turn boundary embeds two voices at once, and the
    # blend is what buried answers are made of. Shorter windows blend less but
    # embed more noisily - ECAPA was trained on longer segments. Whether the
    # trade pays is exactly what this measures.
    "short": {"window": 1.0, "hop": 0.10},
    "very-short": {"window": 0.75, "hop": 0.10},
    # The control. If longer windows also reduce buried answers then blending
    # is not the mechanism and this whole line of attack is wrong.
    "long": {"window": 2.0, "hop": 0.25},
}


def _audio_for(js: Path) -> Path:
    """The 16 kHz wav the worker was fed, which sits beside the cached json."""
    stem = js.name.replace(".gpu.json", "").replace(".json", "")
    for cand in (js.parent / f"{stem}.16k.wav", js.parent / f"{stem}.wav"):
        if cand.exists():
            return cand
    raise SystemExit(f"No 16 kHz wav beside {js} - expected {stem}.16k.wav")


def _embed_cached(audio_path: Path, spans, window: float, hop: float, cache: dict):
    """Embed once per window geometry; variants sharing geometry share the work."""
    key = (window, hop)
    if key not in cache:
        t0 = time.time()
        diar.WINDOW_SEC, diar.HOP_SEC = window, hop
        audio = diar._load_audio(audio_path)
        times, emb = diar._embed_windows(audio, spans)
        cache[key] = (times, emb)
        print(f"    embedded {len(times)} windows at {window}s/{hop}s "
              f"in {time.time() - t0:.0f}s")
    return cache[key]


def run_variant(name: str, cfg: dict, audio_path: Path, words: list[dict],
                spans, cache: dict, num_speakers: int = 2) -> dict:
    diar.WINDOW_SEC, diar.HOP_SEC = cfg["window"], cfg["hop"]
    times, emb = _embed_cached(audio_path, spans, cfg["window"], cfg["hop"], cache)
    if times.size == 0:
        return {"variant": name, "error": "no voiced windows"}

    labels, separation = diar._cluster(emb, num_speakers)
    turns = diar._smooth(times, labels)

    centroids = []
    for k in range(num_speakers):
        members = emb[labels == k]
        c = (members.mean(axis=0) if members.shape[0]
             else np.zeros(emb.shape[1]))
        centroids.append(c / (np.linalg.norm(c) + 1e-9))
    centroids = np.stack(centroids)

    # The same post-processing production runs, so the comparison reflects the
    # transcript a caller would actually receive rather than raw cluster labels.
    labelled = diar.assign_words(words, turns)
    labelled, flips = diar.refine_word_speakers(audio_path, labelled, centroids)
    labelled = diar.realign_by_punctuation(labelled)

    pre = diar.turns_from_words(labelled)
    share: dict[str, float] = {}
    for t in pre:
        share[t["speaker"]] = share.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    dominant = max(share.values()) / (sum(share.values()) or 1.0)
    weak = separation < 0.15 or dominant > 0.85
    labelled, handed = diar.split_buried_answers(
        audio_path, labelled, centroids, weak_voice=weak)

    final = diar.turns_from_words(labelled)
    report = quality.assess({"turns": final}, expected_speakers=num_speakers).to_dict()
    return {
        "variant": name, "windows": int(times.size), "separation": round(separation, 3),
        "flips": flips, "handed": handed, "turns": len(final),
        "buried": report.get("buried_answers", 0),
        "dominant": report.get("dominant_speaker_pct", 0),
        "speakers": report.get("speakers", 0), "verdict": report.get("verdict", "?"),
        "_turns": final,
    }


def diagnose(audio_path: Path, turns: list[dict], words: list[dict]) -> None:
    """Is a merged turn separable at all, or do both halves sound the same?

    A buried answer is a turn whose text holds a question and then a reply. If
    the voice on each side of the question mark is genuinely different, the
    distance below is large and some diarizer can find it. If it is small, the
    two parties are indistinguishable to the embedding model over that span,
    and better clustering is not the answer - the split would have to come
    from language, not voice.
    """
    import torch

    encoder = diar._get_encoder()
    audio = diar._load_audio(audio_path)

    def embed(lo: float, hi: float):
        a = audio[int(lo * diar.SAMPLE_RATE):int(hi * diar.SAMPLE_RATE)]
        if a.size < int(0.4 * diar.SAMPLE_RATE):
            return None
        with torch.no_grad():
            e = encoder.encode_batch(torch.tensor(a).unsqueeze(0)).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    print("\n  buried answers - is the voice actually different across the split?")
    print(f"    {'time':>8}  {'dist':>5}  text")
    seen = 0
    for t in turns:
        text = t.get("text", "")
        if "?" not in text:
            continue
        after = text.split("?", 1)[1].strip()
        if len(after.split()) < 3:
            continue
        seen += 1
        # Locate the question mark in time by walking the turn's own words.
        inside = [w for w in words if t["start"] - 1e-3 <= w["start"] <= t["end"]]
        cut = None
        for w in inside:
            if "?" in str(w.get("word", "")):
                cut = w["end"]
                break
        if cut is None or cut - t["start"] < 0.4 or t["end"] - cut < 0.4:
            print(f"    {t['start']:8.1f}  {'   -':>5}  (too short to test) {text[:48]}")
            continue
        a, b = embed(t["start"], cut), embed(cut, t["end"])
        if a is None or b is None:
            print(f"    {t['start']:8.1f}  {'   -':>5}  (too short to test) {text[:48]}")
            continue
        dist = float(1.0 - np.dot(a, b))
        print(f"    {t['start']:8.1f}  {dist:5.3f}  {text[:60]}")
    if not seen:
        print("    none found")
        return

    # The control, without which the numbers above mean nothing. A large
    # distance between two short segments can come from them being short and
    # noisy rather than from two different people. So run the identical
    # measurement on turns that hold no question at all, split at their
    # midpoint: same model, same segment lengths, one speaker by construction.
    # Whatever the split-half distance is here is the floor - buried answers
    # only carry evidence of a second voice insofar as they exceed it.
    ctrl = []
    for t in turns:
        if "?" in t.get("text", "") or t["end"] - t["start"] < 1.6:
            continue
        mid = (t["start"] + t["end"]) / 2
        a, b = embed(t["start"], mid), embed(mid, t["end"])
        if a is not None and b is not None:
            ctrl.append(float(1.0 - np.dot(a, b)))
        if len(ctrl) >= 40:
            break
    if ctrl:
        c = np.array(ctrl)
        print(f"\n    control: same-speaker turns split at the midpoint, n={len(c)}")
        print(f"      median {np.median(c):.3f}   90th pct {np.percentile(c, 90):.3f}"
              f"   max {c.max():.3f}")
        print("      A buried answer only shows a second voice if it clears this.")
    else:
        print("\n    control: no question-free turns long enough to split")


def run_pyannote(audio_path: Path, words: list[dict], num_speakers: int = 2) -> dict:
    """Score pyannote over the same cached words, on this machine.

    The image is not built yet, but nothing about the comparison requires a
    GPU: the transcript is fixed, so running the pipeline here answers the
    only question that matters - does it merge fewer questions with their
    answers than what is deployed today.
    """
    import os
    from pipeline import pyannote_diar as pyd

    # Locally the weights live in the ordinary Hugging Face cache rather than
    # baked into an image, so the build marker will not exist. Point it at
    # something real instead of loosening the check that protects production.
    marker = Path(".pyannote_ready")
    marker.touch()
    model_id = pyd.VARIANTS["pyannote"][0]
    pyd.VARIANTS["pyannote"] = (model_id, str(marker))
    if not os.environ.get("HF_TOKEN") and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()

    t0 = time.time()
    res = pyd.diarize(audio_path, num_speakers=num_speakers)
    print(f"    pyannote produced {len(res.turns)} raw turns in {time.time() - t0:.0f}s")
    for n in res.notes:
        print(f"    [!] {n}")

    labelled = pyd_assign(words, res.turns)
    labelled = diar.realign_by_punctuation(labelled)
    final = diar.turns_from_words(labelled)
    report = quality.assess({"turns": final}, expected_speakers=num_speakers).to_dict()
    return {
        "variant": "pyannote", "windows": 0, "separation": 0.0,
        "flips": 0, "handed": 0, "turns": len(final),
        "buried": report.get("buried_answers", 0),
        "dominant": report.get("dominant_speaker_pct", 0),
        "speakers": report.get("speakers", 0), "verdict": report.get("verdict", "?"),
        "_turns": final,
    }


def pyd_assign(words: list[dict], turns) -> list[dict]:
    from pipeline import nemo_diar
    return nemo_diar.assign_words(words, turns)


def calibrate(audio_path: Path, spans, lengths=(0.5, 1.0, 1.5, 2.0, 3.0)) -> None:
    """How long must a segment be before ECAPA can tell voices apart here?

    The buried-answer diagnostic and its control both depend on turn labels,
    and those labels come from the diarizer under test - if it merged two
    speakers, the "same-speaker" control contains two speakers too. This
    measurement uses no labels at all.

    Within one continuous run of speech, consecutive segments are almost
    always the same person. Across two runs separated by a long silence, on a
    two-party call, they are often not. So for each segment length we compare:

      within   mean distance between consecutive segments inside one span
      across   mean distance between segments drawn from different spans

    If `across` sits meaningfully above `within`, the embedding carries speaker
    identity at that length and clustering has something to work with. If the
    two are equal, it does not - and no amount of clustering, purification or
    window tuning recovers information that was never in the vector. That is a
    statement about the audio and the encoder, not about our algorithm.
    """
    import torch

    encoder = diar._get_encoder()
    audio = diar._load_audio(audio_path)
    sr = diar.SAMPLE_RATE

    def embed_many(segs):
        if not segs:
            return np.zeros((0, 192))
        out = []
        for i in range(0, len(segs), 64):
            t = torch.from_numpy(np.stack(segs[i:i + 64]))
            with torch.no_grad():
                e = encoder.encode_batch(t).squeeze(1).cpu().numpy()
            out.append(e)
        e = np.concatenate(out, axis=0)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    print(f"\n  segment length vs. speaker information (no turn labels used)")
    print(f"    {'len':>5} {'segs':>5} {'spans':>6} {'within':>8} {'across':>8} "
          f"{'gap':>7}")
    long_spans = [s for s in spans if s[1] - s[0] >= 4.0][:30]
    if len(long_spans) < 2:
        print("    not enough long speech runs to calibrate")
        return

    for L in lengths:
        per_span, segs = [], []
        for lo, hi in long_spans:
            idx = []
            t = lo
            while t + L <= hi:
                a = audio[int(t * sr):int((t + L) * sr)]
                if a.size == int(L * sr):
                    idx.append(len(segs))
                    segs.append(a)
                t += L
            if len(idx) >= 2:
                per_span.append(idx)
        if len(per_span) < 2:
            print(f"    {L:5.2f} {'-':>5} {'-':>6}   too few segments at this length")
            continue

        emb = embed_many(segs)
        within, across = [], []
        for idx in per_span:
            for a, b in zip(idx, idx[1:]):
                within.append(1.0 - float(emb[a] @ emb[b]))
        for i in range(len(per_span)):
            for j in range(i + 1, len(per_span)):
                # One pair per span pair keeps this O(spans^2), not O(segs^2).
                across.append(1.0 - float(emb[per_span[i][0]] @ emb[per_span[j][0]]))
        w, a = float(np.mean(within)), float(np.mean(across))
        print(f"    {L:5.2f} {len(segs):5} {len(per_span):6} {w:8.3f} {a:8.3f} "
              f"{a - w:+7.3f}")
    print("    gap > 0 means the embedding separates speakers at that length.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cached", help="a cached <name>.gpu.json from a worker run")
    p.add_argument("--audio", default=None)
    p.add_argument("--variants", default="baseline,short,very-short,long")
    p.add_argument("--speakers", type=int, default=2)
    p.add_argument("--vad", type=float, default=0.4)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--pyannote", action="store_true",
                   help="score pyannote community-1 over the cached words")
    p.add_argument("--calibrate", action="store_true",
                   help="measure whether ECAPA separates speakers on this audio "
                        "at all, using no turn labels")
    p.add_argument("--diagnose-only", action="store_true",
                   help="skip the variant sweep and diagnose the cached turns; "
                        "answers 'is this fixable at all' in minutes, not hours")
    p.add_argument("--out", default=None, help="write the winning transcript here")
    args = p.parse_args(argv)

    js = Path(args.cached)
    data = json.loads(js.read_text(encoding="utf-8"))
    words = [w for w in data.get("words", []) if w.get("start") is not None]
    if not words:
        raise SystemExit(f"{js} carries no word timestamps")
    audio_path = Path(args.audio) if args.audio else _audio_for(js)

    print(f"=== {js.name} ===")
    print(f"  audio  {audio_path.name}")
    print(f"  words  {len(words)} (held fixed across variants)")

    spans, _ = vad_mod.detect_speech(audio_path, threshold=args.vad)
    speech = [(s.start, s.end) for s in spans]
    print(f"  speech {sum(e - s for s, e in speech):.0f}s in {len(speech)} spans")

    if args.calibrate:
        calibrate(audio_path, speech)

    if args.diagnose_only:
        # The cached turns are what the worker actually returned, so this asks
        # about the transcript the user complained about rather than a variant
        # of it - and it costs a few embeddings instead of a full sweep.
        diagnose(audio_path, data.get("turns") or [], words)
        return 0

    cache: dict = {}
    rows = []
    if args.pyannote:
        print("  -- pyannote")
        try:
            rows.append(run_pyannote(audio_path, words, args.speakers))
        except Exception as exc:
            import traceback; traceback.print_exc()
            rows.append({"variant": "pyannote", "error": f"{type(exc).__name__}: {exc}"})
    for name in args.variants.split(","):
        name = name.strip()
        if name not in VARIANTS:
            print(f"  unknown variant {name!r}; known: {', '.join(VARIANTS)}")
            continue
        print(f"  -- {name}")
        try:
            rows.append(run_variant(name, VARIANTS[name], audio_path, words,
                                    speech, cache, args.speakers))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rows.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"})

    print(f"\n  {'variant':12} {'win':>6} {'sep':>6} {'turns':>6} {'buried':>7} "
          f"{'domin':>7} {'flips':>6} {'split':>6}  verdict")
    print("  " + "-" * 76)
    for r in rows:
        if "error" in r:
            print(f"  {r['variant']:12} FAILED: {r['error'][:50]}")
            continue
        print(f"  {r['variant']:12} {r['windows']:6} {r['separation']:6.3f} "
              f"{r['turns']:6} {r['buried']:7} {r['dominant']:6.0f}% "
              f"{r['flips']:6} {r['handed']:6}  {r['verdict']}")

    ok = [r for r in rows if "error" not in r]
    if not ok:
        return 1
    best = min(ok, key=lambda r: (r["buried"], abs(r["dominant"] - 50)))
    print(f"  -> fewest buried answers: {best['variant']} "
          f"({best['buried']}, baseline "
          f"{next((r['buried'] for r in ok if r['variant'] == 'baseline'), '?')})")

    if args.diagnose:
        diagnose(audio_path, best["_turns"], words)

    if args.out:
        from pipeline import parse as parse_mod
        turns = [parse_mod.Turn(speaker=t["speaker"], start=t["start"], end=t["end"],
                                text=t["text"], words=[]) for t in best["_turns"]]
        tr = parse_mod.Transcript(
            source=f"{js.stem} [{best['variant']}]", turns=turns,
            speakers=sorted({t.speaker for t in turns}),
            duration_sec=0.0, word_count=len(words), mean_word_confidence=0.0,
            low_confidence_words=0, unattributed_words=0)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        parse_mod.write_text(tr, out.with_suffix(".txt"))
        parse_mod.write_html(tr, out.with_suffix(".html"))
        print(f"  wrote {out.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
