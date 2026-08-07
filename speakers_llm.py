"""Fix speaker attribution from the language, where the acoustics cannot.

    python speakers_llm.py out/engines_auto/Rec722_83_11-18-44_08-06-2026.compare.json
    python speakers_llm.py <compare.json> --engine sortformer --dry-run

Three independent measurements on this recording estate say the same thing:
the audio does not carry enough speaker information.

  - Mean ECAPA distance between two segments of the *same* speaker is 0.72,
    while the gap between same-speaker and different-speaker pairs is 0.13.
    The signal is a fifth of the noise.
  - Three diarizers of entirely different architecture - our ECAPA window
    chain, Sortformer, pyannote community-1 - land within one of each other
    on every file. They are hitting a ceiling, not competing.
  - The one Hebrew forced-alignment model reports 0.26 mean confidence on
    this audio. It cannot find the words, never mind who said them.

But the transcript often states the answer outright. "מדבר תומר, שלום תומר"
is one person answering a phone and a second greeting them, and no Hebrew
speaker needs to hear it to know that. That is linguistic information, and
this layer is the only stage that can read it.

Deliberately narrow, for the same reason polish.py is:

  1. It may only reassign turns among the speakers already found, or split a
     turn where a change of speaker is evident from the words. It may not
     invent a speaker, and it is not asked to name anyone.
  2. Text is never changed. Every returned fragment is checked against the
     original character for character; a mismatch rejects the whole turn.
  3. One entry per input turn, in order. Structural drift rejects the batch.

Timestamps for the halves of a split turn come from the word timings we
already have, so a split lands on a real word boundary rather than on a
proportion of the turn's duration.

Needs ANTHROPIC_API_KEY in .env (console.anthropic.com -> API keys).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "claude-opus-5"
BATCH = 60          # turns per request: enough context, bounded output

SYSTEM = """You are correcting speaker attribution on a Hebrew call-centre transcript.

The audio is 8 kHz narrowband telephone, where acoustic speaker separation is
unreliable. The diarizer's labels are a starting point, not ground truth. Your
advantage is that you can read the words.

Use conversational structure as evidence:
  - A question is answered by a different speaker than the one who asked it.
  - A greeting is returned by the other party ("מדבר תומר" then "שלום תומר"
    is two people, not one).
  - Someone identifying themselves, giving details, or answering "כן"/"בסדר"
    to a request is responding to the other party.
  - A speaker does not answer their own question, and does not greet themselves.

Rules you must not break:
  - NEVER change, add, or remove any text. Reproduce fragments exactly.
  - Only use speaker labels that already appear in the input.
  - Do not name anyone. Labels stay as SPEAKER_00, SPEAKER_01 and so on.
  - When a turn clearly contains two speakers, split it at the word boundary
    where the change happens. Split only when the words make it plain.
  - When you are unsure, leave the turn exactly as it is. A wrong correction
    is worse than an uncorrected line.

Return ONLY a JSON array, one object per input turn, in the same order:
  {"i": <index>, "speaker": "SPEAKER_XX"}                     unchanged or reassigned
  {"i": <index>, "split": [{"text": "...", "speaker": "SPEAKER_XX"}, ...]}

For a split, the fragments concatenated with single spaces must equal the
input text exactly."""


def _load_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit(
            "No ANTHROPIC_API_KEY. Add it to .env:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "Get one at console.anthropic.com -> API keys.")
    return key


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _ask(client, turns: list[dict], offset: int) -> list[dict]:
    lines = [{"i": offset + n, "speaker": t["speaker"], "text": t.get("text", "")}
             for n, t in enumerate(turns)]
    speakers = sorted({t["speaker"] for t in turns})

    msg = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"Speakers present: {', '.join(speakers)}\n\n"
                   + json.dumps(lines, ensure_ascii=False, indent=1)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("model returned no JSON array")
    return json.loads(m.group(0))


def _apply(turns: list[dict], edits: list[dict], known: set[str]) -> tuple[list[dict], list[dict]]:
    """Rebuild the turn list from the edits, rejecting anything unsafe."""
    by_index = {e.get("i"): e for e in edits if isinstance(e, dict)}
    out: list[dict] = []
    changes: list[dict] = []

    for i, t in enumerate(turns):
        e = by_index.get(i)
        if not e:
            out.append(t)
            continue

        if "split" in e and isinstance(e["split"], list) and len(e["split"]) > 1:
            frags = e["split"]
            joined = _norm(" ".join(f.get("text", "") for f in frags))
            if joined != _norm(t.get("text", "")):
                out.append(t)                       # text drifted - reject
                continue
            if any(f.get("speaker") not in known for f in frags):
                out.append(t)
                continue
            pieces = _split_turn(t, frags)
            if pieces is None:
                out.append(t)
                continue
            out.extend(pieces)
            changes.append({"i": i, "kind": "split", "from": t["speaker"],
                            "into": [f["speaker"] for f in frags],
                            "text": t.get("text", "")})
            continue

        spk = e.get("speaker")
        if spk in known and spk != t["speaker"]:
            out.append({**t, "speaker": spk})
            changes.append({"i": i, "kind": "reassign", "from": t["speaker"],
                            "to": spk, "text": t.get("text", "")})
        else:
            out.append(t)

    return out, changes


def _split_turn(turn: dict, frags: list[dict]) -> list[dict] | None:
    """Cut a turn at word boundaries, taking times from its own words.

    Falling back to a proportion of the duration would put the boundary
    somewhere no one stopped speaking, which is the failure this whole layer
    exists to fix. Without word timings the split is refused instead.
    """
    words = turn.get("words") or []
    if not words:
        return None

    out: list[dict] = []
    cursor = 0
    for f in frags:
        n = len(_norm(f["text"]).split())
        chunk = words[cursor:cursor + n]
        if not chunk:
            return None
        cursor += n
        out.append({"speaker": f["speaker"],
                    "start": round(float(chunk[0]["start"]), 2),
                    "end": round(float(chunk[-1]["end"]), 2),
                    "text": f["text"], "words": chunk})
    if cursor != len(words):
        return None                                  # word count disagreed
    return out


def _turns_with_words(data: dict, engine: str | None) -> list[dict]:
    """Attach word timings to the chosen engine's turns, by time overlap."""
    if engine and engine != "ecapa":
        alt = (data.get("alternates") or {}).get(engine)
        if not alt:
            raise SystemExit(f"engine {engine!r} not in this response")
        turns = [dict(t) for t in alt.get("turns", [])]
    else:
        turns = [dict(t) for t in data.get("turns", [])]

    words = data.get("words") or []
    for t in turns:
        t["words"] = [w for w in words
                      if w.get("start") is not None
                      and t["start"] - 1e-3 <= float(w["start"]) < t["end"] + 1e-3]
    return turns


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="a *.compare.json from compare_diar.py")
    p.add_argument("--engine", default="sortformer")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="print what would change and write nothing")
    args = p.parse_args(argv)

    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    turns = _turns_with_words(data, args.engine)
    if not turns:
        raise SystemExit("no turns in that file")
    known = {t["speaker"] for t in turns}
    print(f"{Path(args.source).name}  engine={args.engine}  "
          f"turns={len(turns)}  speakers={len(known)}")

    import anthropic
    client = anthropic.Anthropic(api_key=_load_key())

    fixed: list[dict] = []
    changes: list[dict] = []
    for start in range(0, len(turns), BATCH):
        batch = turns[start:start + BATCH]
        try:
            edits = _ask(client, batch, start)
        except Exception as exc:
            print(f"  batch {start}: {type(exc).__name__}: {exc} - kept as is")
            fixed.extend(batch)
            continue
        local = [{**e, "i": e["i"] - start} for e in edits if "i" in e]
        got, ch = _apply(batch, local, known)
        fixed.extend(got)
        changes.extend({**c, "i": c["i"] + start} for c in ch)
        print(f"  turns {start}-{start + len(batch) - 1}: {len(ch)} change(s)")

    print(f"\n{len(changes)} change(s) over {len(turns)} turns "
          f"-> {len(fixed)} turns after splitting")
    for c in changes:
        if c["kind"] == "split":
            print(f"  [{c['i']:4}] SPLIT  {c['from']} -> {' / '.join(c['into'])}")
            print(f"         {c['text'][:88]}")
        else:
            print(f"  [{c['i']:4}] MOVE   {c['from']} -> {c['to']}")
            print(f"         {c['text'][:88]}")

    if args.dry_run:
        return 0

    from pipeline import parse as parse_mod
    out = Path(args.out or f"out/llm_speakers/{Path(args.source).stem}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tr = parse_mod.Transcript(
        source=f"{Path(args.source).stem} [{args.engine} + llm]",
        turns=[parse_mod.Turn(speaker=t["speaker"], start=t["start"], end=t["end"],
                              text=t.get("text", ""), words=[]) for t in fixed],
        speakers=sorted({t["speaker"] for t in fixed}),
        duration_sec=data.get("meta", {}).get("duration_sec", 0.0),
        word_count=len(data.get("words") or []),
        mean_word_confidence=0.0, low_confidence_words=0, unattributed_words=0)
    parse_mod.write_text(tr, out.with_suffix(".txt"))
    parse_mod.write_html(tr, out.with_suffix(".html"))
    Path(out.with_suffix(".changes.json")).write_text(
        json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out.with_suffix('.txt')}")
    print(f"      {out.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
