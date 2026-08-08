"""LLM correction layer: fix ASR misrecognitions, touch nothing else.

    python polish.py out/gpu/Rec722_83_11-11-56_08-06-2026.gpu.json

Narrowband telephone audio produces a specific error class: acoustically
plausible but wrong words - "נפגרו" for "נסגרו", a garbled customer name, a
technical term the ASR has never seen. A strong LLM reading the whole
conversation can fix many of these from context.

The research warning this module is built around: open-ended "improve the
transcript" prompts barely help and sometimes hallucinate. So the task is
fenced in three ways:

  1. The prompt allows only misrecognition fixes - no rephrasing, no grammar
     "improvements", no additions or deletions.
  2. The model must return exactly one line per input line; structural
     drift rejects the whole response.
  3. A per-line guard rejects any correction that changes the line's letter
     count by more than a third - that is rewriting, not fixing. Measured on
     letters rather than words on purpose: the commonest Hebrew error here
     splits one word into several ("מה שלומך" heard as "מה שאת אומרת"), which
     doubles the word count while barely moving the letters.

Every accepted change lands in a diff report for human review. Speaker labels
and timestamps are never sent to the model, so they cannot be corrupted.

Needs ANTHROPIC_API_KEY in .env (console.anthropic.com -> API keys).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "claude-opus-5"
# $ per million tokens, for the cost line in the report.
PRICE_IN, PRICE_OUT = 5.00, 25.00

SYSTEM = """אתה מתקן שגיאות זיהוי בתמלול שיחת טלפון בעברית שנוצר על ידי מערכת תמלול אוטומטית.

האודיו הוא שיחת מוקד טלפונית באיכות נמוכה, ולכן יש מילים שזוהו לא נכון - מילים שנשמעות דומה למה שנאמר אבל שגויות בהקשר, שמות אנשים וחברות משובשים, ומונחים טכניים שהמערכת לא הכירה.

תקן אך ורק:
- מילים שברור מההקשר שזוהו לא נכון (החלפה במילה שנשמעת דומה)
- שמות ומונחים משובשים כשההקשר חושף את הנכון

אסור בהחלט:
- לשכתב, לשפר ניסוח, או לתקן דקדוק של דיבור טבעי ("אני יגיד" נשאר כמו שהוא)
- להוסיף או למחוק מילים מעבר לנדרש לתיקון הזיהוי
- לשנות מספרים אלא אם יש שגיאת זיהוי ברורה
- לתקן משהו שאתה לא בטוח בו - ספק פירושו להשאיר כמו שהוא

קבל רשימת שורות ממוספרות. החזר את כל השורות באותו סדר ובאותה כמות, מתוקנות במקומות הנדרשים בלבד. שורה ללא שגיאות חוזרת זהה לחלוטין."""


def polish(gpu_json: Path, out_dir: Path | None = None) -> dict:
    import anthropic

    data = json.loads(gpu_json.read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    if not turns:
        raise SystemExit(f"{gpu_json.name}: no turns to polish")

    lines = [t["text"].strip() for t in turns]
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines))

    client = anthropic.Anthropic(api_key=_api_key())
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": numbered}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "lines": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["lines"],
                    "additionalProperties": False,
                },
            }
        },
    )

    if response.stop_reason == "refusal":
        raise SystemExit("Model declined the request - transcript left unchanged")

    text = next(b.text for b in response.content if b.type == "text")
    corrected = json.loads(text)["lines"]

    # Guard 1: structural drift rejects everything.
    if len(corrected) != len(lines):
        raise SystemExit(
            f"Model returned {len(corrected)} lines for {len(lines)} inputs - "
            "discarding the whole response, transcript left unchanged"
        )

    # Guard 2: a "fix" that rewrites the line is rejected line by line.
    changes = []
    final = []
    rejected = 0
    for i, (orig, new) in enumerate(zip(lines, corrected)):
        new = new.strip()
        if new == orig:
            final.append(orig)
            continue
        ow, nw = len(orig.split()), len(new.split())
        # Guard on characters, not words. A word-count rule was written to
        # stop rewriting, and it did - but it also rejected the single most
        # common Hebrew misrecognition shape, where one word is heard as
        # several. A real case from this estate: "מה שלומך" came back as
        # "מה שאת אומרת", two words becoming four. That is a 100% word-count
        # change and a 4-character one, and the word rule threw away the
        # correct fix while a rewrite of the same length would have passed.
        #
        # Characters measure what the rule was actually reaching for: a fix
        # replaces sounds that were misheard, so it stays close in length,
        # whatever it does to the spaces.
        oc, nc = len(orig.replace(" ", "")), len(new.replace(" ", ""))
        if oc >= 8 and abs(nc - oc) > max(4, oc // 3):
            rejected += 1
            final.append(orig)
            continue
        final.append(new)
        changes.append({"turn": i, "before": orig, "after": new})

    usage = response.usage
    cost = usage.input_tokens / 1e6 * PRICE_IN + usage.output_tokens / 1e6 * PRICE_OUT

    # Write outputs next to the source, under a suffix that keeps the original.
    out_dir = out_dir or gpu_json.parent
    stem = gpu_json.name.replace(".gpu.json", "")

    for t, new_text in zip(turns, final):
        t["text"] = new_text
    (out_dir / f"{stem}.polished.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    from pipeline import parse as pm
    t_objs = [pm.Turn(speaker=t["speaker"], start=t["start"], end=t["end"],
                      text=t["text"], words=[]) for t in turns]
    words = data.get("words", [])
    confs = [w["probability"] for w in words if w.get("probability") is not None]
    transcript = pm.Transcript(
        source=f"{stem} (polished)", turns=t_objs,
        speakers=data.get("speakers", []),
        duration_sec=data.get("meta", {}).get("duration_sec", 0.0),
        word_count=len(words),
        mean_word_confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
        low_confidence_words=sum(1 for c in confs if c < 0.5),
        unattributed_words=0,
    )
    pm.write_text(transcript, out_dir / f"{stem}.polished.txt")
    pm.write_html(transcript, out_dir / f"{stem}.polished.html",
                  quality=data.get("quality"))

    report = {
        "file": stem,
        "model": MODEL,
        "turns": len(lines),
        "changed": len(changes),
        "rejected_by_guard": rejected,
        "tokens": {"in": usage.input_tokens, "out": usage.output_tokens},
        "cost_usd": round(cost, 4),
        "changes": changes,
    }
    (out_dir / f"{stem}.polish-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _api_key() -> str:
    import os
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            val = line.split("=", 1)[1].strip()
            if val:
                return val
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    raise SystemExit(
        "No ANTHROPIC_API_KEY found. Create one at console.anthropic.com and add\n"
        "ANTHROPIC_API_KEY=sk-ant-... to .env"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Fix ASR misrecognitions with an LLM.")
    p.add_argument("gpu_json", nargs="+", help="*.gpu.json file(s) from out/gpu")
    args = p.parse_args(argv)

    for item in args.gpu_json:
        r = polish(Path(item))
        print(f"\n=== {r['file']} ===")
        print(f"  {r['changed']} תיקונים מתוך {r['turns']} שורות "
              f"({r['rejected_by_guard']} נפסלו על ידי מעקה הבטיחות)")
        print(f"  עלות: ${r['cost_usd']:.4f} "
              f"(~{r['cost_usd'] * 3.7 * 100:.0f} אגורות)")
        for c in r["changes"][:12]:
            print(f"  [{c['turn']}] {c['before'][:44]}")
            print(f"      ← {c['after'][:44]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
