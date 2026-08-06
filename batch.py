"""Batch validation: run the pipeline over many recordings, produce one report.

    python batch.py <folder or files...>          process new files, report all
    python batch.py <folder> --force              reprocess everything
    python batch.py <folder> --report-only        just rebuild the report

Built for the validation round before anything ships: point it at a folder of
real calls, let it grind, and read one organised report instead of thirty
consoles. Files that already have output are skipped, so recordings can be
added to the folder incrementally and only the new ones cost anything.

The report leads with what needs human ears. The quality checks cannot hear -
they judge the shape of the output - so their proper role is triage: CLEAN
files likely need no attention, everything else is listed with its specific
problems and the timestamps to check.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import runpod_client as rp

AUDIO_EXTS = {".wav", ".wmv", ".mp3", ".opus", ".ogg", ".m4a", ".flac", ".aac", ".wma"}


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(
                f for f in sorted(path.iterdir())
                if f.suffix.lower() in AUDIO_EXTS and f.is_file()
            )
        elif path.is_file():
            files.append(path)
    # Skip our own derived audio if someone points this at the out/ folder.
    return [f for f in files if not f.name.endswith((".16k.wav", ".speech.wav"))]


def process_all(files: list[Path], out_dir: Path, force: bool,
                use_gpu: bool = True) -> list[dict]:
    creds = rp.load_credentials()
    rows: list[dict] = []

    endpoint = None
    if use_gpu:
        import gpu
        endpoint = gpu._endpoint_id(None)
    else:
        from run import process as cpu_process

    for i, f in enumerate(files, 1):
        result_path = out_dir / (f"{f.stem}.gpu.json" if use_gpu else f"{f.stem}.json")
        if result_path.exists() and not force:
            print(f"[{i}/{len(files)}] {f.name} - already processed, skipping "
                  "(--force to redo)")
            rows.append(_row_from_json(result_path, f))
            continue

        print(f"[{i}/{len(files)}] {f.name}")
        started = time.time()
        try:
            if use_gpu:
                import gpu
                out = gpu.process(f, endpoint, creds, out_dir)
                result = _gpu_to_result(out)
            else:
                result = cpu_process(f, out_dir, creds=creds)
            rows.append(_row_from_result(result, f, time.time() - started))
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            rows.append({
                "file": f.name, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return rows


def _gpu_to_result(out: dict) -> dict:
    """Adapt the worker's response to the shape the report reader expects.

    Quality is re-scored locally from the returned words rather than trusting
    the verdict stored in the response - verdict logic evolves faster than the
    worker image, and this lets a report round apply today's rules to
    yesterday's cached results without paying for GPU reruns.
    """
    words = out.get("words", [])
    quality = out.get("quality", {})
    turns = out.get("turns", [])
    if words:
        try:
            from pipeline import diarize as diar
            from pipeline import quality as quality_mod
            rebuilt = diar.turns_from_words(
                [w for w in words if w.get("speaker")])
            if rebuilt:
                quality = quality_mod.assess({"turns": rebuilt},
                                             expected_speakers=2).to_dict()
                turns = rebuilt
        except Exception:
            pass
    return {
        "quality": quality,
        "turns": [
            {"speaker": t["speaker"], "start": t["start"], "end": t["end"],
             "text": t.get("text", "")}
            for t in turns
        ],
        "duration_sec": out.get("meta", {}).get("duration_sec", 0.0),
    }


def _row_from_result(result: dict, f: Path, wall: float) -> dict:
    q = result.get("quality", {})
    share = {}
    for t in result.get("turns", []):
        share[t["speaker"]] = share.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    total = sum(share.values()) or 1.0
    return {
        "file": f.name,
        "status": "ok",
        "verdict": q.get("verdict", "unknown"),
        "speakers": q.get("speakers", 0),
        "turns": q.get("turns", 0),
        "words": q.get("words", 0),
        "confidence": q.get("mean_confidence", 0.0),
        "buried": q.get("buried_answers", 0),
        "repeated": q.get("repeated_lines", 0),
        "stretched": q.get("stretched_turns", 0),
        "share": " / ".join(f"{100 * v / total:.0f}%" for v in
                            sorted(share.values(), reverse=True)),
        "problems": q.get("problems", []),
        "duration_sec": result.get("duration_sec", 0.0),
        "wall_sec": round(wall, 1),
    }


def _row_from_json(path: Path, f: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "meta" in data and "quality" in data:   # worker response shape
            data = _gpu_to_result(data)
        return _row_from_result(data, f, 0.0)
    except Exception as exc:
        return {"file": f.name, "status": "failed", "error": f"stale output: {exc}"}


SEVERITY = {"unreliable": 0, "suspect": 1, "unknown": 2, "one-sided": 3, "empty": 4, "clean": 5}


def write_report(rows: list[dict], out_dir: Path) -> Path:
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    ok.sort(key=lambda r: (SEVERITY.get(r["verdict"], 2), -r["buried"]))

    clean = [r for r in ok if r["verdict"] in ("clean", "empty", "one-sided")]
    flagged = [r for r in ok if r["verdict"] not in ("clean", "empty", "one-sided")]
    total_audio_min = sum(r["duration_sec"] for r in ok) / 60

    lines = [
        "# דוח בדיקת תמלול — סבב אימות",
        "",
        f"**{len(rows)} הקלטות** · {total_audio_min:.0f} דקות אודיו · "
        f"{len(clean)} נקיות · {len(flagged)} מסומנות לבדיקה · {len(failed)} נכשלו",
        "",
        "המדדים בודקים את *צורת* הפלט (תשובות קבורות, שורות חוזרות, דוברים "
        "עודפים) — הם טריאז', לא פסק דין. קובץ נקי כנראה תקין; קובץ מסומן "
        "דורש האזנה בנקודות שצוינו.",
        "",
        "## סיכום",
        "",
        "| קובץ | שיפוט | דוברים | חלוקה | תש' קבורות | ביטחון |",
        "|---|---|---|---|---|---|",
    ]
    for r in ok:
        mark = {"clean": "✅", "suspect": "⚠️", "unreliable": "❌",
                "empty": "⬜", "one-sided": "📢"}.get(r["verdict"], "?")
        lines.append(
            f"| {r['file']} | {mark} {r['verdict']} | {r['speakers']} "
            f"| {r['share']} | {r['buried']} | {r['confidence']:.3f} |"
        )
    for r in failed:
        lines.append(f"| {r['file']} | 💥 נכשל | | | | |")

    if flagged:
        lines += ["", "## קבצים שדורשים האזנה", ""]
        for r in flagged:
            lines.append(f"### {r['file']}")
            for p in r["problems"]:
                lines.append(f"- {p}")
            lines.append("")

    if failed:
        lines += ["", "## כשלים", ""]
        for r in failed:
            lines.append(f"- **{r['file']}**: {r.get('error', '?')}")

    lines += [
        "",
        "---",
        f"נוצר על ידי `batch.py` · תמלולים מלאים ב-`{out_dir}/<שם>.txt`",
    ]

    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Validate the pipeline over many recordings.")
    p.add_argument("paths", nargs="+", help="Folder(s) or audio file(s)")
    p.add_argument("--out", default="out")
    p.add_argument("--force", action="store_true", help="Reprocess existing outputs")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--cpu", action="store_true",
                   help="Use the old local pipeline instead of the GPU worker")
    args = p.parse_args(argv)

    files = collect(args.paths)
    if not files:
        print("No audio files found.")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        rows = []
        for f in files:
            for name in (f"{f.stem}.gpu.json", f"{f.stem}.json"):
                if (out_dir / name).exists():
                    rows.append(_row_from_json(out_dir / name, f))
                    break
    else:
        rows = process_all(files, out_dir, args.force, use_gpu=not args.cpu)

    report = write_report(rows, out_dir)
    print(f"\nreport -> {report}")

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)} processed, "
          f"{sum(1 for r in ok if r['verdict'] == 'clean')} clean, "
          f"{sum(1 for r in ok if r['verdict'] != 'clean')} flagged, "
          f"{len(rows) - len(ok)} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
