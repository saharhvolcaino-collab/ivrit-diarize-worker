"""Run every diarizer over one transcript and let the numbers choose.

    python compare_diar.py <recording...>

The three chains share the same ASR output, so any difference between them is
the diarizer alone. That matters more than it sounds: comparing separately
transcribed runs would fold GPU nondeterminism into the comparison and make a
diarizer look better or worse for reasons that have nothing to do with it.

Judge by `buried` - turns holding both a question and its answer. Silhouette
separation measures how tight the clusters are, not whether they are right,
and DER-style numbers need reference labels we do not have. Buried answers
count the failure the transcript actually exhibits.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import parse as parse_mod
from pipeline import preprocess as prep_mod
from pipeline import runpod_client as rp

API = "https://api.runpod.ai/v2"
ENGINES = ["ecapa", "sortformer", "msdd", "pyannote"]


def _submit(blob: str, endpoint: str, creds, engines: list[str],
            num_speakers: int | None = 2, align: str | None = None,
            align_min_score: float = 0.0, verify: bool = False) -> dict:
    payload = {"input": {
        "audio_base64": blob,
        "audio_format": "opus",
        "language": "he",
        "diarize": True,
        "num_speakers": num_speakers,
        "engines": engines,
        "align": align,
        "align_min_score": align_min_score,
        "verify": verify,
    }}
    job = rp._request(f"{API}/{endpoint}/run", creds.api_key, payload)
    state = rp.poll(rp.Credentials(api_key=creds.api_key, endpoint_id=endpoint),
                    job["id"], timeout_sec=2400)
    out = state.get("output") or {}
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _row(name: str, q: dict, meta: dict | None = None) -> str:
    share = ""
    secs = f"{meta.get('seconds', 0):5.1f}s" if meta else "     -"
    return (f"  {name:12} {q.get('speakers', 0):4} {q.get('turns', 0):6} "
            f"{q.get('buried_answers', 0):7} "
            f"{q.get('dominant_speaker_pct', 0):7.0f}% {secs}  "
            f"{q.get('verdict', '?')}")


def compare(audio: Path, endpoint: str, creds, out_dir: Path,
            num_speakers: int | None = 2, align: str | None = None,
            align_min_score: float = 0.0, verify: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {audio.name} ===")

    prep = prep_mod.preprocess(audio, out_dir, make_opus=True)
    blob = base64.b64encode(Path(prep.opus_path).read_bytes()).decode("ascii")

    started = time.time()
    # One request, every engine: same audio, same words, same GPU state.
    out = _submit(blob, endpoint, creds, ENGINES, num_speakers, align,
                  align_min_score, verify)
    (out_dir / f"{audio.stem}.compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    results: dict[str, dict] = {}
    primary_engine = ENGINES[0]
    results[primary_engine] = {
        "quality": out.get("quality", {}), "turns": out.get("turns", []),
        "meta": out.get("diarization", {}),
    }
    for name, alt in (out.get("alternates") or {}).items():
        results[name] = {"quality": alt.get("quality", {}),
                         "turns": alt.get("turns", []),
                         "meta": alt.get("meta", {})}

    ver = (out.get("meta") or {}).get("verification")
    if ver:
        print(f"  two-model check: {ver['agreement_pct']}% agreement "
              f"({ver['agreed']} agreed, {ver['contested']} contested, "
              f"{ver['only_primary']}+{ver['only_secondary']} one-sided)")
        for d in ver["disagreements"][:12]:
            a, b = [v for k, v in d.items() if k.startswith("whisper")][:2]
            print(f"     [{d['start']:7.1f}]  {a!r}  vs  {b!r}")
        if len(ver["disagreements"]) > 12:
            print(f"     ... {len(ver['disagreements']) - 12} more")
    caps = (out.get("meta") or {}).get("capabilities")
    if caps:
        print(f"  worker has: {', '.join(caps)}")
    al = (out.get("meta") or {}).get("alignment")
    if al:
        print(f"  alignment: {al['engine']} moved {al['moved']} words, "
              f"mean score {al['mean_score']}, {al['failed_chunks']} chunk(s) failed")
    print(f"  {'engine':12} {'spk':>4} {'turns':>6} {'buried':>7} "
          f"{'domin':>8} {'time':>7}  verdict")
    print("  " + "-" * 62)
    for name in ENGINES:
        r = results.get(name)
        if not r:
            print(f"  {name:12} (not returned)")
            continue
        if "error" in r.get("meta", {}):
            print(f"  {name:12} FAILED: {str(r['meta']['error'])[:44]}")
            continue
        print(_row(name, r["quality"], r["meta"]))
        for n in r["meta"].get("notes") or []:
            print(f"    [!] {n}")

        # One directory per engine, same filename inside each. Reading the
        # same call across methods is then a matter of opening the same name
        # in two folders, rather than picking suffixes apart in one listing.
        turns = [parse_mod.Turn(speaker=t["speaker"], start=t["start"],
                                end=t["end"], text=t.get("text", ""), words=[])
                 for t in r["turns"]]
        if turns:
            t_obj = parse_mod.Transcript(
                source=f"{audio.name} [{name}]", turns=turns,
                speakers=sorted({t.speaker for t in turns}),
                duration_sec=out.get("meta", {}).get("duration_sec", 0.0),
                word_count=r["quality"].get("words", 0),
                mean_word_confidence=r["quality"].get("mean_confidence", 0.0),
                low_confidence_words=r["quality"].get("low_confidence_words", 0),
                unattributed_words=0,
            )
            eng_dir = out_dir / name
            eng_dir.mkdir(parents=True, exist_ok=True)
            parse_mod.write_text(t_obj, eng_dir / f"{audio.stem}.txt")
            parse_mod.write_html(t_obj, eng_dir / f"{audio.stem}.html",
                                 quality=r["quality"])

    best = min(
        (n for n in results if not results[n].get("meta", {}).get("error")),
        key=lambda n: (results[n]["quality"].get("buried_answers", 99),
                       abs(results[n]["quality"].get("dominant_speaker_pct", 100) - 50)),
        default=None,
    )
    if best:
        print(f"  -> fewest buried answers: {best}")
    print(f"  {time.time() - started:.0f}s total")
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    import gpu as gpu_mod

    p = argparse.ArgumentParser(description="Compare diarization engines.")
    p.add_argument("audio", nargs="+")
    p.add_argument("--out", default="out/diarcmp")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--speakers", default="2",
                   help='"auto" lets each engine count the speakers itself')
    p.add_argument("--align", default=None, choices=[None, "ctc"],
                   help="ctc = re-time words by forced alignment (whisperX's way)")
    p.add_argument("--verify", action="store_true",
                   help="transcribe with a second model too; agreement marks "
                        "what is certain and disagreement locates the errors")
    p.add_argument("--align-min-score", type=float, default=0.0,
                   help="only accept alignments at least this confident; "
                        "below it a word keeps the timestamp it already had")
    args = p.parse_args(argv)

    creds = rp.load_credentials()
    endpoint = gpu_mod._endpoint_id(args.endpoint)
    out_dir = Path(args.out)

    summary: dict[str, dict] = {}
    for item in args.audio:
        try:
            n = None if args.speakers in ("auto", "", "none") else int(args.speakers)
            summary[Path(item).name] = compare(
                Path(item), endpoint, creds, out_dir, n, args.align,
                args.align_min_score, args.verify)
        except Exception as exc:
            print(f"  ERROR {Path(item).name}: {exc}")

    print(f"\n{'=' * 66}\nBURIED ANSWERS BY ENGINE (lower is better)\n{'=' * 66}")
    print(f"  {'file':30} " + " ".join(f"{e:>10}" for e in ENGINES))
    for name, res in summary.items():
        cells = []
        for e in ENGINES:
            q = res.get(e, {}).get("quality")
            cells.append(f"{q['buried_answers']:>10}" if q else f"{'-':>10}")
        print(f"  {name[:30]:30} " + " ".join(cells))

    Path(out_dir, "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
