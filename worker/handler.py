"""RunPod worker: the full proven pipeline - Hebrew ASR + speaker attribution.

This replaces the stock ivrit-ai worker for three verified reasons: it drops
every decoding parameter on the floor (so VAD and pinned decoding are
impossible through it), its diarization labels whole Whisper segments (which
merged both parties of a call into a 96/4 split), and its unpinned decoding
returns different words for the same audio between runs.

Everything here was validated locally on real call-centre recordings first;
this file only moves that chain onto the GPU:

  ASR         faster-whisper, ivrit-ai Hebrew model, decoding pinned
              (temperature 0, fixed beam), built-in Silero VAD with thresholds
              tuned on real calls - hallucination guard AND original-clock
              timestamps, no client-side trimming or remapping needed.
  diarize     ECAPA embeddings in sliding windows, multi-method clustering
              with centroid purification (marginal telephone audio flips
              between correct and degenerate without it).
  refine      each word re-judged against the two voice signatures.
  realign     mid-sentence speaker flips voted away on Whisper's punctuation.
  split       text after a mid-turn question goes to the other party; voice
              veto, overruled by a >=6 dB loudness gap (verified case: answer
              14 dB hotter while the embedding insisted "same speaker").
  quality     shape-based scoring - the decoder's own no-speech probability
              read 0.000 on five minutes of hallucinated text, so the model's
              self-assessment is not usable.

The pipeline/ package is copied into the image and imported as-is - one code
path, tested locally, run on GPU.
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
import traceback
import urllib.request

import runpod

# The full model, not turbo. On ivrit.ai's own leaderboard the non-turbo model
# leads on every eval set (0.051 vs 0.053 WER on eval-d1, 0.081 vs 0.082 on
# kan) - a small margin on clean audio, but those benchmarks were measured on
# clean audio, and ours is 8-bit narrowband telephone where the harder model
# has more room to matter. Turbo is roughly 2x faster; at 3 agorot per hour of
# audio that is not a trade worth making.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "ivrit-ai/whisper-large-v3-ct2")

# Loaded once at module scope. RunPod's FlashBoot snapshots the live process,
# so models loaded lazily inside the handler would be reloaded on every cold
# start - at billed time.
_asr = None


def _load_models() -> None:
    global _asr
    from faster_whisper import WhisperModel
    _asr = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

    # Warm the diarization encoder too (env ECAPA_DEVICE=cuda in the image).
    from pipeline import diarize as diar
    diar._get_encoder()


def _fetch_audio(job_input: dict) -> str:
    """Materialise the request's audio as a 16 kHz mono wav path."""
    suffix = job_input.get("audio_format", "bin")
    fd, raw = tempfile.mkstemp(suffix=f".{suffix}")
    os.close(fd)

    if job_input.get("audio_base64"):
        with open(raw, "wb") as f:
            f.write(base64.b64decode(job_input["audio_base64"]))
    elif job_input.get("audio_url"):
        urllib.request.urlretrieve(job_input["audio_url"], raw)
    else:
        raise ValueError("Provide either audio_base64 or audio_url")

    wav = raw + ".16k.wav"
    # soxr without dither: dither is deliberate random noise and made
    # supposedly identical runs differ all the way down the chain.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", raw,
         "-af", "aresample=resampler=soxr:precision=28:dither_method=none:osr=16000",
         "-ac", "1", "-c:a", "pcm_s16le", wav],
        check=True, capture_output=True,
    )
    os.unlink(raw)
    return wav


def _transcribe(wav: str, job_input: dict) -> tuple[list[dict], dict]:
    segments, info = _asr.transcribe(
        wav,
        language=job_input.get("language", "he"),
        word_timestamps=True,
        # Tuned on real call-centre recordings: 0.4 rather than the usual 0.5,
        # chosen by sweeping and scoring on signals hallucination cannot
        # inflate (speaker count, repeated lines) - never on word count.
        vad_filter=job_input.get("vad_filter", True),
        vad_parameters=dict(
            threshold=float(job_input.get("vad_threshold", 0.4)),
            min_speech_duration_ms=100,
            min_silence_duration_ms=500,
            speech_pad_ms=250,
            max_speech_duration_s=25.0,
        ),
        # The single biggest anti-hallucination lever: one invented segment
        # must not seed a repetition loop through the rest of the recording.
        condition_on_previous_text=False,
        # A single temperature, not the default fallback ladder. The ladder
        # re-decodes with sampling whenever a quality check fails, which makes
        # the output non-reproducible exactly on the hard segments - two runs
        # of the same file differed by 27 words and by 4 buried answers, enough
        # to flip a file's verdict. A failed segment is better left failed and
        # visible than silently re-rolled.
        temperature=[0.0],
        beam_size=int(job_input.get("beam_size", 5)),
        hotwords=job_input.get("hotwords") or None,
    )

    words: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "word": w.word.strip(),
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "probability": round(float(w.probability), 4),
            })

    words, looped = _collapse_repetition_loops(words)
    if looped:
        meta_note = f"collapsed {looped} decoder repetition loop(s)"
    else:
        meta_note = None

    meta = {
        "model": WHISPER_MODEL,
        "language": info.language,
        "duration_sec": round(float(info.duration), 2),
        "duration_after_vad_sec": round(
            float(getattr(info, "duration_after_vad", info.duration)), 2),
    }
    if meta_note:
        meta["notes"] = [meta_note]
    return words, meta


def _collapse_repetition_loops(
    words: list[dict], max_cycles: int = 3
) -> tuple[list[dict], int]:
    """Cut the decoder's repetition loops out of the word stream.

    Letter-by-letter dictation over a phone line - someone spelling out an
    email address, slowly, with pauses - produces periodic audio that can trap
    Whisper's decoder in a loop: one real call came back with ".u .p" repeated
    over forty times. condition_on_previous_text=False blocks contamination
    *between* windows but not a loop *within* one, so this cleans up after it.

    Detection is deterministic: a cycle of one or two normalised tokens
    repeating more than max_cycles consecutive times is not speech (nobody
    says "u p" forty times), so everything past the third cycle goes. Genuine
    short repetitions - "לא, לא, לא", "רגע רגע" - stay, because three cycles
    are kept.
    """
    def norm(w: dict) -> str:
        return (w.get("word") or "").strip(" .,!?").lower()

    out: list[dict] = []
    removed = 0
    loops = 0
    i = 0
    n = len(words)
    while i < n:
        cut = False
        for cycle_len in (1, 2):
            if i + cycle_len * (max_cycles + 1) > n:
                continue
            cycle = [norm(words[i + k]) for k in range(cycle_len)]
            if not all(cycle):
                continue
            reps = 1
            j = i + cycle_len
            while (j + cycle_len <= n
                   and [norm(words[j + k]) for k in range(cycle_len)] == cycle):
                reps += 1
                j += cycle_len
            if reps > max_cycles:
                keep = i + cycle_len * max_cycles
                out.extend(words[i:keep])
                removed += j - keep
                loops += 1
                i = j
                cut = True
                break
        if not cut:
            out.append(words[i])
            i += 1
    return out, loops


def _run_nemo_engines(
    wav: str, words: list[dict], engines: list[str], num_speakers: int
) -> dict:
    """Run whichever NeMo diarizers were requested, each on the same words.

    Failures are captured per engine rather than raised: one model being
    unavailable should not cost the caller the transcript, and a partial
    comparison is still a comparison.
    """
    from pipeline import quality

    wanted = [e for e in engines if e in ("sortformer", "msdd")]
    if not wanted:
        return {}

    from pipeline import nemo_diar

    out: dict = {}
    for name in wanted:
        try:
            fn = (nemo_diar.diarize_sortformer if name == "sortformer"
                  else nemo_diar.diarize_msdd)
            res = fn(wav, num_speakers=num_speakers)
            labelled = nemo_diar.assign_words(words, res.turns)
            keep = ("word", "start", "end", "speaker", "probability")
            labelled = [{k: w.get(k) for k in keep} for w in labelled]

            from pipeline import diarize as diar
            labelled = diar.realign_by_punctuation(labelled)
            turns = diar.turns_from_words(labelled)
            report = quality.assess({"turns": turns}, expected_speakers=num_speakers)

            out[name] = {
                "words": labelled,
                "turns": [{"speaker": t["speaker"], "start": round(t["start"], 2),
                           "end": round(t["end"], 2), "text": t["text"]}
                          for t in turns],
                "speakers": sorted({t["speaker"] for t in turns}),
                "quality": report.to_dict(),
                "meta": {"seconds": res.seconds,
                         "dropped_speech_sec": res.dropped_speech_sec,
                         "notes": res.notes},
            }
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-1200:]}
    return out


def handler(job):
    started = time.time()
    job_input = job.get("input") or {}
    wav = None
    try:
        wav = _fetch_audio(job_input)

        t0 = time.time()
        words, meta = _transcribe(wav, job_input)
        meta["asr_sec"] = round(time.time() - t0, 2)

        result: dict = {"meta": meta, "words": words}

        if job_input.get("diarize", True) and words:
            from pipeline import diarize as diar
            from pipeline import quality

            num_speakers = int(job_input.get("num_speakers", 2))

            # Sortformer and MSDD both replace the sliding-window embedder with
            # a network that segments at frame resolution and was trained on
            # telephone speech. They are requested by name so one call can run
            # several and return them side by side for comparison.
            engines = job_input.get("engines") or ["ecapa"]
            if isinstance(engines, str):
                engines = [engines]
            extra = _run_nemo_engines(wav, words, engines, num_speakers)
            # Promote the first engine that actually produced turns. Naming the
            # first requested one unconditionally would crash on a failed
            # engine, whose entry holds an error and nothing else - and MSDD is
            # now allowed to be absent from the image entirely.
            lead = next((e for e in engines
                         if e in extra and "error" not in extra[e]), None)
            if lead and "ecapa" not in engines:
                primary = extra[lead]
                result.update({
                    "words": primary["words"],
                    "turns": primary["turns"],
                    "speakers": primary["speakers"],
                    "quality": primary["quality"],
                    "diarization": {"engine": lead, **primary["meta"]},
                })
                result["alternates"] = {
                    k: {"turns": v.get("turns", []),
                        "quality": v.get("quality", {}),
                        "meta": v.get("meta") or {"error": v.get("error")}}
                    for k, v in extra.items() if k != lead
                }
                result["meta"]["total_sec"] = round(time.time() - started, 2)
                return result

            t0 = time.time()
            # Feed the diarizer the same speech the ASR heard. faster-whisper
            # applies VAD internally, so without this the diarizer alone sees
            # the silence, hold music and line noise - and embeds the channel
            # instead of the speakers.
            from pipeline import vad as vad_mod
            spans, _total = vad_mod.detect_speech(
                wav, threshold=float(job_input.get("vad_threshold", 0.4))
            )
            speech_spans = [(s.start, s.end) for s in spans]
            local = diar.diarize(wav, num_speakers=num_speakers,
                                 speech_spans=speech_spans)
            labelled = diar.assign_words(words, local.turns)
            flips = handed = 0
            if local.centroids is not None:
                labelled, flips = diar.refine_word_speakers(
                    wav, labelled, local.centroids)
            labelled = diar.realign_by_punctuation(labelled)
            if local.centroids is not None and num_speakers == 2:
                # When the two voices barely separate - or one label swallowed
                # the call - the centroids are not evidence, and the splitter
                # must not let them veto. Loudness decides alone there.
                pre = diar.turns_from_words(labelled)
                share: dict[str, float] = {}
                for t in pre:
                    share[t["speaker"]] = share.get(t["speaker"], 0.0) + \
                        (t["end"] - t["start"])
                dominant = max(share.values()) / (sum(share.values()) or 1.0)
                weak = local.separation < 0.15 or dominant > 0.85
                labelled, handed = diar.split_buried_answers(
                    wav, labelled, local.centroids, weak_voice=weak)

            keep = ("word", "start", "end", "speaker", "probability")
            labelled = [{k: w.get(k) for k in keep} for w in labelled]
            turns = diar.turns_from_words(labelled)

            report = quality.assess(
                {"turns": turns}, expected_speakers=num_speakers)

            result.update({
                "words": labelled,
                "turns": [
                    {"speaker": t["speaker"], "start": round(t["start"], 2),
                     "end": round(t["end"], 2), "text": t["text"]}
                    for t in turns
                ],
                "speakers": sorted({t["speaker"] for t in turns}),
                "quality": report.to_dict(),
                "diarization": {
                    "separation": local.separation,
                    "windows": local.window_count,
                    "words_rejudged": flips,
                    "answers_handed_over": handed,
                    "seconds": round(time.time() - t0, 2),
                    "notes": local.notes,
                },
            })

            # ECAPA leads when it was asked for, but the NeMo runs already
            # happened over these same words - dropping them would throw away
            # the only comparison that holds the transcript constant.
            if extra:
                result["alternates"] = {
                    k: {"turns": v.get("turns", []),
                        "quality": v.get("quality", {}),
                        "meta": v.get("meta") or {"error": v.get("error")}}
                    for k, v in extra.items()
                }

        result["meta"]["total_sec"] = round(time.time() - started, 2)
        return result

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}
    finally:
        if wav and os.path.exists(wav):
            os.unlink(wav)


# cuBLAS only guarantees run-to-run reproducibility when it is given a fixed
# workspace; without this, the same audio decodes differently between runs and
# Whisper amplifies it - one differing timestamp token moves the next 30-second
# window and re-decodes the rest of the file. Measured here as 620 vs 637 words
# on identical input. Must be set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

if not torch.cuda.is_available():
    # Better a dead worker than one silently billing CPU-speed inference.
    raise SystemExit("GPU health check failed: CUDA is not available")

_load_models()
print("models ready", flush=True)

runpod.serverless.start({"handler": handler})
