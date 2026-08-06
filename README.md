# Hebrew call transcription + speaker diarization

A process, not a product. The goal is to establish whether a self-hosted
RunPod pipeline beats the current transcription vendor on real call-centre
audio, and to measure the difference rather than assert it.

## What we are working with

Two real recordings, and they do not share a format:

| | `Rec601` | `Rec617` |
|---|---|---|
| Codec | GSM 6.10 (`gsm_ms`) | PCM 8-bit linear |
| Sample rate | 8000 Hz | 11025 Hz |
| Bitrate | 13 kbps | 88 kbps |
| Measured bandwidth | 3559 Hz | 2845 Hz |
| Duration | 9.8 min | 2.7 min |
| Speech / non-speech | 39.5% / 60.5% | 54.3% / 45.7% |
| Estimated SNR | 19 dB | 32.4 dB |
| Speech segments | 256 (median 0.24 s) | 48 (median 1.91 s) |
| Energy bimodality | 1.34 | 1.68 |

All mono. Both narrowband. The estate is heterogeneous, so the pipeline
detects format per file instead of assuming one.

### What that implies

**The codec hurts speaker identity more than transcription.** GSM 6.10 is a
parametric vocoder - it transmits filter coefficients and an excitation
signal, so what comes out is a synthetic reconstruction rather than a
recording. 8-bit linear PCM is differently bad: uniform quantisation noise
means quiet speech is buried, where G.711 mu-law would have companded it.
Both destroy the fine spectral detail voiceprints depend on. Expect the
voiceprint stage to be the weakest link, and measure it separately.

**Silence is the main hallucination risk.** Whisper fabricates text on
non-speech at very high rates (~100% on pure silence in published tests).
`Rec601` is 60% non-speech with quarter-second speech fragments - the
signature of a silence-suppressing recorder. VAD is not an optimisation
here, it is the primary guard, together with `condition_on_previous_text=False`
so one hallucination cannot seed a repetition loop through the rest of the file.

**Energy bimodality is the one free advantage.** In both files the two
parties sit at clearly different levels, which gives diarization a cue
beyond timbre alone.

## Decisions, and why

### Do not use the ivrit-ai worker as-is

`ivrit-ai/runpod-serverless` is the obvious starting point - actively
maintained, models baked into the image, diarization flag present. We take
its two good ideas (the Hebrew-tuned model, baking weights into the image)
and reject the rest, for three reasons:

1. **It silently drops decoding parameters.** In `ivrit-py`, `audio.py:1069`
   accepts `**kwargs` and forwards none of them. `beam_size`, `vad_filter`,
   and `initial_prompt` travel all the way to the GPU and are discarded at the
   final hop, with no error. You always get library defaults while believing
   you tuned something. VAD in particular is the setting we most need.
2. **It pins pyannote 3.1**, which is the weak choice for two-speaker
   telephony (see below).
3. **No voiceprint stage**, which is a hard requirement here.

### Diarization: Sortformer over pyannote

On CALLHOME - telephone speech, 8 kHz, narrowband, mono, two speakers, i.e.
the same audio family as ours - the published gap is large:

| Model | DER |
|---|---|
| pyannote 3.1 | 28.5% |
| pyannote community-1 | 26.7% |
| NVIDIA Streaming Sortformer v2 | ~5.3% |

The protocols differ (Sortformer reports with a 0.25 s collar, pyannote with
0), which accounts for some of the gap but nowhere near all of it.

Sortformer's known failure is collapse above 4 speakers - 41% DER. We have
exactly 2, so the weakness does not apply. Use `v2` specifically: `v1` is
CC-BY-NC and cannot be used commercially.

This is a hypothesis to test, not a settled result. The benchmark is English
telephony; nobody has published a Hebrew DER for any diarizer. We run both
models on real audio and decide from that.

### Preprocessing: resample, and stop

No denoising. Three independent 2025-26 studies found neural speech
enhancement makes Whisper worse - in one, enhanced audio lost to raw noisy
audio in 40 of 40 configurations. Whisper's noise robustness was learned from
unprocessed audio; cleaning moves the signal out of distribution.

No bandwidth extension. Every 8 kHz to 16 kHz super-resolution model is
generative. Inventing the wrong fricative fabricates the exact cue the model
needs. None of them report Whisper WER.

No AGC or compression. Raising the noise floor between words feeds the
hallucination failure mode.

Peak normalisation only - pure linear gain - and even that is skipped when
the peak is already at full scale, as it is in both sample files.

### Transport

OPUS at 24 kbps carries the 9.8-minute call in 1.5 MB, so base64 lands around
2 MB against RunPod's hard 10 MB cap on `/run`. Calls up to roughly 45 minutes
can go inline; beyond that they need a presigned URL. No object storage
needed for the proof of concept.

## Pipeline

```
probe       detect format, bandwidth and channel layout per file
preprocess  soxr resample to 16 kHz mono, peak normalise
vad         Silero VAD, threshold 0.4 - cut silence, keep a timeline map
transcribe  ivrit-ai/whisper-large-v3-turbo-ct2 on RunPod, with diarization
diarize     second opinion: ECAPA voice embeddings in 1.5 s sliding windows
            (0.25 s hop), agglomerative clustering to the known speaker count.
            CPU-only, runs beside the pipeline, no GPU worker needed.
realign     mid-sentence speaker flips voted away sentence-by-sentence,
            using Whisper's own Hebrew punctuation
arbitrate   both attributions are scored by the quality checks and the less
            faulted one wins, per file - neither source wins everywhere
remap       put timestamps back on the original recording's clock
collapse    fold spurious minor speakers into the two real ones
quality     score the transcript for hallucination and attribution artefacts
```

One command: `python run.py <recording>`

Only transcription touches RunPod. Everything else is CPU work that belongs on
the calling server: the voice-embedding model is ~20 MB and a call takes
seconds to diarize, so shipping it to a GPU would cost more in cold start than
it saves in compute. When a custom worker image is eventually built, the same
logic moves into the worker unchanged (`worker/handler.py` already mirrors it).

### Why two diarizers and an arbiter

The worker labels speakers per Whisper segment, and Whisper segments never
break on a speaker change - so on one call it merged both parties into a
96%/4% split with answers buried inside the asker's turns ("what is your
name?" and the reply "Mendy", same speaker). The local window pass fixes
exactly that, but on another call - where the worker was right, and the 88/12
split was confirmed correct by ear - the local pass drifted to 94/6. Opposite
failure modes, automatically distinguishable by the buried-answer check, so
the choice is made per file:

| file | worker said | local said | arbiter chose | outcome |
|---|---|---|---|---|
| Rec720 | 94/6, 5 buried | 53/47, 3 buried | **local** | both parties recovered |
| Rec617 | 51/49, 2 buried | 53/47, 3 buried | **worker** | kept the better one |
| Rec601 | 88/12, 1 buried | 94/6, 3 issues | **worker** | kept the confirmed split |

## Results

Measured on both sample recordings. "Baseline" is the stock RunPod template
with the full recording sent as-is; "final" is this pipeline.

| | baseline | final |
|---|---|---|
| **Rec601** (15% speech) | | |
| verdict | unreliable | **clean** |
| word confidence | 0.645 | **0.883** |
| weak words | 30.3% | **2.9%** |
| turns stretched over silence | 5 | **0** |
| speakers found | 8 | **2** |
| GPU seconds | 14.5 | **6.3** |
| **Rec617** (91% speech) | | |
| verdict | clean | clean |
| word confidence | 0.943 | 0.938 |
| speakers found | 3 | **2** |

The gain is concentrated exactly where the theory said it would be. On the
sparse recording the baseline invented five minutes of Knesset boilerplate -
`תודה רבה, אדוני היושב-ראש` repeated eight times - and scattered it across
eight speakers. On the dense recording, where there is little silence to cut,
the two are equivalent. VAD trimming is not a general accuracy improvement; it
removes a specific failure that silence causes.

Cost, measured: about **0.3 agorot per minute** of audio, or roughly 5 agorot
for a ten-minute call.

## What we found along the way

**The stock worker discards decoding parameters.** `ivrit-py` accepts
`**kwargs` at `audio.py:1069` and forwards none of them. `vad_filter`,
`beam_size` and `hotwords` reach the GPU and are dropped there with no error.
This is why VAD runs locally here rather than being switched on in the request.

**Its pyannote engine does not work.** Passing
`diarization_args={"engine": "pyannote"}` fails with a PyTorch 2.6
`weights_only` error. The only functioning diarizer in that image is its own
`ivrit` engine - embeddings plus clustering, with no overlap handling.

**`min_speakers`/`max_speakers` change nothing.** After trimming, the engine
already returns two speakers, so bounding it is a no-op. Verified, not assumed.

**Whisper's own silence detector is useless here.** Every hallucinated segment
came back with `no_speech_prob = 0.000`. Quality has to be judged on the shape
of the output - repeated lines, turns far below conversational speed, excess
speakers - because the model's self-assessment does not fire.

**Word count is a trap as a quality metric.** A VAD threshold of 0.3 produced
the most words of any setting and was by far the worst output: six speakers and
a line repeated six times. Thresholds were chosen on signals hallucination
cannot inflate.

**The pipeline is deterministic.** Five identical runs per file returned
identical results, standard deviation zero on every metric. An apparent
variance turned out to be a hard-coded CLI default overriding the tuned one.

## Not done

**Speaker boundary precision.** The diarizer still misses changes inside a
Whisper segment - one call has `איזה יוזר?` attributed to the wrong party
because the segment holding it never got split. Fixing this needs a diarizer
run independently of the ASR, which means a custom worker.

**Real speaker names.** Voiceprint enrollment. Expect this to be the hardest
part on this audio: GSM 6.10 and 8-bit PCM destroy the fine spectral detail
speaker embeddings depend on, and no Hebrew-specific embedding model exists.

**Measurement against the current vendor.** Needs a reference transcript and
the vendor's output on the same file.

`worker/` holds a Dockerfile and handler for the custom worker - ivrit.ai
Whisper plus Sortformer plus pyannote community-1, with word assignment by
probability mass. It is written but **has never been built or run**, and
Sortformer's four-speaker output width cannot be narrowed by any parameter, so
the collapse to two speakers there is untested code.

## Measuring

A pipeline change that improves DER but not tcpWER is not an improvement, so
we report the set:

- **tcpWER** - speaker-attributed word error rate with a time constraint
- **DI-cpWER** - the same with speaker errors removed, i.e. pure ASR quality
- **tcpWER - DI-cpWER** - the diarization tax, in WER points
- **WDER** - fraction of correctly recognised words given the wrong speaker
- **DER** at collar 0 with overlap included, for the diarizer alone

Computed with `meeteval`. Confidence intervals bootstrapped over *sessions*,
not words - errors correlate within a call, and bootstrapping over words
badly understates the interval.

Hebrew normalisation must be identical on both sides before scoring: strip
niqqud, strip bidi and zero-width controls, normalise geresh/gershayim, and
fix one convention for digits versus spelled-out numbers. That last one alone
can move Hebrew WER by several points. Do not normalise away final letter
forms - they are correct orthography and folding them hides real errors.

## Usage

```bash
python pipeline/probe.py      recording.wav      # what is this file
python pipeline/analyze.py    recording.wav      # how damaged is it
python pipeline/preprocess.py recording.wav --out out
```
