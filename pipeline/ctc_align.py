"""Word timestamps from the waveform, the way whisperX does it.

Whisper times words from decoder cross-attention. That signal has no idea
whether a moment contains sound, so across a pause it stretches - measured on
one of our own calls, the word "שלום" was reported at 3.00s and spoken at
4.10s, having been dragged back across a second of digital silence. Every
consumer of those timestamps then reasons about silence as if it were speech:
turn splitting on gaps, word-to-speaker overlap, the buried-answer splitter.

Forced alignment answers the same question from the audio instead. An acoustic
model emits a per-frame distribution over characters, and Viterbi finds the
single most likely path through those frames that spells the transcript we
already know. There is nothing to stretch: a frame either looks like a letter
or it does not.

This is the one real capability whisperX has that we did not. Their ASR path
is weaker than ours - the batched decoder never consults temperature fallback,
compression ratio, log-prob or no-speech thresholds, so those options are
accepted and ignored - and they already reached the same diarizer we did. The
alignment stage is the part worth taking.

    ASR text  ---------------------\\
                                    >--- Viterbi over CTC emissions ---> spans
    wav2vec2 emissions ------------/

Two things are deliberately conservative here.

The model is `imvladikon/wav2vec2-xls-r-300m-hebrew`, which is what whisperX
hard-codes for Hebrew - but it was trained on ~69 hours of clean 16 kHz
broadcast and lecture speech, with no telephone data at all, and reports 23%
WER. Our audio is 8 kHz narrowband upsampled, which leaves the 4-8 kHz band
empty; the one controlled comparison in the literature (CrisperWhisper,
Interspeech 2024) has wav2vec2 alignment falling *below* Whisper's own DTW
under noise. So this never overwrites a timestamp it is not confident about,
and it is measured rather than assumed - `align_words` reports how many words
it moved and the mean path score, so the alternative can be scored against
the timestamps we already have instead of replacing them on reputation.

Its vocabulary is 30 characters - the Hebrew alphabet, and nothing else. No
digits, no Latin, no punctuation, which a call-centre transcript is full of.
Rather than dropping those words, out-of-vocabulary characters become a
wildcard that matches whatever the frame looks like, so "25 שקל" and "CRM"
stay in the path and keep their neighbours aligned instead of derailing them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_ID = os.environ.get("CTC_ALIGN_MODEL",
                          "imvladikon/wav2vec2-xls-r-300m-hebrew")
SAMPLE_RATE = 16000
# Long enough to give Viterbi context, short enough to bound wav2vec2 memory.
MAX_CHUNK_SEC = 20.0

_model = None
_labels: dict[str, int] = {}
_blank = 0


@dataclass
class AlignReport:
    words: list[dict]
    moved: int
    mean_score: float
    failed_chunks: int
    notes: list[str]


def _load():
    """Load the CTC model once and cache its character vocabulary."""
    global _model, _labels, _blank
    if _model is not None:
        return _model

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    vocab = processor.tokenizer.get_vocab()
    _labels = {c.lower(): i for c, i in vocab.items()}
    _blank = vocab.get("<pad>", vocab.get("[PAD]", 0))
    _model = model
    return model


def _emissions(audio: np.ndarray):
    import torch

    model = _load()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(audio).unsqueeze(0).to(device)
        logits = model(x).logits
        return torch.log_softmax(logits, dim=-1)[0].cpu()


def _tokenise(text: str) -> tuple[list[int], list[int]]:
    """Map text to vocabulary ids, returning (tokens, index of word per token).

    Characters the model has never seen - digits, Latin, punctuation - become
    -1, the wildcard. Dropping them instead would silently delete a word from
    the path and shift every timing after it.
    """
    tokens: list[int] = []
    owner: list[int] = []
    for wi, word in enumerate(text.split()):
        for ch in word.lower():
            if ch in _labels:
                tokens.append(_labels[ch])
                owner.append(wi)
            elif re.match(r"[\w\d]", ch):        # unknown but pronounced
                tokens.append(-1)
                owner.append(wi)
            # pure punctuation contributes no audio and is skipped
    return tokens, owner


def _align_chunk(emission, tokens: list[int]):
    """Viterbi through the trellis; returns per-token frame index or None."""
    import torch
    import torchaudio.functional as F

    if not tokens:
        return None

    # The wildcard column: for each frame, the best score any real character
    # could achieve there. A token that matches it is saying "something is
    # being said here and I cannot tell you what", which is exactly true of a
    # digit spoken to a model with no digits.
    emis = emission
    star = emis[:, 1:].max(dim=-1, keepdim=True).values
    emis = torch.cat([emis, star], dim=-1)
    star_id = emis.shape[-1] - 1
    ids = torch.tensor([[star_id if t < 0 else t for t in tokens]],
                       dtype=torch.int32)
    try:
        paths, scores = F.forced_align(emis.unsqueeze(0), ids, blank=_blank)
    except Exception:
        return None
    return paths[0].tolist(), scores[0].exp().tolist()


def align_words(audio_path: str | Path, words: list[dict],
                *, min_score: float = 0.0) -> AlignReport:
    """Re-time `words` against the audio. Text is never changed."""
    import soundfile as sf

    if not words:
        return AlignReport(words, 0, 0.0, 0, ["no words to align"])

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != SAMPLE_RATE:
        return AlignReport(words, 0, 0.0, 0,
                           [f"expected {SAMPLE_RATE} Hz audio, got {sr}"])

    try:
        _load()
    except Exception as exc:
        return AlignReport(words, 0, 0.0, 0,
                           [f"alignment model unavailable: {type(exc).__name__}: {exc}"])

    # Chunk on the gaps the current timestamps already show. They are imprecise,
    # which is the whole problem, but they are good enough to decide where a
    # chunk may safely be cut - and a chunk boundary in a pause costs nothing.
    chunks: list[list[int]] = [[]]
    for i, w in enumerate(words):
        if chunks[-1]:
            first = words[chunks[-1][0]]
            if (float(w["end"]) - float(first["start"]) > MAX_CHUNK_SEC
                    and float(w["start"]) - float(words[i - 1]["end"]) > 0.15):
                chunks.append([])
        chunks[-1].append(i)
    chunks = [c for c in chunks if c]

    out = [dict(w) for w in words]
    moved = 0
    failed = 0
    scores: list[float] = []

    for chunk in chunks:
        lo = max(0.0, float(words[chunk[0]]["start"]) - 0.20)
        hi = min(len(audio) / SAMPLE_RATE, float(words[chunk[-1]]["end"]) + 0.20)
        seg = audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
        if seg.size < 400:                       # wav2vec2 receptive minimum
            failed += 1
            continue

        text = " ".join(str(words[i].get("word", "")).strip() for i in chunk)
        tokens, owner = _tokenise(text)
        emission = _emissions(seg)
        aligned = _align_chunk(emission, tokens)
        if aligned is None or not tokens:
            failed += 1
            continue
        path, path_scores = aligned

        # forced_align returns one label per frame; frames map back to time by
        # the ratio between the emission length and the audio it came from.
        ratio = (hi - lo) / max(emission.shape[0], 1)
        # Recover which token each frame belongs to: forced_align emits the
        # tokens in order, so the k-th distinct non-blank run is token k.
        spans: dict[int, list[float]] = {}
        tok_i = -1
        prev = _blank
        for frame, (label, score) in enumerate(zip(path, path_scores)):
            if label != _blank and label != prev:
                tok_i += 1
            prev = label
            if label == _blank or tok_i < 0 or tok_i >= len(owner):
                continue
            wi = chunk[owner[tok_i]] if owner[tok_i] < len(chunk) else None
            if wi is None:
                continue
            t0, t1 = lo + frame * ratio, lo + (frame + 1) * ratio
            if wi not in spans:
                spans[wi] = [t0, t1, 0.0, 0]
            spans[wi][1] = t1
            spans[wi][2] += float(score)
            spans[wi][3] += 1

        for wi, (s, e, tot, n) in spans.items():
            if e <= s or n == 0:
                continue
            score = tot / n
            scores.append(score)
            if score < min_score:
                continue
            if abs(s - float(out[wi]["start"])) > 0.02 or \
               abs(e - float(out[wi]["end"])) > 0.02:
                moved += 1
            out[wi] = {**out[wi], "start": round(s, 3), "end": round(e, 3),
                       "align_score": round(score, 3)}

    notes = [f"forced alignment moved {moved}/{len(words)} words "
             f"across {len(chunks)} chunk(s)"]
    if failed:
        notes.append(f"{failed} chunk(s) failed to align and kept their "
                     "original timestamps")
    return AlignReport(out, moved,
                       float(np.mean(scores)) if scores else 0.0, failed, notes)
