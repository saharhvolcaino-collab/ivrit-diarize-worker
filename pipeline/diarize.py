"""Diarization by voice, at a resolution the ASR's segmentation cannot blur.

The worker already separates speakers by voice - it computes a speaker
embedding and clusters. The failure is one of granularity: it computes one
embedding per Whisper segment, and Whisper segments break on prosody, never on
speaker change. When two people fall inside one segment, that segment's
embedding is a blend of both voices and clustering returns a single answer for
it. On one of our calls this buried five question-and-answer exchanges,
including "what is your name?" and the reply, inside one speaker's turn.

    Whisper segment:  "what is your name?   Mendy."
                       |___ voice A ____|  |_ B _|
                       one embedding for the whole span -> one speaker

This module ignores Whisper's boundaries entirely. It slides a fixed window
across the audio, embeds each window, and clusters those. A change mid-sentence
now falls between two windows and is detectable, because no window straddles it
for long.

Window length is the trade-off. Speaker embeddings need roughly 1.5 seconds of
voice to be stable; shorter windows are noisy and cluster badly. The hop is
what actually sets boundary precision, so it is much smaller than the window -
windows overlap heavily, and a change is localised to within one hop.

Runs on CPU. ECAPA-TDNN is small, and a few hundred windows take well under a
minute, so this needs no GPU and no custom worker.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
WINDOW_SEC = 1.5
HOP_SEC = 0.25
# Below this a window holds too little voice for a stable embedding.
MIN_VOICED_RATIO = 0.5

_encoder = None


def _get_encoder():
    """ECAPA-TDNN, loaded once. Ungated, so no HuggingFace token is needed.

    Runs on CPU by default; set ECAPA_DEVICE=cuda inside the GPU worker, where
    the same code embeds a full call in a couple of seconds.
    """
    global _encoder
    if _encoder is None:
        import os
        import torch

        device = os.environ.get("ECAPA_DEVICE", "cpu")

        # Multi-threaded CPU reductions are not bit-stable, and this clustering
        # sits on a knife edge on narrowband audio - thread noise alone flipped
        # a recording between a correct 53/47 split and a degenerate one
        # between two identical runs. Single-threaded embedding makes the CPU
        # path reproducible; on CUDA the kernels are what they are and the
        # throughput is the point.
        if device == "cpu":
            torch.set_num_threads(1)

        import shutil
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier

        # speechbrain materialises checkpoints into savedir via symlink, and on
        # Windows creating a symlink needs a privilege normal users don't hold.
        # So the download is done here instead, and real copies are placed where
        # speechbrain expects them - when every file already exists at the
        # destination, its fetch step never attempts a link.
        savedir = Path.home() / ".cache" / "speechbrain-ecapa"
        savedir.mkdir(parents=True, exist_ok=True)
        snapshot = Path(snapshot_download("speechbrain/spkrec-ecapa-voxceleb"))
        for f in snapshot.iterdir():
            target = savedir / f.name
            if f.is_file() and not target.exists():
                shutil.copy2(f, target)
        # The pretrainer asks for the label encoder under a renamed destination
        # (label_encoder.txt fetched as label_encoder.ckpt), so satisfy that
        # name too or the fetch for it will fall back to symlinking.
        txt, ckpt = savedir / "label_encoder.txt", savedir / "label_encoder.ckpt"
        if txt.exists() and not ckpt.exists():
            shutil.copy2(txt, ckpt)

        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": device},
        )
    return _encoder


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[Turn]
    speakers: list[str]
    window_count: int
    separation: float
    # One voice signature per speaker, row k belonging to SPEAKER_k. Kept so a
    # later pass can re-judge individual words against them.
    centroids: "np.ndarray | None" = None
    notes: list[str] = field(default_factory=list)


def _load_audio(path: str | Path) -> np.ndarray:
    # mkstemp hands back an open descriptor. Leaving it open locks the file on
    # Windows and the cleanup below then fails.
    fd, name = tempfile.mkstemp(suffix=".diar.wav")
    import os
    os.close(fd)
    tmp = Path(name)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-af", f"aresample=resampler=soxr:precision=28:osr={SAMPLE_RATE}",
             "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True,
        )
        with wave.open(str(tmp), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        tmp.unlink(missing_ok=True)


def _voiced_mask(audio: np.ndarray) -> np.ndarray:
    """Cheap per-sample energy gate, used only to skip near-silent windows.

    Embedding silence produces a vector that describes the room, not a person,
    and those vectors pull cluster centroids around. This is deliberately crude
    - real speech detection already happened upstream.
    """
    frame = int(0.02 * SAMPLE_RATE)
    usable = (audio.size // frame) * frame
    if usable == 0:
        return np.zeros(audio.size, dtype=bool)
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    live = db[db > -80.0]
    floor = float(np.percentile(live, 20)) if live.size else -60.0
    active = db > floor + 6.0
    mask = np.repeat(active, frame)
    return np.pad(mask, (0, audio.size - mask.size), constant_values=False)


def _spans_to_mask(spans: list[tuple[float, float]], size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    for start, end in spans:
        a = max(0, int(start * SAMPLE_RATE))
        b = min(size, int(end * SAMPLE_RATE))
        if b > a:
            mask[a:b] = True
    return mask


def _embed_windows(
    audio: np.ndarray, speech_spans: list[tuple[float, float]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centre times, L2-normalised embeddings) for each voiced window.

    `speech_spans` should carry a real VAD's output. Without it this falls back
    to a crude energy gate, which lets hold music and line noise through - and
    those windows embed the channel rather than a person, pulling both speaker
    centroids toward each other. On one 14-minute support call that was the
    difference between separable and not: the diarizer was fed 856 seconds
    where the ASR only ever heard 529.
    """
    import torch

    encoder = _get_encoder()
    voiced = (_spans_to_mask(speech_spans, audio.size) if speech_spans
              else _voiced_mask(audio))
    win = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(HOP_SEC * SAMPLE_RATE)

    batch, times = [], []
    for start in range(0, max(audio.size - win, 0) + 1, hop):
        chunk = audio[start:start + win]
        if chunk.size < win:
            break
        if voiced[start:start + win].mean() < MIN_VOICED_RATIO:
            continue
        batch.append(chunk)
        times.append((start + win / 2) / SAMPLE_RATE)

    if not batch:
        return np.array([]), np.zeros((0, 192))

    embeddings = []
    # Batched so a long call does not hold every window in memory at once.
    for i in range(0, len(batch), 64):
        tensor = torch.from_numpy(np.stack(batch[i:i + 64]))
        with torch.no_grad():
            out = encoder.encode_batch(tensor).squeeze(1).cpu().numpy()
        embeddings.append(out)

    emb = np.concatenate(embeddings, axis=0)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return np.array(times), emb


def _cluster(emb: np.ndarray, n_speakers: int) -> tuple[np.ndarray, float]:
    """Cluster windows into speakers, and report how separable they were.

    On narrowband telephone audio the two voices are often barely separable,
    and a single clustering attempt sits on a knife edge - the same recording
    swung between a 53/47 split and a degenerate 95/5 across runs differing by
    one VAD span. Two defences against that:

    First, several clustering methods are tried and the one with the best
    silhouette wins. Speaker embeddings form elongated clusters, which average-
    linkage handles best in theory, but on marginal data another method
    sometimes finds the structure it misses.

    Second, centroid purification: drop the most ambiguous members of each
    cluster, recompute the voice signatures from the confident core, and
    reassign every window to its nearest signature. Mixed windows near turn
    boundaries drag centroids toward each other; purification pulls them back
    apart. Two rounds is enough for this to converge in practice.

    The returned separation score matters as much as the labels - when the
    clusters are not really distinct the caller should distrust the labels
    rather than average over them.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score

    if emb.shape[0] < max(n_speakers, 4):
        return np.zeros(emb.shape[0], dtype=int), 0.0

    def score(labels: np.ndarray) -> float:
        if len(set(labels.tolist())) < 2:
            return -1.0
        try:
            return float(silhouette_score(emb, labels, metric="cosine"))
        except ValueError:
            return -1.0

    attempts: list[np.ndarray] = [
        AgglomerativeClustering(
            n_clusters=n_speakers, metric="cosine", linkage="average"
        ).fit_predict(emb),
        AgglomerativeClustering(
            n_clusters=n_speakers, metric="cosine", linkage="complete"
        ).fit_predict(emb),
        KMeans(n_clusters=n_speakers, n_init=10, random_state=0).fit_predict(emb),
    ]
    labels = max(attempts, key=score)

    for _ in range(2):
        centroids = []
        for k in range(n_speakers):
            members = emb[labels == k]
            if members.shape[0] < 3:
                break
            sims = members @ (members.mean(axis=0) /
                              (np.linalg.norm(members.mean(axis=0)) + 1e-9))
            keep = members[sims >= np.quantile(sims, 0.2)]
            c = keep.mean(axis=0)
            centroids.append(c / (np.linalg.norm(c) + 1e-9))
        else:
            reassigned = np.argmax(emb @ np.stack(centroids).T, axis=1)
            if score(reassigned) >= score(labels):
                labels = reassigned
            continue
        break

    return labels, round(max(score(labels), 0.0), 3)


def _smooth(times: np.ndarray, labels: np.ndarray, min_turn: float = 0.6) -> list[Turn]:
    """Turn per-window labels into turns, dropping runs too short to be real.

    Windows overlap, so a single misclassified one produces a flicker rather
    than a turn. Anything shorter than a plausible utterance is absorbed into
    its neighbours.
    """
    if times.size == 0:
        return []

    turns: list[Turn] = []
    cur_label = int(labels[0])
    start_t = times[0]
    prev_t = times[0]

    for t, lab in zip(times[1:], labels[1:]):
        if int(lab) != cur_label:
            turns.append(Turn(start=float(start_t), end=float((prev_t + t) / 2),
                              speaker=f"SPEAKER_{cur_label:02d}"))
            cur_label = int(lab)
            start_t = (prev_t + t) / 2
        prev_t = t
    turns.append(Turn(start=float(start_t), end=float(prev_t + HOP_SEC),
                      speaker=f"SPEAKER_{cur_label:02d}"))

    # Absorb flickers, then rejoin neighbours that became identical.
    kept: list[Turn] = []
    for turn in turns:
        if turn.end - turn.start < min_turn and kept:
            kept[-1].end = turn.end
        else:
            kept.append(turn)

    merged: list[Turn] = []
    for turn in kept:
        if merged and merged[-1].speaker == turn.speaker:
            merged[-1].end = turn.end
        else:
            merged.append(turn)
    return merged


def diarize(
    audio_path: str | Path,
    num_speakers: int = 2,
    speech_spans: list[tuple[float, float]] | None = None,
) -> DiarizationResult:
    audio = _load_audio(audio_path)
    times, emb = _embed_windows(audio, speech_spans)

    if times.size == 0:
        return DiarizationResult([], [], 0, 0.0,
                                 ["No voiced windows found - nothing to diarize."])

    labels, separation = _cluster(emb, num_speakers)
    turns = _smooth(times, labels)

    # A mean of many window embeddings is a far more stable description of a
    # voice than any single window - these are what short words get compared
    # against in the refinement pass.
    centroids = []
    for k in range(num_speakers):
        members = emb[labels == k]
        if members.shape[0] == 0:
            centroids.append(np.zeros(emb.shape[1]))
            continue
        c = members.mean(axis=0)
        centroids.append(c / (np.linalg.norm(c) + 1e-9))

    result = DiarizationResult(
        turns=turns,
        speakers=sorted({t.speaker for t in turns}),
        window_count=int(times.size),
        separation=round(separation, 3),
        centroids=np.stack(centroids),
    )

    if separation < 0.10:
        result.notes.append(
            f"Voices are barely separable (silhouette {separation:.2f}). The two "
            "parties sound alike to the embedding model, or the codec has stripped "
            "the detail that distinguishes them. Treat the labels as unreliable."
        )
    elif separation < 0.20:
        result.notes.append(
            f"Weak voice separation (silhouette {separation:.2f}) - expect errors "
            "around turn boundaries."
        )
    return result


def refine_word_speakers(
    audio_path: str | Path,
    words: list[dict],
    centroids: np.ndarray,
    *,
    margin: float = 0.08,
    pad_sec: float = 0.15,
    min_span_sec: float = 0.40,
) -> tuple[list[dict], int]:
    """Re-judge each word's speaker against the two voice signatures.

    Sliding windows cannot isolate an utterance shorter than the window - a
    half-second answer sitting between two turns of the other party is always
    outvoted by its neighbours, which is exactly how "what is your name?" and
    the name that answers it end up on the same speaker.

    So after clustering has produced a stable signature per voice, each word's
    own audio span is embedded and compared to both. The window label is only
    overturned when the word's own evidence is decisive (cosine margin above
    the threshold): short spans give noisy embeddings, and a noisy flip is
    worse than an inherited neighbour label. Words shorter than min_span_sec
    are padded symmetrically - some neighbour audio leaks in, which is another
    reason the override bar is high rather than a coin flip.

    Returns the relabelled words and how many were actually flipped.
    """
    import torch

    encoder = _get_encoder()
    audio = _load_audio(audio_path)
    total = audio.size / SAMPLE_RATE

    spans: list[tuple[int, float, float]] = []
    for i, w in enumerate(words):
        start, end = float(w["start"]), float(w["end"])
        if end - start < min_span_sec:
            centre = (start + end) / 2
            start, end = centre - min_span_sec / 2, centre + min_span_sec / 2
        start = max(0.0, start - pad_sec)
        end = min(total, end + pad_sec)
        if end - start >= 0.25:
            spans.append((i, start, end))

    if not spans:
        return words, 0

    out = [dict(w) for w in words]
    flipped = 0
    for batch_start in range(0, len(spans), 64):
        batch = spans[batch_start:batch_start + 64]
        # Cut first, size the buffer after - deriving the buffer width from the
        # times re-does the float-to-index rounding and loses by one sample.
        pieces = [audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)] for _, s, e in batch]
        longest = max(p.size for p in pieces)
        chunk = np.zeros((len(batch), longest), dtype=np.float32)
        lengths = []
        for row, piece in enumerate(pieces):
            chunk[row, :piece.size] = piece
            lengths.append(piece.size / longest if longest else 1.0)
        with torch.no_grad():
            emb = encoder.encode_batch(
                torch.from_numpy(chunk),
                wav_lens=torch.tensor(lengths, dtype=torch.float32),
            ).squeeze(1).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

        sims = emb @ centroids.T                      # (batch, n_speakers)
        for row, (idx, _, _) in enumerate(batch):
            order = np.argsort(-sims[row])
            best = int(order[0])
            decisive = float(sims[row][order[0]] - sims[row][order[1]]) >= margin
            new_label = f"SPEAKER_{best:02d}"
            if decisive and out[idx].get("speaker") != new_label:
                out[idx]["speaker"] = new_label
                flipped += 1

    return out, flipped


def split_buried_answers(
    audio_path: str | Path,
    words: list[dict],
    centroids: np.ndarray,
    *,
    veto_margin: float = 0.06,
    loudness_gap_db: float = 6.0,
    max_tail_embed_sec: float = 2.0,
    weak_voice: bool = False,
) -> tuple[list[dict], int]:
    """Hand the text after a mid-turn question to the other party.

    The most stubborn attribution failure on this audio is a question and its
    answer inside one turn - "what is your name? Mendy" - which survives both
    window clustering (the answer is too short to win a window) and per-word
    refinement (a half-second word rarely produces decisive evidence on its
    own). But conversation itself is the strongest prior available: in a
    two-party call, speech following a question mark belongs to the other
    party almost by definition.

    So the prior does the flipping and the voice only holds veto power: the
    tail's audio is compared to both voice signatures, and the flip is
    suppressed only when the tail clearly matches the asker - which is what a
    rhetorical or self-answered question looks like. Indecisive audio defers
    to the conversational prior, inverting the refinement pass's burden of
    proof exactly because here the prior, not the acoustics, carries the
    information.

    The veto itself can be overruled by loudness. On telephone recordings the
    near and far parties sit at very different levels - and on one verified
    exchange the answer ran 14 dB hotter than the question while the embedding
    still insisted both were the asker. Narrowband codecs strip much of what
    the embedding relies on, but they cannot hide a level gap, so a gap above
    loudness_gap_db means two different people regardless of what the
    embedding thinks.

    Only defined for two speakers, and words must carry local cluster labels
    (SPEAKER_k matching centroid row k). Returns words and the number of
    tails handed over.
    """
    import torch

    # centroids may be None. In weak_voice mode the similarity veto is never
    # consulted - loudness decides alone - so the voice signatures are dead
    # weight there, and demanding them would bar every engine that does not
    # produce ECAPA embeddings from a pass it does not need. Outside that
    # mode the veto is the point, so no centroids means nothing to do.
    if centroids is None:
        if not weak_voice:
            return words, 0
    elif centroids.shape[0] != 2:
        return words, 0

    audio = _load_audio(audio_path)
    total = audio.size / SAMPLE_RATE

    out = [dict(w) for w in words]
    turns = turns_from_words(out)

    # Collect (word index range, tail span, question span) per buried
    # question, then embed all tails in one batch.
    jobs: list[tuple[list[int], float, float, float, float]] = []
    cursor = 0
    for turn in turns:
        turn_len = len(turn["words"])
        indices = list(range(cursor, cursor + turn_len))
        cursor += turn_len

        local_q = [
            j for j, w in enumerate(turn["words"])
            if (w.get("word") or "").strip().endswith("?") and j < turn_len - 1
        ]
        for j in local_q:
            tail_idx = indices[j + 1:]
            # A later question inside the same turn starts its own job.
            for k, w in enumerate(turn["words"][j + 1:], start=j + 1):
                if (w.get("word") or "").strip().endswith("?") and k < turn_len - 1:
                    tail_idx = indices[j + 1:k + 1]
                    break
            if not tail_idx:
                continue
            start = float(out[tail_idx[0]]["start"])
            end = min(float(out[tail_idx[-1]]["end"]), start + max_tail_embed_sec)
            if end - start < 0.25:
                centre = (start + end) / 2
                start = max(0.0, centre - 0.25)
                end = min(total, centre + 0.25)
            q_end = float(turn["words"][j]["end"])
            q_start = max(float(turn["words"][max(0, j - 6)]["start"]), q_end - 2.0)
            jobs.append((tail_idx, start, end, q_start, q_end))

    if not jobs:
        return out, 0

    pieces = [audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)] for _, s, e, _, _ in jobs]

    # Skipped entirely when there are no signatures to compare against, which
    # also spares a batch through the encoder that nothing would have read.
    sims = None
    if centroids is not None:
        encoder = _get_encoder()
        longest = max(p.size for p in pieces)
        chunk = np.zeros((len(pieces), longest), dtype=np.float32)
        lengths = []
        for row, piece in enumerate(pieces):
            chunk[row, :piece.size] = piece
            lengths.append(piece.size / longest if longest else 1.0)
        with torch.no_grad():
            emb = encoder.encode_batch(
                torch.from_numpy(chunk),
                wav_lens=torch.tensor(lengths, dtype=torch.float32),
            ).squeeze(1).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        sims = emb @ centroids.T

    def _rms_db(x: np.ndarray) -> float:
        if x.size == 0:
            return -120.0
        r = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        return 20.0 * float(np.log10(r + 1e-12))

    handed = 0
    for row, (tail_idx, _, _, q_start, q_end) in enumerate(jobs):
        current = out[tail_idx[0]].get("speaker")
        if current not in ("SPEAKER_00", "SPEAKER_01"):
            continue
        cur_k = int(current[-2:])
        other_k = 1 - cur_k

        voice_veto = (sims is not None
                      and float(sims[row][cur_k] - sims[row][other_k]) > veto_margin)
        gap = abs(
            _rms_db(audio[int(q_start * SAMPLE_RATE):int(q_end * SAMPLE_RATE)])
            - _rms_db(pieces[row])
        )
        if weak_voice:
            # Degenerate clustering or near-identical voices: the signatures
            # cannot be trusted in either direction, so loudness decides alone.
            # Telephone parties sit at different levels; a same-speaker
            # continuation does not jump levels mid-sentence.
            if gap < 3.0:
                continue
        elif voice_veto and gap < loudness_gap_db:
            continue
        for idx in tail_idx:
            out[idx]["speaker"] = f"SPEAKER_{other_k:02d}"
        handed += 1

    return out, handed


SENTENCE_END = ".?!"


def realign_by_punctuation(words: list[dict], max_span: int = 50) -> list[dict]:
    """Vote away speaker changes that land mid-sentence.

    A change of speaker in the middle of a sentence is almost always a
    diarization error - people interrupt between clauses, not between a verb
    and its object. Wherever a flip occurs and the preceding word carries no
    sentence-ending punctuation, the whole enclosing sentence is put to a
    majority vote, and only a clear majority (>50%) may relabel it.

    This is the standard fix for boundary jitter from window-based diarization,
    and it leans on Whisper's own Hebrew punctuation - which is exactly why the
    window hop can stay small: precision comes from here, not from the windows.
    """
    if len(words) < 3:
        return words

    def ends_sentence(i: int) -> bool:
        text = (words[i].get("word") or "").strip()
        return bool(text) and text[-1] in SENTENCE_END

    out = [dict(w) for w in words]
    k = 0
    n = len(out)
    while k < n - 1:
        if out[k]["speaker"] != out[k + 1]["speaker"] and not ends_sentence(k):
            left = k
            steps = 0
            while left > 0 and not ends_sentence(left - 1) and steps < max_span:
                left -= 1
                steps += 1
            right = k
            steps = 0
            while right < n - 1 and not ends_sentence(right) and steps < max_span:
                right += 1
                steps += 1

            labels = [out[i]["speaker"] for i in range(left, right + 1)]
            winner = max(set(labels), key=labels.count)
            if labels.count(winner) > len(labels) // 2:
                for i in range(left, right + 1):
                    out[i]["speaker"] = winner
            k = right + 1
        else:
            k += 1
    return out


def turns_from_words(words: list[dict], split_gap: float = 1.0) -> list[dict]:
    """Group labelled words into turns, cutting on every speaker change.

    This is the piece the stock pipeline lacks: Whisper's segments never break
    on a speaker change, so building turns from the words instead is what lets
    a mid-segment change survive into the transcript.
    """
    if not words:
        return []
    turns: list[dict] = []
    cur = {"speaker": words[0]["speaker"], "start": words[0]["start"],
           "end": words[0]["end"], "words": [words[0]]}
    for w in words[1:]:
        if w["speaker"] == cur["speaker"] and (w["start"] - cur["end"]) <= split_gap:
            cur["end"] = w["end"]
            cur["words"].append(w)
        else:
            turns.append(cur)
            cur = {"speaker": w["speaker"], "start": w["start"],
                   "end": w["end"], "words": [w]}
    turns.append(cur)
    for t in turns:
        t["text"] = " ".join(x["word"] for x in t["words"]).strip()
    return turns


def assign_words(words: list[dict], turns: list[Turn]) -> list[dict]:
    """Give each word the speaker whose turn covers most of it.

    Assignment is per word rather than per segment, so a speaker change inside
    a Whisper segment survives instead of being averaged away.
    """
    out: list[dict] = []
    for w in words:
        overlap: dict[str, float] = {}
        for t in turns:
            lo, hi = max(w["start"], t.start), min(w["end"], t.end)
            if hi > lo:
                overlap[t.speaker] = overlap.get(t.speaker, 0.0) + (hi - lo)

        if overlap:
            best, amount = max(overlap.items(), key=lambda kv: kv[1])
            span = max(w["end"] - w["start"], 1e-6)
            out.append({**w, "speaker": best,
                        "speaker_confidence": round(min(amount / span, 1.0), 3)})
        else:
            # Words falling in a gap take the nearest turn's speaker - dropping
            # them would silently lose real content from the transcript.
            mid = (w["start"] + w["end"]) / 2
            nearest = min(
                turns,
                key=lambda t: 0.0 if t.start <= mid <= t.end
                else min(abs(mid - t.end), abs(t.start - mid)),
                default=None,
            )
            out.append({**w, "speaker": nearest.speaker if nearest else None,
                        "speaker_confidence": 0.0})
    return out
