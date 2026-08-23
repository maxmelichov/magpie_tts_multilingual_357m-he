"""renikud_onnx — single-file wrapper for the RenikudPlus ONNX models.

Usage: put this file next to your script, then `from renikud_onnx import G2P`.
Requires: onnxruntime, numpy.
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import onnxruntime as ort



import sys
import warnings

#: max_position_embeddings (2048) minus CLS + SEP. Measured: 2046 OK, 2047 raises.
SAFE_CHUNK_CHARS = 2046

#: Hebrew letters alef..tav. Hebrew punctuation (geresh, gershayim, maqaf) is
#: outside this range and is legitimately passed through by the decoder.
HEBREW_LETTER_RE = re.compile(r"[א-ת]")

_SENTENCE_ENDS = ".!?;\n\r…׃"
_CLAUSE_ENDS = ",:–—"


class RawHebrewLeakError(ValueError):
    """Raw Hebrew letters survived into what should be an IPA string (§130a)."""


def _split_after(text: str, terminators: str) -> list[str]:
    pieces: list[str] = []
    start = i = 0
    n = len(text)
    while i < n:
        if text[i] in terminators:
            j = i + 1
            while j < n and text[j] in terminators:
                j += 1
            while j < n and text[j].isspace():
                j += 1
            pieces.append(text[start:j])
            start = i = j
        else:
            i += 1
    if start < n:
        pieces.append(text[start:])
    return pieces


def _split_words(text: str) -> list[str]:
    pieces: list[str] = []
    start = i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            pieces.append(text[start:j])
            start = i = j
        else:
            i += 1
    if start < n:
        pieces.append(text[start:])
    return pieces


def _pack(pieces: list[str], limit: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > limit:
            out.append(cur)
            cur = piece
        else:
            cur += piece
    if cur:
        out.append(cur)
    return out


def split_for_decode(text: str, limit: int = SAFE_CHUNK_CHARS,
                     warn_on_hard_cut: bool = True) -> list[str]:
    """Split into decode windows of at most `limit` characters, losslessly.

    ``"".join(split_for_decode(t)) == t``; ``len(text) <= limit`` returns
    ``[text]`` unchanged.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if len(text) <= limit:
        return [text]

    out: list[str] = []
    for sent_chunk in _pack(_split_after(text, _SENTENCE_ENDS), limit):
        if len(sent_chunk) <= limit:
            out.append(sent_chunk)
            continue
        for clause_chunk in _pack(_split_after(sent_chunk, _CLAUSE_ENDS), limit):
            if len(clause_chunk) <= limit:
                out.append(clause_chunk)
                continue
            for word_chunk in _pack(_split_words(clause_chunk), limit):
                if len(word_chunk) <= limit:
                    out.append(word_chunk)
                    continue
                if warn_on_hard_cut:
                    warnings.warn(
                        f"[long-input] a {len(word_chunk)}-character run with no "
                        f"whitespace exceeds the {limit}-character decode window; "
                        f"cutting it mid-token (§131).",
                        RuntimeWarning, stacklevel=2,
                    )
                out.extend(word_chunk[i:i + limit]
                           for i in range(0, len(word_chunk), limit))

    assert "".join(out) == text, "split_for_decode is not lossless"
    return out


def check_no_raw_hebrew(output: str, text: str, mode: str = "warn",
                        where: str = "phonemize") -> str:
    """Fail loudly when raw Hebrew letters survive into an IPA string (§130a)."""
    if mode == "ignore":
        return output
    leaked = HEBREW_LETTER_RE.findall(output)
    if not leaked:
        return output
    sample = "".join(dict.fromkeys(leaked))[:20]
    msg = (
        f"[{where}] RAW HEBREW IN OUTPUT (§130a): {len(leaked)} Hebrew letter(s) "
        f"({sample}) survived into what should be an IPA string — the tail of the "
        f"input was never transcribed. input={len(text)} chars, "
        f"output={len(output)} chars."
    )
    if mode == "raise":
        raise RawHebrewLeakError(msg)
    print(f"WARNING: {msg}", file=sys.stderr)
    return output







ALEF_ORD = ord("א")
TAF_ORD = ord("ת")
STRESS_MARK = "ˈ"
ORTHOGRAPHIC_MARKERS = ("'", '"')
NEG = -1e30


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def _is_hebrew(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def normalize_graphemes(text: str) -> str:
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r'[״""]', '"', text)
    return text


class G2P:
    def __init__(self, model_path: str, exact_map: bool = True,
                 chunk_chars: int | None = SAFE_CHUNK_CHARS) -> None:
        # §131: `chunk_chars` is the long-input decode window. The default 2,046
        # is this wrapper's measured hard wall (2,048 positions - CLS/SEP); pass
        # 448 for output parity with src/infer.phonemize, or None to disable
        # windowing and let onnxruntime raise past the wall as it did before.
        self._chunk_chars = chunk_chars
        self._session = ort.InferenceSession(model_path)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab: dict[str, int] = json.loads(meta["vocab"])
        self._consonant_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["consonant_vocab"]).items()}
        self._vowel_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["vowel_vocab"]).items()}
        self._cls_id = int(meta["cls_token_id"])
        self._sep_id = int(meta["sep_token_id"])
        self._letter_constraints: dict[str, list[int]] = {
            k: v for k, v in json.loads(meta["letter_consonant_constraints"]).items()
        }
        self._geresh_map: dict[str, str] = json.loads(meta.get("geresh_map", "{}"))
        # Models exported with --gender-inputs expose FiLM speaker conditioning.
        # Older 2-input models don't; phonemize() then ignores the arguments.
        input_names = {i.name for i in self._session.get_inputs()}
        self.supports_gender = {"speaker", "target_speaker"} <= input_names

        # Exact structured-cascade MAP decode. Needs the conditioning column
        # blocks of the cascade heads, which models exported before this feature
        # don't carry; those fall back to greedy argmax + margin stress.
        cond_keys = ("vowel_cond_consonant", "stress_cond_consonant", "stress_cond_vowel")
        self.supports_exact_map = all(k in meta for k in cond_keys)
        self.exact_map_default = exact_map and self.supports_exact_map
        if self.supports_exact_map:
            # [C, V], [C, S], [V, S]
            self._wv_c = np.asarray(json.loads(meta["vowel_cond_consonant"]), dtype=np.float32)
            self._ws_c = np.asarray(json.loads(meta["stress_cond_consonant"]), dtype=np.float32)
            self._ws_v = np.asarray(json.loads(meta["stress_cond_vowel"]), dtype=np.float32)
            self._cond_softmax = meta.get("cascade_cond", "softmax") == "softmax"
            n_cons = self._wv_c.shape[0]
            # [letter, consonant] True = this consonant class is illegal here.
            self._forbidden = np.ones((TAF_ORD - ALEF_ORD + 1, n_cons), dtype=bool)
            for letter, allowed in self._letter_constraints.items():
                self._forbidden[ord(letter) - ALEF_ORD, list(allowed)] = False

    def _tokenize(self, text: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Tokenize character by character, return ids, mask, and offset mapping."""
        normalized = unicodedata.normalize("NFD", text)
        unk_id = self._vocab.get("[UNK]", 0)
        ids = [self._cls_id]
        offsets = [(0, 0)]  # CLS
        for i, c in enumerate(normalized):
            ids.append(self._vocab.get(c, unk_id))
            offsets.append((i, i + 1))
        ids.append(self._sep_id)
        offsets.append((0, 0))  # SEP
        mask = [1] * len(ids)
        return ids, mask, offsets

    def _best_stress_per_word(
        self,
        offsets: list[tuple[int, int]],
        text: str,
        stress_logits: np.ndarray,
        vowel_preds: np.ndarray,
    ) -> set[int]:
        word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}
        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            for word_idx, (ws, we) in enumerate(word_spans):
                if ws <= start < we:
                    words[word_idx].append(tok_idx)
                    break
        stressed: set[int] = set()
        for toks in words.values():
            if not toks:
                continue
            # Stress must sit with a vowel (CˈV); never emit trailing ˈ.
            vowel_toks = [
                t for t in toks
                if self._vowel_vocab.get(int(vowel_preds[t]), "∅") != "∅"
            ]
            if not vowel_toks:
                continue
            # Margin (yes − no), not the raw yes-logit: raw logits differ in scale
            # across tokens (mirrors src/decoder.py).
            stressed.add(max(vowel_toks, key=lambda t: stress_logits[t, 1] - stress_logits[t, 0]))
        return stressed

    def _exact_map(
        self,
        offsets: list[tuple[int, int]],
        text: str,
        consonant_logits: np.ndarray,
        vowel_logits: np.ndarray,
        stress_logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, set[int]]:
        """Closed-form MAP over the joint cascade energy (see outputs/exp_v4/eval_exact_map.py).

            E(c, v, s) = log P(c) + log P(v | c) + log P(s | c, v)

        The graph emits vowel/stress logits conditioned on the model's own
        upstream softmax; subtracting that conditioning back out recovers the
        unconditioned head logits, after which conditioning on ANY (c, v) is an
        exact column addition. No early argmax anywhere, so the one-stress-per-
        word constraint is free to flip both the vowel and the consonant.
        """
        cond_c = _softmax(consonant_logits) if self._cond_softmax else _onehot(
            consonant_logits.argmax(-1), consonant_logits.shape[-1]
        )
        base_v = vowel_logits - cond_c @ self._wv_c              # [S, V]
        cond_v = _softmax(vowel_logits) if self._cond_softmax else _onehot(
            vowel_logits.argmax(-1), vowel_logits.shape[-1]
        )
        base_s = stress_logits - cond_c @ self._ws_c - cond_v @ self._ws_v  # [S, 2]

        logc = _log_softmax(consonant_logits)                    # [S, C]
        logv = _log_softmax(base_v[:, None, :] + self._wv_c[None, :, :])  # [S, C, V]
        logs = _log_softmax(
            base_s[:, None, None, :]
            + self._ws_c[None, :, None, :]
            + self._ws_v[None, None, :, :]
        )                                                        # [S, C, V, 2]
        E = logc[:, :, None, None] + logv[:, :, :, None] + logs  # [S, C, V, 2]

        # Per-letter consonant legality, and "stress needs a vowel".
        for t, (start, end) in enumerate(offsets):
            if end - start == 1 and _is_hebrew(text[start:end]):
                E[t, self._forbidden[ord(text[start]) - ALEF_ORD]] = NEG
        E[:, :, 0, 1] = NEG

        S, C, V = E.shape[:3]
        flat_u = E[:, :, :, 0].reshape(S, C * V)
        flat_s = E[:, :, :, 1].reshape(S, C * V)
        arg_u, arg_s = flat_u.argmax(-1), flat_s.argmax(-1)
        best_u = flat_u[np.arange(S), arg_u]
        best_s = flat_s[np.arange(S), arg_s]
        gain = best_s - best_u

        char_tok = {
            start: t
            for t, (start, end) in enumerate(offsets)
            if end - start == 1 and _is_hebrew(text[start:end])
        }
        stressed: set[int] = set()
        for m in re.finditer(r"\S+", text):
            toks = [char_tok[i] for i in range(m.start(), m.end()) if i in char_tok]
            cands = [t for t in toks if best_s[t] > NEG / 2]
            if cands:
                stressed.add(max(cands, key=lambda t: gain[t]))

        flat = np.where(np.isin(np.arange(S), list(stressed)), arg_s, arg_u)
        return flat // V, flat % V, stressed

    def phonemize(
        self,
        text: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
        on_hebrew_leak: str = "warn",
    ) -> str:
        """Convert unvocalized Hebrew text to IPA.

        §131: inputs longer than `chunk_chars` (default 2,046 — the encoder's
        2,048-position table minus CLS/SEP, measured) are split on sentence, then
        comma, then word boundaries and decoded window by window. Inputs at or
        below it are byte-for-byte unchanged from the pre-§131 wrapper. A
        `[א-ת]`-in-output backstop then reports (default) or raises on any raw
        Hebrew that survives into the IPA string.
        """
        text = normalize_graphemes(text)
        normalized = unicodedata.normalize("NFD", text)
        windows = (split_for_decode(normalized, limit=self._chunk_chars)
                   if self._chunk_chars else [normalized])
        # The windows are contiguous substrings, so concatenating their outputs
        # preserves the original separators by construction.
        result = "".join(
            self._phonemize_window(w, speaker, target_speaker, exact_map)
            for w in windows
        )
        return check_no_raw_hebrew(result, text, mode=on_hebrew_leak,
                                   where="renikud_onnx.G2P.phonemize")

    def _phonemize_window(
        self,
        normalized: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
    ) -> str:
        """One session run + decode over an already normalized/NFD window."""
        ids, mask, offsets = self._tokenize(normalized)

        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
        }
        if self.supports_gender:
            feeds["speaker"] = np.array([speaker], dtype=np.int64)
            feeds["target_speaker"] = np.array([target_speaker], dtype=np.int64)

        consonant_logits, vowel_logits, stress_logits = self._session.run(
            ["consonant_logits", "vowel_logits", "stress_logits"],
            feeds,
        )
        # logits shape: [1, seq_len, num_classes]
        use_exact = self.exact_map_default if exact_map is None else exact_map
        if use_exact and not self.supports_exact_map:
            raise ValueError(
                "this model was exported without the cascade conditioning metadata; "
                "re-export it, or pass exact_map=False for the greedy decode"
            )
        if use_exact:
            consonant_preds, vowel_preds, stressed_positions = self._exact_map(
                offsets,
                normalized,
                consonant_logits[0].astype(np.float32),
                vowel_logits[0].astype(np.float32),
                stress_logits[0].astype(np.float32),
            )
        else:
            consonant_preds = consonant_logits[0].argmax(axis=-1)
            vowel_preds = vowel_logits[0].argmax(axis=-1)
            stressed_positions = self._best_stress_per_word(
                offsets, normalized, stress_logits[0], vowel_preds
            )

        result = []
        prev_end = 0

        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                # CLS, SEP — skip
                if end > start:
                    prev_end = end
                continue

            # Pass through any characters skipped by the tokenizer
            if start > prev_end:
                result.append(normalized[prev_end:start])

            char = normalized[start:end]
            prev_end = end

            if not _is_hebrew(char):
                if char in ORTHOGRAPHIC_MARKERS:
                    pass
                else:
                    result.append(char)
                continue

            cid = int(consonant_preds[tok_idx])
            allowed = self._letter_constraints.get(char)
            if allowed is not None and cid not in allowed:
                cid = max(allowed, key=lambda x: consonant_logits[0][tok_idx, x])
            consonant = self._consonant_vocab.get(cid, "∅")

            # Geresh rule: if next char is apostrophe, force geresh consonant variant
            if char in self._geresh_map and end < len(normalized) and normalized[end] == "'":
                consonant = self._geresh_map[char]

            vowel = self._vowel_vocab.get(int(vowel_preds[tok_idx]), "∅")
            stress = tok_idx in stressed_positions

            # Assemble IPA chunk: [consonant][ˈ][vowel]
            # Exception: word-final ח with vowel a — furtive patah flips to [ˈ]aχ
            word_final = end >= len(normalized) or not normalized[end].isalpha()
            chunk = ""
            if char == "ח" and word_final and vowel == "a":
                if stress:
                    chunk += STRESS_MARK
                chunk += "aχ"
            else:
                if consonant != "∅":
                    chunk += consonant
                if stress and vowel != "∅":
                    chunk += STRESS_MARK
                if vowel != "∅":
                    chunk += vowel
            result.append(chunk)

        if prev_end < len(normalized):
            result.append(normalized[prev_end:])

        return "".join(result)
