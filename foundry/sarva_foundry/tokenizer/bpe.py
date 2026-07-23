"""A byte-level BPE tokenizer, written from scratch.

Same family of algorithm as GPT-2/GPT-4's tokenizer (byte-level base
alphabet, so any input round-trips with no <unk>, plus learned merges), but
implemented here from first principles in plain Python — no `tokenizers`,
no `tiktoken`. That's the point: Sarva's "no black boxes" principle means
even the tokenizer is code you can read start to finish.

Algorithm, in one paragraph: every byte of the UTF-8 encoding of the input
text is remapped to one of 256 dedicated printable Unicode characters (see
`_byte_to_unicode`), so training and encoding operate on ordinary strings
without special-casing raw bytes. Text is first split into word-ish chunks
by `_PRETOKENIZE_PATTERN` (an approximation of GPT-2's regex using only the
stdlib `re` module — see its docstring for the one place this diverges from
GPT-2 exactly). Training then repeatedly finds the most frequent adjacent
symbol pair across the whole corpus and merges it into a new symbol, until
the vocabulary reaches the requested size. Encoding replays those merges
greedily, in the order they were learned.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# Approximates GPT-2's pretokenizer regex (`'s|'t|'re|... | ?\p{L}+| ?\p{N}+|
# ...`) using only stdlib `re`, which has no \p{L}/\p{N} Unicode property
# escapes. `[^\W\d_]` = "word character that isn't a digit or underscore",
# i.e. approximately \p{L}; `\d` is Unicode-aware digits by default on `str`
# patterns. This diverges from GPT-2's exact behavior for a handful of
# Unicode categories (e.g. combining marks split from their base letter
# instead of attaching to it) — acceptable for a teaching implementation,
# and documented here rather than silently assumed identical.
_PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)

_NUM_BYTES = 256


def _byte_to_unicode() -> dict[int, str]:
    """A reversible byte<->str mapping so every byte value becomes exactly
    one printable Unicode character. Lifted from the well-known GPT-2
    trick: printable Latin-1 bytes map to themselves; the rest (control
    characters, etc.) get shifted into the Unicode private-use-adjacent
    range starting at 256, so no byte value is ever ambiguous or
    unprintable in the intermediate string representation BPE operates on.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_to_char = dict(zip(printable, printable, strict=True))
    next_free = 2**8
    for b in range(_NUM_BYTES):
        if b not in byte_to_char:
            byte_to_char[b] = next_free
            next_free += 1
    return {b: chr(c) for b, c in byte_to_char.items()}


_BYTE_TO_UNICODE = _byte_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def _text_to_symbols(text: str) -> str:
    """UTF-8 bytes of `text`, each remapped to its dedicated Unicode char."""
    return "".join(_BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


def _symbols_to_text(symbols: str) -> str:
    # errors="replace" (U+FFFD for whatever bytes don't form valid UTF-8),
    # not the default strict mode: `decode()` round-trips against real
    # encoded text perfectly (encode() always produces valid UTF-8 by
    # construction), but a genuinely undertrained or adversarial model
    # can emit ANY token id sequence, including ones whose concatenated
    # bytes are not valid UTF-8 -- a real case, not hypothetical (hit
    # directly while sampling from an early-training-step reasoning-
    # token model; see sarva_foundry.train.reasoning). A tokenizer used
    # for real inference/RL rollouts must decode gracefully in that case,
    # not raise and abort the whole generation.
    return bytes(_UNICODE_TO_BYTE[c] for c in symbols).decode("utf-8", errors="replace")


def _get_pair_counts(word_freqs: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged = pair[0] + pair[1]
    out: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class ByteLevelBPETokenizer:
    """Trainable byte-level BPE tokenizer.

    Vocabulary layout: ids `0..255` are the raw byte symbols (always
    present, so encoding never produces an out-of-vocabulary token — the
    byte-level base alphabet is what gives this tokenizer its "no <unk>"
    guarantee), then learned merges in the order they were trained, then
    any special tokens appended last.
    """

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = dict.fromkeys(_BYTE_TO_UNICODE.values(), 0)
        for i, symbol in enumerate(sorted(self.vocab, key=lambda s: _UNICODE_TO_BYTE[s])):
            self.vocab[symbol] = i
        self.special_tokens: dict[str, int] = {}
        self._merge_rank: dict[tuple[str, str], int] = {}

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (),
    ) -> None:
        """Learn merges from `texts` until `vocab_size` is reached (byte
        alphabet + special tokens count against the budget)."""
        if vocab_size < _NUM_BYTES:
            raise ValueError(f"vocab_size must be >= {_NUM_BYTES} (the byte alphabet alone)")

        special_tokens = list(special_tokens)
        budget = vocab_size - _NUM_BYTES - len(special_tokens)
        if budget < 0:
            raise ValueError("vocab_size too small to fit the requested special_tokens")

        word_freqs: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for chunk in _PRETOKENIZE_PATTERN.findall(text):
                symbols = _text_to_symbols(chunk)
                word_freqs[tuple(symbols)] += 1

        merges: list[tuple[str, str]] = []
        for _ in range(budget):
            pair_counts = _get_pair_counts(word_freqs)
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            merges.append(best_pair)
            # A plain dict/Counter comprehension here would silently drop
            # frequency mass if two distinct pre-merge words happen to
            # collide into the same tuple after this merge — accumulate
            # explicitly so that can never lose counts.
            merged_freqs: Counter[tuple[str, ...]] = Counter()
            for word, freq in word_freqs.items():
                merged_freqs[_merge_word(word, best_pair)] += freq
            word_freqs = merged_freqs

        self.merges = merges
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}
        for pair in merges:
            self.vocab[pair[0] + pair[1]] = len(self.vocab)
        self.special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = len(self.vocab) + len(self.special_tokens)

    def _apply_merges(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        word = symbols
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            ranked = [(self._merge_rank[p], p) for p in pairs if p in self._merge_rank]
            if not ranked:
                break
            _, best_pair = min(ranked)
            word = _merge_word(word, best_pair)
        return word

    def encode(self, text: str) -> list[int]:
        if not self.vocab:
            raise RuntimeError("tokenizer has not been trained or loaded")

        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in self.special_tokens)
            pieces = re.split(f"({pattern})", text)
        else:
            pieces = [text]

        ids: list[int] = []
        for piece in pieces:
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
                continue
            for chunk in _PRETOKENIZE_PATTERN.findall(piece):
                symbols = tuple(_text_to_symbols(chunk))
                for symbol in self._apply_merges(symbols):
                    ids.append(self.vocab[symbol])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        id_to_symbol = {v: k for k, v in self.vocab.items()}
        id_to_special = {v: k for k, v in self.special_tokens.items()}
        parts: list[str] = []
        buffer = ""
        for token_id in ids:
            if token_id in id_to_special:
                if buffer:
                    parts.append(_symbols_to_text(buffer))
                    buffer = ""
                parts.append(id_to_special[token_id])
            else:
                buffer += id_to_symbol[token_id]
        if buffer:
            parts.append(_symbols_to_text(buffer))
        return "".join(parts)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def save(self, path: Path) -> None:
        data = {
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> ByteLevelBPETokenizer:
        data = json.loads(path.read_text())
        tok = cls()
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok._merge_rank = {pair: i for i, pair in enumerate(tok.merges)}
        for pair in tok.merges:
            tok.vocab[pair[0] + pair[1]] = len(tok.vocab)
        tok.special_tokens = dict(data["special_tokens"])
        return tok
