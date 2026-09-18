"""
Byte Pair Encoding (BPE) Tokenizer from Scratch.

This module implements a subword BPE tokenizer using only standard Python
data structures. It handles:
- Training on raw text corpora
- Unsupervised subword vocabulary induction via iterative pair merges
- Conversion of text to integer token IDs (encoding)
- Lossless reconstruction of token IDs back to text (decoding)
- Special token handling (<|unk|>, <|endoftext|>)
- Serialization to and deserialization from disk (JSON)
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer.
    
    Attributes:
        vocab_size (int): Maximum vocabulary size to learn.
        special_tokens (List[str]): List of reserved special token strings.
        token_to_id (Dict[str, int]): Maps string tokens to integer IDs.
        id_to_token (Dict[int, str]): Maps integer IDs back to string tokens.
        merges (List[Tuple[str, str]]): Chronological record of learned pair merges.
        merge_ranks (Dict[Tuple[str, str], int]): Priority rank of each pair merge.
    """

    def __init__(
        self,
        vocab_size: int = 500,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|unk|>", "<|endoftext|>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    def _init_base_vocab(self, text: str) -> None:
        """Initialize vocabulary with special tokens and all unique characters in text."""
        self.token_to_id.clear()
        self.id_to_token.clear()
        self.merges.clear()
        self.merge_ranks.clear()

        # 1. Register special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # 2. Register base characters found in text
        unique_chars = sorted(list(set(text)))
        for ch in unique_chars:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

    def train(self, text: str, verbose: bool = False) -> None:
        """
        Train the BPE tokenizer on a raw text corpus.
        
        Algorithm:
        1. Initialize base vocabulary (special tokens + unique characters).
        2. Pre-tokenize text into whitespace/word/punctuation chunks to preserve formatting.
        3. Form frequency table of word-character tuples.
        4. Iteratively find most frequent adjacent symbol pair across the corpus.
        5. Merge the most frequent pair into a new composite token.
        6. Repeat until reaching target vocab_size or no more pairs can be merged.
        """
        if not text:
            raise ValueError("Training text cannot be empty.")

        self._init_base_vocab(text)

        # Pre-tokenize into chunks: preserves exact formatting, whitespace, and punctuation
        chunks = re.findall(r"\s+|\w+|[^\w\s]", text)
        chunk_counts = Counter(chunks)

        # Represent each unique chunk as a tuple of its current constituent symbols
        # e.g., ('t', 'h', 'e'): 50000
        splits: Dict[Tuple[str, ...], int] = {
            tuple(chunk): count for chunk, count in chunk_counts.items()
        }

        num_merges_needed = self.vocab_size - len(self.token_to_id)
        if verbose:
            print(
                f"Base vocabulary size: {len(self.token_to_id)}. "
                f"Learning up to {num_merges_needed} merges."
            )

        for step in range(num_merges_needed):
            # Count pair frequencies across all chunk tuples weighted by occurrence
            pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
            for word_tuple, freq in splits.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                # No more adjacent pairs available to merge
                break

            # Select pair with highest corpus frequency: (a*, b*) = argmax count(a, b)
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore
            merged_token = best_pair[0] + best_pair[1]

            # Replace occurrences of (best_pair[0], best_pair[1]) in all word splits
            new_splits: Dict[Tuple[str, ...], int] = {}
            p0, p1 = best_pair
            for word_tuple, freq in splits.items():
                new_word: List[str] = []
                i = 0
                n = len(word_tuple)
                while i < n:
                    if i < n - 1 and word_tuple[i] == p0 and word_tuple[i + 1] == p1:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_splits[tuple(new_word)] = freq
            splits = new_splits

            # Record learned merge rule
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = step

            # Add newly formed subword token to vocabulary
            new_id = len(self.token_to_id)
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            if verbose and (step + 1) % 50 == 0:
                print(f"Merge {step + 1}/{num_merges_needed}: {best_pair} -> '{merged_token}'")

    def _encode_chunk(self, chunk: str) -> List[str]:
        """
        Encode a single word/symbol chunk by applying learned merges in order of rank.
        """
        unk_token = self.special_tokens[0]  # default "<|unk|>"

        # Map characters to base symbols (or unk if unseen)
        tokens = [
            ch if ch in self.token_to_id else unk_token
            for ch in chunk
        ]

        # Iteratively merge adjacent pairs with the lowest rank (earliest learned merge)
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            candidates = [
                (self.merge_ranks[pair], i, pair)
                for i, pair in enumerate(pairs)
                if pair in self.merge_ranks
            ]
            if not candidates:
                break

            # Greedily pick the candidate with minimum merge rank (highest priority)
            _, min_idx, best_pair = min(candidates, key=lambda c: c[0])
            merged_token = best_pair[0] + best_pair[1]
            tokens = tokens[:min_idx] + [merged_token] + tokens[min_idx + 2:]

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Convert a raw string into a list of integer token IDs.
        
        Preserves special tokens (<|unk|>, <|endoftext|>) without splitting them.
        """
        if not text:
            return []

        # Split text while isolating special tokens
        pattern = f"({'|'.join(re.escape(tok) for tok in self.special_tokens)})"
        parts = re.split(pattern, text)

        token_ids: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.token_to_id:
                # Direct hit for special tokens or single-symbol matches
                token_ids.append(self.token_to_id[part])
            else:
                # Pre-tokenize regular text into word/whitespace/punctuation chunks
                chunks = re.findall(r"\s+|\w+|[^\w\s]", part)
                for chunk in chunks:
                    subword_tokens = self._encode_chunk(chunk)
                    for t in subword_tokens:
                        token_ids.append(self.token_to_id[t])

        return token_ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of integer token IDs back into the reconstructed text string.
        """
        if not ids:
            return ""
        tokens = [self.id_to_token.get(idx, self.special_tokens[0]) for idx in ids]
        return "".join(tokens)

    def save(self, filepath: str) -> None:
        """Save vocabulary and merge table to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load a trained BPETokenizer from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
        )
        tokenizer.token_to_id = data["token_to_id"]
        # Invert mapping for fast ID -> token string lookup
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.merge_ranks = {pair: idx for idx, pair in enumerate(tokenizer.merges)}
        return tokenizer
