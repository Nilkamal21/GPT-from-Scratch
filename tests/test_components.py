"""
Unit and Integration Tests for NumPyGPT Components.

Phase 1 Suite:
Verifies:
1. Dataset loading and properties (Tiny Shakespeare).
2. BPETokenizer training and vocabulary expansion.
3. Exact lossless round-trip encoding and decoding.
4. Edge cases (empty string, single character, whitespace, punctuation).
5. Special token handling (<|endoftext|>, <|unk|>).
6. Unknown symbol substitution.
7. Serialization and deserialization (save/load roundtrip).
8. Sequence compression ratio over raw character representation.
"""

import os
import tempfile
import pytest
from tokenizer import BPETokenizer


def test_dataset_exists_and_valid():
    """Verify that Tiny Shakespeare dataset exists, is non-empty, and has expected size."""
    dataset_path = "data/input.txt"
    assert os.path.exists(dataset_path), f"Dataset file {dataset_path} not found."
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert len(text) > 1_000_000, f"Expected >1,000,000 chars, got {len(text):,}"
    # Verify Shakespeare-specific substrings exist
    assert "First Citizen:" in text
    assert "Romeo" in text or "ROMEO" in text


def test_bpe_training_and_vocab_growth():
    """Verify BPE vocabulary expands to the requested target size."""
    sample_text = (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles."
    )
    target_vocab = 60
    tokenizer = BPETokenizer(vocab_size=target_vocab)
    tokenizer.train(sample_text)

    # Base unique chars in sample_text + 2 special tokens = 26 + 2 = 28
    # Merges will grow vocab up to target_vocab
    assert len(tokenizer.token_to_id) == target_vocab
    assert len(tokenizer.id_to_token) == target_vocab
    assert len(tokenizer.merges) > 0


def test_bpe_roundtrip_lossless():
    """Verify that decode(encode(text)) == text strictly holds for in-corpus text."""
    with open("data/input.txt", "r", encoding="utf-8") as f:
        sample = f.read(30_000)

    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(sample)

    test_sentences = [
        "First Citizen:\nBefore we proceed any further, hear me speak.",
        "All:\nSpeak, speak.",
        "MENENIUS:\nWhat work's, my countrymen, in hand? where go you",
        "With bats and clubs? The matter? speak, I pray you.",
    ]

    for sentence in test_sentences:
        encoded_ids = tokenizer.encode(sentence)
        decoded_text = tokenizer.decode(encoded_ids)
        assert decoded_text == sentence, (
            f"Round-trip failed!\nExpected: {repr(sentence)}\nGot:      {repr(decoded_text)}"
        )


def test_bpe_edge_cases():
    """Verify BPE handles edge cases: empty strings, single chars, repeated chars, whitespace."""
    text = "Hello world! This is a simple test text for BPE."
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(text)

    # 1. Empty string
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

    # 2. Single known character
    assert tokenizer.decode(tokenizer.encode("H")) == "H"

    # 3. Repeated characters
    repeated = "aaaaa"
    if "a" in tokenizer.token_to_id:
        assert tokenizer.decode(tokenizer.encode(repeated)) == repeated

    # 4. Whitespace and newlines
    ws = "   \n\n  "
    if " " in tokenizer.token_to_id and "\n" in tokenizer.token_to_id:
        assert tokenizer.decode(tokenizer.encode(ws)) == ws


def test_bpe_special_tokens():
    """Verify special tokens like <|endoftext|> are preserved as atomic units."""
    text = "Shakespeare text snippet for training vocabulary."
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(text)

    prompt = "Hello <|endoftext|> world"
    encoded = tokenizer.encode(prompt)
    eot_id = tokenizer.token_to_id["<|endoftext|>"]

    assert eot_id in encoded, "<|endoftext|> was not encoded as a dedicated special token ID"
    decoded = tokenizer.decode(encoded)
    assert "<|endoftext|>" in decoded


def test_bpe_unknown_character():
    """Verify unseen characters map gracefully to <|unk|>."""
    text = "abc def"
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(text)

    # Emoji is not in base text
    encoded = tokenizer.encode("abc 🔥 def")
    unk_id = tokenizer.token_to_id["<|unk|>"]
    assert unk_id in encoded


def test_bpe_save_and_load():
    """Verify serialization to JSON and reloading preserves identical tokenizer state."""
    text = "To be or not to be, that is the question."
    tokenizer = BPETokenizer(vocab_size=60)
    tokenizer.train(text)

    test_input = "To be or not to be"
    original_encoded = tokenizer.encode(test_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.save(save_path)
        assert os.path.exists(save_path)

        loaded_tokenizer = BPETokenizer.load(save_path)
        loaded_encoded = loaded_tokenizer.encode(test_input)
        loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

        assert original_encoded == loaded_encoded
        assert loaded_decoded == test_input


def test_bpe_compression_efficiency():
    """Verify subword merges reduce sequence length relative to character count."""
    with open("data/input.txt", "r", encoding="utf-8") as f:
        corpus = f.read(50_000)

    tokenizer = BPETokenizer(vocab_size=250)
    tokenizer.train(corpus)

    test_text = corpus[:5000]
    num_chars = len(test_text)
    token_ids = tokenizer.encode(test_text)
    num_tokens = len(token_ids)

    # Compression ratio = num_chars / num_tokens
    compression_ratio = num_chars / num_tokens
    # Tokens per sample should be significantly less than characters
    assert num_tokens < num_chars
    assert compression_ratio > 1.3, f"Expected compression ratio > 1.3, got {compression_ratio:.2f}"
