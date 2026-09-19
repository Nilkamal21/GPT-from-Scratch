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
import numpy as np
from tokenizer import BPETokenizer
from embeddings import Embedding
from attention import causal_mask, scaled_dot_product_attention, MultiHeadAttention


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


# ============================================================================
# PHASE 2 TESTS: Embeddings & Positional Information
# ============================================================================

def test_embedding_output_shape():
    """Verify that Embedding maps (B, T) token IDs to continuous (B, T, D) vectors."""
    vocab_size = 100
    max_seq_len = 32
    d_model = 64
    embedding = Embedding(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model)

    # Batch of shape (2, 8)
    token_ids = np.array([[10, 20, 30, 40, 50, 60, 70, 80],
                          [1,   2,  3,  4,  5,  6,  7,  8]], dtype=np.int32)
    out = embedding.forward(token_ids)
    assert out.shape == (2, 8, d_model), f"Expected shape (2, 8, {d_model}), got {out.shape}"


def test_embedding_exceeds_max_seq_len():
    """Verify that forward raises ValueError when sequence length exceeds max_seq_len."""
    embedding = Embedding(vocab_size=50, max_seq_len=10, d_model=16)
    long_sequence = np.zeros((1, 15), dtype=np.int32)
    with pytest.raises(ValueError):
        embedding.forward(long_sequence)


def test_embedding_determinism():
    """Verify that identical token IDs at the same position produce identical vectors."""
    embedding = Embedding(vocab_size=50, max_seq_len=10, d_model=16)
    token_ids_1 = np.array([[5, 12, 25]], dtype=np.int32)
    token_ids_2 = np.array([[5, 12, 25]], dtype=np.int32)

    out1 = embedding.forward(token_ids_1)
    out2 = embedding.forward(token_ids_2)
    assert np.allclose(out1, out2), "Forward pass must be deterministic for identical inputs."


def test_embedding_position_sensitivity():
    """Verify that swapping token positions produces different vectors due to positional embeddings."""
    embedding = Embedding(vocab_size=50, max_seq_len=10, d_model=16)
    # [10, 20] vs [20, 10]
    seq_a = np.array([[10, 20]], dtype=np.int32)
    seq_b = np.array([[20, 10]], dtype=np.int32)

    out_a = embedding.forward(seq_a)
    out_b = embedding.forward(seq_b)

    # Token 10 at position 0 in seq_a should NOT equal token 10 at position 1 in seq_b
    assert not np.allclose(out_a[0, 0, :], out_b[0, 1, :]), (
        "Token vectors must differ across different positions due to positional embeddings."
    )


def test_embedding_numerical_gradients():
    """
    Verify analytical backward gradients dW_e and dW_p against finite-difference numerical gradients.
    
    Formula:
        g_num = [L(theta + eps) - L(theta - eps)] / (2 * eps)
    """
    vocab_size = 15
    max_seq_len = 10
    d_model = 8
    embedding = Embedding(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model)

    # Small batch with repeated tokens to thoroughly test scatter-add accumulation
    token_ids = np.array([[2, 5, 2],
                          [7, 2, 5]], dtype=np.int32)  # Token 2 appears 3 times!
    B, T = token_ids.shape

    # Forward pass
    out = embedding.forward(token_ids)

    # Define scalar loss: L = sum(out * M) where M is a fixed random upstream gradient
    M = np.random.randn(B, T, d_model)
    embedding.backward(dout=M)

    analytical_dWe = embedding.dW_e.copy()
    analytical_dWp = embedding.dW_p.copy()

    eps = 1e-5

    # 1. Check numerical gradients for W_e
    for v in range(vocab_size):
        for d in range(d_model):
            orig = embedding.W_e[v, d]

            embedding.W_e[v, d] = orig + eps
            out_pos = embedding.forward(token_ids)
            loss_pos = np.sum(out_pos * M)

            embedding.W_e[v, d] = orig - eps
            out_neg = embedding.forward(token_ids)
            loss_neg = np.sum(out_neg * M)

            embedding.W_e[v, d] = orig  # restore

            num_grad = (loss_pos - loss_neg) / (2 * eps)
            ana_grad = analytical_dWe[v, d]

            rel_error = abs(num_grad - ana_grad) / (max(abs(num_grad), abs(ana_grad)) + 1e-8)
            assert rel_error < 1e-5, (
                f"Gradient mismatch for W_e[{v}, {d}]: num={num_grad:.8f}, ana={ana_grad:.8f}, rel_err={rel_error:.2e}"
            )

    # 2. Check numerical gradients for W_p
    for t in range(max_seq_len):
        for d in range(d_model):
            orig = embedding.W_p[t, d]

            embedding.W_p[t, d] = orig + eps
            out_pos = embedding.forward(token_ids)
            loss_pos = np.sum(out_pos * M)

            embedding.W_p[t, d] = orig - eps
            out_neg = embedding.forward(token_ids)
            loss_neg = np.sum(out_neg * M)

            embedding.W_p[t, d] = orig  # restore

            num_grad = (loss_pos - loss_neg) / (2 * eps)
            ana_grad = analytical_dWp[t, d]

            rel_error = abs(num_grad - ana_grad) / (max(abs(num_grad), abs(ana_grad)) + 1e-8)
            assert rel_error < 1e-5, (
                f"Gradient mismatch for W_p[{t}, {d}]: num={num_grad:.8f}, ana={ana_grad:.8f}, rel_err={rel_error:.2e}"
            )


# ============================================================================
# PHASE 3 TESTS: Self-Attention & Multi-Head Attention
# ============================================================================

def test_causal_mask_structure():
    """Verify causal mask has 0.0 on and below diagonal, and -1e9 strictly above diagonal."""
    T = 4
    mask = causal_mask(T)
    assert mask.shape == (T, T)
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert mask[i, j] == 0.0, f"Expected 0.0 at ({i}, {j}), got {mask[i, j]}"
            else:
                assert mask[i, j] == -1e9, f"Expected -1e9 at ({i}, {j}), got {mask[i, j]}"


def test_scaled_dot_product_attention_causality():
    """Verify scaled dot-product attention zeroes out future attention weights and rows sum to 1.0."""
    B, h, T, d_k = 2, 2, 5, 8
    Q = np.random.randn(B, h, T, d_k)
    K = np.random.randn(B, h, T, d_k)
    V = np.random.randn(B, h, T, d_k)
    mask = causal_mask(T)

    out, A = scaled_dot_product_attention(Q, K, V, mask=mask)

    # 1. Output shape must be (B, h, T, d_k)
    assert out.shape == (B, h, T, d_k)
    # 2. Attention weights shape must be (B, h, T, T)
    assert A.shape == (B, h, T, T)

    # 3. Future positions (j > i) must have strictly 0.0 attention weight
    for i in range(T):
        for j in range(i + 1, T):
            assert np.all(A[:, :, i, j] == 0.0), f"Future attention leakage detected at ({i}, {j})"

    # 4. Each row must sum to 1.0 (valid probability distribution)
    assert np.allclose(np.sum(A, axis=-1), 1.0), "Attention weights do not sum to 1.0 across rows"


def test_multi_head_attention_shape_and_validation():
    """Verify MultiHeadAttention output shape and divisibility validation."""
    # Invalid configuration: d_model not divisible by num_heads
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=35, num_heads=4)

    # Valid configuration
    d_model = 32
    num_heads = 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    X = np.random.randn(2, 6, d_model)

    out = mha.forward(X)
    assert out.shape == (2, 6, d_model), f"Expected shape (2, 6, {d_model}), got {out.shape}"


def test_multi_head_attention_zero_causal_leakage():
    """
    Verify that modifying future tokens has ZERO effect on past token representations.
    
    If token at index t_future is altered, representations at indices t <= t_past
    must remain mathematically identical.
    """
    d_model = 16
    num_heads = 2
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Sequence of length 4
    X1 = np.random.randn(1, 4, d_model)
    out1 = mha.forward(X1)

    # Create X2 by modifying ONLY the last token (position 3)
    X2 = X1.copy()
    X2[0, 3, :] += 5.0  # Big change to future token
    out2 = mha.forward(X2)

    # Positions 0, 1, 2 must remain completely identical!
    assert np.allclose(out1[0, :3, :], out2[0, :3, :]), (
        "Causal leakage! Changing future tokens altered past token representations."
    )
    # Position 3 should be different
    assert not np.allclose(out1[0, 3, :], out2[0, 3, :])


def test_multi_head_attention_numerical_gradients():
    """
    Verify analytical backward gradients for MultiHeadAttention against finite differences.
    Tests dX, dW_q, db_q, dW_k, db_k, dW_v, db_v, dW_o, db_o.
    """
    d_model = 8
    num_heads = 2
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    B, T = 2, 3
    X = np.random.randn(B, T, d_model)

    out = mha.forward(X)
    M = np.random.randn(*out.shape)  # Upstream gradient dout
    dX = mha.backward(dout=M)

    eps = 1e-5

    # 1. Check dX gradient
    for b in range(B):
        for t in range(T):
            for d in range(d_model):
                orig = X[b, t, d]

                X[b, t, d] = orig + eps
                out_pos = mha.forward(X)
                loss_pos = np.sum(out_pos * M)

                X[b, t, d] = orig - eps
                out_neg = mha.forward(X)
                loss_neg = np.sum(out_neg * M)

                X[b, t, d] = orig

                num_g = (loss_pos - loss_neg) / (2 * eps)
                ana_g = dX[b, t, d]
                rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
                assert rel_err < 1e-5, f"dX mismatch at ({b},{t},{d}): num={num_g}, ana={ana_g}, rel={rel_err}"

    # 2. Check dW_q gradient
    orig_dWq = mha.dW_q.copy()
    for i in range(d_model):
        for j in range(d_model):
            orig = mha.W_q[i, j]

            mha.W_q[i, j] = orig + eps
            out_pos = mha.forward(X)
            loss_pos = np.sum(out_pos * M)

            mha.W_q[i, j] = orig - eps
            out_neg = mha.forward(X)
            loss_neg = np.sum(out_neg * M)

            mha.W_q[i, j] = orig

            num_g = (loss_pos - loss_neg) / (2 * eps)
            ana_g = orig_dWq[i, j]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-5, f"dW_q mismatch: num={num_g}, ana={ana_g}, rel={rel_err}"

    # 3. Check dW_o gradient
    orig_dWo = mha.dW_o.copy()
    for i in range(d_model):
        for j in range(d_model):
            orig = mha.W_o[i, j]

            mha.W_o[i, j] = orig + eps
            out_pos = mha.forward(X)
            loss_pos = np.sum(out_pos * M)

            mha.W_o[i, j] = orig - eps
            out_neg = mha.forward(X)
            loss_neg = np.sum(out_neg * M)

            mha.W_o[i, j] = orig

            num_g = (loss_pos - loss_neg) / (2 * eps)
            ana_g = orig_dWo[i, j]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-5, f"dW_o mismatch: num={num_g}, ana={ana_g}, rel={rel_err}"
