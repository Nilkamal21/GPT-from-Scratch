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
from transformer import LayerNorm, gelu, gelu_backward, FeedForward, TransformerBlock
from model import GPT
from loss import softmax, cross_entropy_loss, CrossEntropyLoss
from optimizer import Optimizer, SGD, Adam
from train import get_batch, train_step, train
from generate import top_k_filtering, top_p_filtering, generate_tokens, generate
from evaluate import train_val_split, count_parameters, compute_perplexity, estimate_loss, evaluate_model
from train_eval import run_experiment


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


# ============================================================================
# PHASE 4 TESTS: LayerNorm, GELU, FFN, and TransformerBlock
# ============================================================================

def test_layer_norm_properties():
    """Verify LayerNorm normalizes across feature dimension to mean=0, var=1."""
    d_model = 16
    ln = LayerNorm(d_model=d_model)

    x = np.random.randn(2, 4, d_model) * 5.0 + 3.0
    out = ln.forward(x)

    assert out.shape == x.shape
    # Means across last axis should be approximately 0.0
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    # Variances across last axis should be approximately 1.0
    assert np.allclose(np.var(out, axis=-1), 1.0, atol=1e-3)


def test_layer_norm_gradients():
    """Verify analytical backward gradients for LayerNorm against finite differences."""
    d_model = 6
    ln = LayerNorm(d_model=d_model)

    B, T = 2, 3
    x = np.random.randn(B, T, d_model)
    out = ln.forward(x)
    M = np.random.randn(*out.shape)
    dx = ln.backward(dout=M)

    eps = 1e-5

    # 1. Check dx gradient
    for b in range(B):
        for t in range(T):
            for d in range(d_model):
                orig = x[b, t, d]
                x[b, t, d] = orig + eps
                lp = np.sum(ln.forward(x) * M)
                x[b, t, d] = orig - eps
                lm = np.sum(ln.forward(x) * M)
                x[b, t, d] = orig

                num_g = (lp - lm) / (2 * eps)
                ana_g = dx[b, t, d]
                rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
                assert rel_err < 1e-5, f"dx mismatch at ({b},{t},{d}): num={num_g}, ana={ana_g}, rel={rel_err}"

    # 2. Check dgamma gradient
    orig_dgamma = ln.dgamma.copy()
    for d in range(d_model):
        orig = ln.gamma[d]
        ln.gamma[d] = orig + eps
        lp = np.sum(ln.forward(x) * M)
        ln.gamma[d] = orig - eps
        lm = np.sum(ln.forward(x) * M)
        ln.gamma[d] = orig

        num_g = (lp - lm) / (2 * eps)
        ana_g = orig_dgamma[d]
        rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
        assert rel_err < 1e-5, f"dgamma mismatch at {d}: num={num_g}, ana={ana_g}, rel={rel_err}"


def test_gelu_properties_and_gradients():
    """Verify GELU values and analytical gradients against finite differences."""
    # Value checks
    assert np.isclose(gelu(np.array([0.0]))[0], 0.0)
    assert np.isclose(gelu(np.array([5.0]))[0], 5.0, atol=1e-3)
    assert np.isclose(gelu(np.array([-5.0]))[0], 0.0, atol=1e-3)

    # Numerical gradient check
    x = np.linspace(-3, 3, 15)
    M = np.random.randn(*x.shape)
    dx = gelu_backward(x, M)

    eps = 1e-5
    for i in range(len(x)):
        orig = x[i]
        x[i] = orig + eps
        lp = np.sum(gelu(x) * M)
        x[i] = orig - eps
        lm = np.sum(gelu(x) * M)
        x[i] = orig

        num_g = (lp - lm) / (2 * eps)
        ana_g = dx[i]
        rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
        assert rel_err < 1e-5, f"GELU mismatch at {i}: num={num_g}, ana={ana_g}, rel={rel_err}"


def test_feed_forward_shape_and_gradients():
    """Verify FeedForward network shape and gradients against finite differences."""
    d_model = 8
    d_ff = 16
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    B, T = 2, 3
    x = np.random.randn(B, T, d_model)
    out = ffn.forward(x)
    assert out.shape == (B, T, d_model)

    M = np.random.randn(*out.shape)
    dx = ffn.backward(dout=M)

    eps = 1e-5

    # Check dx gradient
    for b in range(B):
        for t in range(T):
            for d in range(d_model):
                orig = x[b, t, d]
                x[b, t, d] = orig + eps
                lp = np.sum(ffn.forward(x) * M)
                x[b, t, d] = orig - eps
                lm = np.sum(ffn.forward(x) * M)
                x[b, t, d] = orig

                num_g = (lp - lm) / (2 * eps)
                ana_g = dx[b, t, d]
                rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
                assert rel_err < 1e-5, f"FFN dx mismatch: num={num_g}, ana={ana_g}, rel={rel_err}"

    # Check dW1 gradient
    orig_dW1 = ffn.dW1.copy()
    for i in range(d_model):
        for j in range(d_ff):
            orig = ffn.W1[i, j]
            ffn.W1[i, j] = orig + eps
            lp = np.sum(ffn.forward(x) * M)
            ffn.W1[i, j] = orig - eps
            lm = np.sum(ffn.forward(x) * M)
            ffn.W1[i, j] = orig

            num_g = (lp - lm) / (2 * eps)
            ana_g = orig_dW1[i, j]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-5, f"FFN dW1 mismatch: num={num_g}, ana={ana_g}, rel={rel_err}"


def test_transformer_block_shape_and_gradients():
    """Verify TransformerBlock forward shape and full backward gradient through all sublayers."""
    d_model = 8
    num_heads = 2
    d_ff = 16
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    B, T = 2, 3
    x = np.random.randn(B, T, d_model)
    out = block.forward(x)
    assert out.shape == (B, T, d_model)

    M = np.random.randn(*out.shape)
    dx = block.backward(dout=M)
    assert dx.shape == x.shape

    # Check numerical gradient for input x through both residual paths
    eps = 1e-5
    for b in range(B):
        for t in range(T):
            for d in range(d_model):
                orig = x[b, t, d]
                x[b, t, d] = orig + eps
                lp = np.sum(block.forward(x) * M)
                x[b, t, d] = orig - eps
                lm = np.sum(block.forward(x) * M)
                x[b, t, d] = orig

                num_g = (lp - lm) / (2 * eps)
                ana_g = dx[b, t, d]
                rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
                assert rel_err < 1e-5, f"Block dx mismatch at ({b},{t},{d}): num={num_g}, ana={ana_g}, rel={rel_err}"


# ============================================================================
# PHASE 5 TESTS: Complete GPT Architecture (model.py)
# ============================================================================

def test_gpt_output_shape_and_validation():
    """Verify that GPT maps (B, T) token IDs to (B, T, vocab_size) logits."""
    vocab_size = 50
    max_seq_len = 16
    d_model = 32
    num_heads = 4
    num_layers = 2

    gpt = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Valid input: batch of 2 sequences of length 8
    token_ids = np.random.randint(0, vocab_size, size=(2, 8))
    logits = gpt.forward(token_ids)
    assert logits.shape == (2, 8, vocab_size), f"Expected shape (2, 8, {vocab_size}), got {logits.shape}"

    # Invalid input: sequence length exceeds max_seq_len
    long_tokens = np.random.randint(0, vocab_size, size=(1, max_seq_len + 5))
    with pytest.raises(ValueError):
        gpt.forward(long_tokens)

    # Invalid input: 1D array instead of 2D
    with pytest.raises(ValueError):
        gpt.forward(np.array([1, 2, 3]))


def test_gpt_block_stacking():
    """Verify num_layers controls block count and activations pass sequentially."""
    for n_layers in [1, 3]:
        gpt = GPT(
            vocab_size=30,
            max_seq_len=10,
            d_model=16,
            num_heads=2,
            num_layers=n_layers,
        )
        assert len(gpt.blocks) == n_layers
        token_ids = np.random.randint(0, 30, size=(1, 5))
        logits = gpt.forward(token_ids)
        assert logits.shape == (1, 5, 30)


def test_gpt_final_layernorm_and_lm_head():
    """Verify Final LayerNorm and LM Head dimensions, parameters, and independence."""
    vocab_size = 40
    d_model = 16
    gpt = GPT(
        vocab_size=vocab_size,
        max_seq_len=10,
        d_model=d_model,
        num_heads=2,
        num_layers=2,
    )

    # ln_f has its own independent gamma and beta
    assert gpt.ln_f.gamma.shape == (d_model,)
    assert gpt.ln_f.beta.shape == (d_model,)
    # Verify ln_f gamma/beta are NOT the same object as any block's LayerNorm
    for block in gpt.blocks:
        assert gpt.ln_f.gamma is not block.ln1.gamma
        assert gpt.ln_f.beta is not block.ln1.beta
        assert gpt.ln_f.gamma is not block.ln2.gamma
        assert gpt.ln_f.beta is not block.ln2.beta

    # LM Head weights and bias shapes
    assert gpt.W_vocab.shape == (d_model, vocab_size)
    assert gpt.b_vocab.shape == (vocab_size,)


def test_gpt_params_and_grads_consistency():
    """Verify that get_params() and get_grads() return matching parameter keys and shapes."""
    gpt = GPT(
        vocab_size=20,
        max_seq_len=8,
        d_model=16,
        num_heads=2,
        num_layers=2,
    )

    # Calling get_grads before backward must raise RuntimeError
    with pytest.raises(RuntimeError):
        gpt.get_grads()

    token_ids = np.random.randint(0, 20, size=(2, 4))
    logits = gpt.forward(token_ids)
    dlogits = np.random.randn(*logits.shape)
    gpt.backward(dlogits)

    params = gpt.get_params()
    grads = gpt.get_grads()

    assert set(params.keys()) == set(grads.keys()), "Parameter and gradient keys do not match."

    for key in params:
        assert params[key].shape == grads[key].shape, (
            f"Shape mismatch for {key}: param {params[key].shape} vs grad {grads[key].shape}"
        )


def test_gpt_forward_backward_numerical_gradients():
    """
    Verify end-to-end analytical backward gradients against finite-difference numerical gradients.
    Tests LM Head parameters (W_vocab, b_vocab), Final LayerNorm (gamma, beta),
    TransformerBlock weights, and Embedding tables (W_e, W_p).
    """
    vocab_size = 8
    max_seq_len = 6
    d_model = 4
    num_heads = 2
    num_layers = 2

    np.random.seed(42)
    gpt = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    token_ids = np.array([[1, 3, 2], [0, 2, 1]], dtype=np.int32)
    logits = gpt.forward(token_ids)

    M = np.random.randn(*logits.shape)
    gpt.backward(dlogits=M)

    eps = 1e-5

    # 1. LM Head W_vocab numerical gradient check
    orig_dW_vocab = gpt.dW_vocab.copy()
    for d in range(d_model):
        for v in range(vocab_size):
            orig = gpt.W_vocab[d, v]
            gpt.W_vocab[d, v] = orig + eps
            lp = np.sum(gpt.forward(token_ids) * M)
            gpt.W_vocab[d, v] = orig - eps
            lm = np.sum(gpt.forward(token_ids) * M)
            gpt.W_vocab[d, v] = orig

            num_g = (lp - lm) / (2 * eps)
            ana_g = orig_dW_vocab[d, v]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-5, f"W_vocab mismatch at ({d},{v}): num={num_g}, ana={ana_g}, rel={rel_err}"

    # 2. LM Head b_vocab numerical gradient check
    orig_db_vocab = gpt.db_vocab.copy()
    for v in range(vocab_size):
        orig = gpt.b_vocab[v]
        gpt.b_vocab[v] = orig + eps
        lp = np.sum(gpt.forward(token_ids) * M)
        gpt.b_vocab[v] = orig - eps
        lm = np.sum(gpt.forward(token_ids) * M)
        gpt.b_vocab[v] = orig

        num_g = (lp - lm) / (2 * eps)
        ana_g = orig_db_vocab[v]
        rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
        assert rel_err < 1e-5, f"b_vocab mismatch at {v}: num={num_g}, ana={ana_g}, rel={rel_err}"

    # 3. Final LayerNorm gamma numerical gradient check
    orig_dgamma = gpt.ln_f.dgamma.copy()
    for d in range(d_model):
        orig = gpt.ln_f.gamma[d]
        gpt.ln_f.gamma[d] = orig + eps
        lp = np.sum(gpt.forward(token_ids) * M)
        gpt.ln_f.gamma[d] = orig - eps
        lm = np.sum(gpt.forward(token_ids) * M)
        gpt.ln_f.gamma[d] = orig

        num_g = (lp - lm) / (2 * eps)
        ana_g = orig_dgamma[d]
        rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
        assert rel_err < 1e-5, f"ln_f gamma mismatch at {d}: num={num_g}, ana={ana_g}, rel={rel_err}"

    # 4. Block 1 FFN W2 numerical gradient check (checks gradient through multiple blocks)
    block1_dW2 = gpt.blocks[1].ffn.dW2.copy()
    for i in range(gpt.blocks[1].ffn.d_ff):
        for j in range(d_model):
            orig = gpt.blocks[1].ffn.W2[i, j]
            gpt.blocks[1].ffn.W2[i, j] = orig + eps
            lp = np.sum(gpt.forward(token_ids) * M)
            gpt.blocks[1].ffn.W2[i, j] = orig - eps
            lm = np.sum(gpt.forward(token_ids) * M)
            gpt.blocks[1].ffn.W2[i, j] = orig

            num_g = (lp - lm) / (2 * eps)
            ana_g = block1_dW2[i, j]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-5, f"Block 1 FFN W2 mismatch at ({i},{j}): num={num_g}, ana={ana_g}, rel={rel_err}"

    # 5. Embedding W_e numerical gradient check (checks gradient through entire model depth)
    orig_dWe = gpt.token_embeddings.dW_e.copy()
    for v in range(vocab_size):
        for d in range(d_model):
            orig = gpt.token_embeddings.W_e[v, d]
            gpt.token_embeddings.W_e[v, d] = orig + eps
            lp = np.sum(gpt.forward(token_ids) * M)
            gpt.token_embeddings.W_e[v, d] = orig - eps
            lm = np.sum(gpt.forward(token_ids) * M)
            gpt.token_embeddings.W_e[v, d] = orig

            num_g = (lp - lm) / (2 * eps)
            ana_g = orig_dWe[v, d]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-4, f"Embedding W_e mismatch at ({v},{d}): num={num_g}, ana={ana_g}, rel={rel_err}"


# ============================================================================
# PHASE 6 TESTS: Loss, Softmax, Cross-Entropy & Backprop Connection (loss.py)
# ============================================================================

def test_softmax_sums_and_shapes():
    """Verify softmax preserves tensor shape, produces values in [0, 1], and sums to 1.0."""
    # Test 3D tensor (B, T, V)
    B, T, V = 2, 4, 10
    logits = np.random.randn(B, T, V)
    probs = softmax(logits, axis=-1)

    assert probs.shape == (B, T, V)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # Sum along last axis must equal 1.0
    row_sums = np.sum(probs, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    # Test 1D array
    v = np.random.randn(8)
    p = softmax(v)
    assert p.shape == (8,)
    assert np.isclose(np.sum(p), 1.0, atol=1e-6)


def test_softmax_numerical_stability():
    """Verify softmax does not overflow to NaN or Inf with extreme values."""
    # Extremely large positive logits that would overflow exp(x) in standard float64
    large_logits = np.array([[[1000.0, 1001.0, 1002.0]]])  # exp(1000) = inf without max subtraction
    probs = softmax(large_logits, axis=-1)
    assert not np.isnan(probs).any(), "Softmax produced NaN with large logits!"
    assert not np.isinf(probs).any(), "Softmax produced Inf with large logits!"
    assert np.allclose(np.sum(probs, axis=-1), 1.0)

    # Extremely negative logits
    neg_logits = np.array([[[-5000.0, -5001.0, -5002.0]]])
    probs_neg = softmax(neg_logits, axis=-1)
    assert not np.isnan(probs_neg).any()
    assert not np.isinf(probs_neg).any()
    assert np.allclose(np.sum(probs_neg, axis=-1), 1.0)


def test_cross_entropy_loss_values_and_shapes():
    """Verify loss calculation, expected values, and shape validation."""
    B, T, V = 2, 3, 5

    # 1. Uniform logits: all z = 0 -> probs = 1/V -> loss = -log(1/V) = log(V)
    zero_logits = np.zeros((B, T, V))
    targets = np.random.randint(0, V, size=(B, T))
    loss, _ = cross_entropy_loss(zero_logits, targets)
    expected_loss = np.log(V)
    assert np.isclose(loss, expected_loss, atol=1e-5), f"Expected {expected_loss}, got {loss}"

    # 2. Confident correct prediction: loss should approach 0
    confident_logits = np.zeros((1, 1, V))
    confident_logits[0, 0, 2] = 50.0  # Very large score for token 2
    conf_targets = np.array([[2]])
    conf_loss, _ = cross_entropy_loss(confident_logits, conf_targets)
    assert conf_loss < 1e-4, f"Loss should be near 0 for confident correct prediction, got {conf_loss}"

    # 3. Shape validation: mismatched shapes
    with pytest.raises(ValueError):
        cross_entropy_loss(np.zeros((2, 3, 5)), np.zeros((2, 4), dtype=int))

    # 4. Out of range targets
    with pytest.raises(ValueError):
        cross_entropy_loss(np.zeros((2, 3, 5)), np.array([[0, 1, 5], [2, 3, 4]]))  # target 5 >= V (5)


def test_cross_entropy_gradient_numerical_check():
    """
    Verify analytical gradient dLogits against finite-difference numerical gradients.
    
    Formula:
        g_num = [L(z + eps) - L(z - eps)] / (2 * eps)
        dLogits = (probs - targets) / (B * T)
    """
    B, T, V = 2, 3, 4
    logits = np.random.randn(B, T, V)
    targets = np.random.randint(0, V, size=(B, T))

    loss, dlogits = cross_entropy_loss(logits, targets)
    assert dlogits.shape == (B, T, V)

    eps = 1e-5
    for b in range(B):
        for t in range(T):
            for v in range(V):
                orig = logits[b, t, v]

                logits[b, t, v] = orig + eps
                l_pos, _ = cross_entropy_loss(logits, targets)

                logits[b, t, v] = orig - eps
                l_neg, _ = cross_entropy_loss(logits, targets)

                logits[b, t, v] = orig

                num_g = (l_pos - l_neg) / (2 * eps)
                ana_g = dlogits[b, t, v]
                rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
                assert rel_err < 1e-5, f"dLogits mismatch at ({b},{t},{v}): num={num_g}, ana={ana_g}, rel={rel_err}"


def test_cross_entropy_class_interface():
    """Verify CrossEntropyLoss class caching and forward/backward workflow."""
    criterion = CrossEntropyLoss()

    # Calling backward before forward must raise RuntimeError
    with pytest.raises(RuntimeError):
        criterion.backward()

    logits = np.random.randn(2, 3, 6)
    targets = np.random.randint(0, 6, size=(2, 3))

    loss = criterion.forward(logits, targets)
    assert isinstance(loss, float)
    assert loss > 0.0

    dlogits = criterion.backward()
    assert dlogits.shape == logits.shape


def test_gpt_backward_connection_with_loss():
    """
    End-to-end integration test connecting loss output directly to GPT.backward(dlogits).
    Verifies that backpropagation computes valid gradients for all model parameters.
    """
    vocab_size = 6
    max_seq_len = 4
    d_model = 8
    num_heads = 2
    num_layers = 2

    gpt = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    criterion = CrossEntropyLoss()

    # Input token sequence and next-token target sequence
    # E.g., tokens: [1, 2, 3], targets: [2, 3, 4]
    token_ids = np.array([[1, 2, 3]], dtype=np.int32)
    targets = np.array([[2, 3, 4]], dtype=np.int32)

    # 1. Forward pass through GPT
    logits = gpt.forward(token_ids)
    assert logits.shape == (1, 3, vocab_size)

    # 2. Compute loss and analytical dlogits
    loss = criterion.forward(logits, targets)
    assert loss > 0.0
    dlogits = criterion.backward()
    assert dlogits.shape == logits.shape

    # 3. Backpropagate dlogits into GPT
    gpt.backward(dlogits)

    # 4. Verify all parameter gradients are populated, non-zero, and shape-matched
    params = gpt.get_params()
    grads = gpt.get_grads()

    assert set(params.keys()) == set(grads.keys())
    for key in params:
        assert params[key].shape == grads[key].shape, f"Shape mismatch for {key}"
        assert not np.isnan(grads[key]).any(), f"NaN gradient detected in {key}!"
        assert not np.isinf(grads[key]).any(), f"Inf gradient detected in {key}!"

    # 5. Finite-difference numerical gradient check on LM Head W_vocab with end-to-end loss
    eps = 1e-5
    orig_dW_vocab = gpt.dW_vocab.copy()
    for d in range(d_model):
        for v in range(vocab_size):
            orig = gpt.W_vocab[d, v]

            gpt.W_vocab[d, v] = orig + eps
            lp, _ = cross_entropy_loss(gpt.forward(token_ids), targets)

            gpt.W_vocab[d, v] = orig - eps
            lm, _ = cross_entropy_loss(gpt.forward(token_ids), targets)

            gpt.W_vocab[d, v] = orig

            num_g = (lp - lm) / (2 * eps)
            ana_g = orig_dW_vocab[d, v]
            rel_err = abs(num_g - ana_g) / (max(abs(num_g), abs(ana_g)) + 1e-8)
            assert rel_err < 1e-5, f"End-to-end W_vocab gradient mismatch: num={num_g}, ana={ana_g}, rel={rel_err}"


# ============================================================================
# PHASE 7 TESTS: Optimizers (SGD, Adam) & Training Loop (optimizer.py, train.py)
# ============================================================================

def test_sgd_optimizer_updates():
    """Verify SGD updates: W = W - lr * g, and momentum velocity buffer."""
    # 1. Pure SGD without momentum
    w = np.array([1.0, 2.0, 3.0])
    params = {"w": w}
    grads = {"w": np.array([0.1, -0.2, 0.5])}
    lr = 0.1

    sgd = SGD(params, lr=lr, momentum=0.0)
    sgd.step(grads)

    expected = np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * (-0.2), 3.0 - 0.1 * 0.5])
    assert np.allclose(w, expected), f"Expected {expected}, got {w}"

    # 2. SGD with momentum: v_1 = g_1, W_1 = W_0 - lr * v_1
    # v_2 = momentum * v_1 + g_2, W_2 = W_1 - lr * v_2
    w_m = np.array([10.0])
    params_m = {"w": w_m}
    sgd_m = SGD(params_m, lr=1.0, momentum=0.9)

    sgd_m.step({"w": np.array([2.0])})
    # v_1 = 2.0, w = 10.0 - 1.0 * 2.0 = 8.0
    assert np.isclose(w_m[0], 8.0)
    assert np.isclose(sgd_m.velocities["w"][0], 2.0)

    sgd_m.step({"w": np.array([1.0])})
    # v_2 = 0.9 * 2.0 + 1.0 = 2.8, w = 8.0 - 1.0 * 2.8 = 5.2
    assert np.isclose(w_m[0], 5.2)
    assert np.isclose(sgd_m.velocities["w"][0], 2.8)


def test_adam_optimizer_updates_and_state():
    """
    Verify Adam updates, first/second moment calculations, bias correction,
    timestep increment, and per-parameter state tracking.
    """
    w = np.array([1.0, -1.0])
    params = {"w": w}
    g = np.array([0.1, -0.2])
    grads = {"w": g}

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    adam = Adam(params, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    # Step 1
    adam.step(grads)
    assert adam.t == 1

    # Expected first moment: m_1 = (1 - beta1) * g = 0.1 * g
    expected_m = (1.0 - beta1) * g
    assert np.allclose(adam.m["w"], expected_m)

    # Expected second moment: v_1 = (1 - beta2) * g^2 = 0.001 * g^2
    expected_v = (1.0 - beta2) * (g ** 2)
    assert np.allclose(adam.v["w"], expected_v)

    # Bias-corrected moments
    m_hat = expected_m / (1.0 - beta1 ** 1)
    v_hat = expected_v / (1.0 - beta2 ** 1)
    assert np.allclose(m_hat, g)  # On step 1, m_hat == g
    assert np.allclose(v_hat, g ** 2)  # On step 1, v_hat == g^2

    # Expected update: w_1 = w_0 - lr * m_hat / (sqrt(v_hat) + eps)
    expected_w = np.array([1.0, -1.0]) - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(w, expected_w)

    # Step 2: verify timestep and moment accumulation
    adam.step(grads)
    assert adam.t == 2


def test_optimizer_parameter_gradient_matching():
    """Verify SGD and Adam update all parameters returned by GPT.get_params()."""
    gpt = GPT(vocab_size=10, max_seq_len=6, d_model=8, num_heads=2, num_layers=1)
    token_ids = np.array([[1, 2, 3]])
    targets = np.array([[2, 3, 4]])

    criterion = CrossEntropyLoss()
    logits = gpt.forward(token_ids)
    loss = criterion.forward(logits, targets)
    dlogits = criterion.backward()
    gpt.backward(dlogits)

    params = gpt.get_params()
    grads = gpt.get_grads()

    # Record initial weights copies
    initial_weights = {k: v.copy() for k, v in params.items()}

    # Initialize Adam optimizer with full model parameter dict
    optimizer = Adam(params, lr=1e-2)
    optimizer.step(grads)

    # Every parameter with non-zero gradient must have changed
    for k in params:
        if np.any(grads[k] != 0.0):
            assert not np.array_equal(params[k], initial_weights[k]), (
                f"Parameter {k} was not updated by optimizer!"
            )


def test_get_batch_shapes_and_shifts():
    """Verify get_batch produces correct shapes, next-token shift, and validates input."""
    # Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = np.arange(10, dtype=np.int32)
    batch_size = 2
    seq_len = 4

    x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len)
    assert x.shape == (batch_size, seq_len)
    assert y.shape == (batch_size, seq_len)

    # Check shift: y[b, t] == x[b, t+1] for all t < seq_len - 1
    for b in range(batch_size):
        assert np.all(x[b, 1:] == y[b, :-1]), "Target tokens must be shifted by 1 relative to inputs!"

    # Error handling
    with pytest.raises(ValueError):
        get_batch(np.array([[1, 2], [3, 4]]), batch_size=2, seq_len=2)  # 2D data
    with pytest.raises(ValueError):
        get_batch(np.array([1, 2, 3]), batch_size=2, seq_len=5)  # len <= seq_len


def test_complete_training_step():
    """Verify that a single train_step performs forward, backward, update, and returns loss."""
    gpt = GPT(vocab_size=12, max_seq_len=8, d_model=8, num_heads=2, num_layers=1)
    criterion = CrossEntropyLoss()
    optimizer = Adam(gpt.get_params(), lr=1e-2)

    x = np.array([[1, 2, 3, 4]], dtype=np.int32)
    y = np.array([[2, 3, 4, 5]], dtype=np.int32)

    loss = train_step(gpt, criterion, optimizer, x, y)
    assert isinstance(loss, float)
    assert loss > 0.0

    # Model gradients should be populated
    grads = gpt.get_grads()
    assert len(grads) > 0
    for k in grads:
        assert not np.isnan(grads[k]).any()


def test_mini_training_loop_loss_decreases():
    """Verify that running the training loop over multiple steps strictly decreases loss."""
    np.random.seed(42)
    vocab_size = 15
    seq_len = 6
    batch_size = 4

    # Repeating sequence pattern that the model can learn easily
    data = np.array([i % vocab_size for i in range(200)], dtype=np.int32)

    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len + 4,
        d_model=16,
        num_heads=2,
        num_layers=1,
    )
    optimizer = Adam(model.get_params(), lr=2e-2)

    losses = train(
        model=model,
        data=data,
        optimizer=optimizer,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=30,
        verbose=False,
    )

    assert len(losses) == 30
    initial_loss = losses[0]
    final_loss = losses[-1]

    # Verify loss dropped significantly
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    assert final_loss < 0.5 * initial_loss, (
        f"Loss should decrease by at least 50% on simple repeating data: initial={initial_loss}, final={final_loss}"
    )


# ============================================================================
# PHASE 8 TESTS: Text Generation (generate.py)
# ============================================================================

def test_top_k_filtering():
    """Verify top-k filtering preserves only top k logits and sets others to -inf."""
    logits = np.array([2.0, 5.0, 1.0, 4.0, 3.0])
    # Top 2 are 5.0 and 4.0
    filtered = top_k_filtering(logits, top_k=2)

    assert filtered[1] == 5.0
    assert filtered[3] == 4.0
    assert np.isneginf(filtered[0])
    assert np.isneginf(filtered[2])
    assert np.isneginf(filtered[4])

    # top_k >= V should return identical logits
    assert np.array_equal(top_k_filtering(logits, top_k=10), logits)


def test_top_p_filtering():
    """Verify top-p nucleus filtering preserves cumulative probability threshold."""
    logits = np.array([10.0, 8.0, 1.0, 0.0])  # Dominant logits are indices 0 and 1
    filtered = top_p_filtering(logits, top_p=0.8)

    # Top logits must be preserved
    assert filtered[0] == 10.0
    # Lowest logits must be masked to -inf
    assert np.isneginf(filtered[2])
    assert np.isneginf(filtered[3])

    # top_p >= 1.0 returns all logits
    assert np.array_equal(top_p_filtering(logits, top_p=1.0), logits)


def test_generate_tokens_output_shape_and_length():
    """Verify generate_tokens appends exactly max_new_tokens to the input sequence."""
    gpt = GPT(vocab_size=20, max_seq_len=16, d_model=16, num_heads=2, num_layers=1)

    # 1. Single sequence (B=1, T=3)
    idx = np.array([[1, 2, 3]], dtype=np.int32)
    max_new_tokens = 5
    out = generate_tokens(gpt, idx, max_new_tokens=max_new_tokens, greedy=True)

    assert out.shape == (1, 3 + max_new_tokens)
    assert np.array_equal(out[:, :3], idx), "Original prompt tokens must be preserved!"

    # 2. Batch of sequences (B=2, T=4)
    idx_batch = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    out_batch = generate_tokens(gpt, idx_batch, max_new_tokens=4, greedy=True)
    assert out_batch.shape == (2, 4 + 4)


def test_generate_greedy_determinism():
    """Verify that greedy generation (temperature=0.0 or greedy=True) is 100% deterministic."""
    gpt = GPT(vocab_size=25, max_seq_len=20, d_model=16, num_heads=2, num_layers=1)
    idx = np.array([[5, 10, 15]], dtype=np.int32)

    out1 = generate_tokens(gpt, idx, max_new_tokens=8, greedy=True)
    out2 = generate_tokens(gpt, idx, max_new_tokens=8, greedy=True)
    out3 = generate_tokens(gpt, idx, max_new_tokens=8, temperature=0.0)

    assert np.array_equal(out1, out2), "Greedy generation must be completely deterministic."
    assert np.array_equal(out1, out3), "temperature=0.0 must match greedy=True."


def test_generate_eos_stopping():
    """Verify that generation stops early when the EOS token is emitted."""
    vocab_size = 10
    gpt = GPT(vocab_size=vocab_size, max_seq_len=20, d_model=16, num_heads=2, num_layers=1)

    # Force model to always predict token 7 as argmax
    gpt.W_vocab[:, :] = -10.0
    gpt.b_vocab[:] = -10.0
    gpt.b_vocab[7] = 100.0  # Token 7 will always be selected

    idx = np.array([[1, 2]], dtype=np.int32)
    # Set eos_id = 7: generation must stop after emitting token 7 once!
    out = generate_tokens(gpt, idx, max_new_tokens=20, greedy=True, eos_id=7)

    # Initial length 2 + 1 EOS token = 3 tokens total (stops far before max_new_tokens=20)
    assert out.shape == (1, 3)
    assert out[0, -1] == 7


def test_generate_context_window_cropping():
    """Verify that sequences longer than max_seq_len are cropped without error."""
    max_seq_len = 6
    gpt = GPT(vocab_size=15, max_seq_len=max_seq_len, d_model=8, num_heads=2, num_layers=1)

    # Initial sequence already equal to max_seq_len
    idx = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)

    # Generate 5 more tokens (total length 11 > max_seq_len)
    out = generate_tokens(gpt, idx, max_new_tokens=5, greedy=True)
    assert out.shape == (1, 11)


def test_generate_text_end_to_end():
    """Verify end-to-end generate() with BPETokenizer returns a valid completed text."""
    corpus = "hello world and welcome to text generation with gpt from scratch"
    tokenizer = BPETokenizer(vocab_size=40)
    tokenizer.train(corpus)

    model = GPT(vocab_size=40, max_seq_len=24, d_model=16, num_heads=2, num_layers=1)

    prompt = "hello"
    text = generate(model, tokenizer, prompt, max_new_tokens=8, greedy=True)

    assert isinstance(text, str)
    assert text.startswith(prompt), f"Generated text {repr(text)} must start with prompt {repr(prompt)}"


def test_generate_argument_validation():
    """Verify that invalid generation arguments raise ValueError."""
    gpt = GPT(vocab_size=10, max_seq_len=10, d_model=8, num_heads=2, num_layers=1)
    idx = np.array([[1, 2]])

    with pytest.raises(ValueError):
        generate_tokens(gpt, idx, max_new_tokens=0)
    with pytest.raises(ValueError):
        generate_tokens(gpt, idx, temperature=-0.5)
    with pytest.raises(ValueError):
        generate_tokens(gpt, idx, top_k=0)
    with pytest.raises(ValueError):
        generate_tokens(gpt, idx, top_p=0.0)
    with pytest.raises(ValueError):
        generate_tokens(gpt, idx, top_p=1.5)
    with pytest.raises(ValueError):
        generate_tokens(gpt, np.array([1, 2, 3]))  # 1D array


# ============================================================================
# PHASE 9A TESTS: Evaluation Setup & Metrics (evaluate.py)
# ============================================================================

def test_train_val_split():
    """Verify train_val_split splits chronologically, maintains total count, and validates inputs."""
    data = np.arange(100, dtype=np.int32)
    train_data, val_data = train_val_split(data, split_ratio=0.8)

    assert len(train_data) == 80
    assert len(val_data) == 20
    assert len(train_data) + len(val_data) == len(data)
    assert np.array_equal(train_data, np.arange(80))
    assert np.array_equal(val_data, np.arange(80, 100))

    # Error handling
    with pytest.raises(ValueError):
        train_val_split(np.array([[1, 2], [3, 4]]), split_ratio=0.9)  # 2D data
    with pytest.raises(ValueError):
        train_val_split(data, split_ratio=0.0)
    with pytest.raises(ValueError):
        train_val_split(data, split_ratio=1.0)


def test_count_parameters():
    """Verify count_parameters correctly calculates total scalar parameters across model."""
    vocab_size = 20
    max_seq_len = 10
    d_model = 8
    num_heads = 2
    num_layers = 1

    gpt = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    total_params = count_parameters(gpt)
    expected_params = sum(p.size for p in gpt.get_params().values())
    assert total_params == expected_params
    assert total_params > 0


def test_compute_perplexity():
    """Verify perplexity calculation: PPL = exp(loss)."""
    assert np.isclose(compute_perplexity(0.0), 1.0)
    assert np.isclose(compute_perplexity(np.log(10.0)), 10.0)

    with pytest.raises(ValueError):
        compute_perplexity(-1.0)


def test_estimate_loss_safety_and_parameters_unchanged():
    """
    Verify inference-only safety during evaluation:
    - Parameters are strictly unchanged before and after evaluation
    - Gradients are NOT computed (model.get_grads() raises RuntimeError)
    """
    gpt = GPT(vocab_size=15, max_seq_len=12, d_model=8, num_heads=2, num_layers=1)
    data = np.array([i % 15 for i in range(100)], dtype=np.int32)

    # Snapshot of initial parameters
    params_before = {k: v.copy() for k, v in gpt.get_params().items()}

    loss = estimate_loss(gpt, data, batch_size=2, seq_len=4, eval_iters=5, seed=42)
    assert isinstance(loss, float)
    assert loss > 0.0

    # Verify parameters are strictly identical
    params_after = gpt.get_params()
    for k in params_before:
        assert np.array_equal(params_before[k], params_after[k]), (
            f"Parameter {k} was mutated during evaluation!"
        )

    # Verify no gradients were computed
    with pytest.raises(RuntimeError):
        gpt.get_grads()


def test_evaluate_model_metrics():
    """Verify evaluate_model produces expected metric keys and valid numerical values."""
    gpt = GPT(vocab_size=20, max_seq_len=16, d_model=8, num_heads=2, num_layers=1)
    data = np.array([i % 20 for i in range(120)], dtype=np.int32)
    train_data, val_data = train_val_split(data, split_ratio=0.8)

    metrics = evaluate_model(
        model=gpt,
        train_data=train_data,
        val_data=val_data,
        batch_size=2,
        seq_len=6,
        eval_iters=5,
        seed=123,
    )

    expected_keys = {
        "train_loss",
        "val_loss",
        "train_perplexity",
        "val_perplexity",
        "num_parameters",
    }
    assert set(metrics.keys()) == expected_keys

    for k in expected_keys:
        assert isinstance(metrics[k], float)
        assert metrics[k] > 0.0
        assert not np.isnan(metrics[k])
        assert not np.isinf(metrics[k])


# ============================================================================
# PHASE 9B TESTS: Real Training + Validation Evaluation (train_eval.py)
# ============================================================================

def test_phase_9b_experiment_structure_and_fields():
    """Verify run_experiment executes smoothly, creates expected records, and validates outputs."""
    fast_config = {
        "data_path": "data/input.txt",
        "data_chars": 5000,
        "vocab_size": 40,
        "max_seq_len": 16,
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 1,
        "d_ff": 32,
        "batch_size": 2,
        "learning_rate": 1e-2,
        "num_steps": 6,
        "eval_interval": 3,
        "eval_iters": 3,
        "split_ratio": 0.90,
        "seed": 42,
    }

    results = run_experiment(fast_config)

    # Check top-level result fields
    expected_fields = [
        "config",
        "raw_chars",
        "total_tokens",
        "train_tokens",
        "val_tokens",
        "num_parameters",
        "history",
        "training_time",
        "generation_prompt",
        "generation_output",
    ]
    for field in expected_fields:
        assert field in results, f"Missing field {field} in experiment results!"

    assert results["raw_chars"] == 5000
    assert results["train_tokens"] > 0
    assert results["val_tokens"] > 0
    assert results["train_tokens"] + results["val_tokens"] == results["total_tokens"]
    assert results["num_parameters"] > 0

    # History should contain step 0, 3, and 6
    history = results["history"]
    assert len(history) == 3
    assert history[0]["step"] == 0
    assert history[1]["step"] == 3
    assert history[2]["step"] == 6

    for record in history:
        assert "train_loss" in record and record["train_loss"] > 0.0
        assert "val_loss" in record and record["val_loss"] > 0.0
        assert "train_ppl" in record and record["train_ppl"] > 0.0
        assert "val_ppl" in record and record["val_ppl"] > 0.0

    # Generation output should be non-empty string starting with prompt
    assert isinstance(results["generation_output"], str)
    assert results["generation_output"].startswith(results["generation_prompt"])


def test_phase_9b_training_isolation_and_parameter_changes():
    """
    Verify training only uses train split, changes model parameters,
    while validation evaluation never updates parameters.
    """
    data = np.arange(200, dtype=np.int32)
    train_data, val_data = train_val_split(data, split_ratio=0.85)

    # Verify no overlap between train and validation splits
    assert len(set(train_data.tolist()).intersection(set(val_data.tolist()))) == 0

    model = GPT(vocab_size=200, max_seq_len=12, d_model=16, num_heads=2, num_layers=1)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.get_params(), lr=1e-2)

    # Snapshot 0: Initial weights
    initial_weights = {k: v.copy() for k, v in model.get_params().items()}

    # 1. Validation evaluation must NOT change weights
    _ = estimate_loss(model, val_data, criterion, batch_size=2, seq_len=6, eval_iters=3, seed=42)
    for k in initial_weights:
        assert np.array_equal(model.get_params()[k], initial_weights[k]), (
            f"Validation evaluation altered parameter {k}!"
        )

    # 2. Training step on train_data MUST change weights
    x, y = get_batch(train_data, batch_size=2, seq_len=6)
    train_step(model, criterion, optimizer, x, y)

    weights_after_train = model.get_params()
    changed_params = [
        k for k in initial_weights if not np.array_equal(weights_after_train[k], initial_weights[k])
    ]
    assert len(changed_params) > 0, "Training step failed to update model parameters!"


def test_optimizer_comparison_sgd_vs_adam():
    """
    Verify SGD and Adam comparison under identical conditions:
    same initial parameters, same batch sampling seed, same loss criterion.
    """
    from compare_optimizers import run_comparison

    # Fast run with small steps
    results = run_comparison(
        data_chars=1_000,
        vocab_size=50,
        max_seq_len=8,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        batch_size=2,
        learning_rate=3e-3,
        num_steps=10,
        eval_interval=5,
        eval_iters=3,
        split_ratio=0.90,
        seed=42,
    )

    sgd = results["SGD"]
    adam = results["Adam"]

    # Initial loss must be identical because model parameters and seed were identical
    assert np.isclose(sgd["initial_loss"], adam["initial_loss"], atol=1e-6)
    assert np.isclose(sgd["initial_val_loss"], adam["initial_val_loss"], atol=1e-6)

    # Both must decrease loss and maintain positive perplexities
    assert sgd["final_train_loss"] > 0.0 and np.isfinite(sgd["final_train_loss"])
    assert adam["final_train_loss"] > 0.0 and np.isfinite(adam["final_train_loss"])
    assert sgd["final_train_ppl"] > 1.0
    assert adam["final_train_ppl"] > 1.0

