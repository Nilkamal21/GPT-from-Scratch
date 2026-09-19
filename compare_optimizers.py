"""
Optimizer Comparison Experiment: SGD vs Adam on NumPyGPT.

This script benchmarks SGD and Adam optimizers under strictly identical conditions:
- Same dataset and tokenizer
- Same train/validation split (90/10)
- Same model architecture (2-layer GPT, 36k parameters)
- Same initial model parameters (restored before each experiment)
- Same random seed and batch sampling sequence
- Same number of training steps (100)
- Same batch size (4) and sequence length (32)
- Same learning rate (3e-3)
"""

from typing import Dict, List, Tuple
import copy
import time
import numpy as np

from tokenizer import BPETokenizer
from model import GPT
from loss import CrossEntropyLoss
from optimizer import SGD, Adam
from train import get_batch, train_step
from evaluate import train_val_split, count_parameters, compute_perplexity, estimate_loss


def run_comparison(
    data_path: str = "data/input.txt",
    data_chars: int = 80_000,
    vocab_size: int = 150,
    max_seq_len: int = 32,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 128,
    batch_size: int = 4,
    learning_rate: float = 3e-3,
    num_steps: int = 100,
    eval_interval: int = 20,
    eval_iters: int = 15,
    split_ratio: float = 0.90,
    seed: int = 42,
) -> Dict[str, dict]:
    """
    Executes Experiment A (SGD) and Experiment B (Adam) under strictly identical conditions.
    """
    # 1. Deterministic Tokenization and Data Split
    np.random.seed(seed)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read(data_chars)

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(raw_text)
    token_ids = np.array(tokenizer.encode(raw_text), dtype=np.int32)
    actual_vocab_size = len(tokenizer.token_to_id)

    train_tokens, val_tokens = train_val_split(token_ids, split_ratio=split_ratio)

    # 2. Instantiate Model and Cache Initial Parameters
    np.random.seed(seed)
    model = GPT(
        vocab_size=actual_vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    )
    initial_params = {k: v.copy() for k, v in model.get_params().items()}
    criterion = CrossEntropyLoss()

    # Initial evaluation (Step 0)
    init_train_loss = estimate_loss(
        model, train_tokens, criterion,
        batch_size=batch_size, seq_len=max_seq_len,
        eval_iters=eval_iters, seed=seed,
    )
    init_val_loss = estimate_loss(
        model, val_tokens, criterion,
        batch_size=batch_size, seq_len=max_seq_len,
        eval_iters=eval_iters, seed=seed,
    )
    init_train_ppl = compute_perplexity(init_train_loss)
    init_val_ppl = compute_perplexity(init_val_loss)

    def run_single_optimizer(opt_name: str, opt_class) -> dict:
        # Restore EXACT initial parameters
        for k, v in initial_params.items():
            model.get_params()[k][:] = v.copy()

        opt = opt_class(model.get_params(), lr=learning_rate)

        history: List[Dict[str, float]] = [{
            "step": 0,
            "train_loss": init_train_loss,
            "val_loss": init_val_loss,
            "train_ppl": init_train_ppl,
            "val_ppl": init_val_ppl,
        }]

        # Seed random generator for identical batch sequence
        np.random.seed(seed)
        start_time = time.time()

        for step in range(1, num_steps + 1):
            x, y = get_batch(train_tokens, batch_size=batch_size, seq_len=max_seq_len)
            train_step(model, criterion, opt, x, y)

            if step % eval_interval == 0 or step == num_steps:
                t_loss = estimate_loss(
                    model, train_tokens, criterion,
                    batch_size=batch_size, seq_len=max_seq_len,
                    eval_iters=eval_iters, seed=seed,
                )
                v_loss = estimate_loss(
                    model, val_tokens, criterion,
                    batch_size=batch_size, seq_len=max_seq_len,
                    eval_iters=eval_iters, seed=seed,
                )
                history.append({
                    "step": step,
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "train_ppl": compute_perplexity(t_loss),
                    "val_ppl": compute_perplexity(v_loss),
                })

        elapsed = time.time() - start_time
        final_record = history[-1]

        return {
            "optimizer": opt_name,
            "initial_loss": init_train_loss,
            "initial_val_loss": init_val_loss,
            "final_train_loss": final_record["train_loss"],
            "final_val_loss": final_record["val_loss"],
            "final_train_ppl": final_record["train_ppl"],
            "final_val_ppl": final_record["val_ppl"],
            "history": history,
            "runtime": elapsed,
        }

    sgd_results = run_single_optimizer("SGD", SGD)
    adam_results = run_single_optimizer("Adam", Adam)

    return {
        "SGD": sgd_results,
        "Adam": adam_results,
        "vocab_size": actual_vocab_size,
        "total_params": count_parameters(model),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
    }


if __name__ == "__main__":
    print("NumPyGPT: Optimizer Benchmark (SGD vs Adam)")
    print("=" * 65)

    results = run_comparison()
    sgd = results["SGD"]
    adam = results["Adam"]

    print(f"\nModel & Data Setup:")
    print(f"  Vocabulary Size:    {results['vocab_size']}")
    print(f"  Total Parameters:   {results['total_params']:,}")
    print(f"  Train Tokens (90%): {results['train_tokens']:,}")
    print(f"  Val Tokens (10%):   {results['val_tokens']:,}")

    print("\nComparison Table:")
    print("-" * 88)
    print(
        f"| {'Optimizer':<10} | {'Initial Loss':<12} | {'Final Train Loss':<16} | "
        f"{'Final Val Loss':<14} | {'Final Train PPL':<15} | {'Final Val PPL':<13} |"
    )
    print("-" * 88)
    for res in [sgd, adam]:
        print(
            f"| {res['optimizer']:<10} | "
            f"{res['initial_loss']:<12.4f} | "
            f"{res['final_train_loss']:<16.4f} | "
            f"{res['final_val_loss']:<14.4f} | "
            f"{res['final_train_ppl']:<15.2f} | "
            f"{res['final_val_ppl']:<13.2f} |"
        )
    print("-" * 88)
