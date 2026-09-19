"""
Phase 9B: Real Training + Validation Evaluation for NumPyGPT.

This script executes:
1. Loading the real Shakespeare dataset (data/input.txt).
2. Tokenizing via BPETokenizer.
3. Chronological 90/10 train/validation split.
4. Training GPT model exclusively on the training split using Adam.
5. Periodic evaluation on both train and strictly held-out validation splits.
6. Recording loss and perplexity trajectories.
7. Running a post-training qualitative generation check.
"""

from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from tokenizer import BPETokenizer
from model import GPT
from loss import CrossEntropyLoss
from optimizer import Adam, SGD
from train import get_batch, train_step
from evaluate import train_val_split, count_parameters, compute_perplexity, estimate_loss
from generate import generate


DEFAULT_CONFIG = {
    "data_path": "data/input.txt",
    "data_chars": 80_000,       # Number of characters from input.txt to use
    "vocab_size": 150,          # BPE target vocabulary size
    "max_seq_len": 32,          # Maximum sequence context window (T)
    "d_model": 32,              # Hidden representation dimension (D)
    "num_heads": 4,             # Number of attention heads (h)
    "num_layers": 2,            # Number of stacked TransformerBlocks (N)
    "d_ff": 128,                # Feed-Forward hidden dimension (4 * d_model)
    "batch_size": 4,            # Sequences per training batch (B)
    "learning_rate": 3e-3,      # Adam learning rate
    "num_steps": 100,           # Total training iterations
    "eval_interval": 20,        # Frequency of evaluation steps
    "eval_iters": 15,           # Number of batches to average during evaluation
    "split_ratio": 0.90,        # 90% train, 10% validation
    "seed": 42,                 # Deterministic random seed
    "optimizer": "adam",        # Optimizer: "adam" or "sgd"
}



def run_experiment(config: Optional[dict] = None) -> dict:
    """
    Executes the Phase 9B real training and validation evaluation experiment.
    
    Args:
        config (Optional[dict]): Experiment configuration overriding DEFAULT_CONFIG.
        
    Returns:
        dict: Complete experiment records including config, dataset counts,
              evaluation history, parameter count, and generation sample.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Set deterministic seed
    np.random.seed(cfg["seed"])

    # 1. Load Real Dataset
    with open(cfg["data_path"], "r", encoding="utf-8") as f:
        raw_text = f.read(cfg["data_chars"])

    # 2. Tokenize Dataset with BPETokenizer
    tokenizer = BPETokenizer(vocab_size=cfg["vocab_size"])
    tokenizer.train(raw_text)
    token_ids = np.array(tokenizer.encode(raw_text), dtype=np.int32)
    actual_vocab_size = len(tokenizer.token_to_id)
    cfg["vocab_size"] = actual_vocab_size

    # 3. Deterministic 90/10 Train/Validation Split
    train_tokens, val_tokens = train_val_split(token_ids, split_ratio=cfg["split_ratio"])

    # 4. Instantiate Model, Loss Criterion, and Optimizer
    model = GPT(
        vocab_size=actual_vocab_size,
        max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
    )

    criterion = CrossEntropyLoss()
    opt_name = str(cfg.get("optimizer", "adam")).lower()
    if opt_name == "sgd":
        optimizer = SGD(model.get_params(), lr=cfg["learning_rate"])
    else:
        optimizer = Adam(model.get_params(), lr=cfg["learning_rate"])
    num_params = count_parameters(model)

    # 5. Tracking History
    history: List[Dict[str, float]] = []

    # Initial evaluation before training (Step 0)
    train_loss_0 = estimate_loss(
        model, train_tokens, criterion,
        batch_size=cfg["batch_size"], seq_len=cfg["max_seq_len"],
        eval_iters=cfg["eval_iters"], seed=cfg["seed"]
    )
    val_loss_0 = estimate_loss(
        model, val_tokens, criterion,
        batch_size=cfg["batch_size"], seq_len=cfg["max_seq_len"],
        eval_iters=cfg["eval_iters"], seed=cfg["seed"]
    )
    history.append({
        "step": 0,
        "train_loss": train_loss_0,
        "val_loss": val_loss_0,
        "train_ppl": compute_perplexity(train_loss_0),
        "val_ppl": compute_perplexity(val_loss_0),
    })

    start_time = time.time()

    # 6. Training Loop (Strictly using train_tokens)
    for step in range(1, cfg["num_steps"] + 1):
        # Sample training batch ONLY from train_tokens
        x, y = get_batch(train_tokens, batch_size=cfg["batch_size"], seq_len=cfg["max_seq_len"])
        train_step(model, criterion, optimizer, x, y)

        # Periodic evaluation on both splits
        if step % cfg["eval_interval"] == 0 or step == cfg["num_steps"]:
            # Evaluate train split
            t_loss = estimate_loss(
                model, train_tokens, criterion,
                batch_size=cfg["batch_size"], seq_len=cfg["max_seq_len"],
                eval_iters=cfg["eval_iters"], seed=cfg["seed"]
            )
            # Evaluate validation split (inference only, no gradients or updates)
            v_loss = estimate_loss(
                model, val_tokens, criterion,
                batch_size=cfg["batch_size"], seq_len=cfg["max_seq_len"],
                eval_iters=cfg["eval_iters"], seed=cfg["seed"]
            )
            history.append({
                "step": step,
                "train_loss": t_loss,
                "val_loss": v_loss,
                "train_ppl": compute_perplexity(t_loss),
                "val_ppl": compute_perplexity(v_loss),
            })

    total_training_time = time.time() - start_time

    # 7. Post-Training Qualitative Generation Check
    sample_prompt = "ROMEO:"
    generated_sample = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=sample_prompt,
        max_new_tokens=25,
        temperature=0.8,
        top_k=10,
    )

    return {
        "config": cfg,
        "raw_chars": len(raw_text),
        "total_tokens": len(token_ids),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "num_parameters": num_params,
        "history": history,
        "training_time": total_training_time,
        "generation_prompt": sample_prompt,
        "generation_output": generated_sample,
    }


if __name__ == "__main__":
    print("NumPyGPT Phase 9B: Real Training + Validation Evaluation")
    print("=" * 60)

    results = run_experiment()

    cfg = results["config"]
    print(f"\nDataset & Splits:")
    print(f"  Raw Characters:    {results['raw_chars']:,}")
    print(f"  Total Tokens:      {results['total_tokens']:,}")
    print(f"  Train Split (90%): {results['train_tokens']:,} tokens")
    print(f"  Val Split (10%):   {results['val_tokens']:,} tokens")
    print(f"  Vocabulary Size:   {cfg['vocab_size']}")
    print(f"  Total Parameters:  {results['num_parameters']:,}")

    print(f"\nTraining Results over {cfg['num_steps']} Steps ({results['training_time']:.2f}s):")
    print("-" * 65)
    print(f"| {'Step':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Train PPL':^11} | {'Val PPL':^11} |")
    print("-" * 65)
    for record in results["history"]:
        print(
            f"| {record['step']:^6d} | "
            f"{record['train_loss']:^12.4f} | "
            f"{record['val_loss']:^12.4f} | "
            f"{record['train_ppl']:^11.2f} | "
            f"{record['val_ppl']:^11.2f} |"
        )
    print("-" * 65)

    print(f"\nQualitative Generation Check:")
    print(f"  Prompt: {repr(results['generation_prompt'])}")
    print(f"  Output: {repr(results['generation_output'])}")
    print("\nPhase 9B Experiment Complete!")
