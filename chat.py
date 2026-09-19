"""
ForgeGPT Terminal Interface.

A modern, polished CLI interface for ForgeGPT (Decoder-Only Transformer from Scratch).
"""

import os
import sys
import numpy as np

from model import GPT
from tokenizer import BPETokenizer
from generate import generate
from loss import CrossEntropyLoss
from optimizer import Adam
from train import get_batch, train_step

# ANSI Color & Style Constants
BORDER_COLOR = "\033[38;5;75m"    # Steel blue / cyan
TITLE_COLOR = "\033[1;97m"        # Bold bright white
SUBTITLE_COLOR = "\033[38;5;250m"  # Neutral light gray
DIM_COLOR = "\033[38;5;244m"       # Dim gray
PROMPT_COLOR = "\033[1;38;5;75m"   # Bold cyan
MODEL_TAG_COLOR = "\033[1;38;5;75m" # Bold cyan tag
SEPARATOR_COLOR = "\033[38;5;238m" # Muted horizontal rule
RESET = "\033[0m"


def print_header(width: int = 62) -> None:
    """Renders a clean bordered header for ForgeGPT with enlarged title."""
    line = "─" * width
    title_art = [
        r" ___                     ___ ___ _____ ",
        r"| __|__ _ _ __ _ ___   / __| _ \_   _|",
        r"| _/ _ \ '_/ _` / -_) | (_ |  _/ | |  ",
        r"|_|\___/_| \__, \___|  \___|_|   |_|  ",
    ]
    s2 = "Decoder-Only Transformer from Scratch"
    s3 = "Commands: /clear to reset screen  •  /quit to exit"

    print(f"{BORDER_COLOR}╭{line}╮{RESET}")
    for row in title_art:
        pad = width - len(row) - 2
        print(f"{BORDER_COLOR}│{RESET}  {TITLE_COLOR}{row}{RESET}" + " " * max(0, pad) + f"{BORDER_COLOR}│{RESET}")
    print(f"{BORDER_COLOR}│{RESET}" + " " * width + f"{BORDER_COLOR}│{RESET}")
    print(f"{BORDER_COLOR}│{RESET}  {SUBTITLE_COLOR}{s2}{RESET}" + " " * (width - len(s2) - 2) + f"{BORDER_COLOR}│{RESET}")
    print(f"{BORDER_COLOR}│{RESET}" + " " * width + f"{BORDER_COLOR}│{RESET}")
    print(f"{BORDER_COLOR}│{RESET}  {DIM_COLOR}{s3}{RESET}" + " " * (width - len(s3) - 2) + f"{BORDER_COLOR}│{RESET}")
    print(f"{BORDER_COLOR}╰{line}╯{RESET}\n")



def clear_screen() -> None:
    """Clears the terminal screen and redraws the header."""
    print("\033[2J\033[H", end="", flush=True)
    print_header()


def init_model_and_tokenizer():
    """Initializes and trains the model and tokenizer on the Shakespeare dataset."""
    with open("data/input.txt", "r", encoding="utf-8") as f:
        raw_text = f.read(80_000)

    tokenizer = BPETokenizer(vocab_size=150)
    tokenizer.train(raw_text)
    token_ids = np.array(tokenizer.encode(raw_text), dtype=np.int32)
    actual_vocab_size = len(tokenizer.token_to_id)

    model = GPT(
        vocab_size=actual_vocab_size,
        max_seq_len=32,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=128,
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.get_params(), lr=3e-3)

    np.random.seed(42)
    for _ in range(100):
        x, y = get_batch(token_ids, batch_size=4, seq_len=32)
        train_step(model, criterion, optimizer, x, y)

    return model, tokenizer


def main():
    print_header()

    model, tokenizer = init_model_and_tokenizer()

    separator = f"{SEPARATOR_COLOR}{'─' * 64}{RESET}"

    while True:
        try:
            prompt = input(f"{PROMPT_COLOR}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{DIM_COLOR}Exiting ForgeGPT. Goodbye!{RESET}\n")
            break

        cmd = prompt.strip()
        if cmd == "/quit":
            print(f"\n{DIM_COLOR}Exiting ForgeGPT. Goodbye!{RESET}\n")
            break

        if cmd == "/clear":
            clear_screen()
            continue

        if not cmd:
            continue

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=40,
            temperature=0.8,
            top_k=10,
        )

        # Separate prompt from completion for clean conversational display
        if output.startswith(prompt):
            completion = output[len(prompt):].lstrip("\n")
            if not completion:
                completion = output
        else:
            completion = output

        print(f"\n{MODEL_TAG_COLOR}ForgeGPT{RESET}")
        print(f"{completion}\n")
        print(f"{separator}\n")


if __name__ == "__main__":
    main()
