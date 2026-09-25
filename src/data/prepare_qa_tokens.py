from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


TOKENIZER_PATH = "data/tokenizer/tokenizer.json"

TRAIN_TEXT = Path("data/qa/train.txt")
VAL_TEXT = Path("data/qa/val.txt")

TRAIN_BIN = Path("data/qa/train.bin")
VAL_BIN = Path("data/qa/val.bin")


print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


def tokenize_file(input_path, output_path):

    print(f"\nReading: {input_path}")

    text = input_path.read_text(encoding="utf-8")

    print("Tokenizing...")

    ids = tokenizer.encode(text).ids

    print(f"Tokens: {len(ids):,}")

    tokens = np.array(
        ids,
        dtype=np.uint16,
    )

    tokens.tofile(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"Saved: {output_path}")
    print(f"Size:  {size_mb:.2f} MB")


tokenize_file(
    TRAIN_TEXT,
    TRAIN_BIN,
)

tokenize_file(
    VAL_TEXT,
    VAL_BIN,
)

print("\nQA tokenization complete!")