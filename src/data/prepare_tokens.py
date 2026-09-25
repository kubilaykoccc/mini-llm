from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


INPUT_FILE = Path("data/processed/train.txt")
TOKENIZER_FILE = Path("data/tokenizer/tokenizer.json")

OUTPUT_DIR = Path("data/processed")

TRAIN_OUTPUT = OUTPUT_DIR / "train.bin"
VAL_OUTPUT = OUTPUT_DIR / "val.bin"

TRAIN_RATIO = 0.9


def main():
    print("Loading tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print("Reading training corpus...")

    text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    print("Encoding text into token IDs...")

    encoding = tokenizer.encode(text)
    token_ids = encoding.ids

    print(f"Total tokens: {len(token_ids):,}")

    split_index = int(
        len(token_ids) * TRAIN_RATIO
    )

    train_ids = token_ids[:split_index]
    val_ids = token_ids[split_index:]

    print(f"Training tokens:   {len(train_ids):,}")
    print(f"Validation tokens: {len(val_ids):,}")

    train_array = np.array(
        train_ids,
        dtype=np.uint16
    )

    val_array = np.array(
        val_ids,
        dtype=np.uint16
    )

    train_array.tofile(TRAIN_OUTPUT)
    val_array.tofile(VAL_OUTPUT)

    print("\nTOKEN PREPARATION COMPLETE")
    print("=" * 50)

    print(f"Train file: {TRAIN_OUTPUT}")
    print(f"Val file:   {VAL_OUTPUT}")

    print(
        f"Train size: {TRAIN_OUTPUT.stat().st_size / 1024 / 1024:.2f} MB"
    )

    print(
        f"Val size:   {VAL_OUTPUT.stat().st_size / 1024 / 1024:.2f} MB"
    )


if __name__ == "__main__":
    main()