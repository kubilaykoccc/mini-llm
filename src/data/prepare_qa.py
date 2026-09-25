from pathlib import Path
from datasets import load_dataset
import random


OUTPUT_DIR = Path("data/qa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.txt"
VAL_FILE = OUTPUT_DIR / "val.txt"

VAL_RATIO = 0.10
SEED = 42


print("Loading Dolly 15K...")

dataset = load_dataset(
    "databricks/databricks-dolly-15k",
    split="train",
)

examples = []

for item in dataset:
    instruction = item["instruction"].strip()
    response = item["response"].strip()
    context = item["context"].strip()

    if not instruction or not response:
        continue

    if context:
        text = (
            f"Context: {context}\n"
            f"Question: {instruction}\n"
            f"Answer: {response}\n"
            f"<|endoftext|>"
        )
    else:
        text = (
            f"Question: {instruction}\n"
            f"Answer: {response}\n"
            f"<|endoftext|>"
        )

    examples.append(text)


random.seed(SEED)
random.shuffle(examples)

split_index = int(len(examples) * (1 - VAL_RATIO))

train_examples = examples[:split_index]
val_examples = examples[split_index:]


TRAIN_FILE.write_text(
    "\n\n".join(train_examples),
    encoding="utf-8",
)

VAL_FILE.write_text(
    "\n\n".join(val_examples),
    encoding="utf-8",
)


print(f"Total examples:      {len(examples):,}")
print(f"Training examples:   {len(train_examples):,}")
print(f"Validation examples: {len(val_examples):,}")

print("\nExample:\n")
print(train_examples[0])