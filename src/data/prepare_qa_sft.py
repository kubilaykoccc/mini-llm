import json
import random
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("data/qa_sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"

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

    # This is what the model will SEE.
    if context:
        prompt = (
            f"Context: {context}\n"
            f"Question: {instruction}\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"Question: {instruction}\n"
            f"Answer:"
        )

    # This is what the model should LEARN to generate.
    answer = (
        f" {response}"
        f"<|endoftext|>"
    )

    examples.append({
        "prompt": prompt,
        "answer": answer,
    })


random.seed(SEED)
random.shuffle(examples)

split_index = int(
    len(examples) * (1 - VAL_RATIO)
)

train_examples = examples[:split_index]
val_examples = examples[split_index:]


def save_jsonl(examples, path):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


save_jsonl(
    train_examples,
    TRAIN_FILE,
)

save_jsonl(
    val_examples,
    VAL_FILE,
)


print(f"Total examples:      {len(examples):,}")
print(f"Training examples:   {len(train_examples):,}")
print(f"Validation examples: {len(val_examples):,}")

print("\nExample prompt:\n")
print(train_examples[0]["prompt"])

print("\nExample answer:\n")
print(train_examples[0]["answer"])