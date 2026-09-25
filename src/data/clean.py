from datasets import load_dataset
import hashlib
from pathlib import Path


NUM_DOCUMENTS = 10_000
MIN_LENGTH = 200

OUTPUT_PATH = Path("data/processed/train.txt")


def clean_text(text):
    text = text.strip()

    lines = text.splitlines()
    lines = [line.strip() for line in lines]

    text = "\n".join(lines)

    return text


def main():
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()

    processed = 0
    saved = 0
    too_short = 0
    duplicates = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:

        for sample in dataset:
            text = clean_text(sample["text"])

            processed += 1

            if len(text) < MIN_LENGTH:
                too_short += 1
                continue

            text_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if text_hash in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(text_hash)

            output_file.write(text)
            output_file.write("\n\n<|endoftext|>\n\n")

            saved += 1

            if processed >= NUM_DOCUMENTS:
                break

    print("\nCLEANING COMPLETE")
    print("=" * 50)
    print(f"Processed:   {processed}")
    print(f"Saved:       {saved}")
    print(f"Too short:   {too_short}")
    print(f"Duplicates:  {duplicates}")
    print(f"Output:      {OUTPUT_PATH}")


if __name__ == "__main__":
    main()