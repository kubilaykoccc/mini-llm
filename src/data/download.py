from datasets import load_dataset


def main():
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    for i, sample in enumerate(dataset):
        print(f"\n{'=' * 80}")
        print(f"DOCUMENT {i + 1}")
        print(f"{'=' * 80}")
        print(sample["text"][:1000])

        if i == 4:
            break


if __name__ == "__main__":
    main()