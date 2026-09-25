from datasets import load_dataset
import hashlib
import statistics


NUM_DOCUMENTS = 10_000


def main():
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    lengths = []
    empty_documents = 0
    short_documents = 0
    very_long_documents = 0
    documents_with_url = 0
    duplicate_documents = 0

    seen_hashes = set()

    shortest_text = None
    longest_text = None

    for i, sample in enumerate(dataset):
        text = sample["text"].strip()

        length = len(text)
        lengths.append(length)

        if length > 50_000:
            very_long_documents += 1

        if "http://" in text or "https://" in text or "www." in text:
            documents_with_url += 1

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if text_hash in seen_hashes:
            duplicate_documents += 1
        else:
            seen_hashes.add(text_hash)

        if length == 0:
            empty_documents += 1

        if length < 200:
            short_documents += 1

        if shortest_text is None or length < len(shortest_text):
            shortest_text = text

        if longest_text is None or length > len(longest_text):
            longest_text = text

        if i + 1 >= NUM_DOCUMENTS:
            break

    print("\nDATASET PROFILE")
    print("=" * 50)

    print(f"Documents analyzed:   {len(lengths)}")
    print(f"Empty documents:      {empty_documents}")
    print(f"Short documents:      {short_documents}")
    print(f"Very long documents:  {very_long_documents}")
    print(f"Documents with URLs:  {documents_with_url}")
    print(f"Exact duplicates:     {duplicate_documents}")

    print(f"\nMinimum length:        {min(lengths):,} characters")
    print(f"Maximum length:        {max(lengths):,} characters")
    print(f"Average length:        {statistics.mean(lengths):,.0f} characters")
    print(f"Median length:         {statistics.median(lengths):,.0f} characters")

    print("\nSHORTEST DOCUMENT")
    print("=" * 50)
    print(shortest_text)

    print("\nLONGEST DOCUMENT (first 1000 characters)")
    print("=" * 50)
    print(longest_text[:1000])


if __name__ == "__main__":
    main()