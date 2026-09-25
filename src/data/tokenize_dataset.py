from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path


TRAIN_FILE = "data/processed/train.txt"
OUTPUT_DIR = Path("data/tokenizer")

VOCAB_SIZE = 16_000

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|unk|>",
]


def main():
    tokenizer = Tokenizer(
        BPE(unk_token="<|unk|>")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print("Training tokenizer...")

    tokenizer.train(
        files=[TRAIN_FILE],
        trainer=trainer,
    )

    tokenizer.decoder = ByteLevelDecoder()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "tokenizer.json"

    tokenizer.save(str(output_path))

    print("\nTOKENIZER TRAINING COMPLETE")
    print("=" * 50)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to:        {output_path}")

    test_text = "Machine learning is changing the world."

    encoded = tokenizer.encode(test_text)

    print("\nTEST")
    print("=" * 50)
    print(f"Text:    {test_text}")
    print(f"Tokens:  {encoded.tokens}")
    print(f"IDs:     {encoded.ids}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")


if __name__ == "__main__":
    main()