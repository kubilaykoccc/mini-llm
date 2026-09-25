import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


sys.path.append(
    str(Path(__file__).parent / "model")
)

from gpt import GPT


# ==================================================
# CONFIG
# ==================================================

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256
EMBED_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8

CHECKPOINT = "checkpoints/qa_sft/sft_step_3000.pt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.7
TOP_K = 40


# ==================================================
# DEVICE
# ==================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ==================================================
# TOKENIZER
# ==================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

EOS_ID = tokenizer.token_to_id(
    "<|endoftext|>"
)

print(f"EOS token ID: {EOS_ID}")


# ==================================================
# MODEL
# ==================================================

model = GPT(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
).to(DEVICE)


print(
    f"Loading checkpoint: {CHECKPOINT}"
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("SFT v2 model loaded!\n")


# ==================================================
# QUESTION
# ==================================================

question = input("Question: ")

prompt = (
    f"Question: {question}\n"
    f"Answer:"
)

prompt_ids = tokenizer.encode(
    prompt
).ids

generated = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=DEVICE,
)


# ==================================================
# GENERATION
# ==================================================

with torch.no_grad():

    for _ in range(MAX_NEW_TOKENS):

        x = generated[
            :,
            -CONTEXT_LENGTH:
        ]

        logits, _ = model(x)

        logits = logits[:, -1, :]

        # Temperature
        logits = logits / TEMPERATURE

        # Top-K
        top_values, _ = torch.topk(
            logits,
            TOP_K,
        )

        cutoff = top_values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < cutoff,
            torch.full_like(
                logits,
                float("-inf"),
            ),
            logits,
        )

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        token_id = next_token.item()

        # STOP when model generates EOS
        if token_id == EOS_ID:
            break

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )


# ==================================================
# DECODE ONLY THE ANSWER
# ==================================================

answer_ids = generated[
    0,
    len(prompt_ids):
].tolist()

answer = tokenizer.decode(
    answer_ids
)

print("\n--- ANSWER ---\n")
print(answer)