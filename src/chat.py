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

CHECKPOINT = "checkpoints/qa/qa_final.pt"

TOKENIZER_PATH = (
    "data/tokenizer/tokenizer.json"
)

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.7


# ==================================================
# DEVICE
# ==================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    f"Using device: {DEVICE}"
)


# ==================================================
# TOKENIZER
# ==================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


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
    f"Loading checkpoint: "
    f"{CHECKPOINT}"
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("QA model loaded!\n")


# ==================================================
# QUESTION
# ==================================================

question = input(
    "Question: "
)

prompt = (
    f"Question: {question}\n"
    f"Answer:"
)


# ==================================================
# TOKENIZE
# ==================================================

ids = tokenizer.encode(
    prompt
).ids

x = torch.tensor(
    ids,
    dtype=torch.long,
    device=DEVICE,
).unsqueeze(0)


# ==================================================
# GENERATION
# ==================================================

with torch.no_grad():

    for _ in range(
        MAX_NEW_TOKENS
    ):

        x_cond = x[
            :,
            -CONTEXT_LENGTH:
        ]

        logits, _ = model(
            x_cond
        )

        logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        next_token = (
            torch.multinomial(
                probs,
                num_samples=1,
            )
        )

        x = torch.cat(
            [
                x,
                next_token,
            ],
            dim=1,
        )


# ==================================================
# DECODE
# ==================================================

output = tokenizer.decode(
    x[0].tolist()
)

print("\n--- RESPONSE ---\n")

print(output)