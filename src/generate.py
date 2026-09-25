import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.append(
    str(Path(__file__).parent / "model")
)

from gpt import GPT


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

VOCAB_SIZE = 16_000
CONTEXT_LENGTH = 256
EMBED_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8

CHECKPOINT = "checkpoints/model_step_15000.pt"
TOKENIZER_FILE = "data/tokenizer/tokenizer.json"
PROMPT = "Question: What is artificial intelligence?\nAnswer:"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = GPT(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# --------------------------------------------------
# TOKENIZE PROMPT
# --------------------------------------------------

encoded = tokenizer.encode(PROMPT)

input_ids = torch.tensor(
    [encoded.ids],
    dtype=torch.long,
    device=DEVICE,
)


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

with torch.no_grad():

    for _ in range(MAX_NEW_TOKENS):

        context = input_ids[
            :,
            -CONTEXT_LENGTH:
        ]

        logits, _ = model(context)

        next_token_logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )


# --------------------------------------------------
# DECODE
# --------------------------------------------------

generated_ids = (
    input_ids[0]
    .detach()
    .cpu()
    .tolist()
)

text = tokenizer.decode(
    generated_ids
)

print("\nPROMPT:")
print(PROMPT)

print("\nGENERATED:")
print(text)