import sys
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


sys.path.append(
    str(Path(__file__).parent / "model")
)

from gpt import GPT


# ==================================================
# MODEL CONFIG
# ==================================================

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256
EMBED_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8


# ==================================================
# SFT CONFIG
# ==================================================

BATCH_SIZE = 8

LEARNING_RATE = 3e-5

MAX_STEPS = 3000

EVAL_INTERVAL = 100
EVAL_BATCHES = 20

CHECKPOINT_INTERVAL = 500

SEED = 42


# ==================================================
# PATHS
# ==================================================

BASE_CHECKPOINT = (
    "checkpoints/model_step_30000.pt"
)

TOKENIZER_PATH = (
    "data/tokenizer/tokenizer.json"
)

TRAIN_PATH = (
    "data/qa_sft/train.jsonl"
)

VAL_PATH = (
    "data/qa_sft/val.jsonl"
)

OUTPUT_DIR = Path(
    "checkpoints/qa_sft"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# DEVICE
# ==================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


random.seed(SEED)
torch.manual_seed(SEED)


# ==================================================
# TOKENIZER
# ==================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

eos_id = tokenizer.token_to_id(
    "<|endoftext|>"
)

if eos_id is None:
    raise RuntimeError(
        "<|endoftext|> token not found."
    )

print(f"EOS token ID: {eos_id}")


# ==================================================
# LOAD DATASET
# ==================================================

def load_jsonl(path):

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            examples.append(
                json.loads(line)
            )

    return examples


train_examples = load_jsonl(
    TRAIN_PATH
)

val_examples = load_jsonl(
    VAL_PATH
)


print(
    f"Training examples: "
    f"{len(train_examples):,}"
)

print(
    f"Validation examples: "
    f"{len(val_examples):,}"
)


# ==================================================
# PREPROCESS ONE EXAMPLE
# ==================================================

def encode_example(example):

    prompt_ids = tokenizer.encode(
        example["prompt"]
    ).ids

    answer_ids = tokenizer.encode(
        example["answer"]
    ).ids

    # ------------------------------------------------
    # IMPORTANT
    #
    # x:
    # Question + Answer
    #
    # targets:
    # -100 -100 -100 ... Answer tokens
    #
    # CrossEntropy ignores -100.
    # ------------------------------------------------

    full_ids = (
        prompt_ids
        + answer_ids
    )

    targets = (
        [-100] * len(prompt_ids)
        + answer_ids
    )

    # We need x[t] -> target[t+1].
    #
    # Therefore shift both by one.

    x = full_ids[:-1]
    y = targets[1:]

    # Keep the END of the prompt/answer pair
    # if it exceeds our 256-token context.
    #
    # This preserves the Answer section.

    if len(x) > CONTEXT_LENGTH:

        x = x[-CONTEXT_LENGTH:]
        y = y[-CONTEXT_LENGTH:]

    return x, y


# ==================================================
# CREATE BATCH
# ==================================================

def get_batch(examples):

    selected = random.choices(
        examples,
        k=BATCH_SIZE,
    )

    encoded = [
        encode_example(example)
        for example in selected
    ]

    max_length = max(
        len(x)
        for x, _ in encoded
    )

    max_length = min(
        max_length,
        CONTEXT_LENGTH,
    )

    batch_x = []
    batch_y = []

    for x, y in encoded:

        padding = (
            max_length - len(x)
        )

        # Padding token value does not matter for
        # the loss because padded targets = -100.

        padded_x = (
            x
            + [eos_id] * padding
        )

        padded_y = (
            y
            + [-100] * padding
        )

        batch_x.append(
            padded_x
        )

        batch_y.append(
            padded_y
        )

    x = torch.tensor(
        batch_x,
        dtype=torch.long,
        device=DEVICE,
    )

    y = torch.tensor(
        batch_y,
        dtype=torch.long,
        device=DEVICE,
    )

    return x, y


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


# ==================================================
# LOAD BASE MODEL
# ==================================================

print(
    f"\nLoading base model: "
    f"{BASE_CHECKPOINT}"
)

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "Base model loaded successfully!"
)


# ==================================================
# NEW OPTIMIZER
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# ANSWER-ONLY LOSS
# ==================================================

def calculate_loss(x, y):

    # Do NOT pass y into model here.
    #
    # We want logits only because we're computing
    # our own masked loss.

    logits, _ = model(x)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            VOCAB_SIZE,
        ),
        y.reshape(-1),
        ignore_index=-100,
    )

    return loss


# ==================================================
# EVALUATION
# ==================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    datasets = {
        "train": train_examples,
        "val": val_examples,
    }

    for name, examples in datasets.items():

        losses = []

        for _ in range(
            EVAL_BATCHES
        ):

            x, y = get_batch(
                examples
            )

            loss = calculate_loss(
                x,
                y,
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ==================================================
# SAVE CHECKPOINT
# ==================================================

def save_checkpoint(
    step,
    loss,
):

    path = (
        OUTPUT_DIR
        / f"sft_step_{step}.pt"
    )

    torch.save(
        {
            "step": step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                loss,
        },
        path,
    )

    print(
        f"\nCheckpoint saved: "
        f"{path}\n"
    )


# ==================================================
# TRAIN
# ==================================================

print(
    "\nStarting SFT v2 "
    "(answer-only loss)...\n"
)

model.train()


for step in range(
    1,
    MAX_STEPS + 1,
):

    x, y = get_batch(
        train_examples
    )

    loss = calculate_loss(
        x,
        y,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # Prevent extreme gradient spikes.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()


    if (
        step == 1
        or step % EVAL_INTERVAL == 0
    ):

        losses = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"train: "
            f"{losses['train']:.4f} | "
            f"val: "
            f"{losses['val']:.4f}"
        )


    if (
        step
        % CHECKPOINT_INTERVAL
        == 0
    ):

        save_checkpoint(
            step,
            loss.item(),
        )


# ==================================================
# FINAL MODEL
# ==================================================

FINAL_PATH = (
    OUTPUT_DIR
    / "sft_final.pt"
)

torch.save(
    {
        "step": MAX_STEPS,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss":
            loss.item(),
    },
    FINAL_PATH,
)


print(
    "\nSFT v2 complete!"
)

print(
    f"Final model: "
    f"{FINAL_PATH}"
)