import sys
from pathlib import Path

import numpy as np
import torch


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
# FINE-TUNING CONFIG
# ==================================================

BATCH_SIZE = 8

LEARNING_RATE = 3e-5

MAX_STEPS = 3000

EVAL_INTERVAL = 100
EVAL_STEPS = 20

CHECKPOINT_INTERVAL = 500


# ==================================================
# PATHS
# ==================================================

BASE_CHECKPOINT = "checkpoints/model_step_30000.pt"

TRAIN_PATH = "data/qa/train.bin"
VAL_PATH = "data/qa/val.bin"

OUTPUT_DIR = Path("checkpoints/qa")

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


# ==================================================
# DATA
# ==================================================

print("Loading QA dataset...")

train_data = np.memmap(
    TRAIN_PATH,
    dtype=np.uint16,
    mode="r",
)

val_data = np.memmap(
    VAL_PATH,
    dtype=np.uint16,
    mode="r",
)

print(
    f"Train tokens: {len(train_data):,}"
)

print(
    f"Val tokens:   {len(val_data):,}"
)


def get_batch(data):

    starts = torch.randint(
        0,
        len(data) - CONTEXT_LENGTH - 1,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        torch.from_numpy(
            np.array(
                data[i:i + CONTEXT_LENGTH],
                dtype=np.int64,
            )
        )
        for i in starts
    ])

    y = torch.stack([
        torch.from_numpy(
            np.array(
                data[
                    i + 1:
                    i + CONTEXT_LENGTH + 1
                ],
                dtype=np.int64,
            )
        )
        for i in starts
    ])

    return (
        x.to(DEVICE),
        y.to(DEVICE),
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


# ==================================================
# LOAD BASE MODEL
# ==================================================

print(
    f"\nLoading base checkpoint:\n"
    f"{BASE_CHECKPOINT}"
)

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Base model loaded successfully!")


# ==================================================
# NEW OPTIMIZER
# ==================================================

# IMPORTANT:
#
# We deliberately DO NOT load the optimizer
# from pretraining.
#
# Fine-tuning starts with a fresh optimizer.

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# EVALUATION
# ==================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = {}

    datasets = {
        "train": train_data,
        "val": val_data,
    }

    for name, data in datasets.items():

        batch_losses = []

        for _ in range(EVAL_STEPS):

            x, y = get_batch(data)

            _, loss = model(x, y)

            batch_losses.append(
                loss.item()
            )

        losses[name] = (
            sum(batch_losses)
            / len(batch_losses)
        )

    model.train()

    return losses


# ==================================================
# CHECKPOINT
# ==================================================

def save_checkpoint(step, loss):

    path = (
        OUTPUT_DIR
        / f"qa_step_{step}.pt"
    )

    torch.save(
        {
            "step": step,
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss": loss,
        },
        path,
    )

    print(
        f"\nCheckpoint saved: {path}\n"
    )


# ==================================================
# FINE-TUNING
# ==================================================

print("\nStarting QA fine-tuning...\n")

model.train()


for step in range(
    1,
    MAX_STEPS + 1,
):

    x, y = get_batch(
        train_data
    )

    _, loss = model(
        x,
        y,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()


    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------

    if (
        step == 1
        or step % EVAL_INTERVAL == 0
    ):

        losses = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"train loss: "
            f"{losses['train']:.4f} | "
            f"val loss: "
            f"{losses['val']:.4f}"
        )


    # ----------------------------------------------
    # Checkpoint
    # ----------------------------------------------

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
    / "qa_final.pt"
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


print("\nFine-tuning complete!")

print(
    f"Final model saved to: "
    f"{FINAL_PATH}"
)