import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------
# IMPORT MODEL
# --------------------------------------------------

# src/model klasörünü Python import path'ine ekliyoruz.
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

BATCH_SIZE = 16
LEARNING_RATE = 3e-4

TRAIN_FILE = "data/processed/train.bin"
VAL_FILE = "data/processed/val.bin"

MAX_STEPS = 37_000

EVAL_INTERVAL = 100
EVAL_STEPS = 20

CHECKPOINT_INTERVAL = 500


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# --------------------------------------------------
# DATA
# --------------------------------------------------

train_data = np.memmap(
    TRAIN_FILE,
    dtype=np.uint16,
    mode="r",
)

val_data = np.memmap(
    VAL_FILE,
    dtype=np.uint16,
    mode="r",
)


def get_batch(data):

    max_start = (
        len(data)
        - CONTEXT_LENGTH
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        torch.from_numpy(
            data[
                i : i + CONTEXT_LENGTH
            ].astype(np.int64)
        )
        for i in starts.tolist()
    ])

    y = torch.stack([
        torch.from_numpy(
            data[
                i + 1 : i + CONTEXT_LENGTH + 1
            ].astype(np.int64)
        )
        for i in starts.tolist()
    ])

    return (
        x.to(DEVICE),
        y.to(DEVICE),
    )


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = GPT(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
)

model = model.to(DEVICE)


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# CHECKPOINT SYSTEM
# --------------------------------------------------

CHECKPOINT_DIR = Path("checkpoints")

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_checkpoint(
    step,
    loss,
    filename=None,
):

    if filename is None:
        filename = f"model_step_{step}.pt"

    checkpoint_path = (
        CHECKPOINT_DIR / filename
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
        checkpoint_path,
    )

    print(
        f"\nCheckpoint saved: "
        f"{checkpoint_path}"
    )

    return checkpoint_path


def find_latest_checkpoint():

    checkpoints = list(
        CHECKPOINT_DIR.glob(
            "model_step_*.pt"
        )
    )

    if not checkpoints:
        return None

    def get_step(path):

        return int(
            path.stem.split("_")[-1]
        )

    latest_checkpoint = max(
        checkpoints,
        key=get_step,
    )

    return latest_checkpoint


def load_latest_checkpoint():

    checkpoint_path = (
        find_latest_checkpoint()
    )

    if checkpoint_path is None:

        print(
            "No checkpoint found. "
            "Starting from scratch."
        )

        return 0

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    saved_step = checkpoint["step"]

    print(
        f"Resuming from step "
        f"{saved_step + 1}"
    )

    return saved_step + 1


# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

START_STEP = load_latest_checkpoint()


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data),
    ]:

        losses = []

        for _ in range(EVAL_STEPS):

            x, y = get_batch(data)

            _, loss = model(
                x,
                y,
            )

            losses.append(
                loss.item()
            )

        results[split] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

print("\nStarting training...\n")

model.train()

current_step = START_STEP - 1
last_loss = None


try:

    for step in range(
        START_STEP,
        MAX_STEPS + 1,
    ):

        # ------------------------------
        # GET BATCH
        # ------------------------------

        x, y = get_batch(
            train_data
        )

        # ------------------------------
        # FORWARD PASS
        # ------------------------------

        _, loss = model(
            x,
            y,
        )

        # ------------------------------
        # BACKPROPAGATION
        # ------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # ------------------------------
        # UPDATE PARAMETERS
        # ------------------------------

        optimizer.step()

        # Bu step tamamen tamamlandı.
        current_step = step
        last_loss = loss.item()

        # ------------------------------
        # EVALUATION
        # ------------------------------

        if step % EVAL_INTERVAL == 0:

            losses = estimate_loss()

            print(
                f"Step {step:4d} | "
                f"train loss: "
                f"{losses['train']:.4f} | "
                f"val loss: "
                f"{losses['val']:.4f}"
            )

        # ------------------------------
        # REGULAR CHECKPOINT
        # ------------------------------

        if (
            step > 0
            and
            step % CHECKPOINT_INTERVAL == 0
        ):

            save_checkpoint(
                step,
                last_loss,
            )


# --------------------------------------------------
# CTRL+C
# --------------------------------------------------

except KeyboardInterrupt:

    print(
        "\n\nTraining interrupted!"
    )

    if (
        last_loss is not None
        and current_step >= 0
    ):

        save_checkpoint(
            current_step,
            last_loss,
        )

        print(
            f"Safe to exit.\n"
            f"Training can resume from "
            f"step {current_step + 1}."
        )

    else:

        print(
            "Training stopped before "
            "a step was completed."
        )


# --------------------------------------------------
# NORMAL COMPLETION
# --------------------------------------------------

else:

    if last_loss is not None:

        save_checkpoint(
            current_step,
            last_loss,
        )

    print(
        "\nTraining complete!"
    )