import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerBlock


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim,
        num_layers,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        # Token ID -> vector
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        # Position -> vector
        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim,
        )

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Hidden vector -> vocabulary scores
        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False,
        )


    def forward(self, input_ids, targets=None):

        B, T = input_ids.shape

        if T > self.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.context_length}"
            )

        positions = torch.arange(
            0,
            T,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss


if __name__ == "__main__":

    torch.manual_seed(42)

    VOCAB_SIZE = 16_000
    CONTEXT_LENGTH = 256
    EMBED_DIM = 256
    NUM_LAYERS = 6
    NUM_HEADS = 8

    model = GPT(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    )

    B = 2
    T = 32

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (B, T),
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (B, T),
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Input shape: ", input_ids.shape)
    print("Logits shape:", logits.shape)
    print("Loss:        ", loss.item())

    print(
        f"Parameters:   {parameter_count:,}"
    )

    assert logits.shape == (
        B,
        T,
        VOCAB_SIZE,
    )

    print("GPT test passed!")