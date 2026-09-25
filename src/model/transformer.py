import torch
import torch.nn as nn

from attention import CausalSelfAttention


class MLP(nn.Module):

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(
            embed_dim,
            4 * embed_dim,
        )

        self.gelu = nn.GELU()

        self.fc2 = nn.Linear(
            4 * embed_dim,
            embed_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


if __name__ == "__main__":

    torch.manual_seed(42)

    B = 2
    T = 8
    C = 64
    HEADS = 4

    x = torch.randn(B, T, C)

    block = TransformerBlock(
        embed_dim=C,
        num_heads=HEADS,
    )

    output = block(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == x.shape

    print("Transformer block test passed!")