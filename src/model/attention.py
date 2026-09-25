import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.dropout = dropout

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(B, T, C)

        output = self.out_proj(output)

        return output
    

if __name__ == "__main__":
    torch.manual_seed(42)

    B = 2
    T = 8
    C = 64
    HEADS = 4

    x = torch.randn(B, T, C)

    attention = CausalSelfAttention(
        embed_dim=C,
        num_heads=HEADS,
    )

    output = attention(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == x.shape

    print("Attention test passed!")