###
#   Inspired by triangle multiplicative update implementation from https://github.com/lucidrains/triangle-multiplicative-module
###

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from igfold.utils.general import exists, default


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim=None,
        mix='ingoing',
    ):
        super().__init__()
        assert mix in {'ingoing',
                       'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(
            hidden_dim,
            dim,
        )
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(
            dim,
            hidden_dim,
        )
        self.right_proj = nn.Linear(
            dim,
            hidden_dim,
        )

        self.left_gate = nn.Linear(
            dim,
            hidden_dim,
        )
        self.right_gate = nn.Linear(
            dim,
            hidden_dim,
        )
        self.out_gate = nn.Linear(dim, dim)

        # initialize all gating to be identity

        for gate in (
                self.left_gate,
                self.right_gate,
                self.out_gate,
        ):
            nn.init.constant_(
                gate.weight,
                0.,
            )
            nn.init.constant_(
                gate.bias,
                1.,
            )

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(
            hidden_dim,
            dim,
        )

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(
                mask,
                'b i j -> b i j ()',
            )

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(
            self.mix_einsum_eq,
            left,
            right,
        )

        out = self.to_out_norm(out)
        out = self.to_out(out)
        out = out * out_gate
        return out