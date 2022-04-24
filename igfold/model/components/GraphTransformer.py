###
#   Inspired by graph transformer implementation from https://github.com/lucidrains/graph-transformer-pytorch
###

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from igfold.utils.general import exists, default

List = nn.ModuleList


class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        *args,
        **kwargs,
    ):
        x = self.norm(x)
        return self.fn(
            x,
            *args,
            **kwargs,
        )


# gated residual


class Residual(nn.Module):
    def forward(
        self,
        x,
        res,
    ):
        return x + res


class GatedResidual(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        edge_dim=None,
    ):
        super().__init__()
        edge_dim = default(
            edge_dim,
            dim,
        )

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(
            dim,
            inner_dim,
        )
        self.to_kv = nn.Linear(
            dim,
            inner_dim * 2,
        )
        self.edges_to_kv = nn.Linear(
            edge_dim,
            inner_dim,
        )

        self.to_out = nn.Linear(
            inner_dim,
            dim,
        )

    def forward(
        self,
        nodes,
        edges,
        mask=None,
    ):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(
            2,
            dim=-1,
        )

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(
            lambda t: rearrange(
                t,
                'b ... (h d) -> (b h) ... d',
                h=h,
            ),
            (q, k, v, e_kv),
        )

        ek, ev = e_kv, e_kv

        k, v = map(
            lambda t: rearrange(
                t,
                'b j d -> b () j d ',
            ),
            (k, v),
        )
        k = k + ek
        v = v + ev

        sim = einsum(
            'b i d, b i j d -> b i j',
            q,
            k,
        ) * self.scale

        if exists(mask):
            mask = rearrange(
                mask,
                'b i -> b i ()',
            ) & rearrange(
                mask,
                'b j -> b () j',
            )
            mask = repeat(
                mask,
                "b ... -> (b h) ...",
                h=self.heads,
            )
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum(
            'b i j, b i j d -> b i d',
            attn,
            v,
        )
        out = rearrange(
            out,
            '(b h) n d -> b n (h d)',
            h=h,
        )
        return self.to_out(out)


def FeedForward(dim, ff_mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim),
    )


class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=8,
        with_feedforwards=False,
        norm_edges=False,
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(
            edge_dim,
            dim,
        )
        self.norm_edges = nn.LayerNorm(
            edge_dim) if norm_edges else nn.Identity()

        for _ in range(depth):
            self.layers.append(
                List([
                    List([
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                edge_dim=edge_dim,
                                dim_head=dim_head,
                                heads=heads,
                            )),
                        GatedResidual(dim)
                    ]),
                    List(
                        [PreNorm(
                            dim,
                            FeedForward(dim),
                        ),
                         GatedResidual(dim)]) if with_feedforwards else None
                ]))

    def forward(
        self,
        nodes,
        edges,
        mask=None,
    ):
        edges = self.norm_edges(edges)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(
                attn(
                    nodes,
                    edges,
                    mask=mask,
                ),
                nodes,
            )

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(
                    ff(nodes),
                    nodes,
                )

        return nodes, edges