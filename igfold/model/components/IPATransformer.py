###
#   Inspired by IPA implementation from https://github.com/lucidrains/invariant-point-attention
###

import torch
import torch.nn.functional as F
from torch import nn, einsum
from invariant_point_attention.invariant_point_attention import IPABlock, exists
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix
from einops import rearrange, repeat

from igfold.utils.coordinates import get_ideal_coords, place_o_coords


class IPAEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        **kwargs,
    ):
        super().__init__()

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(IPABlock(
                dim=dim,
                **kwargs,
            ))

    def forward(
        self,
        x,
        *,
        translations=None,
        rotations=None,
        pairwise_repr=None,
        mask=None,
    ):
        for block in self.layers:
            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
                mask=mask,
            )

        return x


class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        stop_rotation_grad=False,
        **kwargs,
    ):
        super().__init__()

        self.stop_rotation_grad = stop_rotation_grad

        # using quaternion functions from pytorch3d
        self.quaternion_to_matrix = quaternion_to_matrix
        self.quaternion_multiply = quaternion_multiply

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ipa_block = IPABlock(
                dim=dim,
                **kwargs,
            )
            linear = nn.Linear(dim, 6)
            torch.nn.init.zeros_(linear.weight.data)
            torch.nn.init.zeros_(linear.bias.data)
            self.layers.append(nn.ModuleList([ipa_block, linear]))

    def forward(
        self,
        single_repr,
        *,
        translations=None,
        quaternions=None,
        pairwise_repr=None,
        mask=None,
    ):
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        # if no initial quaternions passed in, start from identity

        if not exists(quaternions):
            quaternions = torch.tensor(
                [1., 0., 0., 0.],
                device=device,
            )  # initial rotations
            quaternions = repeat(
                quaternions,
                'd -> b n d',
                b=b,
                n=n,
            )

        # if not translations passed in, start from identity

        if not exists(translations):
            translations = torch.zeros(
                (b, n, 3),
                device=device,
            )

        # go through the layers and apply invariant point attention and feedforward

        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)
            if self.stop_rotation_grad:
                rotations = rotations.detach()

            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
                mask=mask,
            )

            # update quaternion and translation

            quaternion_update, translation_update = to_update(x).chunk(
                2,
                dim=-1,
            )
            quaternion_update = F.pad(
                quaternion_update,
                (1, 0),
                value=1.,
            )

            quaternions = quaternion_multiply(
                quaternions,
                quaternion_update,
            )
            translations = translations + einsum(
                'b n c, b n c r -> b n r',
                translation_update,
                rotations,
            )

        ideal_coords = get_ideal_coords().to(device)
        ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=b,
            l=n,
        )

        rotations = quaternion_to_matrix(quaternions)
        points_global = einsum(
            'b n a c, b n c d -> b n a d',
            ideal_coords,
            rotations,
        ) + rearrange(
            translations,
            "b l d -> b l () d",
        )

        points_global = place_o_coords(points_global)

        return points_global, translations, quaternions
