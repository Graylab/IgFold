import torch
from einops import rearrange

from igfold.utils.constants import EPS


def normed_vec(vec, eps=EPS):
    mag_sq = torch.sum(vec**2, dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + eps)
    vec = vec / mag

    return vec


def normed_cross(vec1, vec2, eps=EPS):
    vec1 = normed_vec(vec1, eps=eps)
    vec2 = normed_vec(vec2, eps=eps)
    cross = torch.cross(vec1, vec2, dim=-1)

    return cross


def dist(x_1, x_2, eps=EPS):
    d_sq = (x_1 - x_2)**2
    d = torch.sqrt(d_sq.sum(-1) + eps)

    return d


def dist_mat(c1, c2, dim=-3, eps=EPS):
    c1 = c1.unsqueeze(dim)
    c2 = c2.unsqueeze(dim - 1)
    d = dist(c1, c2, eps=eps)

    return d


def angle(x_1, x_2, x_3, eps=EPS):
    a = normed_vec(x_1 - x_2, eps=eps)
    b = normed_vec(x_3 - x_2, eps=eps)
    ang = torch.arccos((a * b).sum(-1))

    return ang


def dihedral(x_1, x_2, x_3, x_4, eps=EPS):
    b1 = normed_vec(x_1 - x_2, eps=eps)
    b2 = normed_vec(x_2 - x_3, eps=eps)
    b3 = normed_vec(x_3 - x_4, eps=eps)
    n1 = normed_cross(b1, b2, eps=eps)
    n2 = normed_cross(b2, b3, eps=eps)
    m1 = normed_cross(n1, b2, eps=eps)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)

    dih = torch.atan2(y, x)

    return dih


def coords_to_frame(coords, eps=EPS):
    if len(coords.shape) == 3:
        coords = rearrange(
            coords,
            "b (l a) d -> b l a d",
            l=coords.shape[-2] // 4,
        )

    N, CA, C, _ = coords.unbind(-2)
    CA_N = normed_vec(N - CA, eps=eps)
    CA_C = normed_vec(C - CA, eps=eps)
    n1 = CA_N
    n2 = normed_cross(n1, CA_C, eps=eps)
    n3 = normed_cross(n1, n2, eps=eps)
    rot = torch.stack([n1, n2, n3], -1)

    return CA, rot