import torch


def place_fourth_atom(
    a_coord: torch.Tensor,
    b_coord: torch.Tensor,
    c_coord: torch.Tensor,
    length: torch.Tensor,
    planar: torch.Tensor,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])

    return d_coord


def get_ideal_coords(center=False):
    N = torch.tensor([[0, 0, -1.458]], dtype=float)
    A = torch.tensor([[0, 0, 0]], dtype=float)
    B = torch.tensor([[0, 1.426, 0.531]], dtype=float)
    C = place_fourth_atom(
        B,
        A,
        N,
        torch.tensor(2.460),
        torch.tensor(0.615),
        torch.tensor(-2.143),
    )

    coords = torch.cat([N, A, C, B]).float()

    if center:
        coords -= coords.mean(
            dim=0,
            keepdim=True,
        )

    return coords


def place_o_coords(coords):
    N = coords[:, :, 0]
    A = coords[:, :, 1]
    C = coords[:, :, 2]

    o_coords = place_fourth_atom(
        torch.roll(N, shifts=-1, dims=1),
        A,
        C,
        torch.tensor(1.231),
        torch.tensor(2.108),
        torch.tensor(-3.142),
    ).unsqueeze(2)

    coords = torch.cat(
        [coords, o_coords],
        dim=2,
    )

    return coords