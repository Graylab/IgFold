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

    n_vec = torch.cross((b_coord - a_coord).expand(bc_vec.shape), bc_vec, dim=-1)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, torch.cross(n_vec, bc_vec, dim=-1), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral),
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


# psi of an extended beta strand, the typical conformation at antibody chain termini
TERMINAL_O_DIHEDRAL = -0.873  # N-CA-C-O dihedral (radians) = psi(130 deg) + 180 deg


def place_o_coords(coords, seq_lens=None):
    """
    Add backbone O atoms to (b, L, 4, 3) N/CA/C/CB coordinates, giving (b, L, 5, 3).

    Interior O atoms are placed trans to the next residue's N. The last residue of each chain
    (chain ends given by ``seq_lens``; the final residue otherwise) has no next N, so its O is
    placed from its own N-CA-C frame with a beta-strand psi.
    """
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
    )
    terminal_o_coords = place_fourth_atom(
        N,
        A,
        C,
        torch.tensor(1.231),
        torch.tensor(2.108),
        torch.tensor(TERMINAL_O_DIHEDRAL),
    )

    if seq_lens is None:
        terminal_idx = [coords.shape[1] - 1]
    else:
        terminal_idx = [int(i) - 1 for i in torch.cumsum(torch.as_tensor(seq_lens), 0)]
    o_coords[:, terminal_idx] = terminal_o_coords[:, terminal_idx]

    coords = torch.cat(
        [coords, o_coords.unsqueeze(2)],
        dim=2,
    )

    return coords
