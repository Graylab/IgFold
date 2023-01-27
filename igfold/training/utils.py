from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F

from igfold.utils.constants import *
from igfold.utils.general import exists
from igfold.utils.geometry import dist, angle, dihedral


def kabsch(
    mobile,
    stationary,
    return_translation_rotation=False,
):
    X = rearrange(
        mobile,
        "... l d -> ... d l",
    )
    Y = rearrange(
        stationary,
        "... l d -> ... d l",
    )

    #  center X and Y to the origin
    XT, YT = X.mean(dim=-1, keepdim=True), Y.mean(dim=-1, keepdim=True)
    X_ = X - XT
    Y_ = Y - YT

    # calculate convariance matrix
    C = torch.einsum("... x l, ... y l -> ... x y", X_, Y_)

    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = rearrange(W, "... a b -> ... b a")
    else:
        V, S, W = torch.linalg.svd(C)

    # determinant sign for direction correction
    v_det = torch.det(V.to("cpu")).to(X.device)
    w_det = torch.det(W.to("cpu")).to(X.device)
    d = (v_det * w_det) < 0.0
    if d.any():
        S[d] = S[d] * (-1)
        V[d, :] = V[d, :] * (-1)

    # Create Rotation matrix U
    U = torch.matmul(V, W)  #.to(device)

    U = rearrange(
        U,
        "... d x -> ... x d",
    )
    XT = rearrange(
        XT,
        "... d x -> ... x d",
    )
    YT = rearrange(
        YT,
        "... d x -> ... x d",
    )

    if return_translation_rotation:
        return XT, U, YT

    transform = lambda coords: torch.einsum(
        "... l d, ... x d -> ... l x",
        coords - XT,
        U,
    ) + YT
    mobile = transform(mobile)

    return mobile, transform


def do_kabsch(
    mobile,
    stationary,
    align_mask=None,
):
    mobile_, stationary_ = mobile.clone(), stationary.clone()
    if exists(align_mask):
        mobile_[~align_mask] = mobile_[align_mask].mean(dim=-2)
        stationary_[~align_mask] = stationary_[align_mask].mean(dim=-2)
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )
    else:
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )

    return kabsch_xform(mobile)


def kabsch_mse(
    pred,
    target,
    align_mask=None,
    mask=None,
    clamp=0.,
    sqrt=False,
):
    aligned_target = do_kabsch(
        mobile=target,
        stationary=pred.detach(),
        align_mask=align_mask,
    )
    mse = F.mse_loss(
        pred,
        aligned_target,
        reduction='none',
    ).mean(-1)

    if clamp > 0:
        mse = torch.clamp(mse, max=clamp**2)

    if exists(mask):
        mse = torch.sum(
            mse * mask,
            dim=-1,
        ) / torch.sum(
            mask,
            dim=-1,
        )
    else:
        mse = mse.mean(-1)

    if sqrt:
        mse = mse.sqrt()

    return mse


def bond_length_l1(
    pred,
    target,
    mask,
    offsets=[1, 2],
):
    losses = []
    for c in range(pred.shape[0]):
        m, p, t = mask[c], pred[c], target[c]
        for o in offsets:
            m_ = (torch.stack([m[:-o], m[o:]])).all(0)
            pred_lens = torch.norm(p[:-o] - p[o:], dim=-1)
            target_lens = torch.norm(t[:-o] - t[o:], dim=-1)

            losses.append(
                torch.abs(pred_lens[m_] - target_lens[m_], ).mean() / o)

    return torch.stack(losses)


def bb_prmsd_l1(
    pdev,
    pred,
    target,
    align_mask=None,
    mask=None,
):
    aligned_target = do_kabsch(
        mobile=target,
        stationary=pred,
        align_mask=align_mask,
    )
    bb_dev = (pred - aligned_target).norm(dim=-1)
    loss = F.l1_loss(
        pdev,
        bb_dev,
        reduction='none',
    )

    if exists(mask):
        mask = repeat(mask, "b l -> b (l 4)")
        loss = torch.sum(
            loss * mask,
            dim=-1,
        ) / torch.sum(
            mask,
            dim=-1,
        )
    else:
        loss = loss.mean(-1)

    loss = loss.mean(-1).unsqueeze(0)

    return loss

def bond_len_loss(pred, seq_lens, mask, eps=EPS):
    b, l, a, d = pred.shape

    pred_bb = pred[:, :, :3]
    mask = repeat(mask, "b l -> b (l 3)")
    for seq_len in seq_lens:
        mask[:, 3 * seq_len - 1] = 0
    mask_bb = mask[:, :-1] * mask[:, 1:]

    pred_bond_lens = dist(
        rearrange(pred_bb, "b l a d -> b (l a) d")[:, :-1],
        rearrange(pred_bb, "b l a d -> b (l a) d")[:, 1:],
    )
    lit_bond_lens = repeat(
        torch.tensor([BL_N_CA, BL_CA_C, BL_C_N]),
        "bl -> b (l bl)",
        b=b,
        l=l,
    )[:, :-1]
    lit_bond_lens = lit_bond_lens.to(pred_bond_lens.device)

    bl_loss = torch.abs(pred_bond_lens - lit_bond_lens) * mask_bb
    bl_loss = bl_loss.sum(-1) / (mask.sum(-1) + eps)

    return bl_loss


def bond_angle_loss(pred, seq_lens, mask, eps=EPS):
    b, l, a, d = pred.shape

    for seq_len in seq_lens:
        mask[:, seq_len - 1] = 0
    mask_ = mask[:, 1:] * mask[:, :-1]

    N, CA, C, CB = pred.unbind(-2)
    ba_CA_C_N = angle(CA[:, :-1], C[:, :-1], N[:, 1:], eps=eps)
    ba_CA_C_N_loss = 1 - torch.cos(ba_CA_C_N - BA_CA_C_N * np.pi / 180)
    ba_CA_C_N_loss = ba_CA_C_N_loss * mask_

    ba_C_N_CA = angle(C[:, :-1], N[:, 1:], CA[:, 1:], eps=eps)
    ba_C_N_CA_loss = 1 - torch.cos(ba_C_N_CA - BA_C_N_CA * np.pi / 180)
    ba_C_N_CA_loss = ba_C_N_CA_loss * mask_

    loss = ba_CA_C_N_loss + ba_C_N_CA_loss
    loss = loss.sum(-1) / (mask_.sum(-1) + eps)

    return loss


def vdw_clash_loss(pred, mask, tol=1.5, eps=EPS):
    b, l, a, d = pred.shape

    mask_ = repeat(mask, "b l -> b (l a)", a=a)
    mask_ = (mask_.unsqueeze(-1) * mask_.unsqueeze(-2))

    vdw_radii = torch.tensor([VDW_N, VDW_C, VDW_C, VDW_C])
    vdw_radii = repeat(vdw_radii, "a -> b (l a)", b=b, l=l)
    vdw_distances = (vdw_radii.unsqueeze(-2) + vdw_radii.unsqueeze(-3))
    vdw_distances = vdw_distances.to(pred.device)

    pred_ = rearrange(pred, "b l a d -> b (l a) d")
    atomic_distances = (pred_.unsqueeze(-2) - pred_.unsqueeze(-3)).norm(dim=-1)

    loss = (vdw_distances - tol - atomic_distances).clamp(min=0)
    loss = loss.sum(dim=(-1, -2)) / (mask_.sum(dim=(-1, -2)) + eps)

    return loss


def cis_peptide_loss(pred, seq_lens, mask, eps=EPS):
    for seq_len in seq_lens:
        mask[:, seq_len - 1] = 0
    mask_ = mask[:, 1:] * mask[:, :-1]

    N, CA, C, _ = pred.unbind(-2)
    dih = dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:], eps=0)

    loss = 1 - torch.cos(dih - np.pi)
    loss = loss.sum(dim=(-1, -2)) / (mask_.sum(dim=(-1, -2)) + eps)

    return loss


def violation_loss(pred, seq_lens, mask, eps=EPS):
    b, l, a, d = pred.shape

    bl_loss = bond_len_loss(pred, seq_lens, mask, eps=eps)
    ba_loss = bond_angle_loss(pred, seq_lens, mask, eps=eps)
    vdw_loss = vdw_clash_loss(pred, mask)
    cis_loss = cis_peptide_loss(pred, seq_lens, mask, eps=eps)

    loss = bl_loss + ba_loss + vdw_loss + cis_loss

    return loss