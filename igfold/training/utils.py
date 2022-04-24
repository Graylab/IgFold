from einops import rearrange, repeat
import torch
import torch.nn.functional as F

from igfold.utils.general import exists


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
