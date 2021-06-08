import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


def vicreg_loss(vicreg_params, embeddings, labels, sim_loss, small_val=0.0001):
    # VICReg Regularization.
    # Based on: https://paperswithcode.com/paper/vicreg-variance-invariance-covariance
    lbda = vicreg_params["lbda"]
    mu = vicreg_params["mu"]
    nu = vicreg_params["nu"]

    # invariance loss
    anchor_idx, pos_idx, __, __ = lmu.get_all_pairs_indices(labels)
    z_a = embeddings[anchor_idx]
    z_b = embeddings[pos_idx]
    inv_loss = F.mse_loss(z_a, z_b)

    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + small_val)
    std_z_b = torch.sqrt(z_b.var(dim=0) + small_val)
    std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

    # covariance loss
    N, D = embeddings.size()
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)
    cov_loss = (
        cov_z_a.fill_diagonal_(0).pow_(2).sum() / D + cov_z_b.fill_diagonal_(0).pow_(2).sum() / D
    )

    # final loss
    loss = lbda * sim_loss + lbda * inv_loss + mu * std_loss + nu * cov_loss

    return (
        inv_loss,
        std_loss,
        cov_loss,
        loss,
    )
