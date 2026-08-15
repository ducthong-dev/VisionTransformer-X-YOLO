"""Auto-encoder objective.

Implements manuscript Eqs. (5)-(6) with the notation corrected:

    Delta_sparse(W, b) = Delta(W, b) + beta * sum_j KL(rho || rho_hat_j)
    Delta(W, b)        = (1/m) sum_i ||x_hat_i - x_i||^2  +  (lambda/2) * ||W||^2

The manuscript writes Eq. (5) with the same symbol on both sides of the equals
sign (Reviewer #12), and S3.3 states that "x is normal RGB input, while x_hat is
an input image with noise adding", which inverts the denoising convention. Here
the *input* carries the corruption and the *target* is the clean image, which is
what makes the objective a denoising one at all.

The L2 term is applied through the optimizer's weight decay rather than added
to the loss explicitly; both are reported in the run config.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def kl_sparsity(latent: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    """KL(rho || rho_hat) summed over latent channels.

    `latent` is [B, C, H, W] with sigmoid activation. rho_hat_j is the mean
    activation of channel j over the batch and spatial positions, i.e. the
    Bernoulli mean the sparsity target is compared against.
    """
    rho_hat = latent.mean(dim=(0, 2, 3)).clamp(EPS, 1.0 - EPS)
    rho_t = torch.full_like(rho_hat, float(rho))
    kl = rho_t * torch.log(rho_t / rho_hat) + (1 - rho_t) * torch.log((1 - rho_t) / (1 - rho_hat))
    return kl.sum()


def ae_loss(
    recon: torch.Tensor,
    target_clean: torch.Tensor,
    latent: torch.Tensor,
    beta: float = 1e-3,
    rho: float = 0.05,
    sparse: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Reconstruction + optional KL sparsity. Returns (loss, components)."""
    recon_loss = F.mse_loss(recon, target_clean)
    if sparse and beta > 0:
        kl = kl_sparsity(latent, rho)
        total = recon_loss + beta * kl
        return total, {
            "ae_recon": float(recon_loss.detach()),
            "ae_kl": float(kl.detach()),
            "ae_total": float(total.detach()),
        }
    return recon_loss, {
        "ae_recon": float(recon_loss.detach()),
        "ae_kl": 0.0,
        "ae_total": float(recon_loss.detach()),
    }
