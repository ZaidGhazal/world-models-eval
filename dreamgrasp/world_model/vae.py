"""Frame autoencoder: small conv VAE, 128x128x3 -> (latent_dim, 16, 16).

Decision (documented per guide): we train our own small VAE rather than using a frozen
SD-VAE from diffusers. Rationale: SD-VAE is ~84M params tuned for natural images at
256px+; LIBERO frames are 128px renders with a narrow visual distribution, a ~5M-param
VAE reaches good reconstructions there, keeps the core package free of large downloads,
and trains in minutes at tiny scale on MPS. The dynamics model is the studied variable —
the VAE is shared across all tiers, so its absolute quality cancels out of tier comparisons.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


def _up_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class FrameVAE(nn.Module):
    """Downsampling factor 8: 128 -> 16 (or 64 -> 8 in tiny mode — fully convolutional)."""

    def __init__(self, latent_dim: int = 8, base: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            _block(base, base),
            _block(base, base * 2),
            _block(base * 2, base * 4),
        )
        self.to_moments = nn.Conv2d(base * 4, latent_dim * 2, 1)
        self.from_latent = nn.Conv2d(latent_dim, base * 4, 1)
        self.decoder = nn.Sequential(
            _up_block(base * 4, base * 2),
            _up_block(base * 2, base),
            _up_block(base, base),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,3,H,W) in [0,1] -> (mu, logvar), each (B, latent_dim, H/8, W/8)."""
        mu, logvar = self.to_moments(self.encoder(x * 2 - 1)).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z -> frames in [0,1]."""
        return (self.decoder(self.from_latent(z)).tanh() + 1) / 2

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x: torch.Tensor, kl_weight: float = 1e-6) -> tuple[torch.Tensor, dict]:
        mu, logvar = self.encode(x)
        recon = self.decode(self.sample(mu, logvar))
        rec_loss = F.mse_loss(recon, x)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        loss = rec_loss + kl_weight * kl
        return loss, {"rec": rec_loss.item(), "kl": kl.item(), "recon": recon}
