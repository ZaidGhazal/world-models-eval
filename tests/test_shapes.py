"""Shape/round-trip checks for the world model — CPU-only, no dataset needed."""

import torch

from dreamgrasp.world_model.dynamics import DynamicsTransformer
from dreamgrasp.world_model.vae import FrameVAE


def test_vae_shapes_128():
    vae = FrameVAE(latent_dim=8, base=16)
    x = torch.rand(2, 3, 128, 128)
    mu, logvar = vae.encode(x)
    assert mu.shape == (2, 8, 16, 16)
    recon = vae.decode(mu)
    assert recon.shape == x.shape
    assert recon.min() >= 0 and recon.max() <= 1


def test_vae_fully_convolutional_64():
    vae = FrameVAE(latent_dim=8, base=16)
    mu, _ = vae.encode(torch.rand(1, 3, 64, 64))
    assert mu.shape == (1, 8, 8, 8)


def test_dynamics_forward_and_loss():
    dyn = DynamicsTransformer(latent_dim=4, latent_hw=4, d_model=32, n_layers=1, n_heads=2, context=3)
    latents = torch.randn(2, 4, 4, 4, 4)  # B,T=context+1,C,H,W
    actions = torch.randn(2, 4, 7)
    states = torch.randn(2, 4, 8)
    pred, state = dyn.forward(latents[:, :-1], actions[:, :-1])
    assert pred.shape == (2, 3, 4, 4, 4)
    assert state.shape == (2, 3, 8)
    loss, parts = dyn.loss(latents, actions, states)
    assert loss.requires_grad and "latent_mse" in parts


def test_dynamics_rollout_horizon():
    dyn = DynamicsTransformer(latent_dim=4, latent_hw=4, d_model=32, n_layers=1, n_heads=2, context=2)
    seed = torch.randn(1, 2, 4, 4, 4)
    actions = torch.randn(1, 2 + 5 - 1, 7)
    dreamed, states = dyn.rollout(seed, actions, horizon=5)
    assert dreamed.shape == (1, 5, 4, 4, 4)
    assert states.shape == (1, 5, 8)


def test_causal_mask_blocks_future():
    """Changing a future action must not change the prediction at an earlier step."""
    torch.manual_seed(0)
    dyn = DynamicsTransformer(latent_dim=2, latent_hw=2, d_model=16, n_layers=1, n_heads=2, context=3)
    dyn.eval()
    latents = torch.randn(1, 3, 2, 2, 2)
    actions = torch.randn(1, 3, 7)
    with torch.no_grad():
        pred_a, _ = dyn.forward(latents, actions)
        actions2 = actions.clone()
        actions2[:, 2] += 10.0  # perturb only the LAST step's action
        pred_b, _ = dyn.forward(latents, actions2)
    assert torch.allclose(pred_a[:, 0], pred_b[:, 0], atol=1e-5)
    assert torch.allclose(pred_a[:, 1], pred_b[:, 1], atol=1e-5)
