"""Latent dynamics: causal transformer over interleaved (frame-latent tokens, action token) steps.

Sequence layout per timestep: [16x16 latent tokens..., 1 action token] repeated for the
context window. Prediction target: next frame's latent tokens (latent MSE). A small state
head predicts proprio from the frame tokens (used by the dream loop).
"""

import torch
import torch.nn.functional as F
from torch import nn


class DynamicsTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        latent_hw: int = 16,
        action_dim: int = 7,
        state_dim: int = 8,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        context: int = 4,
    ):
        super().__init__()
        self.latent_hw = latent_hw
        self.tokens_per_frame = latent_hw * latent_hw
        self.step_len = self.tokens_per_frame + 1  # + action token
        self.context = context
        self.latent_dim = latent_dim

        self.latent_in = nn.Linear(latent_dim, d_model)
        self.action_in = nn.Linear(action_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, context * self.step_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.latent_out = nn.Linear(d_model, latent_dim)
        self.state_head = nn.Linear(d_model, state_dim)

    def _interleave(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """latents (B,T,C,H,W), actions (B,T,A) -> tokens (B, T*step_len, D)."""
        b, t, c, h, w = latents.shape
        lat = self.latent_in(latents.flatten(3).transpose(2, 3))  # (B,T,HW,D)
        act = self.action_in(actions).unsqueeze(2)  # (B,T,1,D)
        return torch.cat([lat, act], dim=2).flatten(1, 2)

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-step next-frame latents.

        latents (B,T,C,H,W), actions (B,T,A) with T <= context.
        Returns (pred_next_latents (B,T,C,H,W), pred_next_state (B,T,state_dim)):
        prediction at step t attends to steps <= t (causal at frame granularity).
        """
        b, t = latents.shape[:2]
        x = self._interleave(latents, actions) + self.pos_emb[:, : t * self.step_len]
        # Block-causal mask: token in frame i may attend to all tokens in frames <= i.
        frame_id = torch.arange(t * self.step_len, device=x.device) // self.step_len
        mask = frame_id.unsqueeze(0) > frame_id.unsqueeze(1)  # True = masked
        h = self.transformer(x, mask=mask)
        h = h.view(b, t, self.step_len, -1)
        frame_repr = h[:, :, : self.tokens_per_frame]  # token outputs at latent positions
        pred = self.latent_out(frame_repr).transpose(2, 3)
        pred = pred.view(b, t, self.latent_dim, self.latent_hw, self.latent_hw)
        state = self.state_head(frame_repr.mean(dim=2))
        return pred, state

    def loss(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Teacher-forced next-latent MSE over a (context+1)-frame clip."""
        inp, target = latents[:, :-1], latents[:, 1:]
        pred, pred_state = self.forward(inp, actions[:, :-1])
        lat_loss = F.mse_loss(pred, target)
        parts = {"latent_mse": lat_loss.item()}
        loss = lat_loss
        if states is not None:
            state_loss = F.mse_loss(pred_state, states[:, 1:])
            loss = loss + 0.1 * state_loss
            parts["state_mse"] = state_loss.item()
        return loss, parts

    @torch.no_grad()
    def rollout(
        self, latents: torch.Tensor, actions: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive dream: seed with (B,T0,C,H,W) latents, apply actions (B,T0-1+horizon,A).

        Returns (dreamed latents (B,horizon,C,H,W), dreamed states (B,horizon,state_dim)).
        """
        seq = latents
        out_lat, out_state = [], []
        for k in range(horizon):
            t0 = seq.shape[1]
            window = seq[:, -self.context :]
            acts = actions[:, t0 - window.shape[1] : t0]
            pred, state = self.forward(window, acts)
            nxt = pred[:, -1:]
            out_lat.append(nxt)
            out_state.append(state[:, -1:])
            seq = torch.cat([seq, nxt], dim=1)
        return torch.cat(out_lat, dim=1), torch.cat(out_state, dim=1)
