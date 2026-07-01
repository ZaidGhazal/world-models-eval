"""Video success classifier: frozen SigLIP per-frame + temporal pooling + MLP head.

Train on labeled simulator rollout videos (produced by sim_eval with --video-every 1):

  python -m dreamgrasp.eval.success_classifier --videos-dir results/videos \
      --labels results/sim_success.parquet --epochs 5 --out checkpoints/classifier

Videos are matched to labels by their `{task}_seed{seed}.mp4` filename.
Accuracy bounds every downstream claim — the >=90% held-out bar applies at Type 2 scale.
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from dreamgrasp.utils.device import get_device
from dreamgrasp.utils.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
ENCODER = "google/siglip-base-patch16-224"
N_FRAMES = 8


class SuccessClassifier(nn.Module):
    def __init__(self, encoder_name: str = ENCODER):
        super().__init__()
        from transformers import SiglipVisionModel

        self.encoder = SiglipVisionModel.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        d = self.encoder.config.hidden_size
        # temporal pooling: concat(mean, max) over frame embeddings
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, 1))

    def embed_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels (T,3,224,224) normalized -> (T,D)"""
        with torch.no_grad():
            return self.encoder(pixel_values=pixels).pooler_output

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        emb = self.embed_frames(pixels)
        pooled = torch.cat([emb.mean(0), emb.max(0).values])
        return self.head(pooled)

    @torch.no_grad()
    def score_frames(self, frames: np.ndarray) -> float:
        """frames (T,H,W,3) uint8 -> success probability."""
        device = next(self.head.parameters()).device
        pixels = preprocess_frames(sample_frames(frames)).to(device)
        return torch.sigmoid(self.forward(pixels)).item()

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), out_dir / "head.pt")

    @classmethod
    def load(cls, ckpt_dir: Path, device: str) -> "SuccessClassifier":
        model = cls()
        model.head.load_state_dict(torch.load(Path(ckpt_dir) / "head.pt", map_location=device))
        return model.to(device).eval()


def sample_frames(frames: np.ndarray, n: int = N_FRAMES) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return frames[idx]


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 -> SigLIP pixel_values (T,3,224,224), mean/std 0.5."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear")
    return (x - 0.5) / 0.5


def build_manifest(videos_dir: Path, labels_path: Path) -> list[tuple[Path, int]]:
    df = pd.read_parquet(labels_path)
    items = []
    for _, r in df.iterrows():
        path = videos_dir / f"{r['task']}_seed{r['seed']}.mp4"
        if path.exists():
            items.append((path, int(r["success"])))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, default=REPO_ROOT / "results" / "videos")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "results" / "sim_success.parquet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "classifier")
    args = parser.parse_args()

    device = get_device()
    seed_everything(args.seed)
    items = build_manifest(args.videos_dir, args.labels)
    if len(items) < 4:
        raise SystemExit(f"only {len(items)} labeled videos found — need at least 4")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    n_val = max(1, int(args.val_fraction * len(items)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    print(f"{len(train_idx)} train / {len(val_idx)} val videos")

    model = SuccessClassifier().to(device)
    optim = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    # Pre-embed all videos once (encoder is frozen) — the actual training is head-only.
    embeddings, labels = [], []
    for path, label in items:
        frames = np.stack(iio.mimread(path, memtest=False))
        pixels = preprocess_frames(sample_frames(frames)).to(device)
        emb = model.embed_frames(pixels)
        embeddings.append(torch.cat([emb.mean(0), emb.max(0).values]))
        labels.append(label)
    embs = torch.stack(embeddings)
    ys = torch.tensor(labels, dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(train_idx.copy())).to(device)
        total = 0.0
        for i in perm.split(8):
            logits = model.head(embs[i]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, ys[i])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += loss.item() * len(i)
        model.eval()
        with torch.no_grad():
            val_logits = model.head(embs[val_idx]).squeeze(-1)
            val_acc = ((val_logits > 0).float() == ys[val_idx]).float().mean().item()
        print(f"epoch {epoch}: train_loss {total / len(train_idx):.4f} val_acc {val_acc:.3f}", flush=True)

    model.save(args.out)
    print(f"saved head -> {args.out} (val_acc {val_acc:.3f} on {len(val_idx)} videos)")


if __name__ == "__main__":
    main()
