"""Acceptance checks for real Type 2 result artifacts."""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")


def check_sim(path: Path, min_best: float, min_spread: float) -> None:
    df = pd.read_parquet(path)
    require_columns(df, ["checkpoint", "task", "seed", "success", "steps"], path)
    by_ckpt = df.groupby("checkpoint")["success"].mean().sort_values()
    spread = float(by_ckpt.iloc[-1] - by_ckpt.iloc[0])
    best = float(by_ckpt.iloc[-1])
    print("sim success by checkpoint:")
    print(by_ckpt.to_string())
    print(f"best={best:.3f} spread={spread:.3f} rows={len(df)}")
    if best < min_best:
        raise SystemExit(f"T2.2 FAILED: best success {best:.3f} < {min_best:.3f}")
    if spread < min_spread:
        raise SystemExit(f"T2.2 FAILED: spread {spread:.3f} < {min_spread:.3f}")


def check_wm(path: Path) -> None:
    df = pd.read_parquet(path)
    require_columns(df, ["checkpoint", "split", "horizon", "psnr", "ssim", "lpips", "divergence_step"], path)
    required_horizons = {1, 8, 16, 32}
    missing = {}
    for ckpt, group in df.groupby("checkpoint"):
        horizons = set(group["horizon"].astype(int))
        absent = required_horizons - horizons
        if absent:
            missing[ckpt] = sorted(absent)
    if missing:
        raise SystemExit(f"T2.3 FAILED: missing fidelity horizons {missing}")
    by_tier = df.groupby("checkpoint")["divergence_step"].mean().sort_index()
    print("mean divergence step by checkpoint:")
    print(by_tier.to_string())
    if "tier_2" in " ".join(by_tier.index) and "tier_4" in " ".join(by_tier.index):
        wm2 = by_tier[[idx for idx in by_tier.index if "tier_2" in idx]].mean()
        wm4 = by_tier[[idx for idx in by_tier.index if "tier_4" in idx]].mean()
        if wm2 > wm4:
            raise SystemExit(f"T2.3 FAILED: WM-2 divergence {wm2:.3f} > WM-4 {wm4:.3f}")
    print(f"wm rows={len(df)}")


def check_dream(path: Path) -> None:
    df = pd.read_parquet(path)
    require_columns(df, ["checkpoint", "task", "wm_tier", "seed", "dream_success_prob"], path)
    print(f"dream rows={len(df)}")
    print(df.groupby(["wm_tier", "checkpoint"])["dream_success_prob"].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    sim = sub.add_parser("sim")
    sim.add_argument("--path", type=Path, default=REPO_ROOT / "results" / "sim_success.parquet")
    sim.add_argument("--min-best", type=float, default=0.20)
    sim.add_argument("--min-spread", type=float, default=0.25)
    wm = sub.add_parser("wm")
    wm.add_argument("--path", type=Path, default=REPO_ROOT / "results" / "wm_fidelity.parquet")
    dream = sub.add_parser("dream")
    dream.add_argument("--path", type=Path, default=REPO_ROOT / "results" / "dream_success.parquet")
    args = parser.parse_args()

    if args.phase == "sim":
        check_sim(args.path, args.min_best, args.min_spread)
    elif args.phase == "wm":
        check_wm(args.path)
    elif args.phase == "dream":
        check_dream(args.path)


if __name__ == "__main__":
    main()
