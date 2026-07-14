"""Calibration analysis: does dream success predict sim success, and how does that
degrade with world-model quality?

  python -m dreamgrasp.eval.correlate                    # real parquets -> trust-region chart
  python -m dreamgrasp.eval.correlate --synthetic 0.9    # self-test: recover a known correlation

Core quantities:
- Spearman rho between dream and sim success rates across policy checkpoints, per WM tier,
  pooled and per task, with bootstrap 95% CIs.
- Task-level Pearson for the best checkpoint.
- Trust-region chart: x = WM fidelity, y = ranking reliability (rho +/- CI).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]


def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Spearman rho with bootstrap 95% CI over paired resamples."""
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boot.append(stats.spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return float(rho), float(lo), float(hi)


def normalize_task(task: str) -> str:
    """Canonical task id: sim_eval.py writes LIBERO's underscored slug (task.name);
    dream_eval.py writes the LeRobotDataset's natural-language sentence for the same task.
    Both collapse to the same string once whitespace becomes underscores."""
    return task.strip().lower().replace(" ", "_")


def success_rates(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate per (checkpoint, task): mean of the value column."""
    df = df.assign(task=df["task"].map(normalize_task))
    return df.groupby(["checkpoint", "task"])[value_col].mean().reset_index()


def ranking_reliability(
    sim: pd.DataFrame, dream: pd.DataFrame, pooled: bool = True, threshold: float | None = None
) -> dict[str, tuple[float, float, float]]:
    """Per WM tier: Spearman(dream success, sim success) across checkpoints.

    threshold: if set, binarize dream_success_prob at this cutoff (dream "success rate")
    instead of averaging the continuous probability directly — the classifier's own
    train-time decision boundary is 0.5 (logit > 0); RUNBOOK's robustness check asks for
    +/-0.1 around that.
    """
    sim_rates = success_rates(sim, "success")
    out = {}
    for tier, tier_df in dream.groupby("wm_tier"):
        if threshold is not None:
            is_success = (tier_df["dream_success_prob"] > threshold).astype(float)
            tier_df = tier_df.assign(dream_success_prob=is_success)
        dream_rates = success_rates(tier_df, "dream_success_prob")
        merged = dream_rates.merge(sim_rates, on=["checkpoint", "task"], suffixes=("_dream", ""))
        if pooled:
            by_ckpt = merged.groupby("checkpoint")[["dream_success_prob", "success"]].mean()
            out[tier] = spearman_with_ci(
                by_ckpt["dream_success_prob"].to_numpy(), by_ckpt["success"].to_numpy()
            )
        else:
            out[tier] = spearman_with_ci(
                merged["dream_success_prob"].to_numpy(), merged["success"].to_numpy()
            )
    return out


def dream_for_split(dream: pd.DataFrame, sim: pd.DataFrame, suite: str, split: str) -> pd.DataFrame:
    """Subset dream rows whose task belongs to `split` (e.g. "train" vs "heldout") within
    `suite`, matching sim's task naming via normalize_task since dream/sim use different
    conventions for the same task (see 2026-07-13 RUN_LOG finding)."""
    tasks = {normalize_task(t) for t in sim[(sim["suite"] == suite) & (sim["split"] == split)]["task"]}
    return dream[dream["task"].map(normalize_task).isin(tasks)]


def task_level_pearson(sim: pd.DataFrame, dream: pd.DataFrame, checkpoint: str) -> float:
    sim_rates = success_rates(sim[sim.checkpoint == checkpoint], "success")
    dream_rates = success_rates(dream[dream.checkpoint == checkpoint], "dream_success_prob")
    merged = dream_rates.merge(sim_rates, on=["checkpoint", "task"])
    return float(stats.pearsonr(merged["dream_success_prob"], merged["success"]).statistic)


def trust_region_chart(
    reliability: dict[str, tuple[float, float, float]],
    fidelity: dict[str, float],
    out_path: Path,
    label: str = "in-distribution",
    fig=None,
    flag_tier: str | None = None,
    flag_note: str = "",
):
    """x = fidelity per tier, y = rho with CI bars. Returns the matplotlib figure.

    flag_tier: draw this tier's point with an open/starred marker and a text annotation
    (e.g. tier_4's absolute-vs-rank caveat — see RUN_LOG 2026-07-13) so a reader doesn't
    read its rho at face value without the context of why it's uncertain.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = [t for t in sorted(reliability) if t in fidelity]
    x = [fidelity[t] for t in tiers]
    y = [reliability[t][0] for t in tiers]
    yerr = np.array(
        [
            [reliability[t][0] - reliability[t][1] for t in tiers],
            [reliability[t][2] - reliability[t][0] for t in tiers],
        ]
    )
    if fig is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        ax = fig.axes[0]
    line = ax.errorbar(x, y, yerr=np.abs(yerr), marker="o", capsize=4, label=label)
    if flag_tier in tiers:
        i = tiers.index(flag_tier)
        color = line.lines[0].get_color()
        ax.plot(
            x[i], y[i], marker="*", markersize=16, markeredgecolor="black", markerfacecolor=color, zorder=5
        )
        if flag_note:
            ax.annotate(
                flag_note,
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                style="italic",
            )
    ax.set_xlabel("world-model fidelity (divergence step)")
    ax.set_ylabel("ranking reliability (Spearman rho)")
    ax.set_title("Trust region: how good must the world model be?")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def make_synthetic(
    target_rho: float, n_ckpts: int = 20, n_tasks: int = 8, n_seeds: int = 50, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # n_ckpts=20 (not the study's 5): the self-test checks pipeline correctness, and a
    # 5-point Spearman is too high-variance to compare against a target.
    """Fake sim/dream results whose checkpoint-level correlation is `target_rho` by construction:
    dream = mix of true skill and independent noise."""
    rng = np.random.default_rng(seed)
    skill = rng.uniform(0.1, 0.9, n_ckpts)  # true per-checkpoint success rate
    sim_rows, dream_rows = [], []
    w = target_rho  # mixing weight: rho -> ~w for large samples
    noise = rng.uniform(0.1, 0.9, n_ckpts)
    for c in range(n_ckpts):
        for t in range(n_tasks):
            p_sim = np.clip(skill[c] + rng.normal(0, 0.05), 0, 1)
            p_dream = np.clip(w * skill[c] + (1 - w) * noise[c] + rng.normal(0, 0.02), 0, 1)
            for s in range(n_seeds):
                sim_rows.append(
                    {
                        "checkpoint": f"P{c}",
                        "task": f"task{t}",
                        "seed": s,
                        "success": bool(rng.random() < p_sim),
                        "steps": 100,
                    }
                )
            dream_rows.append(
                {
                    "checkpoint": f"P{c}",
                    "task": f"task{t}",
                    "wm_tier": "synthetic",
                    "seed": 0,
                    "dream_success_prob": p_dream,
                }
            )
    return pd.DataFrame(sim_rows), pd.DataFrame(dream_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=Path, default=REPO_ROOT / "results" / "sim_success.parquet")
    parser.add_argument("--dream", type=Path, default=REPO_ROOT / "results" / "dream_success.parquet")
    parser.add_argument("--fidelity", type=Path, default=REPO_ROOT / "results" / "wm_fidelity.parquet")
    parser.add_argument("--synthetic", type=float, default=None, metavar="RHO")
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help="also report rho using dream_success_prob binarized at each threshold (robustness check)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "figures" / "trust_region.png")
    parser.add_argument("--suite", default="libero_spatial", help="suite used to split dream rows by task")
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.synthetic is not None:
        sim, dream = make_synthetic(args.synthetic)
        rel = ranking_reliability(sim, dream)
        rho, lo, hi = rel["synthetic"]
        print(f"target rho ~{args.synthetic}: recovered {rho:.3f} CI=[{lo:.3f},{hi:.3f}]")
        ok = (args.synthetic >= 0.999 and rho > 0.95) or abs(rho - args.synthetic) < 0.35
        print("RECOVERY OK" if ok else "RECOVERY FAILED")
        raise SystemExit(0 if ok else 1)

    sim, dream = pd.read_parquet(args.sim), pd.read_parquet(args.dream)
    dream_train = dream_for_split(dream, sim, args.suite, "train")
    dream_heldout = dream_for_split(dream, sim, args.suite, "heldout")

    print("== in-distribution (train tasks) ==")
    rel = ranking_reliability(sim, dream_train)
    for tier, (rho, lo, hi) in rel.items():
        print(f"{tier}: rho={rho:.3f} CI=[{lo:.3f},{hi:.3f}]")
    if args.threshold_sweep:
        for t in args.threshold_sweep:
            print(f"-- classifier threshold={t} (dream success rate, binarized) --")
            for tier, (rho, lo, hi) in ranking_reliability(sim, dream_train, threshold=t).items():
                print(f"{tier}: rho={rho:.3f} CI=[{lo:.3f},{hi:.3f}]")
    best = success_rates(sim, "success").groupby("checkpoint")["success"].mean().idxmax()
    print(f"task-level Pearson (best={best}): {task_level_pearson(sim, dream_train, best):.3f}")

    rel_heldout = None
    if len(dream_heldout):
        print("== held-out (distribution shift) ==")
        rel_heldout = ranking_reliability(sim, dream_heldout)
        for tier, (rho, lo, hi) in rel_heldout.items():
            print(f"{tier}: rho={rho:.3f} CI=[{lo:.3f},{hi:.3f}]")
    else:
        print("== held-out (distribution shift): no held-out dream rows yet, skipping ==")

    fid = {}
    if args.fidelity.exists():
        fdf = pd.read_parquet(args.fidelity)
        for ckpt, g in fdf.groupby("checkpoint"):
            fid[Path(ckpt).name] = float(g["divergence_step"].mean())
    if fid:
        fig = trust_region_chart(
            rel,
            fid,
            args.out,
            label="in-distribution",
            flag_tier="tier_4",
            flag_note="tier_4: low absolute\ndream-success, see caveat",
        )
        if rel_heldout:
            trust_region_chart(rel_heldout, fid, args.out, label="held-out", fig=fig, flag_tier="tier_4")
        print(f"chart -> {args.out}")
    import wandb

    run = wandb.init(project="world-models-eval", name="calibration_study", mode=args.wandb)
    for tier, (rho, lo, hi) in rel.items():
        run.summary[f"in_distribution/{tier}/spearman_rho"] = rho
        run.summary[f"in_distribution/{tier}/spearman_ci_lo"] = lo
        run.summary[f"in_distribution/{tier}/spearman_ci_hi"] = hi
    if rel_heldout:
        for tier, (rho, lo, hi) in rel_heldout.items():
            run.summary[f"heldout/{tier}/spearman_rho"] = rho
            run.summary[f"heldout/{tier}/spearman_ci_lo"] = lo
            run.summary[f"heldout/{tier}/spearman_ci_hi"] = hi
    run.summary["task_level_pearson_best"] = task_level_pearson(sim, dream_train, best)
    if args.out.exists():
        run.log({"trust_region": wandb.Image(str(args.out))})
    run.finish()


if __name__ == "__main__":
    main()
