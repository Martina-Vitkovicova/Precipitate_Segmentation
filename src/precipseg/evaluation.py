from pathlib import Path
import re
import pandas as pd
from typing import Sequence, Dict, List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

Image = Mask = npt.NDArray[np.generic]
ImgList = Sequence[Image]

METRICS = ["dice", "iou", "precision", "recall", "f1"]


def save_image_results(imgs: ImgList, labels: ImgList, preds: ImgList, run_dir: Path, max_images: int = 6) -> None:
    out_dir = run_dir / "image_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    for img, label, pred in zip(imgs, labels, preds):
        if max_images is not None and i > max_images:
            break

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        for ax, title, im_ in zip(axs, ["img", "mask", "prediction"], [img, label, pred]):
            ax.imshow(im_, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"res{i}.png", dpi=150)
        plt.close(fig)
        i += 1


def collect_runs(root: str | Path) -> Dict[str, List[pd.DataFrame]]:
    root = Path(root)
    grouped = {}

    for path in sorted(root.rglob("metrics.txt")):
        run_dir = path.parent
        config = run_dir.parent.name if re.fullmatch(r"seed[_-]?\d+", run_dir.name) else run_dir.name

        df = pd.read_csv(path, sep=r"\s+|,", engine="python")
        df.columns = df.columns.str.strip().str.lower()
        df = df[["threshold", *METRICS]].copy()
        df["threshold"] = df["threshold"].astype(float).round(4)
        grouped.setdefault(config, []).append(df.sort_values("threshold").reset_index(drop=True))

    if not grouped:
        raise RuntimeError(f"No metrics.txt files found under {root}")

    return grouped


def runs_at_best_threshold(run_dfs: List[pd.DataFrame], threshold: float) -> pd.DataFrame:
    rows = []
    for i, df in enumerate(run_dfs, 1):
        row = df.loc[(df["threshold"] - threshold).abs().idxmin()]
        rows.append({
            "run": i,
            "threshold": float(row["threshold"]),
            **{m: row[m] for m in METRICS},
        })
    return pd.DataFrame(rows).sort_values("dice", ascending=False).reset_index(drop=True)


def summarise_distribution(runs_df: pd.DataFrame, config: str, threshold: float) -> pd.DataFrame:
    s = runs_df["dice"]
    mean = s.mean()
    std = s.std(ddof=1) if len(s) > 1 else 0.0

    return pd.DataFrame([{
        "config": config,
        "best_threshold": threshold,
        "n_runs": len(s),
        "dice_mean": mean,
        "dice_std": std,
        "dice_min": s.min(),
        "dice_p25": s.quantile(0.25),
        "dice_median": s.median(),
        "dice_p75": s.quantile(0.75),
        "dice_max": s.max(),
        "worst_gap": mean - s.min(),
        "n_bad_runs": int((s < mean - 2 * std).sum()) if std > 0 else 0,
    }])


def summarise_config(
    run_dfs: List[pd.DataFrame],
    config: str,
    best_by: str = "dice",
    out_dir: str | Path | None = None,
    digits: int = 4,
) -> Dict[str, pd.DataFrame]:
    agg = pd.concat(run_dfs).groupby("threshold")[METRICS].agg(["mean", "std"]).reset_index()
    agg.columns = ["threshold"] + [m if s == "mean" else f"{m}_std" for m, s in agg.columns[1:]]
    agg = agg.fillna(0).sort_values("threshold").reset_index(drop=True)

    best = agg.loc[agg[best_by].idxmax()]
    summary = pd.DataFrame([{
        "config": config,
        "n_runs": len(run_dfs),
        "best_threshold": float(best["threshold"]),
        **{k: v for m in METRICS for k, v in [
                (f"best_{m}_mean", best[m]),
                (f"best_{m}_std", best[f"{m}_std"])]}
    }])

    best_threshold = float(summary.loc[0, "best_threshold"])
    runs = runs_at_best_threshold(run_dfs, best_threshold)
    dist = summarise_distribution(runs, config, best_threshold)

    result = {
        "agg_metrics": agg.round(digits),
        "summary_ranked": summary.round(digits),
        "runs_at_best_threshold": runs.round(digits),
        "summary_distribution": dist.round(digits),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in result.items():
            df.to_csv(out_dir / f"{name}.csv", index=False)

    return result


def rank_summaries(
    summary_dfs: List[pd.DataFrame],
    out_path: str | Path | None = None,
    digits: int = 4,
) -> pd.DataFrame:
    global_summary = pd.concat(summary_dfs, ignore_index=True).sort_values(
        ["best_dice_mean", "best_iou_mean", "best_f1_mean"],
        ascending=False,
    ).reset_index(drop=True)

    global_summary.insert(0, "rank", range(1, len(global_summary) + 1))
    global_summary = global_summary.round(digits)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        global_summary.to_csv(out_path, index=False)

    return global_summary
