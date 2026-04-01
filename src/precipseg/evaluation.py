from pathlib import Path
import re
import pandas as pd
import seaborn as sns
from typing import Sequence, Dict, List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

Image = Mask = npt.NDArray[np.generic]
ImgList = Sequence[Image]
GroupedData = Dict[str, List[tuple[pd.DataFrame, str]]]

METRICS = ["dice", "iou", "precision", "recall", "f1"]


def save_image_results(imgs: ImgList, labels: ImgList, preds: ImgList, run_dir: Path, max_images: int = 6):
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


def collect_runs(root: str | Path, metrics_file_name: str = "metrics.txt"):
    root = Path(root).resolve()
    grouped = {}
    config_paths = {}

    for path in root.rglob(metrics_file_name):
        run_dir = path.parent

        if re.fullmatch(r"seed[_-]?\d+", run_dir.name):
            config = run_dir.parent.name
            exp_root = run_dir.parent
        else:
            config = run_dir.name
            exp_root = run_dir

        df = pd.read_csv(path, sep=r"\s+|,", engine="python")
        df.columns = df.columns.str.strip().str.lower()

        grouped.setdefault(config, []).append((df.sort_values("threshold").reset_index(drop=True), run_dir.name))

        if config not in config_paths:
            config_paths[config] = exp_root

    return grouped, config_paths


def prepare_master_df(grouped_runs: GroupedData, main_metric: str, config_paths: Dict[str, Path]):
    all_rows = []
    for config, runs in grouped_runs.items():
        exp_path = config_paths[config]
        for df, seed_name in runs:
            best_idx = df[main_metric].idxmax()
            row = df.loc[best_idx].copy()
            row["config"] = config

            match = re.search(r'\d+', seed_name)
            row["run_id"] = int(match.group()) if match else seed_name

            row["seed_folder"] = seed_name
            row["exp_path"] = str(exp_path)
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def save_main_metric(df: pd.DataFrame, main_metric: str, out_dir: Path, plot: bool = True):
    stats = df.groupby("config")[main_metric].agg(["mean", "std", "max", "min"]).reset_index()
    stats["Mean (±Std)"] = stats.apply(lambda x: f"{x['mean']:.4f} (±{x['std']:.4f})", axis=1)
    stats["Stability Gap"] = stats["Stability Gap"] = (stats["max"] - stats["min"]).round(4)

    lb = stats.sort_values("mean", ascending=False).reset_index(drop=True)

    cols = ["config", "Mean (±Std)", "max", "min", "Stability Gap"]
    lb[cols].rename(columns={"max": "Best", "min": "Worst"}).to_csv(out_dir / "01_leaderboard.csv", index=False)

    if plot:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=main_metric, y="config", order=lb["config"], hue="config", palette="viridis",
                    legend=False)
        plt.title(f"Global Stability: {main_metric.upper()}")
        plt.tight_layout()
        plt.savefig(out_dir / "02_stability_plot.png", dpi=150)
        plt.close()
    return lb


def save_full_metrics(df: pd.DataFrame, main_metric: str, out_dir: Path):
    full = df.groupby("config")[METRICS].agg(["mean", "std"]).reset_index()

    formatted_df = pd.DataFrame()
    formatted_df["config"] = full["config"]

    for m in METRICS:
        mean_col = full[(m, "mean")]
        std_col = full[(m, "std")]

        formatted_df[f"{m.capitalize()} (±Std)"] = [
            f"{mean:.4f} (±{std:.4f})" for mean, std in zip(mean_col, std_col)
        ]

    sort_values = full[(main_metric, "mean")]
    formatted_df["_sort_key"] = sort_values
    formatted_df = formatted_df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"]).reset_index(
        drop=True)

    formatted_df.to_csv(out_dir / "03_full_metrics_avg.csv", index=False)
    return formatted_df


def save_extremes(df: pd.DataFrame, main_metric: str, n: int, out_dir: Path, mode: str = "both"):
    """ mode: 'top', 'bottom', or 'both' """

    sorted_df = df.sort_values(main_metric, ascending=False).reset_index(drop=True)

    parts = []
    if mode in ["top", "both"]:
        top = sorted_df.head(n).copy()
        top["group"] = "Top"
        parts.append(top)
    if mode in ["bottom", "both"]:
        bottom = sorted_df.tail(n).copy()
        bottom["group"] = "Bottom"
        parts.append(bottom)

    if not parts: return
    combined = pd.concat(parts).drop_duplicates()

    lead_cols = ["group", "run_id", "config"]
    exclude = lead_cols + ["exp_path", "seed_folder"]
    other_cols = [c for c in combined.columns if c not in exclude]

    final_order = lead_cols + other_cols

    combined = combined[final_order].round(4)
    combined.to_csv(out_dir / f"04_extreme_runs_{mode}_{main_metric}.csv", index=False)

    return combined


def save_individual_reports(df: pd.DataFrame, plot: bool = True):
    for config in df["config"].unique():
        config_data = df[df["config"] == config].copy()
        exp_root = Path(config_data["exp_path"].iloc[0])
        metric_cols = [m for m in METRICS if m in config_data.columns]
        table_cols = ["run_id", "threshold"] + metric_cols

        summary_table = config_data[table_cols].sort_values("run_id").round(4)
        summary_table.to_csv(exp_root / f"all_seeds_summary.csv", index=False)

        if plot:
            melted = config_data.melt(id_vars=["run_id"], value_vars=METRICS,
                                      var_name="Metric", value_name="Score")
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=melted, x="Metric", y="Score", hue="Metric", palette="Set2", legend=False)
            plt.title(f"Metric Distributions: {config}")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(exp_root / f"metric_variance_plot.png", dpi=150)
            plt.close()


def run_full_evaluation(root_data_dir: str | Path, metrics_file_name: str = "metrics.txt", main_metric: str = "dice",
                        n_extremes: int = 5):
    root_path = Path(root_data_dir).resolve()
    global_dir = root_path / "global_reports"
    global_dir.mkdir(parents=True, exist_ok=True)

    grouped_runs, config_paths = collect_runs(root_path, metrics_file_name)
    master_df = prepare_master_df(grouped_runs, main_metric, config_paths)

    save_main_metric(master_df, main_metric, global_dir)
    save_full_metrics(master_df, main_metric, global_dir)
    save_extremes(master_df, main_metric, n_extremes, global_dir, "both")

    save_individual_reports(master_df)

    print(f"All reports generated")
