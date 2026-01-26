import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
import matplotlib.colors as mcolors
import matplotlib as mpl

from src.image_analysis.matrix_analysis import matrix_stats_per_image

cl0_path = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new_divided_me\cluster_0")
cl1_path = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new_divided_me\cluster_1")
cl2_path = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new_divided_me\cluster_2")
cl3_path = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new_divided_me\cluster_3")
cl4_path = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new_divided_me\cluster_4")
cluster_dirs = {0: cl0_path, 1: cl1_path, 2: cl2_path, 3: cl3_path, 4: cl4_path}

FEATURES = ["mean", "std", "p5", "p95", "local_std_std", "grad_energy"]


def compute_or_load_stats(image_root: pathlib.Path, csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load per-image background statistics from CSV if it exists, otherwise compute them.
    """
    if csv_path.exists():
        stats_df = pd.read_csv(csv_path)
        print(f"Loaded stats from: {csv_path}")
        return stats_df

    stats_list = matrix_stats_per_image(image_root)
    stats_df = pd.DataFrame(stats_list)
    print("Stats computed.")
    return stats_df


def get_feature_columns(stats_df: pd.DataFrame) -> list[str]:
    feature_columns = [col for col in FEATURES if col in stats_df.columns]
    if not feature_columns:
        raise ValueError("No valid feature columns found in stats.")
    return feature_columns


def get_feature_matrix(stats_df: pd.DataFrame, feature_columns: list[str], standardize: bool = True) -> np.ndarray:
    feature_matrix = stats_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    if standardize:
        feature_matrix = (feature_matrix - feature_matrix.mean(0)) / (feature_matrix.std(0) + 1e-12)
    return feature_matrix


def run_kmeans(feature_matrix: np.ndarray, n_clusters: int, seed: int):
    kmeans = KMeans(n_clusters=int(n_clusters), n_init=20, random_state=int(seed)).fit(feature_matrix)
    cluster_labels = kmeans.labels_.astype(int)

    silhouette = float("nan")
    if len(np.unique(cluster_labels)) > 1:
        silhouette = float(silhouette_score(feature_matrix, cluster_labels))

    return cluster_labels, silhouette


def show_important_features(stats_df: pd.DataFrame, feature_columns: list[str]):
    print("\nFeature importance (ANOVA across clusters)")
    labels = stats_df["cluster"].to_numpy()

    for feature_name in feature_columns:
        samples = [stats_df.loc[labels == cluster_id, feature_name].values for cluster_id in np.unique(labels)]
        F_value, p_value = f_oneway(*samples)
        print(f"  {feature_name:15s} F={F_value:.2f}  p={p_value:.2e}")


def print_cluster_summary(stats_df: pd.DataFrame, feature_columns: list[str]):
    summary_df = stats_df.loc[:, feature_columns + ["cluster"]].copy()

    group_mean = summary_df.groupby("cluster").mean(numeric_only=True)
    group_std = summary_df.groupby("cluster").std(numeric_only=True)
    summary = group_mean.round(3).astype(str) + " ± " + group_std.round(3).astype(str)

    print(summary.reset_index().to_string(index=False, col_space=18, justify="center"))


def evaluate_k_range(stats_df: pd.DataFrame, feature_columns: list[str], k_values, seed: int, standardize: bool,):
    feature_matrix = get_feature_matrix(stats_df, feature_columns, standardize=standardize)

    print(f"\n=== KMeans sweep ===")
    for n_clusters in k_values:
        cluster_labels, silhouette = run_kmeans(feature_matrix, n_clusters=n_clusters, seed=seed)
        stats_df = stats_df.copy()
        stats_df["cluster"] = cluster_labels
        print("\n" + "-" * 80)
        print(f"k={n_clusters}, silhouette={silhouette:.3f}\n")
        print_cluster_summary(stats_df, feature_columns)
        show_important_features(stats_df, feature_columns)


def load_clustering_from_root_path(divided_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    i = 0
    for subdir in sorted(p for p in divided_root.iterdir() if p.is_dir()):
        cluster_id = i
        stats_list = matrix_stats_per_image(subdir)
        i += 1
        for stats in stats_list:
            row = stats.copy()
            row["cluster"] = cluster_id
            rows.append(row)
    return pd.DataFrame(rows)


def loaded_clustering_stats(fin_root: pathlib.Path, standardize: bool = True):
    stats_df = load_clustering_from_root_path(fin_root)
    feature_columns = get_feature_columns(stats_df)
    feature_matrix = get_feature_matrix(stats_df, feature_columns, standardize=standardize)

    labels = stats_df["cluster"].to_numpy()
    n_clusters = len(np.unique(labels))
    silhouette = float("nan") if n_clusters < 2 else float(silhouette_score(feature_matrix, labels))

    print(f"k={n_clusters}, silhouette={silhouette:.3f}\n")
    print_cluster_summary(stats_df, feature_columns)
    show_important_features(stats_df, feature_columns)

    return stats_df


def apply_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.facecolor": "#ECECF4",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.4,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#BEBEBE",
        "grid.linewidth": 1.6,
        "grid.alpha": 0.4,
    })


def plot_clusters_mean_p95(stats_df: pd.DataFrame, out_pdf: str | None = "cluster_mean_vs_p95.pdf"):
    apply_paper_style()

    clusters = sorted(stats_df["cluster"].unique())
    cmap = mcolors.ListedColormap(plt.cm.Set1.colors[:len(clusters)])
    norm = mcolors.BoundaryNorm(boundaries=list(clusters) + [clusters[-1] + 1], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    sc = ax.scatter(stats_df["mean"], stats_df["p95"], c=stats_df["cluster"], cmap=cmap, norm=norm,
                    s=55, alpha=0.85, edgecolors="black", linewidths=0.6)

    ax.set_xlabel("Mean intensity", fontweight="semibold")
    ax.set_ylabel("p95 intensity", fontweight="semibold")
    ax.set_title("All images coloured by cluster", pad=12, fontweight="bold")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = plt.colorbar(sc, cax=cax, ticks=clusters, spacing="proportional",
                        boundaries=list(clusters) + [clusters[-1] + 1])
    cbar.set_label("Cluster ID", fontsize=11, labelpad=4)
    cbar.ax.tick_params(labelsize=11, length=0)

    plt.tight_layout()
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    LABELED_DIR = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new")
    STATS_CSV = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Baseline\src\image_analysis\matrix_stats_per_image.csv")

    stats_df = compute_or_load_stats(LABELED_DIR, STATS_CSV)
    feature_columns = get_feature_columns(stats_df)
    k_values = range(2, 9)
    evaluate_k_range(stats_df, feature_columns, k_values, seed=12345, standardize=True)

    print("\n\n=== Final clustering summary ===\n")
    FIN_CLUSTERING_PATH = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new_divided_me")
    fin_stats_df = loaded_clustering_stats(FIN_CLUSTERING_PATH, standardize=True)

    plot_clusters_mean_p95(fin_stats_df, out_pdf=None)


# k=6, silhouette=0.558
#
#  cluster        mean               std                 p5                p95           local_std_std       grad_energy
#    0      141.449 ± 1.497      25.79 ± 1.643    102.111 ± 4.014    179.333 ± 3.969     10.268 ± 0.927      101.32 ± 9.457
#    1      123.301 ± 0.753     29.683 ± 3.782       72.4 ± 6.947      168.6 ± 5.562      7.145 ± 0.825    136.495 ± 16.607
#    2      124.051 ± 1.968     23.535 ± 2.287       88.5 ± 2.953      159.7 ± 4.968      9.449 ± 1.303       92.17 ± 8.429
#    3      148.509 ± 2.416     33.154 ± 0.904     92.375 ± 1.768      199.5 ± 2.976      8.404 ± 1.214     158.234 ± 9.564
#    4      126.822 ± 1.681       17.51 ± 1.97      99.25 ± 4.245    153.333 ± 2.462      5.958 ± 1.102     68.389 ± 10.586
#    5      125.803 ± 0.169     43.104 ± 2.438       47.0 ± 4.243       192.5 ± 4.95      10.472 ± 0.81     212.974 ± 5.385
#
# Feature importance (ANOVA across clusters)
#   mean            F=329.86  p=2.94e-34
#   std             F=73.51  p=1.65e-20
#   p5              F=93.91  p=1.18e-22
#   p95             F=151.53  p=5.61e-27
#   local_std_std   F=23.31  p=1.78e-11
#   grad_energy     F=111.77  p=3.30e-24
# ...


# === Final clustering summary ===
#
# k=5, silhouette=0.414
#
#  cluster          mean               std                 p5                p95           local_std_std       grad_energy
#     0       141.617 ± 1.508     26.515 ± 2.767      100.8 ± 5.613       180.8 ± 5.959    10.349 ± 0.912    105.698 ± 16.467
#     1        124.48 ± 1.754     22.018 ± 2.905     90.933 ± 4.334      157.733 ± 5.12     8.542 ± 1.701     86.042 ± 11.447
#     2       127.882 ± 1.259     16.458 ± 1.921    101.714 ± 3.861       153.0 ± 2.449     5.408 ± 1.137     64.534 ± 12.346
#     3        123.718 ± 1.19      31.92 ± 6.288    68.167 ± 11.785    172.583 ± 10.681     7.699 ± 1.515    149.242 ± 33.384
#     4       149.278 ± 1.136      33.17 ± 0.975     92.857 ± 1.215     200.286 ± 2.138       8.02 ± 0.59      160.11 ± 8.595
#
# Feature importance (ANOVA across clusters)
#   mean            F=572.14  p=1.40e-38
#   std             F=29.35  p=3.83e-12
#   p5              F=41.76  p=9.29e-15
#   p95             F=68.27  p=9.32e-19
#   local_std_std   F=14.65  p=8.68e-08
#   grad_energy     F=38.00  p=4.91e-14

