import pathlib
import cv2
import numpy as np
import pandas as pd

from src.image_analysis.image_mask_loading import iter_imgs_and_masks


def get_avg_matrix_color(root_path):
    total_count = 0
    mean = 0.0
    M2 = 0.0  # Sum of squares of differences from the current mean

    for img, mask in iter_imgs_and_masks(root_path):
        mask_bool = mask.astype(bool)

        unmasked_pixels = img[~mask_bool].astype(np.float32)
        n = unmasked_pixels.size

        batch_mean = unmasked_pixels.mean()
        batch_M2 = ((unmasked_pixels - batch_mean) ** 2).sum()

        delta = batch_mean - mean
        total_new = total_count + n
        mean += delta * (n / total_new)
        M2 += batch_M2 + delta ** 2 * total_count * n / total_new
        total_count = total_new

    variance = M2 / total_count
    std_dev = np.sqrt(variance)
    return mean, std_dev


def count_pixels_in_range(root_path, mean, window: float = 30.0):
    total_in_range = 0
    total_masked = 0

    lower = mean - window
    upper = mean + window

    for img, mask in iter_imgs_and_masks(root_path):
        mask_bool = mask.astype(bool)
        unmasked = img[~mask_bool].astype(np.float32)

        in_range = ((unmasked >= lower) & (unmasked <= upper)).sum()
        total_in_range += in_range
        total_masked += unmasked.size

    return total_in_range, total_masked


def get_local_std_std(img: np.ndarray, mask: np.ndarray, k: int = 7) -> float:
    # Estimates spatial variability of the background by computing the std of a k×k local standard deviation map
    img_f = img.astype(np.float32)

    m = cv2.blur(img_f, (k, k))
    m2 = cv2.blur(img_f * img_f, (k, k))
    local_std_map = np.sqrt(np.maximum(m2 - m * m, 0))

    vals = local_std_map[mask == 0]
    return float(vals.std(ddof=0)) if vals.size else float("nan")


def get_grad_energy(img: np.ndarray, mask: np.ndarray, ksize: int = 3) -> float:
    # Average background edge strength (Sobel gradient magnitude).
    img_f = img.astype(np.float32)

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=ksize)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    vals = grad_mag[mask == 0]
    return float(vals.mean()) if vals.size else float("nan")


def matrix_stats_per_image(root_path: pathlib.Path, save_csv_path: pathlib.Path | None = None):
    out = []

    for img, mask, img_path, _ in iter_imgs_and_masks(root_path, with_paths=True):
        bg = img[mask == 0].astype(np.float32)
        if bg.size == 0:
            continue

        mean = float(bg.mean())
        std = float(bg.std(ddof=0))
        p5 = float(np.percentile(bg, 5))
        p95 = float(np.percentile(bg, 95))

        local_std_std = get_local_std_std(img, mask, k=7)
        grad_energy = get_grad_energy(img, mask, ksize=3)

        out.append({
            "path": str(img_path),
            "mean": mean,
            "std": std,
            "p5": p5,
            "p95": p95,
            "local_std_std": local_std_std,
            "grad_energy": grad_energy,
        })

    if save_csv_path is not None:
        df = pd.DataFrame(out)
        df.to_csv(save_csv_path, index=False)

    return out


def print_average_matrix_stats(root_path: pathlib.Path, windows):
    mean, std = get_avg_matrix_color(root_path)

    print("\n=== Background intensity statistics ===")
    print(f"Background mean intensity : {mean:.3f}")
    print(f"Background intensity std  : {std:.3f}")

    for window in windows:
        count_in_range, total_unmasked = count_pixels_in_range(root_path, mean, window)
        frac = 100.0 * count_in_range / max(total_unmasked, 1)

        print(f"Pixels in [mean ± {window:>2}] = [{mean-window:.2f}, {mean+window:.2f}]  : "
              f"{frac:5.1f}% ({count_in_range:,}/{total_unmasked:,})")


def print_per_image_matrix_stats(root_path):
    stats = matrix_stats_per_image(root_path)

    print("\n=== Per-image background statistics ===")
    print(
        f"{'idx':>3s}  "
        f"{'mean':>8s}  {'std':>8s}  {'p5':>6s}  {'p95':>6s}  "
        f"{'lc_std':>8s}  {'grad_E':>8s}"
    )
    print("-" * 65)

    for i, d in enumerate(stats):
        i += 1
        print(
            f"{i:3d}  "
            f"{d['mean']:8.3f}  {d['std']:8.3f}  "
            f"{d['p5']:6.1f}  {d['p95']:6.1f}  "
            f"{d['local_std_std']:8.3f}  {d['grad_energy']:8.3f}"
        )


if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=120)

    LABELED_DIR = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new")

    print_average_matrix_stats(LABELED_DIR, windows=[14, 30, 55])
    print_per_image_matrix_stats(LABELED_DIR)

    # === Background intensity statistics (wo precips) for all labeled images ===
    # Background; mean; intensity: 131.057
    # Background; intensity; std: 27.056
    # Pixels in [mean ± 14] = [117.06, 145.06]: 49.2 % (30, 650, 520 / 62, 314, 492)
    # Pixels in [mean ± 30] = [101.06, 161.06]: 79.7 % (49, 654, 350 / 62, 314, 492)
    # Pixels in [mean ± 55] = [76.06, 186.06]: 94.7 % (58, 982, 766 / 62, 314, 492)

    # === Per-image background statistics ===
    # idx      mean       std      p5     p95    lc_std    grad_E
    # -----------------------------------------------------------------
    #   1   140.417    26.986    97.0   182.0    10.243   114.228
    #   2   141.762    26.902   100.0   182.0    10.742   108.694
    #   3   144.330    26.539   104.0   186.0     9.650   107.885
    #   4   143.125    33.040    89.0   194.0    11.087   145.099
    #   5   142.672    25.556   104.0   179.0    10.338    96.517
    # ...
