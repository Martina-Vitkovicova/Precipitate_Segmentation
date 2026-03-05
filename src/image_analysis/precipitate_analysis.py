from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src.image_analysis.image_mask_loading import iter_imgs_and_masks


def describe(values, name: str, precision: int = 2) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}

    q25, med, q75 = np.percentile(x, [25, 50, 75])
    desc = {
        "count": int(x.size),
        "min": float(x.min()),
        "25%": float(q25),
        "median": float(med),
        "mean": float(x.mean()),
        "75%": float(q75),
        "max": float(x.max()),
        "std": float(x.std(ddof=0)),
    }

    print(f"{name}:")
    for k in ["count", "min", "25%", "median", "mean", "75%", "max", "std"]:
        v = desc[k]
        if k == "count":
            print(f"  {k:>6} = {v:d}")
        else:
            print(f"  {k:>6} = {v:.{precision}f}")
    print()
    return desc


def describe_per_image_precip_count(precip_counts: np.ndarray, decimals: int = 2):
    x = np.asarray(precip_counts, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print("Precipitate count per image:\n  (no data)\n")
        return

    stats = {
        "image_count": int(x.size),
        "min": int(x.min()),
        "25%": np.percentile(x, 25),
        "median": np.percentile(x, 50),
        "mean": x.mean(),
        "75%": np.percentile(x, 75),
        "max": int(x.max()),
        "std": x.std(ddof=0),
    }

    print("Precipitate count per image:")
    print(f"  images = {stats['image_count']}")
    for k in ["mean", "std", "25%", "median", "75%"]:
        print(f"  {k:>6} = {stats[k]:.{decimals}f}")
    for k in ["min", "max"]:
        print(f"  {k:>6} = {stats[k]:d}")
    print()


def collect_precip_count_per_image(root_path: Path) -> pd.DataFrame:
    rows = []
    for img, mask, img_path, mask_path in iter_imgs_and_masks(root_path, with_paths=True):
        mask_bin = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows.append({
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "precip_count": int(len(contours)),
        })
    return pd.DataFrame(rows)


def collect_precip_areas(root_path: Path) -> tuple[np.ndarray, np.ndarray]:
    areas_contour = []
    areas_px = []

    for _, mask in iter_imgs_and_masks(root_path):
        mask_bin = (mask > 0).astype(np.uint8)

        _, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        if stats.shape[0] > 1:
            areas_px.extend(stats[1:, cv2.CC_STAT_AREA].tolist())

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            a = float(cv2.contourArea(cnt))
            if a > 0:
                areas_contour.append(a)

    return np.asarray(areas_contour, dtype=np.float64), np.asarray(areas_px, dtype=np.float64)


def collect_precip_circularity(root_path: Path) -> np.ndarray:
    circs = []

    for _, mask in iter_imgs_and_masks(root_path):
        mask_bin = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area <= 0:
                continue
            perim = float(cv2.arcLength(cnt, True))
            if perim <= 1e-6:
                continue
            circs.append(4.0 * np.pi * area / (perim * perim))

    return np.asarray(circs, dtype=np.float64)


def collect_precip_radii(root_path: Path) -> np.ndarray:
    radii = []

    for _, mask in iter_imgs_and_masks(root_path):
        mask_bin = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (_, _), r = cv2.minEnclosingCircle(cnt)
            if r > 0:
                radii.append(float(r))

    return np.asarray(radii, dtype=np.float64)


def collect_precip_intensity(root_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_px = []
    per_image_means = []
    per_image_zero_percents = []
    non_zero_px = []

    for img, mask in iter_imgs_and_masks(root_path):
        m = mask > 0
        pix = img[m].astype(np.float32)
        if pix.size == 0:
            continue

        all_px.append(pix)
        per_image_means.append(float(pix.mean()))
        per_image_zero_percents.append(float(np.count_nonzero(pix == 0) * 100.0 / pix.size))

        nz = pix[pix > 0]
        if nz.size:
            non_zero_px.append(nz)

    all_px_arr = np.concatenate(all_px) if all_px else np.asarray([], dtype=np.float64)
    non_zero_arr = np.concatenate(non_zero_px) if non_zero_px else np.asarray([], dtype=np.float64)

    return (
        all_px_arr.astype(np.float64),
        np.asarray(per_image_means, dtype=np.float64),
        np.asarray(per_image_zero_percents, dtype=np.float64),
        non_zero_arr.astype(np.float64),
    )


def save_stats_csv(output_dir: Path, **named_arrays):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in named_arrays.items():
        out_path = output_dir / f"{name}.csv"
        pd.DataFrame({name: np.asarray(arr)}).to_csv(out_path, index=False)


def collect_all_stats(root_path: Path) -> dict:
    areas_contour, areas_px = collect_precip_areas(root_path)
    return {
        "areas_px": areas_px,
        "areas_contour": areas_contour,
        "circularity": collect_precip_circularity(root_path),
        "radii": collect_precip_radii(root_path),
        "intensity": collect_precip_intensity(root_path),  # (all_px, means, zero_perc, non_zero)
    }


def print_summaries(count_df, stats: dict):
    if not count_df.empty:
        describe_per_image_precip_count(count_df["precip_count"].values)

    all_px, means, zero_perc, non_zero = stats["intensity"]
    describe(stats["areas_px"], "Precipitate area (px², pixel-count)")
    describe(stats["areas_contour"], "Precipitate area (px², contour-based)")
    describe(stats["radii"], "Precipitate radius (px, min enclosing circle)")
    describe(stats["circularity"], "Precipitate circularity (0-1)")
    describe(all_px, "Intensity of all precipitate pixels (0-255)")
    describe(means, "Mean precipitate intensity per image (0-255)")
    describe(zero_perc, "Zero-valued precipitate pixels per image (%)")
    describe(non_zero, "Non-zero precipitate pixel intensity (0-255)")


def save_outputs(output_dir: Path, count_df, stats: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not count_df.empty:
        count_df.to_csv(output_dir / "precip_count_per_image.csv", index=False)

    all_px, means, zero_perc, non_zero = stats["intensity"]
    save_stats_csv(
        output_dir,
        precip_area_px2=stats["areas_px"],
        precip_area_contour=stats["areas_contour"],
        precip_radius_px=stats["radii"],
        precip_circularity=stats["circularity"],
        precip_intensity_all=all_px,
        precip_intensity_mean_per_image=means,
        precip_zero_percent_per_image=zero_perc,
        precip_intensity_non_zero=non_zero,
    )


if __name__ == "__main__":
    root_path = Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new")
    output_dir = Path(r"C:\Users\vitko\Desktop\CVR\Baseline\src\image_analysis\precipitate_stats")

    count_df = collect_precip_count_per_image(root_path)
    stats = collect_all_stats(root_path)

    print_summaries(count_df, stats)

    SAVE_STATS = True
    if SAVE_STATS:
        save_outputs(output_dir, count_df, stats)
        print(f"Saved CSVs to: {output_dir}")
