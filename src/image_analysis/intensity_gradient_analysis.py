import os
import pathlib
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_normalized_illumination(image_path, blur_sigma_scale):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    height, width = gray.shape

    blur_sigma = blur_sigma_scale * min(height, width)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    illumination_field = blurred / (blurred.mean() + 1e-8)
    return illumination_field, gray.mean()


def fit_illumination_gradient(illumination_field):
    height, width = illumination_field.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float64)

    matrix = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(height * width)], axis=1)
    target_values = illumination_field.astype(np.float64).ravel()

    # fit plane I(x, y) = a*x + b*y + c to the relative illumination field
    a, b, c = np.linalg.lstsq(matrix, target_values, rcond=None)[0]
    return float(a), float(b), float(c)


def analyze_corners(a, b, c, height, width, global_mean, corner_frac):
    ch = max(1, int(height * corner_frac))
    cw = max(1, int(width * corner_frac))

    regions = {
        "top_left":     (slice(0, ch),        slice(0, cw)),
        "top_right":    (slice(0, ch),        slice(width - cw, width)),
        "bottom_left":  (slice(height - ch, height), slice(0, cw)),
        "bottom_right": (slice(height - ch, height), slice(width - cw, width)),
    }

    values = {}
    for name, (ys, xs) in regions.items():
        yy, xx = np.mgrid[ys, xs].astype(np.float64)
        rel_vals = a * xx + b * yy + c
        values[name] = float(rel_vals.mean() * global_mean)

    darkest = min(values.items(), key=lambda kv: kv[1])
    brightest = max(values.items(), key=lambda kv: kv[1])
    spread = float(brightest[1] - darkest[1])

    return values, darkest, brightest, spread


def analyze_illumination(path, corner_frac=0.10, blur_sigma_scale=0.06, verbose=True):
    illum_field, global_mean = get_normalized_illumination(path, blur_sigma_scale)
    height, width = illum_field.shape

    a, b, c = fit_illumination_gradient(illum_field)
    corner_data, darkest, brightest, corner_spread = analyze_corners(a, b, c, height, width, global_mean, corner_frac)

    magnitude = np.hypot(a, b)
    angle = np.degrees(np.arctan2(b, a))

    gray_change_per_100px = magnitude * 255 * 100

    if verbose:
        print(f"--- Illumination Analysis: {path} ---")
        print("Image values in range 0-255")
        print(f"Gradient Direction: {angle:.2f}° (0°=right, -90°=up)")
        print(f"Brightness Slope (per 100px): {gray_change_per_100px:.3f}")
        print(f"Darkest Area: {darkest[0]} = {darkest[1]:.2f}")
        print(f"Brightest Area: {brightest[0]} = {brightest[1]:.2f}")
        print(f"Max Contrast Spread: {brightest[1] - darkest[1]:.2f}")

    return {
        "corner_values": corner_data,
        "darkest_corner": darkest,
        "brightest_corner": brightest,
        "max_corner_spread": corner_spread,
        "gradient_angle_deg": angle,
        "gradient_magnitude_rel_per_px": magnitude,
        "gray_change_per_100px": gray_change_per_100px,
    }


def find_extreme_corners(corner_names, all_metrics, summary):
    for name in corner_names:
        vals = np.array([m["corner_values"][name] for m in all_metrics], dtype=np.float64)
        summary[f"corner_values.{name}_mean"] = float(vals.mean())
        summary[f"corner_values.{name}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    pairs = []
    for m in all_metrics:
        cv = m["corner_values"]
        dark = min(cv, key=cv.get)
        bright = max(cv, key=cv.get)
        pairs.append((dark, bright))

    best_pair, best_count = Counter(pairs).most_common(1)[0]
    summary["max_change_pair"] = best_pair
    summary["max_change_pair_count"] = int(best_count)


def analyze_batch(paths, corner_frac=0.10, blur_sigma_scale=0.06, verbose=True):
    all_metrics = []
    for path in paths:
        illum_info = analyze_illumination(path, corner_frac=corner_frac, blur_sigma_scale=blur_sigma_scale, verbose=False)
        all_metrics.append(illum_info)
        print(f"Processed: {path}")

    scalar_keys = [
        "max_corner_spread",
        "gradient_angle_deg",
        "gradient_magnitude_rel_per_px",
        "gray_change_per_100px",
    ]

    summary = {}
    for k in scalar_keys:
        vals = np.array([m[k] for m in all_metrics], dtype=np.float64)
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    corner_names = list(all_metrics[0]["corner_values"].keys())
    find_extreme_corners(corner_names, all_metrics, summary)

    if verbose:
        def fmt(mu, sd, prec=2):
            return f"{mu:.{prec}f} ± {sd:.{prec}f}"

        print("\n=== Summary across images (mean ± std) ===")
        print("Intensity values in range 0-255")
        print(f"Gradient angle (°):              {fmt(summary['gradient_angle_deg_mean'], summary['gradient_angle_deg_std'])}")
        print(f"Brightness change /100px:        {fmt(summary['gray_change_per_100px_mean'], summary['gray_change_per_100px_std'])}")
        print(f"Max corner Δ:                    {fmt(summary['max_corner_spread_mean'], summary['max_corner_spread_std'])}")
        d, b = summary["max_change_pair"]
        print(f"Most common max-change pair: {d} -> {b} ({summary['max_change_pair_count']}/{len(all_metrics)})")

        print("\nCorner means:")
        for name in corner_names:
            print(f"  {name:12s} {fmt(summary[f'corner_values.{name}_mean'], summary[f'corner_values.{name}_std'])}")

    return all_metrics, summary


def show_illumination(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.06 * min(gray.shape))

    illum = blurred / blurred.mean()  # normalize to ~1.0
    plt.imshow(illum, cmap='inferno')
    plt.title("Relative illumination (bright = brighter area)")
    plt.axis('off')
    plt.colorbar(label='Relative intensity')
    plt.show()


def read_paths(txt: pathlib.Path):
    return [
        pathlib.Path(line.strip())
        for line in txt.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_imgs(root):
    return sorted(root.rglob("img.png"))


if __name__ == "__main__":
    LABELED_DIR = pathlib.Path(r"C:\Users\vitko\Desktop\CVR\Data\labeled_new")
    img_paths = list(LABELED_DIR.rglob('img.png'))

    analyze_batch(img_paths, corner_frac=0.2)

    for img_path in img_paths:
        if os.path.isfile(img_path):
            show_illumination(img_path)


# there is a strong brightness gradient from top-right to bottom-left

# === Summary across labeled images (mean ± std) ===
# Gradient angle (°):                      -40.53 ± 11.17
# Brightness change /100px (gray):         3.98 ± 1.10
# Total change across image:               -30.06 ± 8.07 (if negat, TR brighter)
# TR corner mean:                          143.2417 ± 10.9983
# BL corner mean:                          118.3593 ± 9.7010
# Corner Δ(TR–BL) abs:                     24.8824 ± 7.2427

# === Summary cl0 (mean ± std) ===
# Gradient angle (°):                      -39.24 ± 8.25
# Brightness change /100px (gray):         3.75 ± 0.53
# Total change across image:               -29.60 ± 4.22
# TR corner mean:                          152.4925 ± 2.4070
# BL corner mean:                          128.3827 ± 1.7184
# Corner Δ(TR–BL) abs:                     24.1098 ± 3.2084

# === Summary cl1 (mean ± std) ===
# Gradient angle (°):                      -39.53 ± 17.06
# Brightness change /100px (gray):         3.50 ± 0.79
# Total change across image:               -23.57 ± 5.57
# TR corner mean:                          134.4035 ± 3.3533
# BL corner mean:                          113.3189 ± 5.4964
# Corner Δ(TR–BL) abs:                     21.0846 ± 7.4822

# === Summary cl2 (mean ± std) ===
# Gradient angle (°):                      -37.80 ± 12.14
# Brightness change /100px (gray):         2.70 ± 0.91
# Total change across image:               -26.15 ± 9.37
# TR corner mean:                          137.1981 ± 4.4085
# BL corner mean:                          116.9248 ± 4.8744
# Corner Δ(TR–BL) abs:                     20.2733 ± 9.1312

# === Summary cl3 (mean ± std) ===
# Gradient angle (°):                      -43.69 ± 6.22
# Brightness change /100px (gray):         5.15 ± 0.94
# Total change across image:               -35.64 ± 6.44
# TR corner mean:                          137.9723 ± 2.3060
# BL corner mean:                          108.7719 ± 3.6081
# Corner Δ(TR–BL) abs:                     29.2004 ± 5.4638

# === Summary cl4 (mean ± std) ===
# Gradient angle (°):                      -41.84 ± 2.11
# Brightness change /100px (gray):         4.63 ± 0.16
# Total change across image:               -38.95 ± 1.43
# TR corner mean:                          164.0421 ± 1.7582
# BL corner mean:                          132.7113 ± 2.0465
# Corner Δ(TR–BL) abs:                     31.3307 ± 2.3081


