from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.image_analysis.image_mask_loading import iter_imgs_and_masks


def blur_map_local(img, win=15):
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    k = win if win % 2 == 1 else win + 1
    mean = cv2.boxFilter(lap, -1, (k, k))
    mean2 = cv2.boxFilter(lap * lap, -1, (k, k))

    std = np.sqrt(np.maximum(mean2 - mean ** 2, 0))
    sharp_norm = cv2.normalize(std, None, 0, 1, cv2.NORM_MINMAX)
    blur = 1 - sharp_norm
    return blur


def show_blur_overlay(img, blur, alpha=0.5):
    plt.figure()
    plt.imshow(img, cmap="gray")
    im = plt.imshow(blur, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    plt.colorbar(im, label="Blur level (0 = sharp, 1 = blurred)")
    plt.title("Blur overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compute_dataset_blur_maps(dataset_dir, win=15, blur_threshold=0.70, max_images=None, target_size=(1024, 1024)):
    mean_sum = freq_sum = row_sum = None
    n_images = 0

    for i, (img, _) in enumerate(iter_imgs_and_masks(dataset_dir)):
        if max_images is not None and i >= max_images:
            break

        if target_size is not None and img.shape != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        blur = blur_map_local(img, win)

        if mean_sum is None:
            mean_sum = np.zeros_like(blur, dtype=np.float64)
            freq_sum = np.zeros_like(blur, dtype=np.float64)
            row_sum = np.zeros(blur.shape[0], dtype=np.float64)

        mean_sum += blur
        freq_sum += (blur >= blur_threshold)
        row_sum += blur.mean(axis=1)
        n_images += 1

    mean_blur_map = mean_sum / n_images
    blur_frequency_map = freq_sum / n_images
    mean_row_profile = row_sum / n_images
    return mean_blur_map, blur_frequency_map, mean_row_profile, n_images


def show_results(mean_blur_map, blur_frequency_map, mean_row_profile, blur_threshold):
    print(f"Images processed: {mean_row_profile.size}")
    print(f"Global mean blur: {mean_blur_map.mean():.4f}")
    print(f"Global P95 blur:  {np.percentile(mean_blur_map, 95):.4f}")

    plt.figure()
    plt.imshow(mean_blur_map)
    plt.title("Mean blur map")
    plt.axis("off")
    plt.show()

    plt.figure()
    plt.imshow(blur_frequency_map)
    plt.title(f"Blur frequency map (threshold={blur_threshold})")
    plt.axis("off")
    plt.show()

    plt.figure()
    plt.plot(mean_row_profile)
    plt.title("Mean blur per image row")
    plt.xlabel("row (top → bottom)")
    plt.ylabel("mean blur")
    plt.show()


if __name__ == "__main__":
    dataset_dir = Path("../../segm/real_data")

    win = 15
    blur_threshold = 0.70

    mean_blur_map, blur_frequency_map, mean_row_profile, n_images = \
        compute_dataset_blur_maps(dataset_dir, win, blur_threshold)

    show_results(mean_blur_map, blur_frequency_map, mean_row_profile, blur_threshold)

    for img, _ in iter_imgs_and_masks(dataset_dir):
        blur = blur_map_local(img, win)
        show_blur_overlay(img, blur, alpha=0.5)

    # mostly blurring on the bottom side of the image and near edges

