import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utils.config_parsing import GeneratorConfig
from src.generators.background_gen import generate_matrix
from src.generators.etch_pits_gen import add_etch_pits
from src.generators.precipitates_gen import add_precipitates
from src.utils.type_definitions import Image, RandomState, Mask, FloatRange, IntRange


def generate_image(root_seed: int, img_params_path: str) -> Tuple[Image, Mask]:
    """
    Generate a single synthetic image and precipitate mask from a config file.

    The pipeline applies matrix generation, etch pits, precipitates, configured
    imaging effects, and a final small Gaussian noise term. Randomness is fully
    determined by `root_seed`.
    """
    root_rng = np.random.RandomState(root_seed)
    vor_rng, etch_pits_rng, precip_rng, corner_blur_rng, bottom_blur_rng, illum_rng, gauss_rng = (
        split_random_state(root_rng, 7))

    with open(img_params_path, "r") as file:
        raw_params = json.load(file)

    img_params = GeneratorConfig.from_dict(raw_params)

    matrix = generate_matrix(vor_rng, img_params.general, img_params.matrix)
    image = add_etch_pits(matrix, etch_pits_rng, img_params.general, img_params.etch_pits)
    image, mask = add_precipitates(image, precip_rng, img_params.general, img_params.precipitates)

    effects = img_params.effects
    if effects.corner_blur_enabled:
        image = add_corner_blur(image, corner_blur_rng, effects.corner_blur_radius_range,
                                effects.corner_blur_kernel_size_range)

    if effects.bottom_blur_enabled:
        image = add_bottom_blur(image, bottom_blur_rng, effects.bottom_blur_height_range,
                                effects.bottom_blur_kernel_size_range)

    if effects.illumination_gradient_enabled:
        image = add_illumination_gradient(image, illum_rng, effects.illumination_gradient_strength_range,
                                          effects.illumination_gradient_angle_range)

    image = add_gaussian_noise(image, gauss_rng)
    image = np.clip(image, 0.0, 255.0)

    return image, mask


def split_random_state(parent_rng: RandomState, num: int) -> List[RandomState]:
    """Derive independent RNG streams from a parent RNG."""
    return [np.random.RandomState(int(parent_rng.randint(0, 10**9))) for _ in range(num)]


def save_images(root_path: Path, seed: int, img: Image, mask: Mask, img_params_path: str | None, gen: str) -> None:
    """ Save generated image and mask to disk under a folder named by seed and generation number. """
    root_path = Path(root_path)
    path = root_path / f"seed-{seed}"
    path.mkdir(exist_ok=True, parents=True)
    plt.imsave(path / "img.png", img, cmap='gray', vmin=0, vmax=255)
    plt.imsave(path / "mask.png", mask, cmap='gray', vmin=0, vmax=255)

    assert os.path.isfile(path / "img.png") and os.path.isfile(path / "mask.png")

    if img_params_path:
        save_metadata(img_params_path, seed, gen, path)


def save_metadata(img_params_path, seed, gen, path):
    """ Save image generation parameters and seed to metadata.json in the image folder. """
    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    updated_img_params = img_params.copy()
    updated_img_params["seed"] = seed
    updated_img_params["gen"] = gen

    with open(path / "metadata.json", 'w') as file:
        json.dump(updated_img_params, file, indent=4)


def add_gaussian_noise(image: Image, rng: RandomState) -> Image:
    """ Add small zero-mean Gaussian noise to the whole image. """
    h, w = image.shape
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    noise -= float(noise.mean())

    alpha = float(rng.randint(1, 3)) * 0.01
    return image + noise * (alpha * 127.5)


def add_illumination_gradient(image: Image, rng: RandomState, strength_range: FloatRange,
                              angle_range: FloatRange) -> Image:
    """ Add a directional illumination gradient to the image to simulate microscopy imaging effect. """
    h, w = image.shape
    y = np.linspace(-0.5, 0.5, h)
    x = np.linspace(-0.5, 0.5, w)
    X, Y = np.meshgrid(x, y)

    angle = np.deg2rad(rng.uniform(*angle_range))
    dx, dy = np.cos(angle), np.sin(angle)

    # compute directional gradient in chosen orientation
    grad = dx * X + dy * Y
    grad = (grad - grad.min()) / (grad.max() - grad.min())  # normalize to 0–1

    strength = rng.uniform(*strength_range)
    mask = 1.0 - strength + grad * strength

    return image * mask


def add_corner_blur(image: Image, rng: RandomState, radius_range: IntRange, kernel_size_range: IntRange) -> Image:
    """ Blur a random corner using a circular alpha mask. """
    h, w = image.shape
    corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    corner = corners[int(rng.randint(0, len(corners)))]

    # make circular mask in that corner
    mask = np.zeros((h, w), np.uint8)
    radius = rng.randint(*radius_range)
    cv2.circle(mask, corner, radius, (255, 255, 255), -1)

    # feather edges
    feather = rng.randint(radius // 8, radius // 6)
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    alpha = mask.astype(np.float32) / 255.0

    blur = rng.randint(*kernel_size_range) | 1
    blurred = cv2.GaussianBlur(image, (blur, blur), 0)
    result = (blurred * alpha + image * (1 - alpha)).astype(np.uint8)

    return result


def add_bottom_blur(image: Image, rng: RandomState, height_range: IntRange, kernel_size_range: IntRange) -> Image:
    """ Blur a band near the bottom edge using a feathered alpha mask. """
    h, w = image.shape

    band_size = int(rng.randint(*height_range) + 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h - band_size: h, :] = 255

    # feather the transition toward the interior
    feather = rng.randint(band_size // 6, band_size // 4)
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    alpha = mask.astype(np.float32) / 255.0

    k = int(rng.randint(*kernel_size_range)) | 1
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    out = (blurred.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha))

    return np.clip(out, 0, 255).astype(np.uint8)
