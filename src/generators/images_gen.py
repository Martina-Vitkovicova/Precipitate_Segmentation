import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.generators.background_gen import voronoi_bg
from src.generators.etch_pits_gen import add_etch_pits
from src.generators.precipitates_gen import add_precipitates
from src.type_definitions import Image, RandomState


def generate_image(root_seed, img_params_path):
    rng = np.random.RandomState(root_seed)
    vor_seed = rng.randint(0, 10 ** 5)
    precip_seed = rng.randint(0, 10 ** 5)
    etch_pits_seed = rng.randint(0, 10 ** 5)
    illumination_rng = np.random.RandomState(rng.randint(0, 10 ** 5))
    corner_blur_rng = np.random.RandomState(rng.randint(0, 10 ** 5))
    bottom_blur_rng = np.random.RandomState(rng.randint(0, 10 ** 5))

    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    light_direction = img_params.get("light_direction")
    img_size = img_params.get("img_size")
    background = voronoi_bg(vor_seed, img_size=img_size, light_direction=light_direction,
                            **get_subparams(img_params, "voronoi_"))

    image = add_etch_pits(background, etch_pits_seed, light_direction=light_direction,
                          **get_subparams(img_params, "etch_pits_"))
    image, mask = add_precipitates(image, precip_seed, light_direction=light_direction,
                                   **get_subparams(img_params, "precipitate_"))

    if img_params.get("corner_blur"):
        image = add_corner_blur(image, corner_blur_rng, **get_subparams(img_params, "corner_blur_"))

    if img_params.get("bottom_blur"):
        image = add_bottom_blur(image, bottom_blur_rng, **get_subparams(img_params, "bottom_blur_"))

    if img_params.get("img_illumination_gradient"):
        image = add_illumination_gradient(image, illumination_rng,
                                          **get_subparams(img_params, "img_illumination_gradient_"))
    image = add_gaussian_noise(image, rng)

    return image, mask


def get_subparams(params: dict, prefix: str) -> dict:
    out = {}
    for key, value in params.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
    return out


def save_images(root_path, seed, img, mask, img_params_path=None, gen=1):
    root_path = Path(root_path)
    path = root_path / f"seed-{seed}_gen-{gen}"
    path.mkdir(exist_ok=True, parents=True)
    plt.imsave(path / "img.png", img, cmap='gray', vmin=0, vmax=255)
    plt.imsave(path / "mask.png", mask, cmap='gray', vmin=0, vmax=255)

    assert os.path.isfile(path / "img.png") and os.path.isfile(path / "mask.png")

    if img_params_path:
        save_metadata(img_params_path, seed, gen, path)


def save_metadata(img_params_path, seed, gen, path):
    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    updated_img_params = img_params.copy()
    updated_img_params["seed"] = seed
    updated_img_params["gen"] = gen

    with open(path / "metadata.json", 'w') as file:
        json.dump(updated_img_params, file, indent=4)


def add_gaussian_noise(image: Image, rng: RandomState):
    noise = rng.normal(0, 1, (1024, 1024))
    noise -= noise.mean()
    alpha = rng.randint(1, 3) * 0.01
    new_image = image + noise * alpha * 127.5
    return new_image


def add_illumination_gradient(image, rng, strength_range, angle_range):
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


def add_corner_blur(image: Image, rng: RandomState, radius_range: List[int], kernel_size_range: List[int]) -> Image:
    h, w = image.shape
    corners = [[0, 0], [w, 0], [w, h], [0, h]]
    i = rng.randint(0, len(corners))
    corner = corners[i]

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


def add_bottom_blur(image: Image, rng: RandomState, height_range: List[int], kernel_size_range: List[int]) -> Image:
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
