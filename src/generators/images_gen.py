import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from perlin_numpy import generate_perlin_noise_2d

from src.generators.background_gen import voronoi_bg
from src.generators.precipitates_gen import add_precipitates
from src.type_definitions import Image, RandomState


def generate_image(root_seed, img_params_path):
    rng = np.random.RandomState(root_seed)
    vor_seed = rng.randint(0, 10 ** 5)
    precip_seed = rng.randint(0, 10 ** 5)
    illumination_rng = np.random.RandomState(rng.randint(0, 10 ** 5))

    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    background = voronoi_bg(vor_seed, **img_params)
    image, mask = add_precipitates(background, precip_seed, **img_params)
    # image = add_perlin_noise(image, rng, alpha=0.05, divisor=2)
    image = add_gaussian_noise(image, rng)
    image = add_illumination_gradient(image, illumination_rng, img_params["img_illumination_gradient_strength_range"],
                                      img_params["img_illumination_gradient_angle_range"])

    return image, mask


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


def add_perlin_noise(image: Image, rng: RandomState, alpha: float, divisor: int) -> Image:
    """ Adds perlin noise to the whole image. """

    height, width = image.shape
    res = (height // divisor, width // divisor)
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)
    noise -= noise.mean()
    new_image = image + noise * alpha * 127.5

    return new_image


def add_gaussian_noise(image: Image, rng: RandomState):
    noise = rng.normal(0, 1, (1025, 1025))
    noise = noise[:-1, :-1]
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
    mask = 1.0 - strength + grad * strength * 2.0

    return image * mask
