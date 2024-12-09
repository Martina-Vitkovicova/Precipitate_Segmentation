import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.generators.background_gen import voronoi_bg
from src.generators.precipitates_gen import add_precipitates


def generate_image(root_seed, img_params_path):
    rng = np.random.RandomState(root_seed)
    vor_seed = rng.randint(0, 10 ** 5)
    precip_seed = rng.randint(0, 10 ** 5)

    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    background = voronoi_bg(vor_seed, **img_params)
    image, mask = add_precipitates(background, precip_seed, **img_params)

    return image, mask


def save_images(root_path, seed, img, mask, img_params_path=None, gen=1):
    root_path = Path(root_path)
    path = root_path / f"seed-{seed}_gen-{gen}"
    path.mkdir(exist_ok=True, parents=True)
    plt.imsave(path/"img.png", img, cmap='gray')
    plt.imsave(path/"mask.png", mask, cmap='gray')

    assert os.path.isfile(path/"img.png") and os.path.isfile(path/"mask.png")

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

