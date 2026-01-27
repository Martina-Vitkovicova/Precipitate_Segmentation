from pathlib import Path
import numpy as np

from generators.images_gen import generate_image, save_images
from src.image_visualization import show_image_mask

GENERATION = "cl4"
IMG_PARAMS_PATH = "../config/img_params.json"
IMG_SAVE_PATH = "../../Data/gen"


def _read_seeds_file(path, limit=None):
    seeds = []
    with open(path, "r") as file:
        for _ in file:
            seeds.append(int(file.readline()))
            if limit is not None and len(seeds) >= limit:
                break
    if len(seeds) < limit:
        print("Not enough seeds in seeds.txt — switching to random.")
        rng = np.random.RandomState()
        seeds += [int(rng.randint(0, 10 ** 9)) for _ in range(limit - len(seeds))]

    return seeds


def generate(num=1, gen=1, img_params_path="../config/img_params.json", img_save_path="../../Data/gen",
             show=True, save=True, seed=22, seeds_path="seeds.txt"):
    img_save_path = Path(f"{img_save_path}_{gen}")

    if seed is not None:  # seed available
        if num == 1:
            seeds = [int(seed)]
        else:
            # generate a reproducible sequence from that seed
            rng = np.random.RandomState(int(seed))
            seeds = [int(rng.randint(0, 10 ** 9)) for _ in range(num)]
    elif seeds_path is not None and Path(seeds_path).exists():  # file available
        seeds = _read_seeds_file(seeds_path, limit=num)
    else:  # otherwise random seeds
        rng = np.random.RandomState()
        seeds = [int(rng.randint(0, 10 ** 9)) for _ in range(num)]

    for root_seed in seeds:
        img, mask = generate_image(root_seed, img_params_path)

        if show:
            show_image_mask(img, mask)
        if save:
            save_images(img_save_path, root_seed, img, mask, img_params_path, gen)


if __name__ == "__main__":
    generate(gen=GENERATION,
             img_params_path=IMG_PARAMS_PATH,
             img_save_path=IMG_SAVE_PATH,
             save=False,
             show=True,
             seed=None,
             seeds_path=None,
             num=1)
