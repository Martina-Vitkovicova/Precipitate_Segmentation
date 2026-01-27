import argparse
from pathlib import Path
import numpy as np

from src.generators.images_gen import generate_image, save_images
from src.image_visualization import show_image_mask

GENERATION = "cl4"
IMG_PARAMS_PATH = "../config/img_params.json"
IMG_SAVE_PATH = "../../Data/gen"


def _read_seeds_file(path, limit=None):
    seeds = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(int(line))
            if limit is not None and len(seeds) >= limit:
                break

    if limit is not None and len(seeds) < limit:
        print(f"Not enough seeds in {path} — switching to random.")
        rng = np.random.RandomState()
        seeds += [int(rng.randint(0, 10**9)) for _ in range(limit - len(seeds))]

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


def main():
    p = argparse.ArgumentParser(description="Generate synthetic SEM images and masks.")
    p.add_argument("--num", type=int, default=1, help="Number of images to generate.")
    p.add_argument("--gen", default=GENERATION, help="Generation label (e.g. wo_embossing, gen15).")
    p.add_argument("--config", dest="img_params_path", default=IMG_PARAMS_PATH, help="Path to JSON config.")
    p.add_argument("--out", dest="img_save_path", default=IMG_SAVE_PATH, help="Output directory prefix.")
    p.add_argument("--seed", type=int, default=None, help="Root seed. If num>1, generates a reproducible seed list.")
    p.add_argument("--seeds_path", default=None, help="Path to a text file with one seed per line.")
    p.add_argument("--show", action="store_true", help="Show generated images and masks.")
    p.add_argument("--no-show", dest="show", action="store_false", help="Do not show images/masks.")
    p.set_defaults(show=True)
    p.add_argument("--save", action="store_true", help="Save outputs (img/mask/metadata) to disk.")
    p.add_argument("--no-save", dest="save", action="store_false", help="Do not save outputs.")
    p.set_defaults(save=True)

    args = p.parse_args()
    generate(**vars(args))


if __name__ == "__main__":
    # generate(save=False, seed=None)
    main()
