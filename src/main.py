import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np

from src.generators.images_gen import generate_image, save_images
from src.utils.image_visualization import show_image_mask


DEFAULT_GENERATION = "test"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "../config/base_params.json"
DEFAULT_OUTPUT_DIR = Path("../../Data")


def make_seeds(*, num: int, seed: Optional[int]) -> List[int]:
    """
    Create a list of root seeds.

    - If seed is None, generate `num` random seeds.
    - If seed is provided and num == 1, return [seed].
    - If seed is provided and num > 1, return a reproducible sequence derived from seed.
    """
    if num <= 0:
        return []

    if seed is None:
        rng = np.random.RandomState()
        return [int(rng.randint(0, 10**9)) for _ in range(num)]

    if num == 1:
        return [int(seed)]

    rng = np.random.RandomState(int(seed))
    return [int(rng.randint(0, 10**9)) for _ in range(num)]


def generate(num: int = 1, gen: str = DEFAULT_GENERATION, img_params_path: str = DEFAULT_CONFIG_PATH,
             img_save_path: str = DEFAULT_OUTPUT_DIR, show: bool = True,
             save: bool = True, seed: Optional[int] = None) -> None:
    """
    Generate synthetic image-mask pairs using the configuration file.

    Randomness is determined by `seed`. If `num > 1`, a reproducible list of per-image seeds is derived from `seed`.
    """
    out_dir = Path(img_save_path) / gen
    seeds = make_seeds(num=num, seed=seed)

    for root_seed in seeds:
        img, mask = generate_image(root_seed, img_params_path)

        if show:
            show_image_mask(img, mask)
        if save:
            save_images(out_dir, root_seed, img, mask, img_params_path, gen)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic SEM images and masks.")
    p.add_argument("--num", type=int, default=1, help="Number of images to generate.")
    p.add_argument("--gen", default=DEFAULT_GENERATION, help="Generation label used in the output folder suffix.")
    p.add_argument("--config", dest="img_params_path", default=DEFAULT_CONFIG_PATH, help="Path to JSON config.")
    p.add_argument("--out", dest="img_save_path", default=DEFAULT_OUTPUT_DIR, help="Output directory prefix.")
    p.add_argument("--seed", type=int, default=None, help="Root seed. If omitted, images are generated randomly.")

    p.add_argument("--show", dest="show", action="store_true", help="Show generated images and masks.")
    p.add_argument("--no-show", dest="show", action="store_false", help="Do not show images and masks.")
    p.set_defaults(show=True)

    p.add_argument("--save", dest="save", action="store_true", help="Save outputs to disk.")
    p.add_argument("--no-save", dest="save", action="store_false", help="Do not save outputs.")
    p.set_defaults(save=True)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    generate(**vars(args))


if __name__ == "__main__":
    main()
    # generate(gen="test",
    #          img_params_path="../config/params_cl4.json",
    #          img_save_path="../../Data",
    #          save=False,
    #          show=True,
    #          seed=None,
    #          num=1)
