import numpy as np

from generators.images_gen import generate_image, save_images
from src.image_visualization import show_image_mask

GENERATION = 14
IMG_PARAMS_PATH = "../config/img_params_borders.json"
IMG_SAVE_PATH = "../../Data/gen" + str(GENERATION)


def generate(num=1, gen=14, img_params_path="../config/img_params.json", img_save_path="../../Data/gen",
             show=True, save=True, random_seeds=False, seeds_path="seeds.txt"):
    img_save_path = f"{img_save_path}{gen}"

    with open(seeds_path, "r") as file:
        if not random_seeds and num > sum(1 for _ in file):
            return print("Not enough seeds in seeds.txt")
        file.seek(0)

        for _ in range(num):
            if random_seeds:
                root_seed = np.random.randint(0, 10 ** 5)
            else:
                root_seed = int(file.readline())

            img, mask = generate_image(root_seed, img_params_path)

            if show:
                show_image_mask(img, mask)
            if save:
                save_images(img_save_path, root_seed, img, mask, img_params_path, gen)


if __name__ == "__main__":
    generate(gen=GENERATION, img_params_path=IMG_PARAMS_PATH, img_save_path=IMG_SAVE_PATH, save=False, random_seeds=True, num=10)
