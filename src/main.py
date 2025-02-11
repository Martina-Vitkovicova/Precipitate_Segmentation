import numpy as np

from generators.images_gen import generate_image, save_images
from src.image_visualization import show_image_mask

GENERATION = 14
IMG_PARAMS_PATH = "../config/img_params.json"
IMG_SAVE_PATH = "../../Data/gen" + str(GENERATION)


with open("seeds.txt", "r") as file:
    for _ in range(52):
        root_seed = int(file.readline())
        img, mask = generate_image(root_seed, IMG_PARAMS_PATH)

        show_image_mask(img, mask)
        # save_images(IMG_SAVE_PATH, root_seed, img, mask, IMG_PARAMS_PATH, GENERATION)
