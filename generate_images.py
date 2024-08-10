import os
import numpy as np
from matplotlib import pyplot as plt

from generate_background import voronoi_bg
from generate_precipitates import add_precipitates
from perlin_numpy import generate_perlin_noise_2d


def generate_image(root_seed, img_params):
    rng = np.random.RandomState(root_seed)
    vor_seed = rng.randint(0, 10**5)
    precip_seed = rng.randint(0, 10**5)
    background = voronoi_bg(vor_seed, img_params)
    image, mask = add_precipitates(background, precip_seed, img_params)
    image = add_perlin_noise(image, vor_seed)

    return image, mask


def add_perlin_noise(image, seed):
    np.random.seed(seed)
    width, height = image.shape
    noise = generate_perlin_noise_2d((width, height), (width//4, height//4))
    scaled_noise = (noise + 1) * 255 / 2

    alpha = 0.05
    blended_image = (image * (1 - alpha) + scaled_noise * alpha).astype(np.uint8)

    return blended_image


def save_images(root_path, seed, img, mask, gen=1):
    path = root_path + '/seed_{}_gen_{}'.format(seed, gen)
    os.mkdir(path)
    plt.imsave(path + '/img.png', img, cmap='gray')
    plt.imsave(path + '/mask.png', mask, cmap='gray')

    assert os.path.isfile(path + '/img.png') and os.path.isfile(path + '/mask.png')
