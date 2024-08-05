import numpy as np
import scipy
import skimage
from matplotlib import pyplot as plt
import skimage.draw as skdraw
from perlin_numpy import generate_perlin_noise_2d
from pv_voronoi_gist.colorized_voronoi import voronoi_finite_polygons_2d


def voronoi_bg(img_size, seed, min_regions, max_regions):
    rng = np.random.RandomState(seed)
    image = np.full(img_size, rng.randint(120, 150), dtype=np.float32)
    num_points = rng.randint(min_regions, max_regions)
    points = rng.rand(num_points, 2) * img_size[0]

    vor = scipy.spatial.Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for region in regions:
        polygon_verts = vertices[region]

        # perturbation creates problem with borders - double border + blanks and overlaps
        # perim_row, perim_col = skdraw.polygon_perimeter(polygon_verts[:, 1], polygon_verts[:, 0], shape=img_size)
        # perturbed_row, perturbed_col = apply_wavy_perturbation(np.vstack((perim_row, perim_col)).T, img_size, 2, 0.05)

        row, col = skdraw.polygon(polygon_verts[:, 1], polygon_verts[:, 0], shape=image.shape)

        color = rng.randint(120, 150)
        image[row, col] = color
        border_mask = add_border_width(polygon_verts[:, 1], polygon_verts[:, 0], img_size, 2)
        image[border_mask == 1] = color - 5

    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    return image, regions, vertices


def add_border_width(perturbed_col, perturbed_row, img_size, width):
    perim_row, perim_col = skdraw.polygon_perimeter(perturbed_col, perturbed_row, shape=img_size)

    border_image = np.zeros(img_size)
    border_image[perim_row, perim_col] = 1
    border_mask = skimage.morphology.dilation(border_image, skimage.morphology.disk(width))

    return border_mask


def apply_wavy_perturbation(coords, image_shape, ampl=3, freq=0.05):
    perturbation_sin = ampl * np.sin(freq * np.arange(len(coords)))
    perturbation_sin2 = (ampl + 1) * np.sin(freq - 0.02 * np.arange(len(coords)))
    perturbed_coords = coords + perturbation_sin[:, np.newaxis] + perturbation_sin2[:, np.newaxis]

    perim_row = np.clip(perturbed_coords[:, 0].astype(int), 0, image_shape[0] - 1)
    perim_col = np.clip(perturbed_coords[:, 1].astype(int), 0, image_shape[1] - 1)
    
    return perim_row, perim_col


def add_perlin_noise(image, seed, img_size):
    np.random.seed(seed)
    noise = (generate_perlin_noise_2d(img_size, (img_size[0]//4, img_size[1]//4)) + 1) * 255 / 2

    alpha = 0.05
    blended_image = (image * (1 - alpha) + noise * alpha).astype(np.uint8)

    # plt.imshow(blended_image, cmap='gray')
    # plt.show()

    return blended_image
