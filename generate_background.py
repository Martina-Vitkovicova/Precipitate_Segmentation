import numpy as np
import scipy
import skimage
import skimage.draw as skdraw
from pv_voronoi_gist.colorized_voronoi import voronoi_finite_polygons_2d


def voronoi_bg(seed, img_params):
    rng = np.random.RandomState(seed)
    width, height = img_params["img_size"]
    image = np.zeros((height, width), dtype=np.float32)
    num_points = rng.randint(*img_params["voronoi_regions_num_range"])

    points = np.column_stack((rng.uniform(0, width, num_points), rng.uniform(0, height, num_points)))

    vor = scipy.spatial.Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for region in regions:
        polygon_verts = vertices[region]

        # needs to be fixed
        # perturbation creates problem with borders - double border + blanks and overlaps
        # perim_row, perim_col = skdraw.polygon_perimeter(polygon_verts[:, 1], polygon_verts[:, 0], shape=img_size)
        # perturbed_row, perturbed_col = apply_wavy_perturbation(np.vstack((perim_row, perim_col)).T, img_size, 2, 0.05)

        row, col = skdraw.polygon(polygon_verts[:, 1], polygon_verts[:, 0], shape=image.shape)

        color = rng.randint(*img_params["voronoi_color_range"])
        image[row, col] = color
        image = change_region_border_width(image, color, polygon_verts, img_params["voronoi_border_width"])

    return image


def change_region_border_width(image, color, polygon_verts, width):
    perim_row, perim_col = skdraw.polygon_perimeter(polygon_verts[:, 1], polygon_verts[:, 0], shape=image.shape)

    border_image = np.zeros(image.shape)
    border_image[perim_row, perim_col] = 1
    border_mask = skimage.morphology.dilation(border_image, skimage.morphology.disk(width))

    image[border_mask == 1] = color - 5
    return image


def apply_wavy_perturbation(coords, image_shape, ampl=3, freq=0.05):
    perturbation_sin = ampl * np.sin(freq * np.arange(len(coords)))
    perturbation_sin2 = (ampl + 1) * np.sin(freq - 0.02 * np.arange(len(coords)))
    perturbed_coords = coords + perturbation_sin[:, np.newaxis] + perturbation_sin2[:, np.newaxis]

    perim_row = np.clip(perturbed_coords[:, 0].astype(int), 0, image_shape[0] - 1)
    perim_col = np.clip(perturbed_coords[:, 1].astype(int), 0, image_shape[1] - 1)
    
    return perim_row, perim_col
