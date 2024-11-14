import hashlib
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy
import skimage
import skimage.draw as skdraw

from perlin_numpy import generate_perlin_noise_2d
from src.external_libs.pv_voronoi_gist.colorized_voronoi import voronoi_finite_polygons_2d


def voronoi_bg(seed: int, *, img_size: List[int], voronoi_regions_num_range: List[int],
               voronoi_color_range: List[int], voronoi_border_width: int, **kwargs):
    rng = np.random.RandomState(seed)
    width, height = img_size
    # image = np.zeros((height, width), dtype=np.float32)
    image = np.full((height, width), 118, dtype=np.float32)
    num_points = rng.randint(*voronoi_regions_num_range)

    points = np.column_stack((rng.uniform(0, width, num_points), rng.uniform(0, height, num_points)))

    vor = scipy.spatial.Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    modified_edges = make_curved_edges(regions, vertices, width, height)

    for region in regions:
        y_coords, x_coords = get_curved_region_coords(modified_edges, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*voronoi_color_range)
        image[row, col] = color
        # todo why is the result darker if applying noise to individual regions?
        # alpha = max(0.02, min(0.1, rng.random()))
        # divisor = rng.choice((2, 4))
        # add_perlin_noise(image, rng, alpha, (image.shape[0] // divisor, image.shape[1] // divisor), row, col)
        # plt.imshow(image, cmap="gray")
        # plt.show()

        image = change_region_border_width(image, color, y_coords, x_coords, voronoi_border_width, rng)

    noise = generate_perlin_noise_2d((width, height), (image.shape[0] // 4, image.shape[1] // 4), rng=rng)
    scaled_noise = (noise + 1) * 255 / 2
    alpha = 0.08
    image = (image * (1 - alpha) + scaled_noise * alpha).astype(np.uint8)

    return image


def make_curved_edges(regions, vertices, width, height):
    modified_edges = {}

    for region in regions:
        for j in range(len(region)):
            index1 = region[j]
            index2 = region[(j + 1) % len(region)]
            edge = (index1, index2)
            sorted_edge = tuple(sorted(edge))

            if sorted_edge not in modified_edges:
                edge_seed = get_edge_seed(sorted_edge)
                edge_rng = np.random.RandomState(edge_seed)
                v1, v2 = vertices[sorted_edge[0]], vertices[sorted_edge[1]]
                line_length = np.linalg.norm(v2 - v1)
                base_segments = max(2, int(np.sqrt(line_length) * 0.2))
                num_segments = max(4, min(12, base_segments + edge_rng.randint(-base_segments//2, base_segments//2)))
                noise = max(5, min(10, int(np.sqrt(line_length) * 0.3) + edge_rng.randint(-2, 2)))
                modified_edges[sorted_edge] = divide_edge(v1, v2, edge_rng, width, height, num_segments, noise)

    return modified_edges


def get_curved_region_coords(modified_edges, region):
    y_coords, x_coords = [], []
    for j in range(len(region)):
        v1 = region[j]
        v2 = region[(j + 1) % len(region)]
        edge = (v1, v2)
        sorted_edge = tuple(sorted(edge))
        poly_edge = modified_edges[sorted_edge]
        if edge != sorted_edge:
            poly_edge = poly_edge[::-1]

        x_coords.extend(poly_edge[:, 0])
        y_coords.extend(poly_edge[:, 1])

    return y_coords, x_coords


def divide_edge(start, end, rng, width, height, num_segments=10, noise_level=10):
    start = tuple(map(int, start))
    end = tuple(map(int, end))
    inside, new_start, new_end = cv2.clipLine((0, 0, width, height), start, end)
    if not inside:
        return np.array([np.array(start), np.array(end)])

    noisy_points = [np.array(start)]
    # notodo ad randomness to the spacing - no need bcs noise creates the randomness
    points = np.linspace(new_start, new_end, num_segments+1)

    for i in range(len(points)):
        noise = rng.normal(0, noise_level, size=2)
        noisy_points.append(points[i] + noise)
    noisy_points.append(np.array(end))

    return np.array(noisy_points)


def get_edge_seed(edge):
    edge_bytes = str(edge).encode('utf-8')
    return int(hashlib.sha256(edge_bytes).hexdigest(), 16) % (2 ** 32)


def change_region_border_width(image, color, y_coords, x_coords, width, rng):
    perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)

    border_image = np.zeros(image.shape)
    border_image[perim_row, perim_col] = 1
    # plt.imshow(border_image)
    # plt.show()
    triangle = np.tri(width, width, dtype=np.uint8)
    border_mask = skimage.morphology.dilation(border_image, triangle)
    # plt.imshow(border_mask)
    # plt.show()

    for i in range(4):
        seq_length = rng.randint(50 // (i+1), 100 // (i+1))
        perim_row, perim_col, exclude_row, exclude_col = slice_lists(perim_row, perim_col, seq_length, rng)
        border_image[exclude_row, exclude_col] = 0
        # plt.imshow(border_image)
        # plt.show()
        width += 2
        triangle = np.tri(width, width, dtype=np.uint8)
        n_mask = skimage.morphology.dilation(border_image, triangle)
        # plt.imshow(n_mask)
        # plt.show()
        border_mask = np.maximum(n_mask, border_mask)
        # plt.imshow(border_mask)
        # plt.show()

    # todo add 3d effect to the border
    image[border_mask == 1] = color + rng.choice([-1, 1]) * rng.randint(8, 16)
    return image


def create_triangle(size):
    triangle = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        triangle[i, :i+1] = 1  # Fill in the triangle
    return triangle


def slice_lists(row, col, seq_len, rng):
    new_row = []
    new_col = []
    removed_row = []
    removed_col = []
    i = 0
    total_length = len(row)

    while i < total_length:
        if rng.choice([True, False]):
            # include these
            sequence_length = rng.randint(10, seq_len)
            end_index = min(i + sequence_length, total_length)
            new_row.extend(row[i:end_index])
            new_col.extend(col[i:end_index])
            i = end_index
        else:
            # pause - exclude these
            pause_length = rng.randint(10, seq_len)
            end_index = min(i + pause_length, total_length)
            removed_row.extend(row[i:end_index])
            removed_col.extend(col[i:end_index])
            i = end_index

    return new_row, new_col, removed_row, removed_col


def add_perlin_noise(image, rng, alpha, res, row, col):
    width, height = image.shape
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)
    scaled_noise = (noise + 1) * 255 / 2

    image[row, col] = (image[row, col] * (1 - alpha) + scaled_noise[row, col] * alpha).astype(np.uint8)

