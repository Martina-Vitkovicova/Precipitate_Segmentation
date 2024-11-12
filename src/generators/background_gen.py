import hashlib
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import skimage.draw as skdraw

from generators.precipitates_gen import perturb_precip_contour
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
    modified_edges = make_wiggly_edges(regions, vertices, width, height)

    for region in regions:
        y_coords, x_coords = get_wiggly_region_coords(modified_edges, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*voronoi_color_range)
        image[row, col] = color
        # plt.imshow(image, cmap="gray")
        # plt.show()

        image = change_region_border_width(image, color, y_coords, x_coords, voronoi_border_width, rng)

    return image


def make_wiggly_edges(regions, vertices, width, height):
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
                num_segments = max(2, min(10, base_segments + edge_rng.randint(-base_segments//2, base_segments//2)))
                noise = max(10, min(20, int(np.sqrt(line_length) * 0.3) + edge_rng.randint(-2, 2)))
                modified_edges[sorted_edge] = divide_edge(v1, v2, edge_rng, width, height, num_segments, noise)

    return modified_edges


def get_wiggly_region_coords(modified_edges, region):
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
    # todo ad randomness to the spacing
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
    border_mask = skimage.morphology.dilation(border_image, skimage.morphology.disk(width))
    for i in range(3):
        perim_row, perim_col = slice_lists(perim_row, perim_col, rng.randint(20, 100), rng)
        border_image[perim_row, perim_col] = 0
        width += 2
        triangle = create_triangle(width)
        n_mask = skimage.morphology.dilation(border_image, triangle)
        border_mask = np.maximum(n_mask, border_mask)

    # todo add 3d effect to the border
    image[border_mask == 1] = color + rng.randint(-15, 15)
    return image


def create_triangle(size):
    triangle = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        triangle[i, :i+1] = 1  # Fill in the triangle
    return triangle


def slice_lists(row, col, seq_len, rng):
    new_row = []
    new_col = []
    i = 0

    while i < len(row):
        if rng.choice([True, False]):
            sequence_length = rng.randint(10, seq_len)
            end_index = min(i + sequence_length, len(row))
            new_row.extend(row[i:end_index])
            new_col.extend(col[i:end_index])
            i = end_index
        else:
            # Random length for the pause (at least 1)
            pause_length = rng.randint(10, seq_len)
            i += pause_length

    return new_row, new_col


def apply_wavy_perturbation(coords, image_shape, ampl=3, freq=0.05):
    perturbation_sin = ampl * np.sin(freq * np.arange(len(coords)))
    perturbation_sin2 = (ampl + 1) * np.sin(freq - 0.02 * np.arange(len(coords)))
    perturbed_coords = coords + perturbation_sin[:, np.newaxis] + perturbation_sin2[:, np.newaxis]

    perim_row = np.clip(perturbed_coords[:, 0].astype(int), 0, image_shape[0] - 1)
    perim_col = np.clip(perturbed_coords[:, 1].astype(int), 0, image_shape[1] - 1)
    
    return perim_row, perim_col


def generate_border_width(border_length, rng, segment_length_range, start=1, min_value=0, max_value=20):
    width_list = [start]

    while len(width_list) < border_length:
        # Determine the increasing or decreasing wave
        max_w = 3
        step = rng.randint(0, max_w) if len(width_list) % 2 == 1 else - rng.randint(1, max_w)
        segment_length = rng.randint(*segment_length_range)

        for _ in range(segment_length):
            next_value = width_list[-1] + step
            next_value = max(min_value, min(max_value, next_value))
            width_list.append(next_value)
            if len(width_list) >= border_length:
                # return np.array(width_list).reshape(len(width_list), 1, 2)
                return np.array(width_list)
