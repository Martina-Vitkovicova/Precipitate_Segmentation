import hashlib
from typing import List, Tuple, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import skimage
import skimage.draw as skdraw

from perlin_numpy import generate_perlin_noise_2d

from src.generators.finite_voronoi_gen import get_finite_voronoi
from src.type_definitions import (Image, IndexedRegion, Vertices, Edge, Point, RandomState, Coords, BorderMask,
                                  GradientImage)


def voronoi_bg(seed: int, *, img_size: List[int], voronoi_regions_num_range: List[int], voronoi_color_range: List[int],
               voronoi_border_width: int, voronoi_border_jaggedness: float, voronoi_border_noise: float,
               voronoi_border_3d: bool, voronoi_light_direction: List[int], voronoi_shading_color_range: List[int],
               voronoi_perlin_noise: List[int], **kwargs) -> Image:
    """
    Creates a background matrix image consisting of regions (grains) with borders.

    Parameters
    ----------
    seed:
        int number used for setting random generators.
    img_size:
        [width, height] dimensions of the generated image.
    voronoi_regions_num_range:
        [min, max] number of base voronoi regions (the total region count may be a little different due to truncating
        and adding regions which were not closed before).
    voronoi_color_range:
        [darker, lighter] color range for the regions. Each region has individually generated colour.
    voronoi_border_width:
        int number for base width of the region borders. The borders are processed and widened, so this number is the
        minimum border width in the image.
    voronoi_border_jaggedness:
    voronoi_border_noise:
    voronoi_border_3d:
    voronoi_light_direction:
    voronoi_shading_color_range:
    voronoi_perlin_noise:
    kwargs:
        other parameters not used for the background generation.

    Returns
    -------
    Numpy 2D ndarray image.
    """
    rng = np.random.RandomState(seed)
    width, height = img_size
    # image = np.zeros((height, width), dtype=np.float32)
    image = np.full((height, width), 140, dtype=np.float32)

    num_points = rng.randint(*voronoi_regions_num_range)
    region_generator_points = np.column_stack((rng.uniform(0, width, num_points),
                                               rng.uniform(0, height, num_points)))
    vor = scipy.spatial.Voronoi(region_generator_points)
    regions, vertices = get_finite_voronoi(vor)

    # to generate only one long border
    # p1, p2 = generate_distant_points(rng, 800, width, height)
    # regions, vertices = [[0, 1]], np.array([p1, p2])

    modified_borders = distort_borders(regions, vertices, width, height, voronoi_border_jaggedness,
                                       voronoi_border_noise)

    for region in regions:
        y_coords, x_coords = get_distorted_region_coords(modified_borders, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*voronoi_color_range)
        image[row, col] = color
        # plt.imshow(image, cmap="gray")
        # plt.show()
        image = improve_borders(image, y_coords, x_coords, voronoi_border_width, rng, color, voronoi_border_3d,
                                voronoi_light_direction, voronoi_shading_color_range)
        image = add_perlin_noise(image, rng, row, col, voronoi_perlin_noise)

    return image


def generate_distant_points(rng, min_distance, width, height):
    while True:
        p1, p2 = np.column_stack((rng.uniform(0, width, 2), rng.uniform(0, height, 2)))
        distance = np.linalg.norm(p1 - p2)

        if distance > min_distance:
            return p1, p2


def distort_borders(regions: List[IndexedRegion], vertices: Vertices, width: int, height: int,
                    voronoi_border_jaggedness: float, voronoi_border_noise: float) -> Dict[Edge, Vertices]:
    """ Transforms borders consisting of straight lines to jagged polylines. """

    modified_borders = {}

    for region in regions:
        for i in range(len(region)):
            edge = (region[i], region[(i + 1) % len(region)])
            sorted_edge = tuple(sorted(edge))

            if sorted_edge not in modified_borders:
                edge_seed = get_edge_seed(sorted_edge)
                edge_rng = np.random.RandomState(edge_seed)
                v1, v2 = vertices[sorted_edge[0]], vertices[sorted_edge[1]]
                v1, v2 = tuple(map(int, v1)), tuple(map(int, v2))
                modified_borders[sorted_edge] = divide_edge(v1, v2, edge_rng, width, height, voronoi_border_jaggedness,
                                                            voronoi_border_noise)

    return modified_borders


def get_edge_seed(edge: Edge) -> int:
    """ Create a unique seed to set a random generator with. """

    edge_bytes = str(edge).encode('utf-8')
    return int(hashlib.sha256(edge_bytes).hexdigest(), 16) % (2 ** 32)


def divide_edge(start: Point, end: Point, rng: RandomState, width: int, height: int, voronoi_border_jaggedness: float,
                voronoi_border_noise: float) -> Vertices:
    """ Divides a straight line given by start and end point to number of segments and adds noise to the coordinates
    to make the new line jagged. """

    # if the line doesn't lie inside the image area, it's not necessary to divide it
    # if it lies partially, we divide only the part of the line which lies in the area
    is_inside, new_start, new_end = cv2.clipLine((0, 0, width, height), start, end)
    if not is_inside:
        # todo should I clip this?
        return np.array([np.array(start), np.array(end)])

    noisy_points = [np.array(new_start)]

    line_length = np.linalg.norm(np.array(new_end) - np.array(new_start))
    # todo set a parameter for num of segments aka border jaggedness
    base_segments = max(2, int(np.sqrt(line_length) * voronoi_border_jaggedness))
    # changes in x, y coords to make the border jagged
    noise_level = max(3, min(15, int(np.sqrt(line_length) * voronoi_border_noise) + rng.randint(-2, 2)))
    # to how many segments will the line be divided
    num_segments = max(3, min(15, base_segments + rng.randint(-base_segments // 2, base_segments // 2)))
    # no need to add randomness to the spacing because noise creates the randomness
    points = np.linspace(new_start, new_end, num_segments + 1)

    for i in range(1, len(points) - 1):
        noise = rng.normal(0, noise_level, size=2)
        noisy_points.append(points[i] + noise)
    noisy_points.append(np.array(new_end))

    return np.array(noisy_points)


def get_distorted_region_coords(modified_borders: Dict[Edge, Vertices], region: IndexedRegion) -> Tuple[Coords, Coords]:
    """ Return coordinates of the distorted region border. """

    y_coords, x_coords = [], []
    for j in range(len(region)):
        edge = (region[j], region[(j + 1) % len(region)])
        sorted_edge = tuple(sorted(edge))
        poly_edge = modified_borders[sorted_edge]

        if edge != sorted_edge:
            poly_edge = poly_edge[::-1]

        x_coords.extend(poly_edge[:, 0])
        y_coords.extend(poly_edge[:, 1])

    return np.array(y_coords).astype(int), np.array(x_coords).astype(int)


def improve_borders(image: Image, y_coords: Coords, x_coords: Coords, width: int, rng: RandomState, color: int,
                    voronoi_border_3d: bool, voronoi_light_direction: List[int],
                    voronoi_shading_color_range: List[int]) -> Image:
    """ Changes the border appearance - width and color. """

    border_mask = change_border_width(image, y_coords, x_coords, rng, width)

    if not voronoi_border_3d:
        new_image = image.copy()
        new_image[border_mask == 1] = color + rng.randint(-15, 15)
    else:
        grad_shading = compute_3d_border_effect(border_mask, rng, voronoi_light_direction, voronoi_shading_color_range)
        border_mask = skimage.morphology.dilation(border_mask, skimage.morphology.disk(width))
        new_image = blend_borders_w_bg(border_mask, image, grad_shading)

    return new_image


def compute_3d_border_effect(border_mask: BorderMask, rng: RandomState, voronoi_light_direction: List[int],
                             voronoi_shading_color_range: List[int]) -> GradientImage:
    """ Compute borders colors to make it look like 3D edges. """

    # todo do the 3d effect for the whole image at once?
    height_map = get_height_map(border_mask, 2)
    dx, dy = np.gradient(height_map)
    # dz points perpendicularly away from the surface
    dz = np.ones_like(height_map)
    normals = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm + 1e-8  # avoid division by zero

    # todo light direction correspondence to the emboss?
    light_dir = np.array(voronoi_light_direction)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # transform the color to specified values range
    gradient_shading = np.dot(normals, light_dir).astype(np.float32)
    gradient_shading = np.clip(gradient_shading, 0, 1)
    gradient_shading = ((gradient_shading - gradient_shading.min()) /
                        (gradient_shading.max() - gradient_shading.min() + 1e-8))

    lower, upper = [x + rng.randint(-10, 10) for x in voronoi_shading_color_range]
    gradient_shading = gradient_shading * (upper - lower) + lower
    gradient_shading = ndimage.gaussian_filter(gradient_shading, sigma=1)

    # when should be cast to int?
    return gradient_shading


def change_border_width(image: Image, y_coords: Coords, x_coords: Coords, rng: RandomState, width: int) -> BorderMask:
    """ Gradually thickens and then narrows region borders simulating natural flow. """

    border_image, border_mask = np.zeros_like(image), np.zeros_like(image)
    perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)
    border_image[perim_row, perim_col] = 1
    temp_width = int(width * 0.5)
    i = 0

    while temp_width < width:
        # triangle = np.tri(temp_width, temp_width, dtype=np.uint8)
        disk = skimage.morphology.disk(temp_width)
        n_mask = skimage.morphology.dilation(border_image, disk)

        px_len = len(perim_row) // rng.randint(6, 10)
        # seq_length = rng.randint(30 // (i + 1), 60 // (i + 1))
        perim_row, perim_col, exclude_row, exclude_col = slice_lists(perim_row, perim_col, px_len, rng)
        if len(exclude_col) > 0 and len(exclude_row) > 0:
            border_image[exclude_row, exclude_col] = 0
            border_mask = np.maximum(n_mask, border_mask)
            temp_width += 1
            i += 1
        else:
            break

    return smooth_border_mask(border_mask)


def smooth_border_mask(border_mask: BorderMask):
    # plt.imshow(border_mask, cmap="gray")
    # plt.show()
    border_mask = ndimage.gaussian_filter(border_mask, sigma=3)
    # plt.imshow(border_mask, cmap="gray")
    # plt.show()
    border_mask = border_mask > 0.4
    # plt.imshow(border_mask, cmap="gray")
    # plt.show()

    return border_mask.astype(np.int8)


def slice_lists(row: Coords, col: Coords, seq_len: int, rng: RandomState) -> Tuple[Coords, Coords, Coords, Coords]:
    """ Helper function for slicing border coordinate indices, used for gradually widening and narrowing the border. """

    new_row, new_col, removed_row, removed_col = [], [], [], []
    total_length = len(row)
    i = 0

    while i < total_length:
        sequence_length = rng.randint(10, seq_len)
        end_index = min(i + sequence_length, total_length)

        if rng.choice([True, False]):
            # Include the entire segment.
            new_row.extend(row[i:end_index])
            new_col.extend(col[i:end_index])
        else:
            # Exclude only half the segment, include the other half.
            half = (end_index - i) // 2
            removed_row.extend(row[i:i + half])
            removed_col.extend(col[i:i + half])
            new_row.extend(row[i + half:end_index])
            new_col.extend(col[i + half:end_index])
        i = end_index

    return np.array(new_row), np.array(new_col), np.array(removed_row), np.array(removed_col)


def blend_borders_w_bg(border_mask: BorderMask, image: Image, grad_shading: GradientImage) -> Image:
    """ Blends the 3d border effect image with the original image according to the height map. """
    height_map = get_height_map(border_mask, 2)
    border_mask = border_mask.astype(bool)
    new_image = image.copy()

    new_image[border_mask] = (height_map[border_mask] * grad_shading[border_mask] +
                              (1 - height_map[border_mask]) * image[border_mask])

    return new_image


def add_perlin_noise(image: Image, rng: RandomState, row: Coords, col: Coords, voronoi_perlin_noise: List[int])\
        -> Image:
    """ Adds perlin noise the whole background. """

    new_image = image.copy()
    alpha = rng.randint(*voronoi_perlin_noise) / 100
    divisor = rng.choice((2, 4))
    res = (image.shape[0] // divisor, image.shape[1] // divisor)

    height, width = image.shape
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)
    # * 127.5 so the noise mean is 0
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude

    return new_image


def get_height_map(border_mask: BorderMask, sigma: float) -> Image:
    """ Computes an image with high values in the center of voronoi borders which gradually gets smaller allowing to use
    it as an alpha mask for blending the voronoi borders to the image. """

    distance_map = ndimage.distance_transform_edt(border_mask)
    # dm = (distance_map / distance_map.max() * 255).astype(np.uint8)
    height_map = ndimage.gaussian_filter(distance_map, sigma=sigma)
    # hm = cv2.bilateralFilter(dm, 5, 10, 5)
    # plt.imshow(height_map, cmap="gray")
    # plt.show()

    # normalize the height map to range [0, 1]
    max_distance = np.max(height_map)
    height_map = height_map / max_distance if max_distance > 0 else height_map
    # plt.imshow(height_map, cmap="gray")
    # plt.show()

    return height_map
