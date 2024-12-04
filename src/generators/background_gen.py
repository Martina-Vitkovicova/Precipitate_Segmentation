import hashlib
from typing import List, Tuple, Dict, cast
import cv2
import numpy as np
import scipy
from scipy import ndimage
import skimage
import skimage.draw as skdraw

from perlin_numpy import generate_perlin_noise_2d
from src.external_libs.pv_voronoi_gist.colorized_voronoi import voronoi_finite_polygons_2d
from src.type_definitions import (Image, IndexedRegion, Vertices, Edge, Point, RandomState, Coords, BorderMask,
                                  GradientImage)


def voronoi_bg(seed: int, *, img_size: List[int], voronoi_regions_num_range: List[int],
               voronoi_color_range: List[int], voronoi_border_width: int, **kwargs) -> Image:
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
    kwargs:
        other parameters not used for the background generation.

    Returns
    -------
    Numpy 2D ndarray image.
    """
    rng = np.random.RandomState(seed)
    width, height = img_size
    # image = np.zeros((height, width), dtype=np.float32)
    image = np.full((height, width), 160, dtype=np.float32)
    num_points = rng.randint(*voronoi_regions_num_range)
    region_generator_points = np.column_stack((rng.uniform(0, width, num_points),
                                               rng.uniform(0, height, num_points)))

    vor = scipy.spatial.Voronoi(region_generator_points)
    regions, vertices = cast((List[IndexedRegion], Vertices), voronoi_finite_polygons_2d(vor))
    modified_borders = distort_borders(regions, vertices, width, height)

    for region in regions:
        y_coords, x_coords = get_distorted_region_coords(modified_borders, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*voronoi_color_range)
        image[row, col] = color

        image = improve_borders(image, y_coords, x_coords, voronoi_border_width, rng, color, add_3d_effect=True)
        image = add_perlin_noise(image, rng, row, col)

    return image


def distort_borders(regions: List[IndexedRegion], vertices: Vertices, width: int, height: int) -> Dict[Edge, Vertices]:
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

                line_length = np.linalg.norm(v2 - v1)
                # todo set a parameter for num of segments aka border jaggedness
                base_segments = max(2, int(np.sqrt(line_length) * 0.2))
                # to how many segments will the line be divided
                num_segments = max(4, min(12, base_segments + edge_rng.randint(-base_segments // 2, base_segments // 2)))
                # changes in x, y coords to make the border jagged
                noise = max(5, min(10, int(np.sqrt(line_length) * 0.3) + edge_rng.randint(-2, 2)))

                v1, v2 = tuple(map(int, v1)), tuple(map(int, v2))
                modified_borders[sorted_edge] = divide_edge(v1, v2, edge_rng, width, height, num_segments, noise)

    return modified_borders


def get_edge_seed(edge: Edge) -> int:
    """ Create a unique seed to set a random generator with. """

    edge_bytes = str(edge).encode('utf-8')
    return int(hashlib.sha256(edge_bytes).hexdigest(), 16) % (2 ** 32)


def divide_edge(start: Point, end: Point, rng: RandomState, width: int, height: int, num_segments: int = 10,
                noise_level: int = 10) -> Vertices:
    """ Divides a straight line given by start and end point to number of segments and adds noise to the coordinates
    to make the new line jagged. """

    # if the line doesn't lie inside the image area, it's not necessary to divide it
    # if it lies partially, we divide only the part of the line which lies in the area
    is_inside, new_start, new_end = cv2.clipLine((0, 0, width, height), start, end)
    if not is_inside:
        return np.array([np.array(start), np.array(end)])

    noisy_points = [np.array(start)]
    # no need to add randomness to the spacing because noise creates the randomness
    points = np.linspace(new_start, new_end, num_segments + 1)

    for i in range(len(points)):
        noise = rng.normal(0, noise_level, size=2)
        noisy_points.append(points[i] + noise)
    noisy_points.append(np.array(end))

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
                    add_3d_effect: bool) -> Image:
    """ Gradually thickens and then narrows region borders simulating natural flow. """

    perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)
    border_image, border_mask = np.zeros_like(image), np.zeros_like(image)
    border_image[perim_row, perim_col] = 1
    max_width = 6 - width
    new_image = image.copy()

    for i in range(max_width):
        triangle = np.tri(width, width, dtype=np.uint8)
        n_mask = skimage.morphology.dilation(border_image, triangle)
        seq_length = rng.randint(50 // (i + 1), 100 // (i + 1))
        perim_row, perim_col, exclude_row, exclude_col = slice_lists(perim_row, perim_col, seq_length, rng)
        border_image[exclude_row, exclude_col] = 0
        border_mask = np.maximum(n_mask, border_mask)
        width += 1

    border_mask.astype(int)
    if not add_3d_effect:
        new_image[border_mask == 1] = color + rng.randint(-15, 15)
    else:
        # more dilation for smooth transition from edge to region
        border_mask = skimage.morphology.dilation(border_mask, np.tri(5, 5, dtype=np.uint8))
        grad_shading = compute_3d_border_effect(border_mask)
        new_image = blend_borders_w_bg(border_mask, new_image, grad_shading)

    return new_image


def compute_3d_border_effect(border_mask: BorderMask) -> GradientImage:
    """ Compute borders colors to make it look like 3D edges. """

    # todo do the 3d effect for the whole image at once?
    height_map = get_height_map(border_mask, 0.4)
    dx, dy = np.gradient(height_map)
    # dz points perpendicularly away from the surface
    dz = np.ones_like(height_map)
    normals = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm + 1e-8  # avoid division by zero

    # todo light direction correspondence to the emboss?
    light_dir = np.array([-1, -1, 2])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # transform the color to specified values range
    gradient_shading = np.dot(normals, light_dir).astype(np.float32)
    gradient_shading = np.clip(gradient_shading, 0, 1)
    gradient_shading = ((gradient_shading - gradient_shading.min()) /
                        (gradient_shading.max() - gradient_shading.min() + 1e-8))
    lower, upper = 100, 200
    gradient_shading = gradient_shading * (upper - lower) + lower

    # when should be cast to int?
    return gradient_shading


def slice_lists(row: Coords, col: Coords, seq_len: int, rng: RandomState) -> Tuple[Coords, Coords, Coords, Coords]:
    """ Helper function for slicing border coordinate indices, used for gradually widening and narrowing the border. """

    new_row, new_col, removed_row, removed_col = [], [], [], []
    total_length = len(row)
    i = 0

    while i < total_length:
        sequence_length = rng.randint(10, seq_len)
        if rng.choice([True, False]):
            # include these
            end_index = min(i + sequence_length, total_length)
            new_row.extend(row[i:end_index])
            new_col.extend(col[i:end_index])
        else:
            # pause - exclude these
            end_index = min(i + sequence_length, total_length)
            removed_row.extend(row[i:end_index])
            removed_col.extend(col[i:end_index])
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


def add_perlin_noise(image: Image, rng: RandomState, row: Coords, col: Coords) -> Image:
    """ Adds perlin noise the whole background. """

    new_image = image.copy()
    alpha = max(0.05, min(0.15, rng.random()))
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
    max_distance = np.max(distance_map)

    # normalize the height map to range [0, 1]
    height_map = distance_map / max_distance if max_distance > 0 else distance_map
    height_map = ndimage.gaussian_filter(height_map, sigma=sigma)

    return height_map
