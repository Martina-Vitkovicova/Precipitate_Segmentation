import hashlib
import random
from typing import List, Tuple, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import skimage
import skimage.draw as skdraw

from perlin_numpy import generate_perlin_noise_2d
from scipy.ndimage import distance_transform_edt

from src.generators.finite_voronoi_gen import get_finite_voronoi
from src.type_definitions import (Image, IndexedRegion, Vertices, Edge, Point, RandomState, Coords, BorderMask,
                                  GradientImage)



def voronoi_bg(seed: int, *, img_size: List[int], light_direction: List[int], regions_num_range: List[int],
               color_range: List[int], border_width: int, change_border_width: bool, border_jaggedness: float,
               border_noise: float, border_3d: bool, shading_color_range: List[int], perlin_noise: bool,
               perlin_noise_alpha_range: List[int], perlin_noise_scale_range: List[int],
               gaussian_noise: bool, gaussian_noise_alpha_range: List[int], porous_texture: bool,
               porous_texture_density_range: List[float], porous_texture_blur_range: List[float],
               porous_texture_intensity_range: List[float], porous_texture_size_range: List[int], **kwargs) -> Image:
    """
    Creates a background matrix image consisting of regions (grains) with borders.

    Parameters
    ----------
    seed:
        int number used for setting random generators.
    img_size:
        [width, height] dimensions of the generated image.
    regions_num_range:
        [min, max] number of base voronoi regions (the total region count may be a little different due to truncating
        and adding regions which were not closed before).
    color_range:
        [darker, lighter] color range for the regions. Each region has individually generated colour.
    border_width:
        int number for base width of the region borders. The borders are processed and widened, so this number is the
        minimum border width in the image.
    change_border_width:
    border_jaggedness:
    border_noise:
    border_3d:
    light_direction:
    shading_color_range:
    porous_texture:
    kwargs:
        other parameters not used for the background generation.

    Returns
    -------
    Numpy 2D ndarray image.
    """
    rng = np.random.RandomState(seed)
    perlin_rng = np.random.RandomState(seed + 1)
    borders_rng = np.random.RandomState(seed + 2)
    gauss_rng = np.random.RandomState(seed + 3)
    texture_rng = np.random.RandomState(seed + 4)

    width, height = img_size
    # image = np.zeros((height, width), dtype=np.float32)
    image = np.full((height, width), 140, dtype=np.float32)

    num_points = rng.randint(*regions_num_range)
    region_generator_points = np.column_stack((rng.uniform(0, width, num_points),
                                               rng.uniform(0, height, num_points)))
    vor = scipy.spatial.Voronoi(region_generator_points)
    regions, vertices = get_finite_voronoi(vor)

    modified_borders = distort_borders(regions, vertices, img_size, border_jaggedness, border_noise)
    border_mask = np.zeros_like(image)

    for region in regions:
        y_coords, x_coords = get_distorted_region_coords(modified_borders, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*color_range)
        image[row, col] = color

        if perlin_noise:
            image = add_perlin_noise(image, perlin_rng, row, col, perlin_noise_alpha_range,
                                     perlin_noise_scale_range)
        if gaussian_noise:
            image = add_gaussian_noise(image, gauss_rng, row, col, gaussian_noise_alpha_range)

        perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)
        border_mask[perim_row, perim_col] = 1

    if porous_texture:
        image = add_porous_texture(image, texture_rng, light_direction, porous_texture_density_range,
                                   porous_texture_blur_range, porous_texture_intensity_range,
                                   porous_texture_size_range)

    image = modify_borders(image, border_mask, border_width, change_border_width, borders_rng,
                           130, border_3d, light_direction, shading_color_range)

    return image


def generate_distant_points(rng, min_distance, width, height):
    while True:
        p1, p2 = np.column_stack((rng.uniform(0, width, 2), rng.uniform(0, height, 2)))
        distance = np.linalg.norm(p1 - p2)

        if distance > min_distance:
            return p1, p2


def distort_borders(regions: List[IndexedRegion], vertices: Vertices, img_size: List[int],
                    border_jaggedness: float, border_noise: float) -> Dict[Edge, Vertices]:
    """ Transforms borders consisting of straight lines to jagged polylines. """

    modified_borders = {}

    for region in regions:
        for i in range(len(region)):
            edge = (region[i], region[(i + 1) % len(region)])
            sorted_edge = tuple(sorted(edge))

            if sorted_edge not in modified_borders:
                edge_seed = get_edge_seed(sorted_edge)
                edge_rng = np.random.RandomState(edge_seed)
                v1, v2 = (tuple(int(point) for point in vertices[i])
                          for i in sorted_edge)
                modified_borders[sorted_edge] = divide_edge(v1, v2, edge_rng, img_size, border_jaggedness,
                                                            border_noise)

    return modified_borders


def get_edge_seed(edge: Edge) -> int:
    """ Create a unique seed to set a random generator with. """

    edge_bytes = str(edge).encode('utf-8')
    return int(hashlib.sha256(edge_bytes).hexdigest(), 16) % (2 ** 32)


def divide_edge(start: Point, end: Point, rng: RandomState, img_size: List[int], border_jaggedness: float,
                border_noise: float) -> Vertices:
    """ Divides a straight line given by start and end point to number of segments and adds noise to the coordinates
    to make the new line jagged. """

    # if the line doesn't lie inside the image area, it's not necessary to divide it
    # if it lies partially, we divide only the part of the line which lies in the area
    is_inside, new_start, new_end = cv2.clipLine((0, 0, *img_size), start, end)
    if not is_inside:
        return np.array([np.array(start), np.array(end)])

    noisy_points = [np.array(new_start)]

    line_length = np.linalg.norm(np.array(new_end) - np.array(new_start))

    base_segments = max(1, int(np.sqrt(line_length) * border_jaggedness))
    # changes in x, y coords to make the border jagged
    if border_noise > 0:
        noise_level = max(2, min(30, int(np.sqrt(line_length) * border_noise) + rng.randint(-2, 2)))
    else:
        noise_level = 0
    # to how many segments will the line be divided
    num_segments = max(2, min(15, base_segments + rng.randint(-base_segments // 2, base_segments // 2)))
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


def modify_borders(image: Image, border_mask: BorderMask, width: int, change_width: bool,
                   rng: RandomState, color: int, border_3d: bool, light_direction: List[int],
                   shading_color_range: List[int]) -> Image:
    """ Changes the border appearance - width and color. """

    border_mask = change_border_width(border_mask, rng, width, change_width)

    if not border_3d:
        new_image = image.copy()
        new_image[border_mask > 0] = color + rng.randint(-20, -19)
    else:
        grad_shading = compute_3d_border_effect(border_mask, rng, light_direction, shading_color_range)
        border_mask = skimage.morphology.dilation(border_mask, skimage.morphology.disk(width))
        new_image = blend_borders_w_bg(border_mask, image, grad_shading)

    return new_image


def change_border_width(border_mask, rng: RandomState, width: int, change_width: bool):
    if not change_width:
        disk = skimage.morphology.disk(width)
        return skimage.morphology.dilation(border_mask, disk)

    min_width, max_width = 0.5, width * 2

    contours, _ = cv2.findContours(border_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    width_map = np.zeros(border_mask.shape, dtype=np.float32)

    for contour in contours:
        contour = contour.squeeze()
        length = contour.shape[0]
        if length < 3:
            continue
        num_control_points = int(np.sqrt(length) * 0.7)

        xs = contour[:, 0].astype(int)
        ys = contour[:, 1].astype(int)
        widths = generate_widths(length, rng, num_control_points, min_width, max_width)
        width_map[ys, xs] = widths
        # plt.imshow(width_map)
        # plt.show()

    border_distance, (closest_border_y, closest_border_x) = distance_transform_edt(1 - border_mask, return_indices=True)
    # plt.imshow(dt)
    # plt.show()

    half_distance_band = (width_map[closest_border_y, closest_border_x] * 0.5)
    thickened_borders = (border_distance <= half_distance_band).astype(np.uint8) * 255
    thickened_borders = ndimage.gaussian_filter(thickened_borders, sigma=width//2, truncate=0.6)

    return thickened_borders


def generate_widths(length, rng: RandomState, num_control_points, min_width, max_width):
    control_pts_x = np.linspace(0, length - 1, num_control_points)
    control_pts_y = rng.uniform(min_width, max_width, num_control_points)
    xs = np.arange(length)

    widths = np.interp(xs, control_pts_x, control_pts_y)

    return widths


def compute_3d_border_effect(border_mask: BorderMask, rng: RandomState, light_direction: List[int],
                             shading_color_range: List[int]) -> GradientImage:
    """ Compute borders colors to make it look like 3D edges. """

    # todo do the 3d effect for the whole image at once?
    height_map = get_height_map(border_mask, 1.5)
    dx, dy = np.gradient(height_map)
    # dz points perpendicularly away from the surface
    dz = np.ones_like(height_map)
    normals = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm + 1e-8  # avoid division by zero

    # [x, y, z]: -x -> light from right; y -> light from top; large z -> light from top
    light_dir = np.array(light_direction)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # transform the color to specified values range
    gradient_shading = np.dot(normals, light_dir).astype(np.float32)
    gradient_shading = np.clip(gradient_shading, 0, 1)
    gradient_shading = ((gradient_shading - gradient_shading.min()) /
                        (gradient_shading.max() - gradient_shading.min() + 1e-8))

    gradient_shading = ndimage.gaussian_filter(gradient_shading, sigma=0.35)

    lower, upper = [x + rng.randint(-5, 5) for x in shading_color_range]
    gradient_shading = gradient_shading * (upper - lower) + lower
    # gradient_shading = ndimage.gaussian_filter(gradient_shading, sigma=1)

    # when should be cast to int?
    return gradient_shading


def blend_borders_w_bg(border_mask: BorderMask, image: Image, grad_shading: GradientImage) -> Image:
    """ Blends the 3d border effect image with the original image according to the height map. """
    height_map = get_height_map(border_mask, 1.5)
    border_mask = border_mask.astype(bool)
    new_image = image.copy()

    new_image[border_mask] = (height_map[border_mask] * grad_shading[border_mask] +
                              (1 - height_map[border_mask]) * image[border_mask])

    return new_image


def add_perlin_noise(image: Image, rng: RandomState, row: Coords, col: Coords, perlin_noise: List[int],
                     perlin_noise_scale) -> Image:
    """ Adds perlin noise the individual cells. """
    new_image = image.copy()
    alpha = rng.randint(*perlin_noise) / 100
    divisor = rng.choice(perlin_noise_scale)
    res = (image.shape[0] // divisor, image.shape[1] // divisor)

    height, width = image.shape
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)  # range ~ [-1, 1]
    noise = noise - noise.mean()
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude

    return new_image


def add_gaussian_noise(image: Image, rng: RandomState, row: Coords, col: Coords,
                       gaussian_noise: List[int]) -> Image:
    new_image = image.copy()
    noise = rng.normal(0, 1, (1025, 1025))
    alpha = rng.randint(*gaussian_noise) * 0.03
    noise = noise - noise.mean()
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude
    return new_image


def add_porous_texture(image: Image, rng: RandomState, light_dir, density_range: List[float],
                       blur_size_range: List[float], intensity_range: List[float], size_range: List[float]) -> Image:
    h, w = image.shape

    density = rng.uniform(*density_range)
    blur = rng.uniform(*blur_size_range)
    intensity = rng.uniform(*intensity_range)
    texture = _draw_impulses(h, w, rng, density, size_range)

    light_tex = ndimage.gaussian_filter(texture, blur)
    light_tex -= light_tex.mean()

    dark_tex = ndimage.gaussian_filter(texture, blur * 1.1)
    dx, dy = _compute_shift(light_dir, size_range)
    dark_tex = ndimage.shift(dark_tex, shift=(dy, dx), mode="constant", cval=0.0, order=1)
    dark_tex = -1 * dark_tex  # make dark areas stronger after blur
    dark_tex -= dark_tex.mean()

    combined = (light_tex + dark_tex) * intensity
    return image + combined


def _compute_shift(light_dir, size_range):
    light_x, light_y, _ = light_dir
    light_x, light_y = -light_x, -light_y  # invert axis to match matrix shading
    light_vec = np.array([light_x, light_y], dtype=np.float32)
    light_vec /= np.linalg.norm(light_vec) + 1e-6

    shift_px = (size_range[0] + size_range[1]) / 2.0 + 2.0
    dx = float(light_vec[0] * shift_px)
    dy = float(light_vec[1] * shift_px)

    return dx, dy


def _draw_impulses(h, w, rng, density, size_range):
    texture = np.zeros((h, w), dtype=np.float32)
    num = int(rng.poisson(density * h * w))
    if num <= 0:
        return texture

    y_impulse_coords = rng.randint(0, h, size=num)
    x_impulse_coords = rng.randint(0, w, size=num)
    radii = rng.uniform(size_range[0], size_range[1], size=num)
    colors = rng.lognormal(mean=0.0, sigma=0.4, size=num).astype(np.float32)

    for y, x, r, c in zip(y_impulse_coords, x_impulse_coords, radii, colors):
        radius = int(round(r))
        color = (float(c), float(c), float(c))
        if radius > 0:
            cv2.circle(texture, (int(x), int(y)), radius, color, thickness=-1)

    return texture


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
