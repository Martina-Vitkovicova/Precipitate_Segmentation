import hashlib
from typing import List, Tuple, Dict
import cv2
import numpy as np
from scipy import ndimage
import skimage
import skimage.draw as skdraw

from perlin_numpy import generate_perlin_noise_2d
from scipy.ndimage import distance_transform_edt

from src.utils.config_parsing import GeneralParams, MatrixParams, TextureParams
from src.generators.finite_voronoi_gen import get_finite_voronoi
from src.utils.type_definitions import (Image, IndexedRegion, Vertices, Edge, Point, RandomState, Coords, BorderMask,
                                        GradientImage, IntPair, IntRange, Vec3i)


def generate_matrix(matrix_rng: RandomState, general_params: GeneralParams, params: MatrixParams) -> Image:
    """
    Generate a grayscale image representing grain-like metal matrix background.
    The image is created by tessellating the image domain into Voronoi regions that act as grains.

    Controlled by MatrixParams for grain intensity, border distortion, border width and grain texture.
    GeneralParams supplies the global image size and light direction.

    Stochastic behaviour is fully determined by `matrix_rng`.
    """
    perlin_rng, borders_rng, gauss_rng, texture_rng = split_random_state(matrix_rng, 4)

    img_size = general_params.image_size
    width, height = img_size
    image = np.full((height, width), 140, dtype=np.float32)
    border_mask = np.zeros_like(image)

    num_points = matrix_rng.randint(*params.grains_num_range)
    region_generator_points = np.column_stack((matrix_rng.uniform(0, width, num_points),
                                               matrix_rng.uniform(0, height, num_points)))
    regions, vertices = get_finite_voronoi(region_generator_points)

    modified_borders = distort_borders(regions, vertices, img_size, params.border_jaggedness,
                                       params.border_perturbation_noise)

    for region in regions:
        y_coords, x_coords = get_distorted_region_coords(modified_borders, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        intensity = matrix_rng.randint(*params.grains_intensity_range)
        image[row, col] = intensity

        if params.perlin_noise_enabled:
            image = _add_perlin_noise(image, perlin_rng, row, col, params.perlin_noise_alpha_range,
                                      params.perlin_noise_scales)
        if params.gaussian_noise_enabled:
            image = _add_gaussian_noise(image, gauss_rng, row, col, params.gaussian_noise_alpha_range)

        perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)
        border_mask[perim_row, perim_col] = 1

    if params.porous_texture_enabled:
        porous_texture_params = TextureParams(params.porous_texture_density_range, params.porous_texture_blur_range,
                                              params.porous_texture_intensity_range, params.porous_texture_size_range)
        image = add_porous_texture(image, texture_rng, general_params.light_direction, porous_texture_params)

    image = _modify_borders(image, border_mask, params.border_width, params.border_variable_width_enabled, borders_rng,
                            130, params.border_shading_enabled, general_params.light_direction,
                            params.border_shading_intensity_range)

    return image


def split_random_state(parent_rng: RandomState, num: int) -> List[RandomState]:
    """Derive independent RNG streams from a parent RNG."""
    return [np.random.RandomState(parent_rng.randint(0, 10 ** 9)) for _ in range(num)]


def distort_borders(regions: List[IndexedRegion], vertices: Vertices, img_size: IntPair,
                    border_jaggedness: float, border_noise: float) -> Dict[Edge, Vertices]:
    """ Transforms borders consisting of straight lines to jagged polylines. """
    modified_borders: dict[Edge, Vertices] = {}

    for region in regions:
        for i in range(len(region)):
            edge = (region[i], region[(i + 1) % len(region)])
            sorted_edge: Edge = (min(edge), max(edge))

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


def divide_edge(start: Point, end: Point, rng: RandomState, img_size: IntPair, border_jaggedness: float,
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
    """ Return coordinates of the distorted region border as integer arrays. """
    y_coords, x_coords = [], []
    for j in range(len(region)):
        edge = (region[j], region[(j + 1) % len(region)])
        sorted_edge: Edge = (min(edge), max(edge))
        poly_edge = modified_borders[sorted_edge]

        if edge != sorted_edge:
            poly_edge = poly_edge[::-1]

        x_coords.extend(poly_edge[:, 0])
        y_coords.extend(poly_edge[:, 1])

    return np.array(y_coords).astype(int), np.array(x_coords).astype(int)


def _modify_borders(image: Image, border_mask: BorderMask, width: int, change_width: bool,
                    rng: RandomState, intensity: int, border_3d: bool, light_direction: Vec3i,
                    shading_intensity_range: IntRange) -> Image:
    """ Modify border appearance by thickening and optional directional shading.
    `border_mask` is expected to be uint8 with non-zero border pixels."""
    border_mask = _change_border_width(border_mask, rng, width, change_width)

    if not border_3d:
        new_image = image.copy()
        new_image[border_mask > 0] = intensity + rng.randint(-20, -19)
    else:
        grad_shading = _compute_3d_border_effect(border_mask, rng, light_direction, shading_intensity_range)
        border_mask = skimage.morphology.dilation(border_mask, skimage.morphology.disk(width))
        new_image = _blend_borders_w_bg(border_mask, image, grad_shading)

    return new_image


def _change_border_width(border_mask, rng: RandomState, width: int, change_width: bool) -> BorderMask:
    """Thicken borders with either fixed or locally varying width."""
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
        widths = _generate_widths(length, rng, num_control_points, min_width, max_width)
        width_map[ys, xs] = widths

    border_distance, (closest_border_y, closest_border_x) = distance_transform_edt(1 - border_mask, return_indices=True)

    half_distance_band = (width_map[closest_border_y, closest_border_x] * 0.5)
    thickened_borders = (border_distance <= half_distance_band).astype(np.uint8) * 255
    thickened_borders = ndimage.gaussian_filter(thickened_borders, sigma=width // 2, truncate=0.6)

    return thickened_borders


def _generate_widths(length: int, rng: RandomState, num_control_points: int, min_width: float, max_width: float) \
        -> np.ndarray:
    """ Generate smoothly varying border widths along a border of given length. """
    control_pts_x = np.linspace(0, length - 1, num_control_points)
    control_pts_y = rng.uniform(min_width, max_width, num_control_points)
    xs = np.arange(length)

    widths = np.interp(xs, control_pts_x, control_pts_y)

    return widths


def _compute_3d_border_effect(border_mask: BorderMask, rng: RandomState, light_direction: Vec3i,
                              shading_intensity_range: IntRange) -> GradientImage:
    """ Compute directional shading on borders using a height-map normal approximation. """
    height_map = _get_height_map(border_mask, 1.5)
    dx, dy = np.gradient(height_map)
    # dz points perpendicularly away from the surface
    dz = np.ones_like(height_map)
    normals = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm + 1e-8  # avoid division by zero

    # [x, y, z]: -x -> light from right; y -> light from top; large z -> light from top
    light_dir = np.array(light_direction)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # transform the intensity to specified values range
    gradient_shading = np.dot(normals, light_dir).astype(np.float32)
    gradient_shading = np.clip(gradient_shading, 0, 1)
    gradient_shading = ((gradient_shading - gradient_shading.min()) /
                        (gradient_shading.max() - gradient_shading.min() + 1e-8))

    gradient_shading = ndimage.gaussian_filter(gradient_shading, sigma=0.35)

    lower, upper = [x + rng.randint(-5, 5) for x in shading_intensity_range]
    gradient_shading = gradient_shading * (upper - lower) + lower

    return gradient_shading


def _blend_borders_w_bg(border_mask: BorderMask, image: Image, grad_shading: GradientImage) -> Image:
    """ Blend shaded borders into the image using the height map as alpha. """
    height_map = _get_height_map(border_mask, 1.5)
    border_mask = border_mask.astype(bool)
    new_image = image.copy()

    new_image[border_mask] = (height_map[border_mask] * grad_shading[border_mask] +
                              (1 - height_map[border_mask]) * image[border_mask])

    return new_image


def _add_perlin_noise(image: Image, rng: RandomState, row: Coords, col: Coords, perlin_noise_alpha: IntRange,
                      perlin_noise_scales: List[int]) -> Image:
    """ Adds perlin noise the individual cells. """
    new_image = image.copy()
    alpha = rng.randint(*perlin_noise_alpha) / 100
    divisor = rng.choice(perlin_noise_scales)
    res = (image.shape[0] // divisor, image.shape[1] // divisor)

    height, width = image.shape
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)  # range ~ [-1, 1]
    noise = noise - noise.mean()
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude

    return new_image


def _add_gaussian_noise(image: Image, rng: RandomState, row: Coords, col: Coords,
                        gaussian_noise_alpha: IntRange) -> Image:
    """ Add zero-mean Gaussian noise inside a region mask. """
    new_image = image.copy()
    noise = rng.normal(0, 1, (1025, 1025))
    alpha = rng.randint(*gaussian_noise_alpha) * 0.03
    noise = noise - noise.mean()
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude
    return new_image


def add_porous_texture(image: Image, rng: RandomState, light_dir: Vec3i, params: TextureParams) -> Image:
    """ Add porous-like impulse texture with a directional dark/light offset derived from `light_dir`. """
    h, w = image.shape

    density = rng.uniform(*params.density_range)
    blur = rng.uniform(*params.blur_range)
    intensity = rng.uniform(*params.intensity_range)
    texture = _draw_impulses(h, w, rng, density, params.size_range)

    light_tex = ndimage.gaussian_filter(texture, blur)
    light_tex -= light_tex.mean()

    dark_tex = ndimage.gaussian_filter(texture, blur * 1.1)
    dx, dy = _compute_shift(light_dir, params.size_range)
    dark_tex = ndimage.shift(dark_tex, shift=(dy, dx), mode="constant", cval=0.0, order=1)
    dark_tex = -1 * dark_tex  # make dark areas stronger after blur
    dark_tex -= dark_tex.mean()

    combined = (light_tex + dark_tex) * intensity
    return image + combined


def _compute_shift(light_dir: Vec3i, size_range: IntRange) -> Tuple[float, float]:
    """ Compute directional shift for dark texture based on light direction and average impulse size. """
    light_x, light_y, _ = light_dir
    light_x, light_y = -light_x, -light_y  # invert axis to match matrix shading
    light_vec = np.array([light_x, light_y], dtype=np.float32)
    light_vec /= np.linalg.norm(light_vec) + 1e-6

    shift_px = (size_range[0] + size_range[1]) / 2.0 + 2.0
    dx = float(light_vec[0] * shift_px)
    dy = float(light_vec[1] * shift_px)

    return dx, dy


def _draw_impulses(h: int, w: int, rng: RandomState, density: float, size_range: IntRange) -> Image:
    """ Draw random circular impulses on a blank image. """
    texture = np.zeros((h, w), dtype=np.float32)
    num = int(rng.poisson(density * h * w))
    if num <= 0:
        return texture

    y_impulse_coords = rng.randint(0, h, size=num)
    x_impulse_coords = rng.randint(0, w, size=num)
    radii = rng.uniform(size_range[0], size_range[1], size=num)
    intensities = rng.lognormal(mean=0.0, sigma=0.4, size=num).astype(np.float32)

    for y, x, r, c in zip(y_impulse_coords, x_impulse_coords, radii, intensities):
        radius = int(round(r))
        intensity = (float(c), float(c), float(c))
        if radius > 0:
            cv2.circle(texture, (int(x), int(y)), radius, intensity, thickness=-1)

    return texture


def _get_height_map(border_mask: BorderMask, sigma: float) -> Image:
    """ Computes an image peaking at the center of voronoi borders which gradually get smaller. Used as alpha mask
    for blending the voronoi borders to the image. """

    distance_map = ndimage.distance_transform_edt(border_mask)
    height_map = ndimage.gaussian_filter(distance_map, sigma=sigma)

    # normalize the height map to range [0, 1]
    max_distance = np.max(height_map)
    height_map = height_map / max_distance if max_distance > 0 else height_map

    return height_map
