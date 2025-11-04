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


def voronoi_bg(seed: int, *, img_size: List[int], voronoi_regions_num_range: List[int], voronoi_color_range: List[int],
               voronoi_border_width: int, voronoi_change_border_width: bool, voronoi_border_jaggedness: float,
               voronoi_border_noise: float, voronoi_border_3d: bool, voronoi_light_direction: List[int],
               voronoi_shading_color_range: List[int], voronoi_perlin_noise_alpha_range: List[int],
               voronoi_perlin_noise_scale_range: List[int], voronoi_gaussian_noise_alpha_range: List[int],
               voronoi_porous_texture: bool, voronoi_porous_texture_density_range: List[float],
               voronoi_porous_texture_blur_range: List[float], voronoi_porous_texture_intensity_range: List[float],
               voronoi_porous_texture_size_range: List[float], **kwargs) -> Image:
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
    voronoi_change_border_width:
    voronoi_border_jaggedness:
    voronoi_border_noise:
    voronoi_border_3d:
    voronoi_light_direction:
    voronoi_shading_color_range:
    voronoi_perlin_noise:
    voronoi_perlin_noise_scale:
    voronoi_gaussian_noise:
    voronoi_porous_texture:
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

    num_points = rng.randint(*voronoi_regions_num_range)
    region_generator_points = np.column_stack((rng.uniform(0, width, num_points),
                                               rng.uniform(0, height, num_points)))
    vor = scipy.spatial.Voronoi(region_generator_points)
    regions, vertices = get_finite_voronoi(vor)

    modified_borders = distort_borders(regions, vertices, img_size, voronoi_border_jaggedness, voronoi_border_noise)
    border_mask = np.zeros_like(image)

    for region in regions:
        y_coords, x_coords = get_distorted_region_coords(modified_borders, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*voronoi_color_range)
        image[row, col] = color
        image = add_perlin_noise(image, perlin_rng, row, col, voronoi_perlin_noise_alpha_range,
                                 voronoi_perlin_noise_scale_range)
        # if gauss_rng.randint(1, 5) > 3:
        #     image = add_gaussian_noise(image, gauss_rng, row, col, voronoi_gaussian_noise)

        perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)
        border_mask[perim_row, perim_col] = 1

    image = modify_borders(image, border_mask, voronoi_border_width, voronoi_change_border_width, borders_rng,
                           130, voronoi_border_3d, voronoi_light_direction,
                           voronoi_shading_color_range)
    if voronoi_porous_texture:
        image = add_porous_texture(image, texture_rng, True, voronoi_porous_texture_density_range,
                                   voronoi_porous_texture_blur_range, voronoi_porous_texture_intensity_range, voronoi_porous_texture_size_range)
        image = add_porous_texture(image, texture_rng, False, list(np.array(voronoi_porous_texture_density_range)*0.7),
                                   voronoi_porous_texture_blur_range, list(np.array(voronoi_porous_texture_intensity_range)*0.7),
                                   list(np.array(voronoi_porous_texture_size_range)*0.7))

    return image


def generate_distant_points(rng, min_distance, width, height):
    while True:
        p1, p2 = np.column_stack((rng.uniform(0, width, 2), rng.uniform(0, height, 2)))
        distance = np.linalg.norm(p1 - p2)

        if distance > min_distance:
            return p1, p2


def distort_borders(regions: List[IndexedRegion], vertices: Vertices, img_size: List[int],
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
                v1, v2 = (tuple(int(point) for point in vertices[i])
                          for i in sorted_edge)
                modified_borders[sorted_edge] = divide_edge(v1, v2, edge_rng, img_size, voronoi_border_jaggedness,
                                                            voronoi_border_noise)

    return modified_borders


def get_edge_seed(edge: Edge) -> int:
    """ Create a unique seed to set a random generator with. """

    edge_bytes = str(edge).encode('utf-8')
    return int(hashlib.sha256(edge_bytes).hexdigest(), 16) % (2 ** 32)


def divide_edge(start: Point, end: Point, rng: RandomState, img_size: List[int], voronoi_border_jaggedness: float,
                voronoi_border_noise: float) -> Vertices:
    """ Divides a straight line given by start and end point to number of segments and adds noise to the coordinates
    to make the new line jagged. """

    # if the line doesn't lie inside the image area, it's not necessary to divide it
    # if it lies partially, we divide only the part of the line which lies in the area
    is_inside, new_start, new_end = cv2.clipLine((0, 0, *img_size), start, end)
    if not is_inside:
        # todo should I clip this?
        return np.array([np.array(start), np.array(end)])

    noisy_points = [np.array(new_start)]

    line_length = np.linalg.norm(np.array(new_end) - np.array(new_start))

    base_segments = max(1, int(np.sqrt(line_length) * voronoi_border_jaggedness))
    # changes in x, y coords to make the border jagged
    if voronoi_border_noise > 0:
        noise_level = max(2, min(30, int(np.sqrt(line_length) * voronoi_border_noise) + rng.randint(-2, 2)))
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


def modify_borders(image: Image, border_mask: BorderMask, width: int, change_width: str,
                   rng: RandomState, color: int, voronoi_border_3d: bool, voronoi_light_direction: List[int],
                   voronoi_shading_color_range: List[int]) -> Image:
    """ Changes the border appearance - width and color. """

    border_mask = change_border_width(border_mask, rng, width, change_width)

    if voronoi_border_3d == "False":
        new_image = image.copy()
        new_image[border_mask > 0] = color + rng.randint(-15, -14)
    else:
        grad_shading = compute_3d_border_effect(border_mask, rng, voronoi_light_direction, voronoi_shading_color_range)
        border_mask = skimage.morphology.dilation(border_mask, skimage.morphology.disk(width))
        new_image = blend_borders_w_bg(border_mask, image, grad_shading)

    return new_image


def change_border_width(border_mask, rng: RandomState, width: int, change_width: str):
    if change_width == "False":
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
        num_control_points = int(np.sqrt(length)) // 2

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
    thickened_borders = ndimage.gaussian_filter(thickened_borders, sigma=1, truncate=1)

    return thickened_borders


def generate_widths(length, rng: RandomState, num_control_points, min_width, max_width):
    control_pts_x = np.linspace(0, length - 1, num_control_points)
    control_pts_y = rng.uniform(min_width, max_width, num_control_points)
    xs = np.arange(length)

    widths = np.interp(xs, control_pts_x, control_pts_y)

    return widths


def smooth_border_mask(border_mask: BorderMask):
    # plt.imshow(border_mask, cmap="gray")
    # plt.show()
    border_mask = ndimage.gaussian_filter(border_mask, sigma=2)
    # plt.imshow(border_mask, cmap="gray")
    # plt.show()
    border_mask = border_mask > 0.4
    # plt.imshow(border_mask, cmap="gray")
    # plt.show()

    return border_mask.astype(np.int8)


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


def blend_borders_w_bg(border_mask: BorderMask, image: Image, grad_shading: GradientImage) -> Image:
    """ Blends the 3d border effect image with the original image according to the height map. """
    height_map = get_height_map(border_mask, 2)
    border_mask = border_mask.astype(bool)
    new_image = image.copy()

    new_image[border_mask] = (height_map[border_mask] * grad_shading[border_mask] +
                              (1 - height_map[border_mask]) * image[border_mask])

    return new_image


def add_perlin_noise(image: Image, rng: RandomState, row: Coords, col: Coords, voronoi_perlin_noise: List[int],
                     voronoi_perlin_noise_scale) -> Image:
    """ Adds perlin noise the individual cells. """
    new_image = image.copy()
    alpha = rng.randint(*voronoi_perlin_noise) / 100
    divisor = rng.choice(voronoi_perlin_noise_scale)
    res = (image.shape[0] // divisor, image.shape[1] // divisor)

    height, width = image.shape
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)  # range ~ [-1, 1]
    noise = noise - noise.mean()
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude

    return new_image


def add_gaussian_noise(image: Image, rng: RandomState, row: Coords, col: Coords,
                       voronoi_gaussian_noise: List[int]) -> Image:
    new_image = image.copy()
    noise = rng.normal(0, 1, (1025, 1025))
    alpha = rng.randint(*voronoi_gaussian_noise) * 0.03
    noise = noise - noise.mean()
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude
    return new_image


def add_porous_texture(image: Image, rng: RandomState, dark: bool, density_range: List[float], blur_size_range: List[float],
                       intensity_range: List[float], size_range: List[float]) -> Image:
    # TODO check bool vs string types
    h, w = image.shape
    density = rng.uniform(*density_range)
    blur_size = rng.uniform(*blur_size_range)
    intensity = rng.uniform(*intensity_range)

    # generate impulses and their strengths
    texture = np.zeros((h, w), dtype=np.float32)
    num = rng.poisson(density * h * w)
    impulse_strengths = rng.lognormal(mean=0.0, sigma=0.7, size=num).astype(np.float32)

    # add the impulses to the texture
    y_impulse_coords = rng.randint(0, h, size=num)
    x_impulse_coords = rng.randint(0, w, size=num)
    radii = rng.uniform(size_range[0], size_range[1], size=num)
    for y, x, r, a in zip(y_impulse_coords, x_impulse_coords, radii, impulse_strengths):
        cv2.circle(texture, (int(x), int(y)), int(round(r)), float(a), thickness=-1)
    # texture[y_impulse_coords, x_impulse_coords] = impulse_strengths

    # blur to make smooth pits
    texture = ndimage.gaussian_filter(texture, blur_size)

    # adjust intensity of impulses
    texture = texture * intensity

    if dark:
        texture = -texture
    texture -= texture.mean()  # center around 0

    tex = image + texture
    # tex = (np.clip(tex, 0, 1) * 255).astype(np.uint8)

    return tex


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
