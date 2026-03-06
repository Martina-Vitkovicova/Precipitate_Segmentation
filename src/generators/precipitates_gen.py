from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

from src.utils.config_parsing import GeneralParams, PrecipitatesParams
from src.utils.type_definitions import Image, RandomState, Mask, Contour, IntRange, Vec3i


def add_precipitates(image: Image, precip_rng: RandomState, general_params: GeneralParams,
                     params: PrecipitatesParams) -> Tuple[Image, Mask]:
    """
    Add precipitates - small dark oval shapes - to the image and return the updated image and generated binary
    precipitate mask.

    The generation is controlled by a `PrecipitatesParams` object defining the number,
    size range and distribution, intensity range, and optional embossing behaviour. Global
    illumination information is taken from `GeneralParams` and used only for directional
    shading effects.

    The function modifies the input image in-place and returns:
    - the updated image, and
    - a float mask (shape H×W) with values {0.0, 1.0} indicating precipitate regions.

    Stochastic behaviour is fully determined by `precip_rng`.
    """
    emboss_rng = np.random.RandomState(precip_rng.randint(0, 10 ** 9))
    mask = np.zeros(image.shape)
    precip_num = precip_rng.randint(*params.num_range)
    mask_intensity = (255, 255, 255)

    for _ in range(precip_num):
        temp_mask = np.zeros_like(image, dtype=np.float32)
        radius = _generate_precip_size(precip_rng, params.size_range, params.size_distribution)
        contour = _generate_precipitate_contour(image, radius, precip_rng)

        intensity = precip_rng.randint(*params.intensity_range)
        cv2.fillPoly(image, [contour], intensity)
        cv2.fillPoly(mask, [contour], mask_intensity)
        cv2.fillPoly(temp_mask, [contour], mask_intensity)

        if params.emboss_enabled:
            image = _add_individual_emboss(temp_mask, image, radius, emboss_rng, params.emboss_intensity_range,
                                           general_params.light_direction)
        image = _add_feather_effect(image, temp_mask, radius)

    return image, mask


def _generate_precipitate_contour(image: Image, radius: int, rng: RandomState) -> Contour:
    """ Generate a smooth, closed contour with irregular radial perturbations representing a single precipitate. """
    center = (rng.randint(0, image.shape[1]), rng.randint(0, image.shape[0]))

    num_segments = min(25, max(8, int(radius ** 0.5)))

    # generate random radii with variation
    radius_variation = float(radius ** (1.0 / 3.0))
    radii = radius + rng.uniform(-radius_variation, radius_variation, num_segments)
    radii = np.clip(radii, 1, None)

    # polar to cartesian coordinates
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_segments))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    # parametric spline with periodic boundary conditions
    spline_degree = 3
    num_spline_points = 1000
    spline, _, *_ = splprep([x, y], s=radius * 0.1, per=True, k=min(spline_degree, len(x) - 1))

    sampling_params = np.linspace(0, 1, num_spline_points)
    x_smooth, y_smooth = splev(sampling_params, spline)
    x_coords, y_coords = np.array(x_smooth).astype(np.int32), np.array(y_smooth).astype(np.int32)

    contour = np.vstack((x_coords, y_coords)).T.reshape((-1, 1, 2))

    return contour


def _generate_precip_size(rng: RandomState, size_range: IntRange, distribution: List[int]) -> int:
    """ Sample precipitate radius according to a discrete size distribution."""
    breaks = np.linspace(size_range[0], size_range[1], len(distribution) + 1)
    probabilities = np.array(distribution) / np.sum(distribution)
    index = rng.choice(len(distribution), p=probabilities)

    # sample uniform in area, not radius
    # r1, r2 = breaks[index], breaks[index + 1]
    # radius = np.sqrt(rng.uniform(r1 ** 2, r2 ** 2))

    radius = rng.randint(breaks[index], breaks[index + 1])

    return int(radius)


def _compute_emboss_size(radius: int, rng: RandomState) -> int:
    """Compute an odd emboss kernel size scaled with precipitate radius."""
    base = max(3, int(np.sqrt(radius) * 2))
    jitter = rng.randint(-base // 4, base // 4)
    size = max(5, base + jitter)
    return size | 1


def _add_individual_emboss(temp_mask: Mask, image: Image, radius: int, rng: RandomState,
                           emboss_intensity_range: IntRange, light_direction: Vec3i) -> Image:
    """ Apply directional rim embossing to a single precipitate region. """
    effect_size = _compute_emboss_size(radius, rng)
    emboss = _compute_gradient_emboss(temp_mask, effect_size, light_direction)
    emboss_mask = _create_emboss_mask(temp_mask, effect_size)

    blur_size = int(effect_size * 0.6) | 1
    blurred_rim = cv2.GaussianBlur(emboss_mask.astype(float), (blur_size, blur_size), sigmaX=0)

    # nullify emboss outside the rim
    emboss *= blurred_rim
    emboss = cv2.GaussianBlur(emboss.astype(float), (blur_size // 4 | 1, blur_size // 4 | 1), sigmaX=0)

    emboss_low = min(max(emboss_intensity_range[0] + rng.randint(-10, 10), 0), 255)
    emboss_high = min(max(emboss_intensity_range[1] + rng.randint(-10, 10), 0), 255)
    emboss_rescaled = _adjust_image_value_range(emboss, emboss_low, emboss_high)

    alpha = _adjust_image_value_range(blurred_rim, 0, 1)

    temp_mask = temp_mask.astype(bool)
    image[temp_mask] = (alpha[temp_mask] * emboss_rescaled[temp_mask] + (1 - alpha[temp_mask]) * image[temp_mask])

    # add some noise to brake uniformity
    rim = np.clip(blurred_rim, 0.0, 1.0).astype(np.float32)
    rim = cv2.dilate(rim, np.ones((blur_size, blur_size), np.float32), iterations=1)
    noise = np.random.normal(0, 7, image.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    image = image.astype(np.float32)
    image += rim * noise
    image = np.clip(image, 0, 255)

    return image


def _compute_gradient_emboss(temp_mask: Mask, size: int, light_direction: Vec3i) -> Image:
    """ Compute a scalar emboss field from mask gradients and a light direction (approx. Lambertian dot)."""
    # need to make the mask smaller since the gradient is computed around the border
    erosion = max(1, int(size * 0.4))
    temp_mask = cv2.erode(temp_mask, np.ones((erosion, erosion), np.uint8), iterations=1)

    light_dir = np.array(light_direction, dtype=np.float32)
    light_dir[1] *= -1  # invert axis to match matrix shading
    light_dir[0] *= -1
    light_dir /= np.linalg.norm(light_dir) + 1e-8

    depth = 20
    unit_normals = _get_unit_normals(temp_mask, size, depth)

    emboss = np.tensordot(light_dir, unit_normals, axes=(0, 0))  # [-1, 1]
    # rescale the range to get dark shadows
    emboss = (1 + emboss) * 0.65 - 0.3  # (1 + emboss) -> [0, 2], *0.65 -> [0, 1.3], -0.3 -> [-0.3, 1.0]

    return emboss


def _create_emboss_mask(temp_mask: Mask, size: int) -> Mask:
    """Create a rim mask (morphological gradient) defining where embossing is applied."""
    erosion = max(1, int(size / 2))
    temp_mask = cv2.erode(temp_mask, np.ones((erosion, erosion), np.uint8), iterations=1)

    band_size = max(1, size // 2) | 1
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_size, band_size))
    emboss_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_GRADIENT, struct_elem).astype(bool)

    return emboss_mask


def _get_unit_normals(temp_mask: Mask, size: int, depth: int) -> NDArray[np.float32]:
    """ Estimate normals from a blurred mask gradient, returns array shaped (3, H, W)."""
    # blur the mask to increase the gradient area
    kernel_size = max(1, (size // 2) | 1)
    temp_mask_blurred = cv2.GaussianBlur(temp_mask.astype(np.float32), (kernel_size, kernel_size), 1.0)

    grad_x, grad_y = np.gradient(temp_mask_blurred)
    grad_x, grad_y = grad_x * (depth / 100.0), grad_y * (depth / 100.0)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
    unit_normals = np.stack((grad_x / grad_magnitude, grad_y / grad_magnitude, 1.0 / grad_magnitude), axis=0)

    return unit_normals


def _adjust_image_value_range(image: Image, low: int, high: int) -> Image:
    """Rescale values linearly to [low, high], leaves constant images unchanged."""
    image = np.clip(image, 0, 255)
    image_min, image_max = image.min(), image.max()
    if image_max <= image_min:
        return image
    normalized = (image - image_min) / (image_max - image_min)
    scaled = normalized * (high - low) + low
    return scaled


def _add_feather_effect(image: Image, temp_mask: Mask, radius: int) -> Image:
    """Blend precipitate with image to soften edges."""
    dilation_size = max(radius // 4, 3)
    erosion_size = max(radius // 6, 2)
    dilated_mask = cv2.dilate(temp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)))
    eroded_mask = cv2.erode(temp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)))

    loop_mask = cv2.absdiff(dilated_mask, eroded_mask)
    blurred_mask = cv2.GaussianBlur(loop_mask.astype(np.float32) / 255.0, (dilation_size | 1, dilation_size | 1), 1)
    blurred_image = cv2.GaussianBlur(image, (erosion_size | 1, erosion_size | 1), 1)

    feathered_image = (image * (1 - blurred_mask) + blurred_image * blurred_mask)

    return feathered_image
