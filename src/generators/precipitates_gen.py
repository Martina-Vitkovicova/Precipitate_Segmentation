from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

from src.type_definitions import Image, Color, Point, RandomState, Mask, Contour, Coords


def add_precipitates(image: Image, precip_rng: RandomState, *, light_direction: List[int], color_range: Color,
                     num_range: List[int],
                     size_range: List[int], size_distribution: List[int],
                     shading_color_range: List[int], **kwargs) \
        -> Tuple[Image, Mask]:
    emboss_rng = np.random.RandomState(precip_rng.randint(0, 10**9))
    mask = np.zeros(image.shape)
    precip_num = precip_rng.randint(*num_range)
    mask_color = (255, 255, 255)

    for _ in range(precip_num):
        temp_mask = np.zeros_like(image)
        radius = generate_precip_size(precip_rng, size_range, size_distribution)
        contour = generate_precipitate_contour(image, radius, precip_rng)

        color = precip_rng.randint(*color_range)
        cv2.fillPoly(image, [contour], color)
        cv2.fillPoly(mask, [contour], mask_color)
        cv2.fillPoly(temp_mask, [contour], mask_color)

        image = add_individual_emboss(temp_mask, image, radius, emboss_rng, shading_color_range,
                                      light_direction)
        image = add_feather_effect(image, temp_mask, radius)

    return image, mask


def generate_precipitate_contour(image: Image, radius: int, rng: RandomState) -> Contour:
    center = (rng.randint(0, image.shape[1]), rng.randint(0, image.shape[0]))

    num_segments = min(25, max(8, int(radius ** 0.5)))

    # generate random radii with variation
    radius_variation = radius ** (1 / 3)
    radii = radius + rng.uniform(-radius_variation, radius_variation, num_segments)
    radii = np.clip(radii, 1, None)

    # polar to cartesian coordinates
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_segments))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    # parametric spline with periodic boundary conditions
    spline_degree = 3
    num_spline_points = 1000
    tck, u, *_ = splprep([x, y], s=radius*0.1, per=True, k=min(spline_degree, len(x) - 1))

    u_fine = np.linspace(0, 1, num_spline_points)
    x_smooth, y_smooth = splev(u_fine, tck)
    x_coords, y_coords = np.array(x_smooth).astype(np.int32), np.array(y_smooth).astype(np.int32)

    contour = np.vstack((x_coords, y_coords)).T.reshape((-1, 1, 2))

    return contour


def generate_precip_size(rng: RandomState, size_range: List[int], distribution: List[int]) -> int:
    breaks = np.linspace(size_range[0], size_range[1], len(distribution) + 1)
    probabilities = np.array(distribution) / np.sum(distribution)
    index = rng.choice(len(distribution), p=probabilities)

    # sample uniform in area, not radius
    # r1, r2 = breaks[index], breaks[index + 1]
    # radius = np.sqrt(rng.uniform(r1 ** 2, r2 ** 2))

    radius = rng.randint(breaks[index], breaks[index + 1])

    return int(radius)


def compute_emboss_size(radius: int, rng: RandomState) -> int:
    # size = int(0.6 * radius) + rng.randint(-radius//5, radius//5)
    # size = max(5, size)
    base = max(3, int(np.sqrt(radius) * 2))
    jitter = rng.randint(-base // 4, base // 4)
    size = max(5, base + jitter)
    return size | 1


def add_individual_emboss(temp_mask: Mask, image: Image, radius: int, rng: RandomState,
                          shading_color_range: List[int], light_direction: List[int]) -> Image:
    effect_size = compute_emboss_size(radius, rng)
    shading = compute_gradient_shading(temp_mask, effect_size, light_direction)
    shading_mask = create_shading_mask(temp_mask, effect_size)

    blur_size = int(effect_size * 0.6) | 1
    blurred_rim = cv2.GaussianBlur(shading_mask.astype(float), (blur_size, blur_size), sigmaX=0)

    # nullify shading outside the rim
    shading *= blurred_rim
    shading = cv2.GaussianBlur(shading.astype(float), (blur_size // 4 | 1, blur_size // 4 | 1), sigmaX=0)

    shading_low = min(max(shading_color_range[0] + rng.randint(-10, 10), 0), 255)
    shading_high = min(max(shading_color_range[1] + rng.randint(-10, 10), 0), 255)
    shading_rescaled = adjust_image_value_range(shading, shading_low, shading_high)

    alpha = adjust_image_value_range(blurred_rim, 0, 1)

    temp_mask = temp_mask.astype(bool)
    image[temp_mask] = (alpha[temp_mask] * shading_rescaled[temp_mask] + (1 - alpha[temp_mask]) * image[temp_mask])

    # add some noise to brake uniformity
    rim = np.clip(blurred_rim, 0.0, 1.0).astype(np.float32)
    rim = cv2.dilate(rim, np.ones((blur_size, blur_size), np.float32), iterations=1)
    noise = np.random.normal(0, 7, image.shape).astype(np.float32)
    image = image.astype(np.float32)
    image += rim * noise

    return image


def compute_gradient_shading(temp_mask: Mask, size: int, light_direction: List[int]) -> Image:
    # need to make the mask smaller since the gradient is computed around the border
    erosion = max(1, int(size / 2))
    temp_mask = cv2.erode(temp_mask, np.ones((erosion, erosion), np.uint8), iterations=1)

    light_dir = np.array(light_direction, dtype=np.float32)
    light_dir[1] *= -1  # invert axis to match matrix shading
    light_dir[0] *= -1
    light_dir /= np.linalg.norm(light_dir) + 1e-8

    depth = 20
    unit_normals = get_unit_normals(temp_mask, size, depth)

    shading = np.tensordot(light_dir, unit_normals, axes=(0, 0))  # [-1, 1]
    # rescale the range to get dark shadows
    shading = (1 + shading) * 0.65 - 0.3  # (1 + shading) -> [0, 2], *0.65 -> [0, 1.3], -0.3 -> [-0.3, 1.0]

    return shading


def create_shading_mask(temp_mask: Mask, size: int) -> Mask:
    erosion = max(1, int(size / 2))
    temp_mask = cv2.erode(temp_mask, np.ones((erosion, erosion), np.uint8), iterations=1)

    band_size = max(1, size // 2) | 1
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_size, band_size))
    shading_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_GRADIENT, struct_elem).astype(bool)

    return shading_mask


def get_unit_normals(temp_mask: Mask, size: int, depth: int) -> NDArray[np.float32]:
    # blur the mask to increase the gradient area
    kernel_size = max(1, (size // 2) | 1)
    temp_mask_blurred = cv2.GaussianBlur(temp_mask.astype(np.float32), (kernel_size, kernel_size), 1.0)

    grad_x, grad_y = np.gradient(temp_mask_blurred)
    grad_x, grad_y = grad_x * (depth / 100.0), grad_y * (depth / 100.0)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
    unit_normals = np.stack((grad_x / grad_magnitude, grad_y / grad_magnitude, 1.0 / grad_magnitude), axis=0)

    return unit_normals


def adjust_image_value_range(image: Image, low: int, high: int) -> Image:
    image = np.clip(image, 0, 255)
    image_min, image_max = image.min(), image.max()
    if image_max == image_min:
        return image
    normalized = (image - image_min) / (image_max - image_min)
    scaled = normalized * (high - low) + low
    return scaled


def add_feather_effect(image: Image, temp_mask: Mask, radius: int) -> Image:
    dilation_size = max(radius // 4, 3)
    erosion_size = max(radius // 6, 2)
    dilated_mask = cv2.dilate(temp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)))
    eroded_mask = cv2.erode(temp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)))

    loop_mask = cv2.absdiff(dilated_mask, eroded_mask)
    blurred_mask = cv2.GaussianBlur(loop_mask.astype(np.float32) / 255.0, (dilation_size | 1, dilation_size | 1), 1)
    blurred_image = cv2.GaussianBlur(image, (erosion_size | 1, erosion_size | 1), 1)

    feathered_image = (image * (1 - blurred_mask) + blurred_image * blurred_mask)

    return feathered_image
