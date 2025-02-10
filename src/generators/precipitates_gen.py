from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

from src.type_definitions import Image, Color, Point, RandomState, Mask, Contour, Coords


def add_precipitates(image: Image, seed: int, *, precipitate_color: Color, precipitate_num_range: List[int],
                     precipitate_size_range: List[int], precipitate_size_distribution: List[int], **kwargs) \
        -> Tuple[Image, Mask]:
    rng = np.random.RandomState(seed)
    mask = np.zeros(image.shape)
    precip_num = rng.randint(*precipitate_num_range)
    color = precipitate_color
    mask_color = (255, 255, 255)

    for _ in range(precip_num):
        temp_mask = np.zeros_like(image)
        radius = generate_precip_size(rng, precipitate_size_range, precipitate_size_distribution)
        contour = generate_precipitate_contour(image, radius, rng)

        cv2.fillPoly(image, [contour], color)
        cv2.fillPoly(mask, [contour], mask_color)
        cv2.fillPoly(temp_mask, [contour], mask_color)
        image = add_individual_emboss(temp_mask, image, radius, rng)

    image = add_feather_effect(image, mask)

    return image, mask


def generate_precipitate_contour(image: Image, radius: int, rng: RandomState) -> Contour:
    center = (rng.randint(0, image.shape[1]), rng.randint(0, image.shape[0]))

    num_segments = min(20, max(8, int(radius ** 0.5)))
    radius_variation = radius ** 1 / 3

    x, y = generate_shape_coords(num_segments, radius, radius_variation, center)
    contour = np.vstack((x, y)).T.reshape((-1, 1, 2))

    # contour[:, :, 0] = np.clip(contour[:, :, 0], 0, image.shape[1] - 1)
    # contour[:, :, 1] = np.clip(contour[:, :, 1], 0, image.shape[0] - 1)

    return contour


def generate_shape_coords(num_points: int, radius: int, radius_variation: float, center: Point, spline_degree: int = 3,
                          num_spline_points: int = 1000) -> Tuple[Coords, Coords]:
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))

    # generate random radii with variation
    radii = radius + np.random.uniform(-radius_variation, radius_variation, num_points)
    radii = np.clip(radii, 0.5 * radius, None)

    # polar to cartesian coordinates
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # parametric spline with periodic boundary conditions
    tck, u, *_ = splprep([x, y], s=10, per=True, k=min(spline_degree, len(x) - 1))
    u_fine = np.linspace(0, 1, num_spline_points)
    x_smooth, y_smooth = splev(u_fine, tck)
    x_coords, y_coords = np.array(x_smooth).astype(np.int32), np.array(y_smooth).astype(np.int32)

    return x_coords, y_coords


def add_individual_emboss(temp_mask: Mask, image: Image, radius: int, rng: RandomState) -> Image:
    effect_size = compute_emboss_size(radius, rng)

    shading, shading_mask = compute_gradient_shading(temp_mask, effect_size)

    blurred_border = cv2.GaussianBlur(shading_mask.astype(float), (effect_size, effect_size), sigmaX=0)
    shading_strength = rng.randint(80, 200)
    blurred_border = adjust_image_value_range(blurred_border, 20, shading_strength)
    alpha = blurred_border.astype(np.float32) / 255.0

    temp_mask = temp_mask.astype(bool)
    image[temp_mask] = (alpha[temp_mask] * shading[temp_mask] + (1 - alpha[temp_mask]) * image[temp_mask])

    return image


def compute_gradient_shading(temp_mask: Mask, size: int) -> Tuple[Image, Mask]:
    # need to make the mask smaller since the gradient is computed around the border
    erosion = int(size // 4)
    temp_mask = cv2.erode(temp_mask, np.ones((erosion, erosion), np.uint8), iterations=1)
    elevation = np.pi / 2.5  # radians
    azimuth = np.pi / 2.2  # where is the light source
    depth = 20  # how high is the light source - changes the light strength

    # Compute unit incident light direction
    gd = np.cos(elevation)
    light_dir = np.array([gd * np.cos(azimuth), gd * np.sin(azimuth), np.sin(elevation)])

    unit_normals = get_unit_normals(temp_mask, size, depth)

    # Compute shading using dot product
    shading = 255.0 * np.tensordot(light_dir, unit_normals, axes=(0, 0))
    only_shading = remove_shading_background(shading)
    shading_mask = (only_shading > 0).astype(bool)

    ksize = size // 2 | 1
    shading_blurred = cv2.GaussianBlur(only_shading, (ksize, ksize), sigmaX=0)
    shading_normalized = adjust_image_value_range(shading_blurred, 40, 240)

    return shading_normalized, shading_mask


def remove_shading_background(shading: Image) -> Image:
    image_8bit = cv2.convertScaleAbs(shading)
    _, border = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    eroded = cv2.erode(border, np.ones((2, 2), np.uint8), iterations=1)
    shading[eroded == 0] = 0

    return shading


def get_unit_normals(temp_mask: Mask, size: int, depth: int) -> NDArray[np.float32]:
    # blur the mask to increase the gradient area
    temp_mask_blurred = cv2.GaussianBlur(temp_mask, (size, size), 2.0)
    grad_x, grad_y = np.gradient(temp_mask_blurred)
    grad_x, grad_y = grad_x * depth / 100, grad_y * depth / 100

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


def compute_emboss_size(radius: int, rng: RandomState) -> int:
    size = int(radius ** 1 / 2) + 1
    size += rng.randint(- size // 2, size // 2)
    size = size if size % 2 == 1 else size + 1

    return size


def generate_precip_size(rng: RandomState, size_range: List[int], distribution: List[int]) -> int:
    breaks = np.ceil(np.linspace(size_range[0], size_range[1], len(distribution) + 1)).astype(int)
    probabilities = np.array(distribution) / np.sum(distribution)
    chosen_index = rng.choice(len(distribution), p=probabilities)

    size = rng.randint(breaks[chosen_index], breaks[chosen_index + 1])

    return size


def add_feather_effect(image: Image, mask: Mask) -> Image:
    dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    loop_mask = cv2.absdiff(dilated_mask, eroded_mask)
    blurred_mask = cv2.GaussianBlur(loop_mask.astype(np.float32) / 255.0, (3, 3), 1)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 2)

    feathered_image = (image * (1 - blurred_mask) + blurred_image * blurred_mask)

    return feathered_image
