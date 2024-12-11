import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from PIL import ImageFilter, Image
from scipy.interpolate import splprep, splev

from src.generators.background_gen import blend_borders_w_bg


def add_precipitates(image, seed, *, precipitate_color, precipitate_num_range, precipitate_size_ranges,
                     precipitate_sizes_distribution, **kwargs):
    rng = np.random.RandomState(seed)
    mask = np.zeros(image.shape)
    precip_num = rng.randint(*precipitate_num_range)
    color = precipitate_color
    mask_color = (255, 255, 255)

    for _ in range(precip_num):
        temp_mask = np.zeros_like(image)
        center = (rng.randint(0, image.shape[1]), rng.randint(0, image.shape[0]))
        # todo adjust the size generation for spline approach
        size = generate_precip_size(rng, precipitate_size_ranges, precipitate_sizes_distribution)

        num_segments = min(20, max(8, int(size[0] ** 0.5)))
        rv = size[0] ** 1 / 3
        x, y = generate_shape(num_segments, size[0], rv, center)

        contour = np.vstack((x, y)).T.astype(np.int32).reshape((-1, 1, 2))
        contour[:, :, 0] = np.clip(contour[:, :, 0], 0, image.shape[1] - 1)
        contour[:, :, 1] = np.clip(contour[:, :, 1], 0, image.shape[0] - 1)

        cv2.fillPoly(image, [contour], color)
        cv2.fillPoly(mask, [contour], mask_color)
        cv2.fillPoly(temp_mask, [contour], mask_color)
        image, mask = add_individual_emboss(temp_mask, image, mask, size, mask_color, rng)
        # plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        # plt.show()

    return image, mask


def generate_shape(num_points, radius, radius_variation, center, spline_degree=3, num_spline_points=1000):
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))

    # generate random radii with variation
    radii = radius + np.random.uniform(-radius_variation, radius_variation, num_points)
    radii = np.clip(radii, 0.5 * radius, None)  # so radii are positive and not too small

    # polar to cartesian coordinates
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # parametric spline with periodic boundary conditions
    tck, u, *_ = splprep([x, y], s=10, per=True, k=min(spline_degree, len(x) - 1))
    u_fine = np.linspace(0, 1, num_spline_points)
    x_smooth, y_smooth = splev(u_fine, tck)

    return x_smooth, y_smooth


def perturb_precip_contour(image, mask, rng, color, mask_color):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
        _, _, width, height = cv2.boundingRect(cont)
        noise_deviation = min(width, height) / 2 + 1
        smoothing = noise_deviation * 0.8
        noise = generate_smooth_noise(cont, rng, noise_deviation, smoothing)
        cont += noise

    cv2.drawContours(image, contours, -1, color, -1)
    cv2.drawContours(mask, contours, -1, mask_color, -1)


def generate_smooth_noise(contour, rng, max_noise, smoothing):
    noise = np.array(rng.normal(0, max_noise, (len(contour), 1, 2))).astype(np.float32)
    smooth_noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=smoothing, sigmaY=smoothing,
                                    borderType=cv2.BORDER_REFLECT)

    return smooth_noise.astype(np.int32)


def add_individual_emboss(temp_mask, image, mask, size, mask_color, rng):
    effect_size = compute_emboss_size(*size, rng)

    shading = compute_gradient_shading(temp_mask, effect_size)
    border = get_border_from_mask(temp_mask, effect_size, 0.8)
    ksize = (effect_size) | 1
    blurred_border = cv2.GaussianBlur(border, (ksize, ksize), sigmaX=0)
    blurred_border = adjust_image_value_range(blurred_border, 20, 205)
    # plt.imshow(border, cmap="gray", vmin=0, vmax=255)
    plt.show()
    alpha = blurred_border.astype(np.float32) / 255.0
    # plt.imshow(alpha, cmap="gray", vmin=0, vmax=1)
    plt.show()

    temp_mask = temp_mask.astype(bool)
    image[temp_mask] = (alpha[temp_mask] * shading[temp_mask] + (1 - alpha[temp_mask]) * image[temp_mask])
    # np.copyto(mask, mask_color[0], where=().astype(bool))

    return image, mask


def get_border_from_mask(temp_mask, size, scaled):
    kernel = np.ones((size, size), np.uint8)
    if scaled < 1:
        new_temp_mask = cv2.erode(temp_mask, kernel, iterations=1)
    else:
        new_temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
    eroded = cv2.erode(new_temp_mask, kernel, iterations=1)
    border = new_temp_mask - eroded

    # todo fix small precips not finding the border
    if border.max == 0:
        return temp_mask

    return border


def compute_gradient_shading(temp_mask, size):
    elevation = np.pi / 2.5  # radians
    azimuth = np.pi / 2.2  # where is the light source
    depth = 20  # how high is the light source - changes the light strength

    # Compute unit incident light direction
    gd = np.cos(elevation)
    light_dir = np.array([gd * np.cos(azimuth), gd * np.sin(azimuth), np.sin(elevation)])

    unit_normals = get_unit_normals(temp_mask, size, depth)

    # Compute shading using dot product
    shading = 255 * np.tensordot(light_dir, unit_normals, axes=(0, 0))
    only_shading = remove_shading_background(shading)
    shading_blurred = cv2.GaussianBlur(only_shading, (size, size), sigmaX=0)
    shading_normalized = adjust_image_value_range(shading_blurred, 40, 240)
    # plt.imshow(shading_blurred, cmap="gray", vmin=0, vmax=255)
    # plt.show()

    return shading_normalized


def remove_shading_background(shading):
    image_8bit = cv2.convertScaleAbs(shading)
    _, border = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    eroded = cv2.erode(border, np.ones((2, 2), np.uint8), iterations=1)
    shading[eroded == 0] = 0

    return shading


def get_unit_normals(temp_mask, size, depth):
    # blur the mask to increase the gradient area
    temp_mask_blurred = cv2.GaussianBlur(temp_mask, (size, size), 2.0)
    grad_x, grad_y = np.gradient(temp_mask_blurred)
    grad_x, grad_y = grad_x * depth / 100, grad_y * depth / 100

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
    unit_normals = np.stack((grad_x / grad_magnitude, grad_y / grad_magnitude, 1.0 / grad_magnitude), axis=0)

    return unit_normals


def adjust_image_value_range(image, low, high):
    image = np.clip(image, 0, 255)
    image_min, image_max = image.min(), image.max()
    if image_max == image_min:
        return image
    normalized = (image - image_min) / (image_max - image_min)
    scaled = normalized * (high - low) + low
    return scaled


def compute_emboss_size(precip_size_x, precip_size_y, rng):
    size = int(((precip_size_x + precip_size_y) / 2) ** 1 / 2) + 1
    size += rng.randint(- size // 2, size // 2)
    size = size if size % 2 == 1 else size + 1

    return size


def add_emboss_effect(image, mask):
    grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.degrees(np.arctan2(grad_y, grad_x))
    direction = ((direction + 180) % 360)

    # copy just the effect of emboss, otherwise the image will be mostly grey
    emboss_effect = magnitude > 128

    image[emboss_effect] = direction[emboss_effect]
    mask[emboss_effect] = 255

    return image, mask


def generate_precip_size(rng, size_ranges, distribution, joined_size_distribution=True):
    sizes = []
    for size_range in size_ranges:
        breaks = np.ceil(np.linspace(start=size_range[0], stop=size_range[1], num=len(distribution) + 1))
        intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
        total = sum(distribution)
        probabilities = [p / total for p in distribution]

        chosen_interval = rng.choice(len(intervals), p=probabilities)
        low, high = intervals[chosen_interval]
        sizes.append(rng.randint(low, high))

        if joined_size_distribution:
            distribution = adjust_precip_size_distribution(chosen_interval, distribution)

    return sizes


def adjust_precip_size_distribution(interval, distribution):
    # if the size along one axis is big, adjust the distribution so that it's more probable that the other side
    # will be big too
    if interval > len(distribution) // 2:
        distribution = list(distribution)
        value_to_add = distribution[0] // 2
        for i in range(len(distribution) // 2, len(distribution)):
            decay_rate = 0.6
            distribution[i] += value_to_add
            value_to_add *= decay_rate

    return distribution
