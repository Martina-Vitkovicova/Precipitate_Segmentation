import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFilter, Image


def add_precipitates(image, seed, *, img_size, precipitate_color, precipitate_num_range, precipitate_size_ranges,
                     precipitate_sizes_distribution, **kwargs):
    # to do: change the shape generation, probably bezier curves
    rng = np.random.RandomState(seed)

    assert [image.shape[1], image.shape[0]] == img_size
    mask = np.zeros(image.shape)
    precip_num = rng.randint(*precipitate_num_range)
    angle = rng.randint(0, 180)
    color = precipitate_color
    mask_color = (255, 255, 255)

    for _ in range(precip_num):
        center = (rng.randint(0, image.shape[1]), rng.randint(0, image.shape[0]))
        size = generate_precip_size(rng, precipitate_size_ranges,
                                    precipitate_sizes_distribution)
        image = cv2.ellipse(image, center, size, angle, 0, 360, color, -1)
        mask = cv2.ellipse(mask, center, size, angle, 0, 360, mask_color, -1)

    perturb_precip_contour(image, mask, rng, color, mask_color)
    image, mask = add_emboss_effect(image, mask)

    return image, mask


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


def add_emboss_effect(image, mask):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

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
