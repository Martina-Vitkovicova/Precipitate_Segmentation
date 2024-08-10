import cv2
import numpy as np
from PIL import ImageFilter, Image


def add_precipitates(image, seed, img_params):
    # to be done: change the shape generation, probably bezier curves
    rng = np.random.RandomState(seed)

    assert (image.shape[1], image.shape[0]) == img_params["img_size"]
    mask = np.zeros(image.shape)
    precip_num = rng.randint(*img_params["precipitate_num_range"])
    angle = rng.randint(0, 180)
    color = img_params["precipitate_color"]
    mask_color = (255, 255, 255)

    for _ in range(precip_num):
        center = (rng.randint(0, image.shape[1]), rng.randint(0, image.shape[0]))
        size = generate_precip_size(rng, img_params["precipitate_size_ranges"],
                                    img_params["precipitate_sizes_distribution"])
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
    # pad and then crop an image 1 px because of the emboss filter border effect
    image = np.pad(image, pad_width=1, mode='edge')
    pil_image = Image.fromarray(image.astype('uint8'), 'L')

    mask = np.pad(mask, pad_width=1, mode='edge')
    pil_mask = Image.fromarray(mask.astype('uint8'), 'L')

    kernel = (
        1, 0, 0,
        0, 0, 0,
        0, 0, -1
    )

    ImageFilter.EMBOSS.filterargs = ((3, 3), 1, 128, kernel)
    embossed_image = np.array(pil_image.filter(ImageFilter.EMBOSS))
    emboss_image_effect = embossed_image != 128
    # copy just the effect of emboss, otherwise the image will be mostly grey
    image[emboss_image_effect] = embossed_image[emboss_image_effect]
    image = image[1:-1, 1:-1]

    embossed_mask = np.array(pil_mask.filter(ImageFilter.EMBOSS))
    embossed_mask_effect = (embossed_mask != 128) * np.uint8(255)
    mask = np.where((mask == 255) | (embossed_mask_effect == 255), 255, 0).astype(np.uint8)
    mask = mask[1:-1, 1:-1]

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
