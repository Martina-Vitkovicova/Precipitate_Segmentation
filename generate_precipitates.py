import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image


def add_precipitates(image, seed, min_precipitate_num, max_precipitate_num):
    rng = np.random.RandomState(seed)
    num = rng.randint(min_precipitate_num, max_precipitate_num)
    mask = np.zeros(image.shape)

    for _ in range(num):
        center = (rng.randint(20, image.shape[0] - 20), rng.randint(20, image.shape[1] - 20))
        size = (generate_precip_size((1, 42), [2200, 30, 25, 15, 10, 6, 3, 2, 1, 1], rng),
                generate_precip_size((1, 48), [2200, 30, 25, 15, 10, 6, 3, 2, 1, 1], rng))
        # size = (rng.randint(1, 16), rng.randint(1, 12))
        angle = rng.randint(0, 180)
        color = (40, 40, 40)

        perturb_contour_image_mask(image, mask, rng, center, size, angle, color)
        image = cv2.ellipse(image, center, size, angle, 0, 360, color, -1)
        mask = cv2.ellipse(mask, center, size, angle, 0, 360, (255, 255, 255), -1)

    image, mask = add_emboss(image, mask)

    return image, mask


def perturb_contour_image_mask(image, mask, rng, center, size, angle, color):
    ellipse_contour = cv2.ellipse(np.zeros(image.shape, dtype=np.uint8), center, size, angle, 0, 360, color, 4)
    contours, _ = cv2.findContours(ellipse_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
        noise = np.array(rng.normal(0, 5, (len(cont), 2)))
        smooth_noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_REFLECT)
        for i, point in enumerate(cont):
            point[0][0] += smooth_noise[i][0]
            point[0][1] += smooth_noise[i][1]

    cv2.drawContours(image, contours, -1, color, -1)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)


def add_emboss(image, mask):
    # pad and then crop an image 1 px because of emboss filter border effect
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


def generate_precip_size(interval, distribution, rng):
    breaks = np.ceil(np.linspace(start=interval[0], stop=interval[1], num=len(distribution) + 1))
    intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
    total = sum(distribution)
    probabilities = [p / total for p in distribution]

    chosen_interval = rng.choice(len(intervals), p=probabilities)
    low, high = intervals[chosen_interval]
    size = rng.randint(low, high)

    return size



