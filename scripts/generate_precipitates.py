import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image


def add_precipitates(image, seed, img_params):
    rng = np.random.RandomState(seed)

    assert image.shape == img_params["img_size"]
    mask = np.zeros(image.shape)
    num = rng.randint(*img_params["precipitate_num_range"])

    for _ in range(num):
        center = (rng.randint(0, image.shape[0]), rng.randint(0, image.shape[1]))
        size = generate_precip_size(rng, img_params["precipitate_size_ranges"],
                                    img_params["precipitate_sizes_distribution"])
        angle = rng.randint(0, 180)
        color = img_params["precipitate_color"]

        perturb_contour_image_mask(image, mask, center, size, angle, color)
        image = cv2.ellipse(image, center, size, angle, 0, 360, color, -1)
        mask = cv2.ellipse(mask, center, size, angle, 0, 360, (255, 255, 255), -1)

    image, mask = add_emboss(image, mask)

    return image, mask


def perturb_contour_image_mask(image, mask, center, size, angle, color):
    ellipse_contour = cv2.ellipse(np.zeros(image.shape, dtype=np.uint8), center, size, angle, 0, 360, color, 4)
    contours, _ = cv2.findContours(ellipse_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
        ampl = (size[0] + size[1]) / 8
        freq = 0.08

        perturbation_sin = ampl * np.sin(freq * np.arange(len(cont)))
        perturbation = np.stack((perturbation_sin, perturbation_sin), axis=-1)
        smooth_noise = cv2.GaussianBlur(perturbation, (0, 0), sigmaX=ampl/2, sigmaY=ampl/2,
                                        borderType=cv2.BORDER_REFLECT).reshape(len(cont), 1, 2).astype(np.int32)
        cont += smooth_noise

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


def generate_precip_size(rng, size_ranges, distribution):
    sizes = []
    for size_range in size_ranges:
        breaks = np.ceil(np.linspace(start=size_range[0], stop=size_range[1], num=len(distribution) + 1))
        intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
        total = sum(distribution)
        probabilities = [p / total for p in distribution]

        chosen_interval = rng.choice(len(intervals), p=probabilities)
        low, high = intervals[chosen_interval]
        sizes.append(rng.randint(low, high))

        # if the size along one axis is big, adjust the distribution so that it's more probable that the other side
        # will be big too
        if chosen_interval > len(distribution) // 2:
            distribution = list(distribution)
            value_to_add = distribution[0] // 2
            for i in range(len(distribution) // 2, len(distribution)):
                decay_rate = 0.6
                distribution[i] += value_to_add
                value_to_add *= decay_rate

    return sizes


