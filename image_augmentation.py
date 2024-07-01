import os
import random
import shutil

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf

from baseline_model import ImageDataset, DATA_DIR


def save_augmented_images(data_dir, transformations, image_name='image.png', mask_name='mask.png'):
    dataset = ImageDataset(data_dir)

    idx = 0
    for t in transformations:
        # s_image = transforms.ToPILImage()(img)
        # s_mask = transforms.ToPILImage()(mask)
        # s_image.show()
        # s_mask.show()

        for img, mask in dataset:
            a_image, a_mask = t(img, mask)

            a_subdir = os.path.join(data_dir, f"aug_{idx}")
            if os.path.exists(a_subdir):
                shutil.rmtree(a_subdir)
            os.makedirs(a_subdir, exist_ok=True)

            a_image = transforms.ToPILImage()(a_image)
            a_mask = transforms.ToPILImage()(a_mask)
            a_image.save(os.path.join(a_subdir, image_name))
            a_mask.save(os.path.join(a_subdir, mask_name))
            idx += 1


def rotate_images(img, mask):
    angle = random.randint(1, 359)
    a_image = ttf.rotate(img, angle)
    a_mask = ttf.rotate(mask, angle)

    return a_image, a_mask


def horizontal_flip(img, mask):
    a_image = ttf.hflip(img)
    a_mask = ttf.hflip(mask)

    return a_image, a_mask


def vertical_flip(img, mask):
    a_image = ttf.vflip(img)
    a_mask = ttf.vflip(mask)

    return a_image, a_mask


def change_brightness(img, mask):
    factor = random.uniform(0.1, 2)
    a_image = ttf.adjust_brightness(img, factor)
    a_mask = mask

    return a_image, a_mask


def change_contrast(img, mask):
    factor = random.uniform(0.1, 2)
    a_image = ttf.adjust_contrast(img, factor)
    a_mask = mask

    return a_image, a_mask


TRANSFORMS = [rotate_images, horizontal_flip, vertical_flip, change_brightness, change_contrast]
save_augmented_images(DATA_DIR, TRANSFORMS)
