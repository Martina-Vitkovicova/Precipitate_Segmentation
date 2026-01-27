import pathlib

import cv2


def load_img_and_mask(img_path: pathlib.Path):
    mask_path = img_path.with_name("mask.png")
    if not mask_path.is_file():
        return None, None

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None

    return img, mask


def iter_imgs_and_masks(root_path: pathlib.Path):
    for img_path in sorted(root_path.rglob("img.png")):
        mask_path = img_path.with_name("mask.png")
        if not mask_path.is_file():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        yield img, mask
