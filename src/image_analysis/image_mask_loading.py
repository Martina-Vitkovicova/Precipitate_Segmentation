import pathlib
import cv2


def iter_imgs_and_masks(root_path: pathlib.Path, with_paths: bool = False):
    for img_path in sorted(root_path.rglob("img.png")):
        mask_path = img_path.with_name("mask.png")
        if not mask_path.is_file():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        if with_paths:
            yield img, mask, img_path, mask_path
        else:
            yield img, mask
