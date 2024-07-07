import os
import pathlib
import torch

from baseline_model import LitUnet
import matplotlib.pyplot as plt
from denoiseg import evaluation as ev
import denoiseg.visualization as vis


def evaluate_model(iu=None):
    root = pathlib.Path('../Data/test/')
    gt_paths = [os.path.join(root, subdir, "mask.png") for subdir in os.listdir(root)]
    img_paths = [os.path.join(root, subdir, "img.png") for subdir in os.listdir(root)]
    # img_paths = [root / f"{p.stem}.png" for p in gt_paths]

    imgs_test = list(map(iu.load_image, img_paths))
    mask_test = list(map(iu.load_mask, gt_paths))

    checkpoint = torch.load('segmentation_model.pth')
    model_s = LitUnet()
    model_s.load_state_dict(checkpoint)
    model_s.eval()

    predictions, metrics = ev.evaluate_images(model_s, imgs_test, mask_test, device='cpu')

    for i, (img, gt, pred, met) in enumerate(zip(imgs_test, mask_test, predictions, metrics)):

        segms = []
        for im in pred[1:]:
            im = im.copy()
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            im[im < .5] = 0
            im[im >= .5] = 1
            segms.append(im)

        show_imgs = [img, gt, *segms, *pred]
        vis.plot_row(show_imgs, vmin_vmax=(0, 1), figsize=(50, 10))
        plt.suptitle(f'Metric: {met}', y=0.72)
        plt.tight_layout()
        plt.show()
