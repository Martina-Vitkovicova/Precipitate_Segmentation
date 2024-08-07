import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

from dataset_preparation import load_images


def visualise_results(test_dir, predictions, show=True):
    imgs_test, mask_test = load_images(test_dir)

    if show:
        for img, mask, pred in zip(imgs_test, mask_test, predictions):
            segm = pred.copy()
            segm = (segm - np.min(segm)) / (np.max(segm) - np.min(segm))
            segm = np.where(segm < 0.5, 0, 1)

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(50, 10))

            ax1.imshow(img, cmap="gray")
            ax2.imshow(mask, cmap=colors.ListedColormap(['black', 'white']))
            ax3.imshow(pred, cmap="gray")
            ax4.imshow(segm, cmap=colors.ListedColormap(['black', 'white']))

            # mng = plt.get_current_fig_manager()
            # mng.window.setGeometry(50,100,640, 545)
            ax2.format_coord = Formatter(img)
            plt.show()

            # center_window(fig, 1400, 400)


def center_window(fig, width, height):
    root = tk.Tk()
    position_x = (root.winfo_screenwidth() - width) // 2
    position_y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{position_x}+{position_y}")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.mainloop()


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im[int(y)][int(x)]
        return 'x={}, y={}, z={}'.format(x, y, z)
