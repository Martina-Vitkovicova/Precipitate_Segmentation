from matplotlib import pyplot as plt


def show_image_mask(img, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].axis('off')
    plt.show()
