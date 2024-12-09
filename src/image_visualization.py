from matplotlib import pyplot as plt


def show_image_mask(img, mask):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].axis('off')
    plt.show()
