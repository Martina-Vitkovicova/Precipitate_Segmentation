---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Image Synthesis

```python
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("../src")
    
import numpy as np
from matplotlib import pyplot as plt
from generators.images_gen import generate_image, save_images

GENERATION = 56
IMG_PARAMS_PATH = "../config/img_params.json"
IMG_SAVE_PATH = "../../Data/gen" + str(GENERATION)

for i in range(5):
    root_seed = np.random.randint(0, 10 ** 5)
    img, mask = generate_image(root_seed, IMG_PARAMS_PATH)

    save_images(IMG_SAVE_PATH, root_seed, img, mask, IMG_PARAMS_PATH, GENERATION)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].axis('off')

    plt.show()


```

```python

```

```python

```
