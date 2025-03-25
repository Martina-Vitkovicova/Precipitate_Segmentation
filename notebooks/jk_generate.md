---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: precsynth
    language: python
    name: precsynth
---

```python
%load_ext autoreload
%autoreload 2

%cd ../src
```

```python
import numpy as np

root_seed = 52308
rng = np.random.RandomState(seed = root_seed)

seeds = rng.randint(0,2**16,size = 1024)
```

```python
import json

IMG_PARAMS_PATH = "../config/jk_params.json"
with open(IMG_PARAMS_PATH) as f:
    params = json.load(f)

#params['voronoi_color_range'] = [30,120]
params
```

```python
from tqdm.cli import tqdm

from pathlib import Path
import imageio
```

```python

from generators.images_gen import generate_image, save_images
from image_visualization import show_image_mask

imgs = np.zeros((0,*params['img_size']))
masks = np.zeros((0,*params['img_size']))

import multiprocessing as mp


def fn(ps):
    seed, params = ps[0],ps[1]
    img, mask = generate_image(seed, params)

    img_path = f'/tmp/{seed}.png'
    imageio.imwrite(img_path, np.uint8(img))

    mask_path = f'/tmp/{seed}_mask.png'
    imageio.imwrite(mask_path ,np.uint8(mask))
    return img_path,mask_path


from tqdm.contrib.concurrent import process_map

mp_params = [(seed,params) for seed in seeds]
r = process_map(fn, mp_params, max_workers=4)

```

```python

```

```python
import matplotlib.pyplot as plt

def plot_pair(img,mask):
    _,axs = plt.subplots(1,2,figsize = (8,4))
    axs[0].imshow(img,vmin=0,vmax=1,cmap='gray')
    axs[1].imshow(mask,vmin=0,vmax=1,cmap='gray')

plot_pair(*img_pairs[1])
```

It would make sense to blur them a bit first to make sure that all the outliers are supressed

```python

```
