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

```python
%load_ext autoreload
%autoreload 2
```

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

```python
import pathlib
root = pathlib.Path('C:\\Users\\vitko\\Desktop\\CVR\\Data\\labeled')
training_output_root = pathlib.Path('C:\\Users\\vitko\\Desktop\\CVR')
```

```python

from external_libs import delisa_loader

x, y = delisa_loader.load_pairs(root)

assert len(x) == len(y)
assert len(x) > 0

print("Found training pairs", len(x))

test_x, test_y = delisa_loader.load_pairs(root / '../../test')
assert len(test_x) == len(test_y)
assert len(test_x) > 0
print("Found testing pairs", len(test_x))
```

```python
import numpy as np
import denoiseg.configuration as cfg
default_params = cfg.get_default_config()

custom = {
    "patch_size":256,
    "epochs":200,
    "patience":20,
    "dataset_repeat":100,
    "denoise_enabled":False,
    "validation_set_percentage":.2,
    "model":{
        "filters":8,
        "depth":5,
    },
    "note":"added filters",
    "augumentation":{
        "brightness_contrast":True,
        "noise_val":0.001,
        "flip_vertical": True,
        "flip_horizontal": True,
        "rotate_deg": 190,
        "elastic":True,
        "blur_sharp_power": 1,
    }
}

train_params = cfg.merge(default_params,custom)
```

```python
import denoiseg.visualization as vis

vis.sample_ds(
    x,
    y,
    train_params,
    #weightmaps= weightmaps,  # Use this when you want to add weightmaps. Its usefull to force CNN to separate objects
    n=5
)
```

# Training

```python
import segmentation_models_pytorch as smp

encoder = "resnet18"
decoder_attention_type = None #"scse"
activation = 'sigmoid'


#POZOR: tady jsou 4 output vrstvy
# 1. foreground
# 2. background
# 3. borders
# denoise (to neřešíme)


model = smp.Unet(
    encoder_name=encoder,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
    decoder_attention_type=decoder_attention_type,
    activation=activation
)
```

```python
import denoiseg.segmentation as seg
from tqdm.contrib.logging import logging_redirect_tqdm

with logging_redirect_tqdm():
    checkpoint, out_losses =seg.run_training(
        x,
        y,
        train_params, 
        training_output_root,
        device = device,
        #weightmaps = weightmaps,
        model = model
    )

```

```python
import denoiseg.visualization as vis
import matplotlib.pyplot as plt

vis.plot_loss(out_losses)
plt.show()
```

# Evaluate

```python
import denoiseg.training as tr
from tqdm.auto import tqdm
import denoiseg.evaluation as ev

# checkpoint = "C:\\Users\\vitko\\Desktop\\CVR\\20240804214414\\model-checkpoint-best.pth"
model = torch.load(checkpoint)
predictions, metrics = ev.evaluate_images(
    model, 
    test_x,
    test_y,
    device = device
)

print(f'Mean IoU {np.mean(metrics):.5f}')
```

```python
print(f"x,y,prediction")
import denoiseg.visualization as vis
for i,(img,gt,pred,met) in enumerate(zip(test_x,test_y,predictions,metrics)):
    
    segms = []
    for im in pred[1:]:
        im = im.copy()
        im = (im - np.min(im))/(np.max(im)-np.min(im))
        im[im<.5] = 0
        im[im>=.5] = 1
        segms.append(im)

    show_imgs = [img,gt,segms[0]]
    vis.plot_row(show_imgs,vmin_vmax = (0,1),figsize=(50,20))
    plt.suptitle(f'Metric: {met}')
    plt.tight_layout()
    plt.show()
```

# Evaluation

The following code represent instance evalutaion with different thresholds. The results are visualized ina  form or precision recall curve.

Each point represents mean evaluation for a given model across the whole test set. The evaluation encapsulates two compontens:
- precision = TP/(TP+FN)
- recall = TP/(TP + TN)
- The final value $f_1$ is it's harmonic mean. 

Each datapoint shows $f_1$ and $t$ which reffers the the threshold this value has been obtained with

```python
import denoiseg.instance_analysis as ia

# HACK bacward compatibility problem
preds = [p[1:] for p in predictions]
evaluations = ia.evaluate_match(test_x,test_y,preds)
mean_evaluations = ia.mean_evaluations(evaluations)
vis.plot_precision_recall_curve(mean_evaluations)
```

# Image Synthesis

```python
%load_ext
autoreload
%autoreload
2

import numpy as np
from matplotlib import pyplot as plt
from src.generators.images_gen import generate_image, save_images

IMG_PARAMS_PATH = "../../config/img_params.json"
IMG_SAVE_PATH = "../../../Data/gen3"

for i in range(5):
    root_seed = np.random.randint(0, 10 ** 5)
    img, mask = generate_image(root_seed, IMG_PARAMS_PATH)

    save_images(IMG_SAVE_PATH, root_seed, img, mask, IMG_PARAMS_PATH, 3)

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
