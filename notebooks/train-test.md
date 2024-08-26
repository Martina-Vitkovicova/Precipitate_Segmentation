---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: gridsegmentation
    language: python
    name: gridsegmentation
---

```python
%load_ext autoreload
%autoreload 2
%cd ../scripts
```

```python
import logging
logging.basicConfig(level=logging.INFO)


```

```python
from pathlib import Path
real_root = Path('/home/knotek/jupyterlab/data/delisa_2023/')
synt_root = Path('/home/knotek/jupyterlab/data/delisa_synt/')
```

```python
import cv2
import numpy as np

def filter_small(mask,area_limit):
    if area_limit == 0:
        return mask
    
    old_dtype = mask.dtype
    mask = np.uint8(mask)
    n,lbs = cv2.connectedComponents(mask)
    base = np.zeros_like(mask,dtype=np.uint8)
    for i in range(1,n):
        component = np.uint8(lbs==i)
        size = np.sum(component)
        if size >= area_limit:
            base = base | component
    return base.astype(old_dtype)
```

```python
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import semseg.io as io

small_filter = 4
img_paths = list(synt_root.rglob('*/**/img.png'))
test_img_paths = list(real_root.rglob('*/**/img.png'))

model_path = Path('/home/knotek/jupyterlab/training/delisa_2023_synth/model.pth')
label_paths = [ p.parent/'mask.png' for p in img_paths]

img_names = [p.parent.stem for p in img_paths]
imgs = list(map(io.read_image_normalized,tqdm(img_paths)))
labels = list(map(io.read_image_normalized,tqdm(label_paths)))
labels = [ filter_small(label,small_filter) for label in tqdm(labels,desc='filtering small')]

test_label_paths = [ p.parent/'mask.png' for p in test_img_paths]

test_names = [p.parent.stem for p in test_img_paths]
test_imgs = list(map(io.read_image_normalized,tqdm(test_img_paths)))
test_labels = list(map(io.read_image_normalized,tqdm(test_label_paths)))
test_labels = [ filter_small(label,small_filter) for label in tqdm(test_labels,desc='filtering small')]
```

```python
imgs = sum([imgs]*10,[]) # HACK fix batch problems
labels = sum([labels]*10,[])
```

# Prepare DataLoaders

```python
import semseg.dataset as ds
patch_size = 128
x_channels = 3
train_augumentation_fn = ds.setup_augumentation(
    patch_size = patch_size,
    elastic=True,
    brightness_contrast=True,
    blur_sharp_power=1,
    flip_horizontal=False,
    flip_vertical=False,
    noise_value=0.01,
    rotate_deg=10
    )
val_augumentation_fn = ds.setup_augumentation(patch_size = patch_size)

train_dl,val_dl = ds.prepare_dataloaders(
    imgs,
    labels,
    train_augumentation_fn,
    val_augumentation_fn,
    batch_size= 32,
    val_size = 0.25,
    x_channels=x_channels
)
```

# Model

```python
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=x_channels,
    classes=3,
    activation="sigmoid"
)
```

# Loss

```python
from semseg.training import FocalLoss

loss = FocalLoss(alpha = .8,gamma=2)
```

# Training



```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
from tqdm.contrib.logging import logging_redirect_tqdm
from semseg.training import train

checkpoint_path = model_path.parent/f"checkpoint-{model_path.name}"
with logging_redirect_tqdm():
    train_loss,val_loss = train(
        model,
        train_dl,
        val_dl,
        loss,
        checkpoint_path=checkpoint_path,
        patience=20,
        scheduler_patience= 10,
        device = device
    )

model_path.parent.mkdir(exist_ok=True,parents=True)
torch.save(model, model_path)
```

# Test

```python
import semseg.prediction as prd

test_preds = prd.segment_many(model,test_imgs,device = device,x_channels=x_channels)
```

```python
test_labels = [filter_small(lbl,area_limit=small_filter) for lbl in test_labels]
```

```python
for img,label,pred in zip(test_imgs,test_labels,test_preds):
    fig,axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(img,vmin=0,vmax=1)
    axs[1].imshow(label,vmin=0,vmax=1)
    axs[2].imshow(pred,vmin=0,vmax=1)
    title = f"min={np.min(pred):.3f} max={np.max(pred):.3f}"
    plt.suptitle(title)
    for ax in axs:
        ax.set_xlim((0,250))
        ax.set_ylim((750,1000))
    plt.show()
    
    break
```

```python
import numpy as np
import semseg.instance_matching as im

def measure_labeled(thr, labels,predictions,use_tqdm=False):
    precisions = []
    recalls = []
    data = zip(labels,predictions)
    if use_tqdm:
        data = list(tqdm(data))
        
    for label,prediction in data:
        pred_thr = np.uint8(prediction>=thr)
        pred_thr = filter_small(pred_thr,small_filter)
        label_thr = np.uint8(label)
        
        precision,recall,f1 = im.measure_precision_recall_f1(pred_thr,label_thr,component_limit=1000)
        precisions.append(precision)
        recalls.append(recall)
        
    mean_precision = np.nanmean(precisions)
    mean_recall = np.nanmean(recalls)
    mean_f1 = 2* (mean_precision*mean_recall)/(mean_precision + mean_recall)
    return mean_precision,mean_recall, mean_f1


# mean_precision,mean_recall, mean_f1 = measure_labeled(.5,test_labels,test_preds)
# print(f"{thr=:.2f} {mean_precision=:.2f} {mean_recall=:.2f} {mean_f1=:.2}")
```

```python
for thr in np.linspace(.3,.6,4):
    mean_precision,    mean_recall,    mean_f1 = measure_labeled(thr,test_labels,test_preds)
    print(f"{thr=:.2f} {mean_precision=:.2f} {mean_recall=:.2f} {mean_f1=:.2}")
```

```python

```
