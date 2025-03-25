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
from tqdm.cli import tqdm
import imageio
from pathlib import Path
import numpy as np

def imread(p):
    img = imageio.imread(p)[:,:]
    match img.shape:
        case h,w,d:
            img  = img[:,:,0]
        case h,w:
            img = img
        case _:
            raise Exception()
    mx = np.max(img)
    if mx <=1:
        return img
    elif mx <= 255:
        return img/255
    elif mx <= 256**2 -1:
        return img/(256**2 -1)

def stem_to_key(stem):
    if "mask" in stem:
        return stem.split('_')[0],'mask'
    else:
        return stem, "img"

targets = list( Path('/home/knotek/source/Martina-Vitkovicova/Precipitate_Segmentation/dataset_20250213/').glob('*.png'))

img_dict = {}
for p in tqdm(targets):
    k1,k2 = stem_to_key(p.stem)
    d = img_dict.setdefault(k1,{})
    d[k2] = imread(p)


img_pairs = [(k, v['img'],v['mask']) for k,v in img_dict.items() if "img" in v and "mask" in v]
```

```python
model_path = Path('/tmp/jk_models')
```

```python
all_names = [name for name,_,_ in img_pairs] 
all_imgs = [img for _,img,_ in img_pairs]
all_masks = [mask for _,_,mask in img_pairs]
```

```python
import matplotlib.pyplot as plt

def plot_pair(img,mask):
    _,axs = plt.subplots(1,2,figsize = (8,4))
    axs[0].imshow(img,vmin=0,vmax=1,cmap='gray')
    axs[1].imshow(mask,vmin=0,vmax=1,cmap='gray')


plot_pair(all_imgs[0],all_masks[0])
```

```python
from tqdm.cli import tqdm
import matplotlib.pyplot as plt
import semseg.io as io

import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path

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
import numpy as np

rng = np.random.RandomState(seed =123)
n = len(img_pairs)

idcs = rng.permutation(np.arange(n))

test_idcs = idcs[:100]
train_idcs = idcs[100:]

imgs = [all_imgs[i] for i in train_idcs]
labels = [all_masks[i] for i in train_idcs]
```

```python
import semseg.dataset as ds
patch_size = 128
x_channels = 3

train_augumentation_fn = ds.setup_augumentation(
    patch_size = patch_size,
    #elastic=True,
    brightness_contrast=True,
    blur_sharp_power=1,
    flip_horizontal=False,
    flip_vertical=False,
    #noise_value=0.01,
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

```python
import torch
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=x_channels,
    classes=3,
    activation="sigmoid"
)
from semseg.training import FocalLoss

loss = FocalLoss(alpha = .8,gamma=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm.contrib.logging import logging_redirect_tqdm
from semseg.training import train


checkpoint_path = model_path.parent/f"checkpoint-{model_path.name}"

```

```python
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

```python
import semseg.prediction as prd

model = torch.load(model_path,weights_only = False)

test_imgs = [all_imgs[i] for i in test_idcs]
test_labels = [all_masks[i] for i in test_idcs]

test_preds = prd.segment_many(model,test_imgs,device = device,x_channels=x_channels)
```

```python
for img,label,pred in zip(test_imgs[:3],test_labels,test_preds):
    fig,axs = plt.subplots(1,3,figsize=(18,7))
    axs[0].imshow(img,vmin=0,vmax=1)
    axs[1].imshow(label,vmin=0,vmax=1)
    axs[2].imshow(pred,vmin=0,vmax=1)
    # title = f"min={np.min(pred):.3f} max={np.max(pred):.3f}"
    # plt.suptitle(title)
    # for ax in axs:
    #     ax.set_xlim((0,250))
    #     ax.set_ylim((750,1000))
    plt.show()
```

```python
import numpy as np
import semseg.instance_matching as im

def measure_labeled(thr, labels,predictions,use_tqdm=False):
    precisions = []
    recalls = []
    data = zip(labels,predictions)
    if use_tqdm:
        data = tqdm(list(data))
        
    for label,prediction in data:
        pred_thr = np.uint8(prediction>=thr)
        #pred_thr = filter_small(pred_thr,small_filter)
        label_thr = np.uint8(label)
        
        precision,recall,f1 = im.measure_precision_recall_f1(pred_thr,label_thr,component_limit=1000)
        precisions.append(precision)
        recalls.append(recall)
        
    mean_precision = np.nanmean(precisions)
    mean_recall = np.nanmean(recalls)
    mean_f1 = 2* (mean_precision*mean_recall)/(mean_precision + mean_recall)
    return mean_precision,mean_recall, mean_f1


thr = .5
mean_precision,mean_recall, mean_f1 = measure_labeled(thr,test_labels,test_preds,use_tqdm = True)
print(f"{thr=:.2f} {mean_precision=:.2f} {mean_recall=:.2f} {mean_f1=:.2}")
```

# REAL data

```python
import semseg

delisa_2023_path = Path('/home/knotek/data/segmentation_delisa/delisa_2023/labeled/')
delisa_img_paths = list(delisa_2023_path.rglob('img.png'))

delisa_imgs = [ imread(p) for p in tqdm(delisa_img_paths)]
delisa_masks = [ imread(p/'..'/'mask.png') for p in tqdm(delisa_img_paths)]
```

```python
plot_pair(delisa_imgs[5],delisa_masks[5])
```

```python
delisa_preds = prd.segment_many(model,delisa_imgs,device = device,x_channels=x_channels)
```

```python
for img,label,pred in zip(delisa_imgs[:10],delisa_masks,delisa_preds):
    fig,axs = plt.subplots(1,3,figsize=(18,7))
    axs[0].imshow(img,vmin=0,vmax=1)
    axs[1].imshow(label,vmin=0,vmax=1)
    axs[2].imshow(pred,vmin=0,vmax=1)
    # title = f"min={np.min(pred):.3f} max={np.max(pred):.3f}"
    # plt.suptitle(title)
    # for ax in axs:
    #     ax.set_xlim((0,250))
    #     ax.set_ylim((750,1000))
    plt.show()
```
