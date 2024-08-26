---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: computer-vision
    language: python
    name: .venv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import logging
logging.basicConfig(level=logging.INFO)

import semseg.io as io
```

```python
from pathlib import Path
real_root = Path('/home/jry/data/delisa_2024/')
synt_root = Path('/home/jry/data/delisa_synt/')
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

small_filter = 4
img_paths = list(synt_root.rglob('*/**/img.png'))
test_img_paths = list(real_root.rglob('*/**/img.png'))

model_path = Path('/home/jry/jupyterlab/training/delisa_2024_synth/model.pth')
label_paths = [ p.parent/'mask.png' for p in img_paths]

img_names = [p.parent.stem for p in img_paths]
imgs = list(map(io.read_image_normalized,tqdm(img_paths)))
labels_prec = list(map(io.read_image_normalized,tqdm(label_paths)))
labels_prec = [ filter_small(label,small_filter) for label in tqdm(labels_prec,desc='filtering small')]

test_label_paths = [ p.parent/'label.png' for p in test_img_paths]

test_names = [p.parent.stem for p in test_img_paths]
test_imgs = list(map(io.read_image_normalized,tqdm(test_img_paths)))
test_labels_prec = list(map(io.read_image_normalized,tqdm(test_label_paths)))
test_labels_prec = [ filter_small(label,small_filter) for label in tqdm(test_labels_prec,desc='filtering small')]
```

```python
# TODO load artifacts
# TODO change this to load the real labels, the following line uses just the precipitate labels
labels_art = [l for l in labels_prec]
```

```python
imgs = sum([imgs]*10,[]) # HACK fix batch problems
labels_prec = sum([labels_prec]*10,[])
labels_art = sum([labels_art]*10,[])
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
```

# Adding new Dataset

In this cell, there is the code that adds new output channel.

```python
from torch.utils.data import Dataset
from typing import List
import numpy.typing as npt
from torch.utils.data import DataLoader

import sklearn.model_selection as ms
from semseg.dataset import SegmentationDataset, DataLoader,AugTransform,TrainingPair

class SegmentationDatasetWArtifacts(Dataset):
    def __init__(
        self,
        images: List[npt.NDArray],
        labels_prec: List[npt.NDArray],
        labels_art: List[npt.NDArray],
        transform: AugTransform,
        x_channels = 1,
    ):
        self.images = [np.float32(img) for img in images]
        self.labels_prec = [np.float32(label) for label in labels_prec]
        self.labels_art = [np.float32(label) for label in labels_art]
        
        self.transform = transform
        self.x_channels = x_channels

    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, idx) ->TrainingPair:
        image = self.images[idx]
        label_pred = self.labels_prec[idx]
        label_art = self.labels_art[idx]

        # HACK: pack two label so they pass throught the same augumentation
        mask = np.dstack([label_pred,label_art])
        transformed = self.transform(image=image, mask=mask)
        image_aug = transformed["image"]
        label_aug = transformed["mask"]
        label_prec_aug = label_aug[:,:,0]
        artifacts = label_aug[:,:,1]
        
        
        
        # Just precipitates
        precipitates = np.float32(label_prec_aug > 0)
        # Background
        background = 1 - precipitates
        # Precipitate Borders
        border = _get_border(precipitates)

        # TODO the following 4 channels precipitates may need some re-thinking
        # for example, background/border is generated using using precipitates
        # maybe the artifacts should be taken into account as well

        y = np.stack([precipitates, background, border,artifacts])
        x = np.stack([image_aug]*self.x_channels)

        return {
            "x": x,
            "y": y,
        }

def _get_border(foreground_label: npt.NDArray) -> npt.NDArray:
    fg_int = np.uint8(foreground_label)
    kernel = np.ones((3, 3))
    eroded = cv2.morphologyEx(fg_int, cv2.MORPH_ERODE, kernel)
    return np.float32(fg_int - eroded)





val_size: float = 0.25
seed: int = 123
x_channels = 1
batch_size = 32

# pack
lbls_combined = list(zip(labels_art,labels_prec))
img_train, img_val, label_combined_train, label_combined_val = ms.train_test_split(
imgs, lbls_combined, test_size=val_size, random_state=seed
)

label_art_train = []
label_prec_train = []

# unpack 
for art,prec in label_combined_train:
    label_art_train.append(art)
    label_prec_train.append(prec)


label_art_val= []
label_prec_val = []

for art,prec in label_combined_val:
    label_art_val.append(art)
    label_prec_val.append(prec)


dataset_train = SegmentationDatasetWArtifacts(
    img_train,
    label_prec_train,
    label_art_train,
    train_augumentation_fn,
    x_channels = x_channels
)
dataset_val = SegmentationDatasetWArtifacts(
    img_val,
    label_prec_val,
    label_art_train,
    val_augumentation_fn,
    x_channels,
)

train_dl = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
)
val_dl = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
)
```

```python
visu_dl = DataLoader(
    dataset_train,
    batch_size=1,
)

for img_train in visu_dl:
    _,axs = plt.subplots(1,5,figsize=(20,4))
    img = np.squeeze(img_train['x'][0])
    lbls = img_train['y'][0]
    axs[0].imshow(img)
    axs[0].set_title('image')
    titles = ['precipiates','background','borders','artifacts']
    ys = img_train['y'][0]
    for ax,title,y in zip(axs[1:],titles,ys):
        ax.imshow(y)
        ax.set_title(title)
    break
```

# Model

```python
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=x_channels,
    classes=4, # THIS number has been changed from 3 to 4 to account for artifacts
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
#torch.save(model, model_path)
```

# Test

```python
import semseg.prediction as prd

test_preds = prd.segment_many(model,test_imgs,device = device,x_channels=x_channels)
```

```python
test_labels = [filter_small(lbl,area_limit=small_filter) for lbl in test_labels]
for img,label,pred in zip(test_imgs,test_labels,preds):
    fig,axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(img,vmin=0,vmax=1)
    axs[1].imshow(label,vmin=0,vmax=1)
    axs[2].imshow(pred,vmin=0,vmax=1)
    title = f"min={np.min(pred):.3f} max={np.max(pred):.3f}"
    plt.suptitle(title)
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
        data = list(tqdm(data))
        
    for label,prediction in data:
        pred_thr = np.uint8(prediction>=thr)
        pred_thr = filter_small(pred_thr,small_filter)
        label_thr = np.uint8(label)
        
        precision,recall,f1 = im.measure_precision_recall_f1(pred_thr,label_thr,component_limit=1000)
        precisions.append(precision)
        recalls.append(recall)
        
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = 2* (mean_precision*mean_recall)/(mean_precision + mean_recall)
    return mean_precision,mean_recall, mean_f1


# mean_precision,mean_recall, mean_f1 = measure_labeled(.5,test_labels,test_preds)
# print(f"{thr=:.2f} {mean_precision=:.2f} {mean_recall=:.2f} {mean_f1=:.2}")
```

```python
for thr in np.linspace(.05,.6,12):
    mean_precision,    mean_recall,    mean_f1 = measure_labeled(thr,test_labels,test_preds)
    print(f"{thr=:.2f} {mean_precision=:.2f} {mean_recall=:.2f} {mean_f1=:.2}")

```

```python
import pandas as pd

def extract_individuals(mask):
    
    n,arr = cv2.connectedComponents(mask)
    
    masks = (np.uint8(arr == i) for i in range(1,n))
    areas = np.array([np.sum(mask) for mask in masks])
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    
    return pd.DataFrame(areas[None].T,columns=['area_px'])

all_masks = list(map(io.read_image_normalized,tqdm(label_paths)))
all_masks = [ filter_small(label,small_filter) for label in tqdm(labels)]
all_masks = test_preds + all_masks
all_names = test_names + img_names

best_thr = .2
precipitates_list = []
for pred in tqdm(all_masks,desc='extracting'):
    mask = np.uint8(pred>=best_thr)
    precipitates = extract_individuals(mask)
    precipitates_list.append(precipitates)

ums = [int(name.split('_')[0]) for name in all_names]
for um,pred,df in zip(ums,all_masks,precipitates_list):
    img_width = pred.shape[0]
    px_to_um = um/img_width
    df['area_um'] = df['area_px'] * px_to_um

all_dfs = pd.concat(precipitates_list)
xs,bins = plt.hist(all_dfs['area_um'],color = 'black',bins = 250,density=True)[:2]

for df in precipitates_list:
    plt.hist(df['area_um'],alpha = .1,bins=bins,density=True)
    
xs = np.linspace(0,len(bins)-1,6).astype(int)
plt.xlabel("$\mu$m$^2$");


```

```python
groups = {}
for um,df in zip(ums, precipitates_list):
    g = groups.setdefault(um, [])
    g.append(df)

dfs_dict = {}
for um,group in groups.items():
    df = pd.concat(group)
    df = df[df['area_um']!=0]
    dfs_dict[um] = df

dense = False
areas_250 = dfs_dict[250]['area_um']
bins = np.linspace(0,np.max(areas_250),2000)

relative_diff = len(groups[40])/len(groups[250])
relative_area = 250**2/40**2
print(f"{relative_diff=} {relative_area=}")
areas_40_scaled = np.concatenate([dfs_dict[40]['area_um']] * int(relative_area * relative_diff))

bins_250 = plt.hist(areas_250,density=dense,bins = bins,alpha = .6,label='250 $\mu m$')[1]
bins_40 =  plt.hist(areas_40_scaled,density=dense,bins=bins,alpha = .6,label='40 $\mu m$')[1]

plt.xlim(0,30)
plt.xlabel('$\mu m^2$')
plt.ylabel('Count per bin')
plt.legend();
```

```python

```
