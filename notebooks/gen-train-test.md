---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python (training_env)
    language: python
    name: training_env
---

```python
import logging
logging.basicConfig(level=logging.INFO)
```

# Image Synthesis

```python
from pathlib import Path
import sys
sys.path.append("../src")
from main import generate

num = 2  # Number of images
gen = 16  # Generation number

save_path = "../../data"
params_path = Path("../config/img_params.json")

show = True  # Set to True if you want to show images
save = True  # Set to True if you want to save images

random_seeds = False  # Set to True for random seed generation, otherwise seeds from seeds_path txt are used in order
seeds_path = "../src/seeds.txt"
start_seed = 0 # Use seeds from index start_seed

generate(num=num, gen=gen, img_params_path=params_path, img_save_path=save_path,
         show=show, save=save, random_seeds=random_seeds, seeds_path=seeds_path, start_seed=start_seed)
```

# Load Images

```python
from pathlib import Path

model_path = Path('../../models/gen_train_test-synth_200-gen15-try3')
```

```python
test_root = Path('../../data/test')
# data_root = save_path.value
# data_root = Path('../image_generation/Data/gen14')
data_root = Path('../../data/gen15')
```

```python
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import semseg.io as io

img_paths = list(data_root.rglob('*/**/img.png'))
# img_paths = random.sample(img_paths, 52)

test_img_paths = list(test_root.rglob('*/**/img.png'))

label_paths = [ p.parent/'mask.png' for p in img_paths]

img_names = [p.parent.stem for p in img_paths]
imgs = list(map(io.read_image_normalized,tqdm(img_paths)))
labels = list(map(io.read_image_normalized,tqdm(label_paths)))

test_label_paths = [ p.parent/'mask.png' for p in test_img_paths]

test_names = [p.parent.stem for p in test_img_paths]
test_imgs = list(map(io.read_image_normalized,tqdm(test_img_paths)))
test_labels = list(map(io.read_image_normalized,tqdm(test_label_paths)))
```

```python
for i in range(2):
    plt.imshow(imgs[random.randint(0, len(img_names))], cmap="gray", vmin=0, vmax=1)
    plt.show()
```

# Prepare DataLoaders

```python
imgs = sum([imgs]*10,[]) # HACK fix batch problems
labels = sum([labels]*10,[])
```

```python
import semseg.dataset as ds

patch_size = 128
x_channels = 3
train_augumentation_fn = ds.setup_augumentation(
    patch_size = patch_size,
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
import torch

# model = torch.load(model_path)
test_preds = prd.segment_many(model,test_imgs,device = device,x_channels=x_channels)
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

small_filter = 4
test_labels = [filter_small(lbl,area_limit=small_filter) for lbl in test_labels]
```

```python
for img,label,pred in zip(test_imgs,test_labels,test_preds):
    fig,axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(img,vmin=0,vmax=1, cmap="gray")
    axs[1].imshow(label,vmin=0,vmax=1, cmap="gray")
    axs[2].imshow(pred,vmin=0,vmax=1, cmap="gray")
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

# Save Results

```python
path = model_path.name.split("/", 1)[-1]
results_path = Path("../../results")
results_path = results_path / path
metrics_path = results_path / "pres-rec-f1.txt"

def save_results():
    results_path.mkdir(exist_ok=True, parents=True)
    with open(metrics_path, "w") as file:
        file.write("Threshold Mean_precision Mean_recall Mean_f1 \n")
        for thr in np.linspace(.1,.95,10):
            mean_precision,    mean_recall,    mean_f1 = measure_labeled(thr,test_labels,test_preds)
            print(f"{thr=:.2f} {mean_precision=:.2f} {mean_recall=:.2f} {mean_f1=:.2}")
            file.write(f"{thr:.2f} {mean_precision:.2f} {mean_recall:.2f} {mean_f1:.2}\n")
            
save_results()
```

```python
# with open(results_path / "epoch_num", "w") as file:
#     file.write(f"Number of epochs: {last_epoch[0] + 1}")
```

```python
import numpy as np

def save_image_results():
    i = 1
    for img, label, pred in zip(test_imgs, test_labels, test_preds):
        fig, axs = plt.subplots(1,3,figsize=(16,4))
        for ax, title, im in zip(axs, ["img", "mask", "prediction"], [img, label, pred]):
            ax.imshow(im,vmin=0,vmax=1, cmap='gray')
            ax.set_title(title)

        plt.tight_layout()
        fig.savefig(results_path/f"res{i}.png")
        title = f"min={np.min(pred):.3f} max={np.max(pred):.3f}"
        plt.suptitle(title)
        plt.show()
        i += 1
    
save_image_results()

```

```python
import semseg.instance_matching as im
import semseg.prediction as prd
import torch
import numpy as np


def evaluate_images(model, imgs, gts, foreground_thr=0.5, metric=None, device="cpu"):
    if metric is None:
        metric = im.prepare_iou(foreground_thr=foreground_thr)

    preds = prd.segment_many(model, imgs, device=device, x_channels=x_channels)

    metrics = []
    for gt, pred in zip(gts, preds):
        fg = pred
        metric_res = metric(gt,fg)
        metrics.append(metric_res)
    return preds, np.squeeze(metrics)

def save_IoU(): 
    path = results_path / "IoU.txt"
    
    with open(path, "w") as file:
        file.write("Threshold Mean_IoU \n")
        for thr in np.linspace(.1, 0.95, 10):
            predictions, metrics = evaluate_images(
                model, 
                test_imgs,
                test_labels,
                device = device,
                foreground_thr=thr
            )

            print(f'Mean IoU {np.mean(metrics):.5f} for threshold {thr:.2f}')

            file.write(f'{thr:.2f} {np.mean(metrics):.2f} \n')


save_IoU()

```

```python
print("done")
```

```python
exit
```
