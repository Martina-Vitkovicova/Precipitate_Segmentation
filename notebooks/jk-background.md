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
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from perlin_numpy import generate_perlin_noise_2d

from generators.background_gen import voronoi_bg
from generators.precipitates_gen import add_precipitates
from type_definitions import Image, RandomState
from datetime import datetime

```

```python
import numpy as np

def _norm(arr):
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_min == a_max:
        return arr
    else:
        return (arr - a_min)/(a_max - a_min)
```

```python
import imageio
```

```python
import skimage.transform as tr

# delisa_2023_path = Path('/home/knotek/data/segmentation_delisa/delisa_2023/labeled/')
# paths = list(delisa_2023_path.rglob('*.png'))
picked_path = Path('/home/knotek/data/segmentation_delisa/delisa_2023/labeled/DELISA_LTO_08_02_450_4_49/img.png')

img = imageio.imread(picked_path) 
img = _norm(img)
picked_img = img [270:398,515:643]

_,ax = plt.subplots(1,1,figsize = (10,10))
ax.imshow(picked_img)
picked_img.shape
```

```python
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def interpolate_spline(pts,subsample_points = 250):
    points = list(pts)
    points.append(points[0])
    points_arr = np.array(points)
    
    n = len(points)
    t = np.linspace(0,1,n)
    cs = CubicSpline(t, points_arr, bc_type='periodic')
    
    t_subsampled = np.linspace(0,1,subsample_points)
    return cs(t_subsampled)[:]

def scale(pts,factor):

    center = np.mean(pts,axis=0)
    c = pts - center
    c = c*factor
    return c + center

prec_coors_pts = np.array([
    [122,72],
    [125,67],
    [128,72],
    [125,77],
    [123,78],
    [121,77],
]) - [15,0]
prec_coors_pts = scale(prec_coors_pts,1.2)



grain_elevation_pts = np.clip(np.array([
    [0,0],[40,15],[98,134],[98,144],[0,144]
])- [16,0],0,127)


grain_pts = np.array([
    [30,7],
    [70,8],
    [123,12],
    [131,54],
    [103,120],
    #[99,129],
    [90,123],
    [63,98],
    #[60,95],
    [63,78],
    [55,60],
    [35,40],
    [22,20],
])- [16,0]

precipitate_xy = interpolate_spline(prec_coors_pts)
grain_xy = interpolate_spline(grain_pts)

_,ax = plt.subplots(1,1,figsize = (10,10))
ax.imshow(picked_img,cmap='gray')
ax.plot(precipitate_xy.T[0],precipitate_xy.T[1],c='r')
ax.plot(grain_xy.T[0],grain_xy.T[1],c='yellow')
ax.plot(grain_elevation_pts.T[0],grain_elevation_pts.T[1],c='green')
```

# Texture

```python
import scipy.ndimage as ndi
import cv2
import bezier

def render_bitmask(polygon_xy,shape):
    canvas = np.zeros(shape,dtype=np.uint8)
    cv2.fillPoly(canvas, [polygon_xy.astype(int)], color=(255))
    return canvas/255

grain_canvas = render_bitmask(grain_xy,picked_img.shape)
step_map = render_bitmask(grain_elevation_pts,picked_img.shape)
precipitate_fg = render_bitmask(precipitate_xy,picked_img.shape)
```

```python
background_mask = 1-grain_canvas
foreground_mask = grain_canvas
background= step_map
foreground = grain_canvas

def min_blend(background,the_min):
    return np.minimum(background,the_min)

def max_blend(background,the_max):
    return np.maximum(background,the_max)



def blend(background,grain,precipitate):
    cmb = min_blend(
        background+1,
        (1-grain)*(np.max(background)+1)
    )
    
    prec_level = precipitate * (np.max(cmb) + 1)
    return max_blend(cmb,prec_level)    

cmb = blend(background,grain_canvas,precipitate_fg)
names =['combined','background','foreground mask','precipitate']
imgs = [cmb,background,foreground,precipitate_fg]
fig,axs = plt.subplots(1,len(imgs),figsize=(len(imgs)*4,4))
for ax,img,name in zip(axs,imgs,names):
    ax.imshow(img,cmap='gray')
    ax.set_title(name)
```

```python

fg_sigma = 1
bg_sigma = .8
prec_sigma = .8

fg_mask_bl = ndi.gaussian_filter(foreground_mask,fg_sigma)
bg_bl = ndi.gaussian_filter(background,bg_sigma)
precipitate_bl = ndi.gaussian_filter(precipitate_fg,prec_sigma)

cmb = blend(bg_bl,fg_mask_bl,precipitate_bl)

names =['combined','background','foreground mask','prec']
imgs = [cmb,bg_bl,fg_mask_bl,precipitate_bl]

fig,axs = plt.subplots(1,len(imgs),figsize = (len(imgs)*4,4))
for ax,img,name in zip(axs,imgs,names):
    ax.imshow(img,cmap='gray')
    ax.set_title(name)
plt.suptitle('blurred')
plt.show()
```

```python
ns = (np.random.random(cmb.shape) -.5)*2
```

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import noise

h,w = picked_img.shape
size = np.maximum(h,w)
roughness = 8
rng = np.random.RandomState(seed = 123)
pic1 = noise.pink_noise(size, rng.normal, 8)[:h,:w] *2
pic2 = noise.pink_noise(size, rng.normal, 6)[:h,:w]
back_gen = noise.pink_noise(size, rng.normal, .8)[:h,:w]

_,axs = plt.subplots(1,3,figsize = (8,12))

c1 = [pic1,pic2,back_gen]

for ax,img in zip(axs,c1):
    ax.imshow(img)
```

```python
np.max(fg_mask_bl),np.min(fg_mask_bl)
```

```python
foreground_tex = pic1*.2 
background_tex = pic2*.2+ (4+bg_bl)/5

depth_map_no_noise = (foreground_tex * fg_mask_bl + background_tex * (1-fg_mask_bl))

# in min blending, zero background will effectivelly disable 
precipitate_bl_inf_bg = precipitate_bl.copy()
precipitate_bl_inf_bg[precipitate_bl_inf_bg<.001] = -np.inf

depth_map_no_noise = max_blend(depth_map_no_noise, precipitate_bl_inf_bg*3)
depth_map = depth_map_no_noise + back_gen*.3

_,axs = plt.subplots(1,3,figsize = (12,12))
axs[0].imshow(picked_img,cmap='gray')
axs[1].imshow(depth_map_no_noise,cmap='gray')
axs[2].imshow(depth_map,cmap='gray')
np.max(depth_map)
```

```python
import shading
    
normals = shading.calc_normals(depth_map)
```

```python
def vis_normals(normal_vectors):
    return  normal_vectors/2 + .5

n = normals
plt.imshow(vis_normals(n))
plt.show()
```

```python
    

def _index(arr,y,x):
    return np.array([x,y,arr[y,x]])

def _unit(vec):
    d = np.linalg.norm(vec)
    return vec/d



normals = calc_normals(depth_map)
#normals = calc_normals(depth_map_origin)

shaded = shading.assign_orientation(normals,5/4*np.pi, strength = 1)



imgs = [depth_map,vis_normals(normals), shaded,picked_img]

_,axs  = plt.subplots(1,len(imgs),figsize = (20,5))
for ax,img in zip(axs,imgs):
    ax.imshow(img,cmap='gray')
axs[2].set_title('JUST dept map, no texture')
```

```python

```

```python
def add_salt_peper(img, p):
    img = img.copy()
    sp = np.random.rand(*img.shape[:2])

    salt = sp> 1-p
    peper = sp < p

    img[salt] = 1
    img[peper] = 0
    return img

    
    
    
bck_texture = _norm(np.clip(ndi.gaussian_filter((np.random.rand(*shaded.shape)+.5)/2,2) + pic2 ,.3,.8) )/10 +.4
grain_texture = add_salt_peper(ndi.gaussian_filter(np.random.rand(*shaded.shape),.5)/4 + .25, .05)/4 +.4
prec_texture = np.ones_like(shaded) *.2


textured= (bck_texture*(1-fg_mask_bl) + grain_texture*fg_mask_bl)*(1-precipitate_bl) + prec_texture*precipitate_bl

imgs = [textured, bck_texture,grain_texture,prec_texture]
_,axs = plt.subplots(1,len(imgs),figsize = (len(imgs)*4,4))
for ax,img in zip(axs,imgs):
    ax.imshow(img,cmap='gray',vmin=0,vmax=1)
```

```python
final = np.clip( textured + shaded - np.mean(shaded),0,1)

imgs = [picked_img,final, textured,_norm(shaded)]
names = ['original','synth','texture','shading']

_,axs = plt.subplots(1,len(imgs),figsize = (len(imgs)*4,4))
for ax,img,name in zip(axs,imgs,names):
    ax.imshow(img,cmap='gray',vmin=0,vmax=1)
    ax.set_title(name)
axs[2].imshow(imgs[2],cmap='gray')
```

```python

```
