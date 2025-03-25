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
import matplotlib.pyplot as plt

def plot_pair(img,mask):
    _,axs = plt.subplots(1,2,figsize = (8,4))
    axs[0].imshow(img,vmin=0,vmax=1,cmap='gray')
    axs[1].imshow(mask,vmin=0,vmax=1,cmap='gray')
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


# def fn(ps):
#     seed, params = ps[0],ps[1]
#     img, mask = generate_image(seed, params)

#     img_path = f'/tmp/{seed}.png'
#     imageio.imwrite(img_path, np.uint8(img))

#     mask_path = f'/tmp/{seed}_mask.png'
#     imageio.imwrite(mask_path ,np.uint8(mask))
#     return img_path,mask_path


# from tqdm.contrib.concurrent import process_map

# mp_params = [(seed,params) for seed in seeds]
# r = process_map(fn, mp_params, max_workers=4)

```

```python

```

```python
from scipy.stats import qmc

def poisson(radius, n=None, rng= None,d = 2):
    if n is None:
        # n=1000 => fill all radius allows
        n = 10000

    if rng is None:
        rng = np.random.RandomState()

    
    seed = rng.randint(2**32)
    engine = qmc.PoissonDisk(d=d, radius=radius,seed = seed,ncandidates=5)
    samples = engine.fill_space()
    rng.shuffle( samples)
    return samples[:n]
```

```python
import shapely
from shapely.plotting import plot_polygon
from generators.finite_voronoi_gen import get_finite_voronoi

from type_definitions import (Image, IndexedRegion, Vertices, Edge, Point, RandomState, Coords, BorderMask,
                                  GradientImage)

from typing import List, Dict
```

```python
import numpy.typing as npt
```

```python

import warnings
from dataclasses import dataclass

def _sort2(a,b):
    if a < b:
        return a,b
    else:
        return b,a     

def deduplicate_edges_from_vertices(regions_vertices_idcs):
        edges_dedup = []
        encountered = {}
        region_edges = []
        for vs in regions_vertices_idcs:
            edges_oriented = []
            
            for a,b in zip(vs,np.roll(vs,-1)):
                edge_id = _sort2(a,b)
                reverted = edge_id != (a,b)
                if edge_id in encountered:
                    dedup_edge_idc = encountered[edge_id]
                else:
                    dedup_edge_idc = len(edges_dedup)
                    encountered[edge_id] = dedup_edge_idc
                    edges_dedup.append(edge_id)

                edges_oriented.append((dedup_edge_idc,reverted))
            
            region_edges.append(edges_oriented)
        
        return edges_dedup,region_edges
        
class VoronoiData():
    def __init__(self,edges,cell_edges):
        self.edges = edges
        self.cell_edges = cell_edges
        
    @classmethod
    def from_points(cls,points):
        vor = scipy.spatial.Voronoi(points)
        cell_vertices_idcs, vertices = get_finite_voronoi(vor)
        edges_dedup,cell_edges = deduplicate_edges_from_vertices(cell_vertices_idcs)
        edges = [ Edge(np.array([vertices[v1],vertices[v2]])) for v1,v2 in edges_dedup]
        
        return cls(edges,cell_edges)


    def collect_cells(self):
        polygons = []
        for reg in self.cell_edges:
            is_reversed_arr = [r for _,r in reg]
            reg_edges = [self.edges[edge_idx] for edge_idx,_ in reg]
            
            edges_directed = [ np.flip(e.vertices,axis=0) if r else e.vertices
                 for e,r in zip(reg_edges,is_reversed_arr)
            ]

            poly_vtcs = np.vstack([v[:-1] for v in edges_directed])
            polygon = shapely.Polygon(poly_vtcs)
            polygons.append(polygon)
        return polygons

    def render_masks(self,resolution):
        cells = self.collect_cells()    
        if len(cells)>255:
            warnings.warn(f"Too many cells found ({len(cells)}). Only 255 are supported")
            # todo encode value to RGB color and back.

        # to avoid overlaping polygons, each of them is painted to canvas
        # and subsequently extracted
        
        canvas = np.zeros([resolution]*2,dtype=np.uint8)
        
        for i,cell in enumerate(cells):
            xy01 = np.array(cell.exterior.coords.xy)
            xy = (xy01 * resolution).astype(int)
            cv2.fillPoly(canvas,[xy.T],color=i+1)

        return [np.uint8(canvas==i) for i in range(1,np.max(canvas) +1)]


    def plot(self, ax=None):
        if ax is None:
            _,ax = plt.subplots(1,1)
            ax.set_ylim((0,1))
            ax.set_xlim((0,1))
        
        for p in self.collect_cells():
            plot_polygon(p,ax=ax,color = 'gray')
        
        

@dataclass
class Edge():
    vertices:npt.NDArray
```

```python
def _norm(arr):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    return (arr - min_arr)/(max_arr - min_arr)
    
def sample_line(edge_length,min_piece_length,n=None,rng = None):
    if n is None:
        n = 1000
        
    radius = min_piece_length/edge_length
    samples_01 = list(sorted(poisson(radius,d=1,n=n+2,rng=rng)))
    if len(samples_01) <2:
        return np.array([0,edge_length/2,edge_length])[None].T
    else:
        return _norm(samples_01)*edge_length
```

```python

from scipy.interpolate import CubicSpline

def smooth_spline(points,smooth_n = 20):
    
    points_arr = np.array(points)
    n = len(points)
    t = np.linspace(0,1,n)
    cs = CubicSpline(t, points_arr)
    t_subsampled = np.linspace(0,1,smooth_n)
    
    return cs(t_subsampled)[:]


def subdivide_edge(start,end,min_piece_length,jaggedness,n=3,rng = None):
    if rng is None:
        rng = np.random.RandomState()
    
    start,end = np.stack([start,end])
    diff = end-start
    edge_length = np.linalg.norm(diff)

    if edge_length < min_piece_length*2:
        # don't subdivide too small pieces
        return np.stack([start,end])
    
    smp = sample_line(edge_length,min_piece_length=min_piece_length,n=n,rng=rng)
    smp = smp/edge_length
    jaggedness = np.minimum(jaggedness, edge_length/10)
    rnd = (rng.rand(len(smp)) - .5)*2
    rnd[[0,-1]] = 0

    subdivided_straight_line = start + smp*diff
    dx,dy = diff
    normal = np.array([-dy,dx])
    shifts = np.stack([rnd]*2).T * normal
    
    d = np.linalg.norm(shifts,axis=1) 

    # prevent errors during divison
    d[d==0] = 1
    shifts =shifts/np.expand_dims(d,axis=1) * jaggedness

    return subdivided_straight_line + shifts

def distort_edge(
    edge:Edge, 
    min_piece_length = .05,
    jaggedness = .1,
    smooth = True,
    rng = None
):
    s,e = edge.vertices[[0,-1]]
    
    xy = subdivide_edge(
        s,e,
        min_piece_length = min_piece_length,
        jaggedness = jaggedness,
        rng = rng
    )
    if smooth:
        xy = smooth_spline(xy)
    return Edge(xy)


    
y_start = .6
y_end = .4
x_start = .4
x_end = .6
s = np.array([x_start,y_start])
e = np.array([x_end,y_end])

edge = Edge(np.array([s,e]))
rng = np.random.RandomState(seed= 123)
edge_smooth = distort_edge(edge,rng= rng)

rng = np.random.RandomState(seed= 123)
edge_dist = distort_edge(edge,smooth=False,rng=rng)


plt.plot([x_start,x_end],[y_start,y_end])
plt.xlim((.38,.62))
plt.ylim((.38,.62))



plt.plot(edge_smooth.vertices.T[0],edge_smooth.vertices.T[1])
plt.plot(edge_dist.vertices.T[0]+.05,edge_dist.vertices.T[1])
```

```python
import scipy


def distort_cell_edges(
    self:VoronoiData,
    min_piece_length = .05,
    jaggedness = .1,
    smooth=True,
    rng=None,
):
    distorted_edges = [
        distort_edge(
            e,
            min_piece_length=min_piece_length,
            jaggedness=jaggedness,
            smooth=smooth,
            rng=rng
        ) 
        for e in self.edges
    ]

    return VoronoiData(distorted_edges ,self.cell_edges)
    

grain_size_min = .15
min_piece_length = .02
jaggedness = .01
rng= np.random.RandomState(seed=123)
grain_points = poisson(grain_size_min,rng = rng)
voronoi_data = VoronoiData.from_points(grain_points)
rng= np.random.RandomState(seed=123)
voronoi_distorted = distort_cell_edges(
    voronoi_data,
    smooth=False,
    min_piece_length=min_piece_length,
    jaggedness=jaggedness,
    rng=rng
)
rng= np.random.RandomState(seed=123)
voronoi_distorted_smooth = distort_cell_edges(
    voronoi_data,
    smooth=True,
    min_piece_length=min_piece_length,
    jaggedness=jaggedness,
    rng=rng
)

_,axs = plt.subplots(1,3,figsize = (17,5))

def plot_voronoi(data,ax = None):
    if ax is None:
        _,ax = plt.subplots(1,1)
    ax.plot(grain_points.T[0],grain_points.T[1],'x')
    ax.set_ylim((0,1))
    ax.set_xlim((0,1))
    
    data.plot(ax=ax)

    return ax

plot_voronoi(voronoi_data,axs[0])
plot_voronoi(voronoi_distorted,axs[1])
plot_voronoi(voronoi_distorted_smooth,axs[2])
plt.show()
```

```python
import cv2
import shading

voronoi = voronoi_distorted_smooth

width = 1024

masks = np.stack(voronoi.render_masks(width),dtype=np.float32)
rng = np.random.RandomState(seed = 123)
elevations = rng.rand(len(masks),1,1)

# TODO weird lines here. use dilation to overlap masks, sum it and blur afterwards

depth_map = np.max(masks * elevations,axis=0)   
normals = shading.calc_normals(depth_map)
shaded = shading.assign_orientation(normals,5/4*np.pi, strength = 1)
shaded = np.pad(shaded,(0,1),mode='edge')
```

```python


def plot_imgs(imgs,names=[],figsize=None):
    if figsize is None:
        figsize = (len(imgs)*6,6)
        
    _,axs = plt.subplots(1,len(imgs),figsize= figsize)
    for ax,img,name in zip(axs,imgs,names):
        ax.imshow(img,cmap='gray')
        ax.set_title(name)

imgs = [depth_map,normals/2 +.5,shaded]
names = ['depth map','normals','shaded']
plot_imgs(imgs,names)
```

```python
import noise
# todo optimize performance by reducing the texture to the size of the cell + padding

h,w = width,width

textures = []
for i,mask in enumerate(masks):
    roughness = 8 
    rng = np.random.RandomState(seed = 123 + i)
    tex = noise.pink_noise(width, rng.normal, roughness)[:h,:w]
    textures.append(tex)
```

```python
import scipy.ndimage as ndi

def uniform(low,high,rng,shape=1):
    if rng is None:
        rng = np.random.RandomState()
    
    if np.array(shape).shape != ():
        rnd = rng.rand(*shape)
    else:
        rnd = rng.rand(shape)
        
    return rnd*(high-low) + low
    


blur_sigmas = uniform(.8,1.2,rng,shape=len(masks))
blurred_masks = [ ndi.gaussian_filter(mask,blur_sigmas)  for mask,blur_sigmas in zip(masks,blur_sigmas)]

```

```python
elevations = rng.rand(len(textures))*3
depth_map = np.sum([ mask*(texture*.7 + elevation) for mask,texture,elevation in zip(masks,np.roll(textures,1),elevations)],axis=0)
normals = shading.calc_normals(depth_map)
shaded = shading.assign_orientation(normals,5/4*np.pi, strength = 1)
shaded = np.pad(shaded,(0,1),mode='edge')

depth_map_b = np.sum([ mask*(texture*.7 + elevation) for mask,texture,elevation in zip(blurred_masks,np.roll(textures,1),elevations)],axis=0)

depth_map_b = ndi.gaussian_filter(depth_map_b,1)
normals_b = shading.calc_normals(depth_map_b)
shaded_b = shading.assign_orientation(normals_b,5/4*np.pi, strength = 1)
shaded_b = np.pad(shaded_b,(0,1),mode='edge')


imgs = [depth_map,normals/2 +.5,shaded]
names = ['depth map','normals','shaded']
plot_imgs(imgs,names)
plt.show()

imgs = [depth_map_b,normals_b/2 +.5,shaded_b]
names = ['depth map','normals','shaded']
plot_imgs(imgs,names)
```

```python
img = np.clip(_norm(depth_map)/5 +.4 + shaded,0,1)
img_b = np.clip(_norm(depth_map_b)/5 +.4 + shaded_b,0,1)

_,axs = plt.subplots(2,1,figsize = (16,24))
axs[0].imshow(img,cmap='gray')
axs[1].imshow(img_b,cmap='gray')
```

```python
from blob_masks import blob_mask
# package https://github.com/SINTEF/python-blob-masks/tree/main

xy = poisson(.05,rng = rng)

precipitate_points = xy
blob_sizes = uniform(15,64,rng,shape=len(precipitate_points))
blobs = [blob_mask(seed,size=int(size)) for seed,size in enumerate(blob_sizes)]
```

```python
def generate_precipitates(width,precipitate_points):
    precipitates_mask = np.zeros((width,width))
    blob_sizes = uniform(15,64,rng,shape=len(precipitate_points))
    blobs = [blob_mask(seed,size=int(size)) for seed,size in enumerate(blob_sizes)]
    for (x,y), blob in zip(precipitate_points*width,blobs):
        blob = np.array(blob)
        
        x = abs(int(x))
        y = abs(int(y))
        h,w = blob.shape
        hh,ww = precipitates_mask.shape
        
        crop = precipitates_mask[max(y,0):min(y+h,hh),max(x,0):min(x+h,ww)]
        precipitates_mask[max(y,0):min(y+h,hh),max(x,0):min(x+h,ww)] = np.maximum(blob[:crop.shape[0],:crop.shape[1]],crop)
    return precipitates_mask

precipitates_mask = generate_precipitates(width,precipitate_points)

precipitates_mask_sharp = ndi.gaussian_filter(precipitates_mask,1,mode='nearest')

precipitates_mask = ndi.gaussian_filter(precipitates_mask,2,mode='nearest')
plt.imshow(precipitates_mask)
plt.show()
```

```python
depth_map_full = np.maximum(depth_map_b,precipitates_mask*(np.max(depth_map_b)+1))

normals_full = shading.calc_normals(depth_map_full)
shaded_full = shading.assign_orientation(normals_full,5/4*np.pi, strength = 1)
shaded_full = np.pad(shaded_full,(0,1),mode='edge')



imgs = [depth_map_full,normals_full/2 +.5,shaded_full]
names = ['depth map','normals','shaded']
plot_imgs(imgs,names)
```

```python
texture_bck = _norm(depth_map)*.3 +.4
texture = np.minimum(1 - precipitates_mask_sharp *.8,texture_bck)

res= np.clip(texture + shaded_full,0,1)

_,ax = plt.subplots(1,1,figsize = (16,16))
ax.imshow(res,cmap='gray')
```

```python
from tqdm.cli import tqdm
import noise

width = 1024
textures = []
for i in tqdm(range(100)):
    roughness = 8 
    rng = np.random.RandomState(seed = 123 + i)
    tex = noise.pink_noise(width, rng.normal, roughness) #[:h,:w]
    textures.append(tex)
```

```python
blob_sizes = uniform(15,64,rng,shape=200)
all_blobs = [blob_mask(seed,size=int(size)) for seed,size in tqdm(list(enumerate(blob_sizes))]
```

```python
import shading
import cv2
from blob_masks import blob_mask
# package https://github.com/SINTEF/python-blob-masks/tree/main

    
def gen( rng):
    width = 1024
    grain_size_min = .15
    min_piece_length = .02
    jaggedness = .01
    
    grain_points = poisson(grain_size_min,rng = rng)
    voronoi = VoronoiData.from_points(grain_points)
    voronoi_distorted_smooth = distort_cell_edges(
        voronoi,
        smooth=True,
        min_piece_length=min_piece_length,
        jaggedness=jaggedness,
        rng=rng
    )
    
    masks = np.stack(voronoi_distorted_smooth.render_masks(width),dtype=np.float32)
    
    
    ###
    
    blur_sigmas = uniform(.8,1.2,rng,shape=len(masks))
    blurred_masks = [ ndi.gaussian_filter(mask,blur_sigmas)  for mask,blur_sigmas in zip(masks,blur_sigmas)]
    
    ####
    
    elevations = rng.rand(len(textures))*3
    depth_map = np.sum([ mask*(texture*.7 + elevation) for mask,texture,elevation in zip(masks,np.roll(textures,1),elevations)],axis=0)
    
    normals = shading.calc_normals(depth_map)
    shaded = shading.assign_orientation(normals,5/4*np.pi, strength = 1)
    shaded = np.pad(shaded,(0,1),mode='edge')
    depth_map_b = np.sum([ mask*(texture*.7 + elevation) for mask,texture,elevation in zip(blurred_masks,np.roll(textures,1),elevations)],axis=0)
    depth_map_b = ndi.gaussian_filter(depth_map_b,1)
    
    
    ###
    
    n = 68
    precipitate_points = poisson(.05,rng = rng)[:n]
    # blob_sizes = uniform(15,64,rng,shape=len(precipitate_points))
    # blobs = [blob_mask(seed,size=int(size)) for seed,size in enumerate(blob_sizes)]
    idcs = np.arange(len(all_blobs))
    rng.shuffle(idcs)
    blobs = [all_blobs[i] for i in  idcs[:n]]
    
    precipitates_mask_sharpest = generate_precipitates(width,precipitate_points)
    precipitates_mask_sharp = ndi.gaussian_filter(precipitates_mask_sharpest,1,mode='nearest')
    precipitates_mask = ndi.gaussian_filter(precipitates_mask_sharpest,2,mode='nearest')
    
    ###
    
    depth_map_full = np.maximum(depth_map_b,precipitates_mask*(np.max(depth_map_b)+1))
    
    normals_full = shading.calc_normals(depth_map_full)
    shaded_full = shading.assign_orientation(normals_full,5/4*np.pi, strength = 1)
    shaded_full = np.pad(shaded_full,(0,1),mode='edge')
    
    
    ###
    
    texture_bck = _norm(depth_map)*.3 +.4
    texture = np.minimum(1 - precipitates_mask_sharp *.8,texture_bck)
    res= np.clip(texture + shaded_full,0,1)

    
    return res, precipitates_mask_sharpest
    

rng = np.random.RandomState(seed = 123)
img,mask = gen(rng)

plt.imshow(img)
plt.show()
plt.imshow(mask)
```

```python
from tqdm.cli import tqdm

#dataset = [gen(np.random.RandomState(seed = i)) for i in tqdm(range(200))]
dataset = []
for i in tqdm(range(200)):
    try:
        d = gen(np.random.RandomState(seed = i))
        dataset.append(d)
    except:
        print('fail ',i)
```

```python
np.max(dataset[0][0])
```

```python
import imageio.v2 as imageio
from pathlib import Path
root = Path('ds_prec_synth')
paths = [root/f'{i:03}' for i in range(len(dataset))]

for p,(img,mask) in zip(paths, dataset):
    p.mkdir(exist_ok=True,parents=True)
    img_path = p/'img.png'
    mask_path = p/'mask.png'

    imageio.imwrite(img_path, np.uint8(np.dstack([img]*3)*255))
    imageio.imwrite(mask_path, np.uint8(np.dstack([mask]*3)*255))

```

```python
p
```
