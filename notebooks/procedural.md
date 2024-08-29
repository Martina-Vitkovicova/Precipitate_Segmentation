---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
---

```python
%load_ext autoreload
%autoreload 2
```

Mitsuba
===

reference [here](https://mitsuba.readthedocs.io/en/stable/index.html)

```python
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

# switch to scalar_ad_rdb if you use laptop or other machine without GPU
# mi.set_variant("scalar_ad_rgb")
```

# Create Procedural Mesh

Tutorial page [here](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/mesh_io_and_manipulation.html)


## Create Mesh

Create in-memory object and save it to the disk.

```python
import drjit as dr
# Disk with a wavy fringe parameters
N = 100
frequency = 12.0
amplitude = 0.4

# Generate the vertex positions
theta = dr.linspace(mi.Float, 0.0, dr.two_pi, N)
x, y = dr.sincos(theta)
z = amplitude * dr.sin(theta * frequency)
vertex_pos = mi.Point3f(x, y, z)

# Move the last vertex to the center
vertex_pos[dr.eq(dr.arange(mi.UInt32, N), N - 1)] = 0.0

# Generate the face indices
idx = dr.arange(mi.UInt32, N - 1)
face_indices = mi.Vector3u(N - 1, (idx + 1) % (N - 2), idx % (N - 2))

#create empty mesh
mesh = mi.Mesh(
    "your_object_name",
    vertex_count=N,
    face_count=N - 1,
    has_vertex_normals=False,
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = dr.ravel(vertex_pos)
mesh_params["faces"] = dr.ravel(face_indices)


# Store this to disk to make scene setup easier for later
mesh.write_ply("wavydisk.ply")
```

<!-- #region -->
# Material

The basic material is diffuse. The one used below


Notable types:
- [Smooth Metal](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-conductor-conductor)
- [Rough Metal](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#rough-conductor-material-roughconductor)
- [Plastic](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-plastic-material-plastic)

Remember that mitsuba simulates photons, not electorns threfore other types of materials can be also viable option here.
Full supported material list [here](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html)
<!-- #endregion -->

```python
material = {
    "type": "diffuse",
    "reflectance": {
        "type": 'rgb',
        "value": [0.2, 0.25, 0.7]
    },
}
```

# Compose Scene

setup
- [camera](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html)
- [light](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html)
- objects with their material (see above)
- and much more

```python
mesh_dict = {
    "type": "ply",
    "filename": "wavydisk.ply",
    "bsdf": material,
}


res_x = 800
res_y = 600
camera = {
    "type": "perspective",
    "to_world": mi.ScalarTransform4f.look_at(
        origin=[0, -5, 5], target=[0, 0, 0], up=[0, 0, 1]
    ),
    "film": {
        "type": "hdrfilm",
        "pixel_format": "rgba",
        "width": res_x,
        "height": res_y,
        "rfilter": {"type": "box"},
    }
}


light = {
        'type': 'directional',
        'direction': [-.1, -.5, -1.0],
        'irradiance': {
            'type': 'rgb',
            'value': 1.0,
        }
    }
#light = {"type": "constant"}  # TRY ME and notice
# denoiser will start working
# alpha will stop working

scene = mi.load_dict(    {
    "type": "scene",
    "integrator": {"type": "path"},
    "light":light,
    "sensor": camera,
    "wavydisk": mesh_dict,
},parallel=False)

img = mi.render(scene)
```

```python
import matplotlib.pyplot as plt
import numpy as np

bitmap = np.array(mi.Bitmap(img).convert(
    mi.Bitmap.PixelFormat.RGBA, 
    mi.Struct.Type.Float32, 
    srgb_gamma=True
))
bitmap = np.clip(0, 1, bitmap)
plt.imshow(bitmap)
```

# Alpha

Alpha doesn't work now :(

```python
#the render above is already setup ho have alpha channel
# background
plt.imshow(np.ones_like(bitmap) * [1,.3,0,1] )

# foreground withnout any change
plt.imshow(bitmap)
```

# Denoised

```python
h,w = bitmap.shape[:2]
denoiser = mi.OptixDenoiser(
    input_size=[w, h], albedo=False, normals=False, temporal=False
)
plt.imshow(denoiser(img))

```

```python

```
