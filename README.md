# Synthetic SEM Image Generator

## Overview
This generator produces synthetic SEM-like images of irradiated stainless steel microstructures
with corresponding segmentation masks. 
It was developed primarily for precipitate segmentation under limited annotated data, 
but can also be used as an experimental framework for studying the impact of controlled visual features 
on neural network training.

A minimal end-to-end segmentation training pipeline implemented as a Jupyter notebook
demonstrates how synthetic images can be combined with real data to train and evaluate a neural network segmentation model.


## Modelled Image Components
Image generation is controlled via a JSON configuration file. The main configurable components are:

### Matrix and Grain Structure
Represents the polycrystalline metal matrix forming the image background, subdivided into grains with visible boundaries and texture variations.
- Voronoi-based grain partitioning
- Configurable number and size of grains
- Configurable grain boundary width, jaggedness, noise and shading
- Configurable texture composed of Perlin noise, Gaussian noise and porous pattern.

### Precipitates
Often very small, dark features located on top of the matrix surface, usually exhibiting local shading.
- Spline-based shapes with embossing
- Configurable number, size range, size distribution and intensity range.

### Etch Pits
Recessed surface defects in the matrix with pronounced shading.
- Elliptical shapes with perturbations
- Configurable number, size range, intensity range, shading and texture.

### Imaging Artefacts
Non-material effects introduced by the imaging process.
- Global illumination gradients
- Corner and bottom blur effects
- Directional lighting
- Noise.


## Outputs
The generator produces datasets consisting of directories with:
- `img.png` – synthetic grayscale SEM-like image  
- `mask.png` – binary precipitate mask  
- `metadata.json` – generation metadata  

Dataset size and image size are configurable.


## Configuration and Reproducibility
- Generator behaviour is fully defined by JSON configuration files
- Randomness is controlled via explicit random seeds
- Identical configurations and seeds produce identical datasets
- Example configuration files are provided and can be used directly.


## Intended Usage
Typical workflow:
1. Select or modify a JSON configuration  
2. Generate a synthetic dataset offline  
3. Train segmentation models using the generated data  
4. Evaluate performance on real images 


### Requirements

Python 3.10 or 3.11

Install dependencies:
```bash
pip install -r requirements.txt
```


### Example Usage
Dataset generation is performed by running `main.py` as a module from the project root.

A minimal example that generates one image-mask pair, displays it and then saves it to disk with metadata:

```bash
python -m src.main --num 1 --gen experiment1 --config config/base_params.json --save --show
```

For programmatic use (e.g. from scripts or notebooks), the generator logic is exposed via the `generate()` function in `src/main.py`.

### Segmentation Training Pipeline

A minimal end-to-end segmentation workflow is provided in the notebook:

`notebooks/simple_gen-train-test.ipynb`

The notebook demonstrates the complete pipeline:

1. Load real and synthetic datasets
2. Prepare training and validation dataloaders
3. Train a U-Net segmentation model
4. Evaluate predictions on a held-out test dataset
5. Save evaluation metrics and example predictions

## Dataset Analysis and Clustering
The real dataset was analysed in terms of:
- precipitate counts, sizes, shapes and intensities
- matrix intensity characteristics
- imaging artefacts (blur, illumination gradients, ...)

Based on this analysis, images were divided into five distinct visual clusters with cluster-specific configurations provided.


## Limitations
- The generator does not explicitly simulate microscope physics and all visual features observed in real images
- Example configurations are tailored to the analysed dataset.
