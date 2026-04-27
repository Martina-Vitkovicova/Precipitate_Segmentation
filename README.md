# Synthetic SEM Image Generator

## Overview

This repository contains a generator for synthetic SEM-like images of irradiated stainless steel microstructures with corresponding precipitate segmentation masks.

The generator was developed primarily for precipitate segmentation under limited annotated data,
but it can also be used as an experimental framework for studying the impact of controlled visual features 
on neural network training.

Example synthetic datasets created with this generator are available on Hugging Face:
[synthetic-sem-steel-precipitates](https://huggingface.co/datasets/research-centre-rez/synthetic-sem-steel-precipitates)

## Repository Structure

```text
config/              JSON configuration files and parameter documentation
notebooks/           Example notebooks for generation, training, and evaluation
segm/real_data/      Real SEM dataset used for segmentation experiments
src/
├── generators/      Synthetic image feature generators
├── image_analysis/  Real-image analysis and clustering utilities
├── precipseg/       Segmentation training and evaluation code
├── utils/           Shared utilities for configuration parsing and more
└── main.py          Dataset generation interface
```


## Modelled Image Components

Image generation is controlled through JSON configuration files. Detailed parameter documentation is available in [`config/README.md`](config/README.md).

| Component | Description                                                                                                                       |
|---|-----------------------------------------------------------------------------------------------------------------------------------|
| Matrix and grain structure | Polycrystalline metal matrix forming the image background, subdivided into grains with visible boundaries and texture variations. |
| Precipitates | Usually small, dark particle-like features located on top of the matrix surface, often with local shading.                        |
| Etch pits | Recessed surface defects in the matrix with pronounced shading.                                                                   |
| Imaging artefacts | Non-material effects introduced by the imaging process, such as blur, illumination variation, and noise.                          |



## Outputs
The generator produces datasets consisting of directories with:
- `img.png` – synthetic grayscale SEM-like image  
- `mask.png` – binary precipitate mask  
- `metadata.json` – generation metadata

## Reproducibility

Generator behaviour is controlled by JSON configuration files and explicit random seeds. Using the same configuration and seed produces the same generated dataset.


## Requirements

Python 3.10 or 3.11

Install dependencies:
```bash
pip install -r requirements.txt
```


## Example Usage
Dataset generation is performed by running `main.py` as a module from the project root.

A minimal example that generates one image-mask pair and saves it to disk with metadata:

```bash
python -m src.main --num 1 --gen experiment1 --config config/base_params.json --save --out "./segm/data"
```

For programmatic use (e.g. from scripts or notebooks), the generator logic is exposed via the `generate()` function in `src/main.py`.

### Segmentation and Evaluation Pipeline

Two notebooks are provided in the `notebooks/` directory:

```text
notebooks/simple_gen-train-test.ipynb
notebooks/multi_train-eval.ipynb
```

`simple_gen-train-test.ipynb` demonstrates a minimal single-run workflow, including synthetic image generation, dataset preparation, model training, and evaluation.

`multi_train-eval.ipynb` supports systematic multi-run comparative experiments, enabling evaluation of different training strategies, preprocessing choices, and dataset configurations.

The training and evaluation steps in these notebooks rely on the `precipseg` package.


## Dataset Analysis and Clustering

The real SEM dataset was analysed using image-level and precipitate-level characteristics, including precipitate count, size, shape, intensity, matrix brightness, blur, and illumination variation.

Based on this analysis, the images were divided into five visual clusters. The corresponding cluster-specific configuration files are provided in the `config/` directory.

## Limitations
- The generator does not explicitly simulate microscope physics and all visual features observed in real images
- Example configurations are tailored to the analysed dataset.
