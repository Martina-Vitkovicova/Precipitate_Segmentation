# Synthetic Image Generation for Precipitate Segmentation
Segmentation of precipitates in scanning electron microscopy (SEM) images is challenging due to limited annotated data and substantial variation in image appearance.
This project investigates the use of synthetic SEM-like images to improve deep-learning-based segmentation of such images. It covers the complete workflow, from real-image analysis and preprocessing to synthetic data generation, model training, parameter selection, and experimental evaluation.

The project was developed as part of my master's thesis [*Image Synthesis for Precipitate Segmentation Neural Network*](https://dspace.cuni.cz/handle/20.500.11956/209787?locale-attribute=en).

## Repository Structure

```text
config/              JSON configuration files and parameter documentation
notebooks/           Example notebooks for generation, training, and evaluation
results_examples/    Qualitative segmentation results comparisons
segm/real_data/      Real SEM dataset used for segmentation experiments
src/
├── generators/      Synthetic image feature generators
├── image_analysis/  Real-image analysis and clustering utilities
├── precipseg/       Segmentation training and evaluation code
├── utils/           Shared utilities
└── main.py          Dataset generation interface
```

## Project Overview
The project follows four main steps:

1. **Real-image analysis:** Analyse real SEM images to characterise their visual properties and guide synthetic data generation.
2. **Synthetic data generation:** Generate SEM-like images with corresponding pixel-accurate precipitate masks using configurable parameters.
3. **Model training:** Train models using different training configurations and dataset compositions.
4. **Evaluation and comparison:**  Assess segmentation performance quantitatively using evaluation metrics and qualitatively by inspecting predictions and comparing results.


## Synthetic Image Generator
Generator behaviour is controlled by JSON configuration files and random seeds, supporting reproducible experiments with different parameter settings and dataset compositions.

Each generated sample contains:

- `img.png` — synthetic grayscale SEM-like image
- `mask.png` — binary precipitate segmentation mask
- `metadata.json` — metadata describing the generation

The generator models the following components:

| Component | Description |
|---|---|
| Matrix and grain structure | Metal matrix with grains, boundaries, and texture variations |
| Precipitates | Particle-like features with configurable geometry and appearance |
| Etch pits | Recessed surface features that can resemble precipitates |
| Imaging effects | Selected effects such as blur, illumination variation, and noise |

Detailed parameter documentation is available in [`config/README.md`](config/README.md).

Examples of generated synthetic datasets, together with further information about the generator parameters, are available in the [Hugging Face](https://huggingface.co/datasets/research-centre-rez/synthetic-sem-steel-precipitates) repository.

## Training and Evaluation Workflow

The `precipseg` package provides the training and evaluation pipeline, and it covers:
- **Data preparation:** Real and synthetic image–mask pairs are loaded, normalized, and augmented consistently during training.
- **Model training:** A U-Net with a pretrained ResNet-18 encoder is trained using configurable training settings and loss functions.
- **Evaluation:** Predictions are assessed using pixel-level overlap metrics (Dice and IoU) and instance-level detection metrics (precision, recall, and F1).
- **Experimental comparison:** Multiple runs are used to compare training configurations and assess performance variability.

### Notebooks

- [`simple_gen-train-test.ipynb`](notebooks/simple_gen-train-test.ipynb) — demonstrates a single generation-to-training-and-evaluation workflow.
- [`multi_train-eval.ipynb`](notebooks/multi_train-eval.ipynb) — runs comparative experiments across multiple configurations and repeated runs.

## Results

The results compare two training setups: a baseline model trained on 45 real SEM images and a model trained on the same 45 real images augmented with 450 synthetic images from the best-performing configuration. Each setup was evaluated across 20 runs.

| Metric | Real only | Real + synthetic | Change |
|---|---:|---:|---:|
| Dice | 69.70 ± 4.83% | **78.25 ± 0.60%** | +8.55 pp |
| IoU | 54.59 ± 5.39% | **64.88 ± 0.88%** | +10.29 pp |
| F1 | 59.76 ± 3.26% | **61.82 ± 1.59%** | +2.06 pp |
| Precision | 71.17 ± 5.49% | 60.05 ± 3.57% | −11.12 pp |
| Recall | 59.83 ± 6.76% | **76.59 ± 2.67%** | +16.76 pp |

*Values are mean ± standard deviation across 20 runs; changes are percentage-point (pp) differences.*

Overall, the results indicate that the selected synthetic-data configuration improved pixel-level segmentation quality and instance detection recall, while highlighting a trade-off between detecting missed precipitates and avoiding false detections. Statistical testing supported the improvements in Dice, IoU, instance F1, and recall. The thesis provides the full experimental results and statistical analysis.

### Qualitative Comparison
![Comparison of precipitate segmentation results](results_examples/comparison_close.png)
This figure shows an illustrative example of the segmentation results. Additional comparison images are provided in results_examples/, they are not selected to represent either the best or the worst cases, but to provide a representative view of the model's behaviour and its remaining limitations.


## Getting Started

Create and activate a Python environment compatible with the project, then install the dependencies:

```bash
pip install -r requirements.txt
```

Generate one synthetic image–mask sample and save it with metadata:

```bash
python -m src.main --num 1 --gen experiment1 --config config/base_params.json --save --out "./segm/data"
```

For programmatic generation, see the `generate()` function in `src/main.py`. The notebooks provide examples of how generation connects to model training and evaluation.

## Limitations
- The generator does not explicitly simulate microscope physics and all visual features observed in real images
- Example configurations are tailored to the analysed dataset.
