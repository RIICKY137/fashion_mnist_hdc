# Hyperdimensional Computing for Fashion-MNIST

This repository provides an end-to-end implementation of a Hyperdimensional Computing (HDC) model for classifying the Fashion-MNIST dataset. The pipeline is implemented primarily with NumPy and does not rely on deep learning frameworks for model construction or training.

The project is intended to provide:

- A lightweight and interpretable HDC image-classification pipeline
- A from-scratch implementation of binary HDC operations
- A reproducible baseline for Fashion-MNIST experiments
- A foundation for evaluating more advanced HDC encoding and learning methods

## Project Overview

The implementation covers the complete classification workflow:

1. Construction of a binary HDC library
2. Encoding of Fashion-MNIST images using position and intensity hypervectors
3. Training of class prototypes through majority-vote bundling
4. Classification using Hamming similarity
5. Visualization of learned class hypervectors
6. Evaluation of preprocessing and encoding improvements

This repository also serves as a foundation for future experiments aimed at improving classification accuracy while preserving the computational efficiency and interpretability of HDC models.

## Repository Structure

```text
fashion_mnist_hdc/
│
├── src/
│   ├── hdc.py                # Core HDC operations
│   ├── load_fashion.py       # Fashion-MNIST dataset loader
│   ├── encode_fashion.py     # Training and prototype generation
│   ├── test_fashion.py       # Model evaluation
│   ├── visualize.py          # Class hypervector visualization
│   └── v1.py                 # Archived implementation
│
├── data/                     # Generated hypervectors (not included)
│   ├── pixel_hvs.npy
│   ├── value_hvs.npy
│   └── class_hv.npy
│
├── hdc_env/                  # Local virtual environment (ignored)
├── requirements.txt
└── README.md
```

## Hyperdimensional Computing

Hyperdimensional Computing represents information using high-dimensional vectors, typically containing between 5,000 and 20,000 dimensions.

This project uses binary hypervectors and three primary operations:

- **Binding:** Combines two hypervectors using bitwise XOR
- **Bundling:** Aggregates multiple hypervectors using majority voting
- **Similarity:** Compares hypervectors using Hamming similarity

The classification pipeline operates as follows:

1. Encode each image as a single hypervector.
2. Bundle the training hypervectors belonging to each class.
3. Use the resulting bundled vectors as class prototypes.
4. Encode each test image using the same procedure.
5. Compare the test hypervector with every class prototype.
6. Predict the class with the highest similarity score.

HDC models are often studied for their computational simplicity, robustness to noise, interpretability, and suitability for resource-constrained hardware.

## Requirements

The project requires Python 3 and the packages listed in `requirements.txt`.

Although the HDC model itself is implemented with NumPy, `torchvision` is currently used to download and load the Fashion-MNIST dataset.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RIICKY137/fashion_mnist_hdc.git
cd fashion_mnist_hdc
```

### 2. Create a Virtual Environment

Creating a virtual environment is recommended but not required.

On Windows:

```bash
python -m venv hdc_env
hdc_env\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv hdc_env
source hdc_env/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

## Training

Run the following command from the repository root:

```bash
python src/encode_fashion.py
```

The training process generates three sets of hypervectors:

- `pixel_hvs.npy`: Hypervectors representing pixel positions
- `value_hvs.npy`: Hypervectors representing pixel intensity values
- `class_hv.npy`: Learned prototype hypervectors for the ten Fashion-MNIST classes

The current implementation uses:

- 10,000-dimensional binary hypervectors
- Per-pixel position-intensity binding
- Image denoising and normalization
- Majority-vote bundling for class-prototype construction

The generated files are stored in the `data/` directory.

## Evaluation

After training, run:

```bash
python src/test_fashion.py
```

The evaluation script reports:

- The total number of evaluated test samples
- Example predictions and corresponding ground-truth labels
- Overall classification accuracy

The current implementation achieves approximately 60% test accuracy. This result should be treated as an experimental baseline rather than a definitive performance limit for HDC on Fashion-MNIST.

Performance depends on factors including preprocessing, hypervector dimensionality, intensity representation, encoding strategy, and prototype-update rules.

## Visualization

To visualize the learned class hypervectors, run:

```bash
python src/visualize.py
```

The script reshapes the 10,000-dimensional binary class prototypes into two-dimensional bitmap representations. These visualizations provide a qualitative view of the structures encoded within each class hypervector.

The displayed bitmaps are projections of the learned hypervectors and should not be interpreted as direct reconstructions of the original Fashion-MNIST images.

## Experimental Results

| Version | Description | Approximate Accuracy |
|---------|-------------|---------------------:|
| v1 | Raw per-pixel aggregation | 10% |
| v2 | Corrected binary conversion | 25% |
| v3 | Normalization and denoising | 60% |

These results reflect different stages of implementation and are not necessarily controlled comparisons.

For rigorous evaluation, future experiments should record:

- Random seeds
- Preprocessing parameters
- Hypervector dimensions
- Dataset splits
- Software dependencies
- Runtime conditions

## Planned Improvements

Future work may investigate:

- Bipolar hypervectors using `-1` and `+1`
- Multi-level or continuous intensity encoding
- Edge-based and local-feature representations
- Multiply-add-permute encoding
- Region-based and pixel-group binding
- Sparse random indexing
- Iterative class-prototype updates
- Quantized and bit-packed implementations
- Runtime and memory benchmarking
- Ablation studies for individual preprocessing steps

Future experimental implementations may be organized under an `experiments/` directory.

## Reproducibility

For reproducible experiments, subsequent versions should document:

- Random seeds
- Dataset and library versions
- Training and test sample counts
- Hypervector dimensionality
- Preprocessing parameters
- Encoding configuration
- Hardware and runtime measurements

Generated hypervectors are not included in the repository and must be recreated by running the training script.

## Contributing

Contributions, issue reports, and technical discussions are welcome.

Relevant research directions include:

- Alternative binding and bundling operators
- Improved image-encoding strategies
- Efficient binary implementations
- Prototype-learning algorithms
- Applications to additional computer-vision datasets
- Hybrid neural-network and HDC architectures

Contributions should include a clear description of the proposed change and, where applicable, reproducible evaluation results.

## License

This project is released under the MIT License. See the `LICENSE` file for details.
