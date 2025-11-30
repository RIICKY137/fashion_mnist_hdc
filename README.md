This repository contains an end-to-end implementation of a Hyperdimensional Computing (HDC) model applied to the Fashion-MNIST dataset. The goal of this project is to build an interpretable, lightweight, and efficient HDC pipeline using only NumPy, without relying on deep learning frameworks.

The project walks through:

Building a binary HDC library from scratch

Encoding Fashion-MNIST images using position + intensity hypervectors

Training class prototypes with majority voting

Performing inference with Hamming similarity

Visualizing learned class hypervectors

Experimenting with improvements (denoising, normalization, etc.)

This implementation also serves as a foundation for future experiments targeting >90% accuracy, and as a clean research example of applying Hyperdimensional Computing to visual classification tasks.

🚀 Project Structure
fashion_mnist_hdc/
│
├── src/
│   ├── hdc.py                # core HDC operations (bind, bundle, similarity)
│   ├── load_fashion.py       # dataset loader (torchvision)
│   ├── encode_fashion.py     # training script → generates class hypervectors
│   ├── test_fashion.py       # inference script → evaluates test accuracy
│   ├── visualize.py          # visualizes class hypervectors as images
│   └── v1.py                 # optional archived version
│
├── data/                     # generated HDC vectors (not included in repo)
│   ├── pixel_hvs.npy
│   ├── value_hvs.npy
│   └── class_hv.npy
│
├── hdc_env/                  # local virtual environment (ignored)
│
├── requirements.txt
└── README.md

🧠 Hyperdimensional Computing Overview

Hyperdimensional Computing (HDC) represents information using extremely high-dimensional binary vectors (usually 5,000–20,000 dimensions).
Key operations include:

Binding — combines two hypervectors (XOR for binary HDC)

Bundling — majority vote aggregation across vectors

Similarity — Hamming similarity for classification

For classification tasks, the process is:

Encode each image into a single hypervector

Bundle all hypervectors of the same class to form one prototype

Compute similarity between test image hypervectors and class prototypes

Predict the class with the highest similarity score

The HDC paradigm is:

Lightweight

Energy-efficient

Interpretable

Highly noise tolerant

Suitable for edge/embedded devices

📦 Installation
1. Clone the repository
git clone https://github.com/RIICKY137/fashion_mnist_hdc.git
cd fashion_mnist_hdc

2. Create & activate a virtual environment (optional)

Windows:

python -m venv hdc_env
hdc_env\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

🏋️ Training the HDC Model

Training generates three essential hypervector sets:

pixel_hvs.npy — hypervectors for each pixel position

value_hvs.npy — hypervectors for pixel intensity values (0–255)

class_hv.npy — learned class prototype vectors

Run:

python src/encode_fashion.py


The model uses:

DIM = 10,000

Per-pixel encoding (position ⊕ value)

Image denoising and normalization

Majority-vote class bundling

🧪 Testing & Accuracy

Run:

python src/test_fashion.py


This script outputs:

Total test samples

Prediction vs. ground truth for sample images

Overall model accuracy

Current accuracy: ~0.60
This matches expected performance for an HDC model with early-stage improvements.

Typical HDC ranges:

40–50% baseline (no preprocessing)

60–70% with denoising/normalization

85–93% with advanced encoding techniques

🎨 Visualizing Class Hypervectors

Use:

python src/visualize.py


This converts the 10,000-dimensional class hypervectors into 2D binary bitmap images so you can visually inspect what patterns each class prototype has learned.

📈 Current Results
Model Version	Notes	Accuracy
v1	raw per-pixel sum	~0.10
v2	corrected binarization	~0.25
v3	normalization + denoising	~0.60
🔭 Future Improvements

Potential upgrades targeting 85–90% accuracy:

Bipolar hypervectors (−1/+1 representation)

Multi-level pixel intensity encoding

Edge-based or feature-extracted embeddings

MAP (multiply-add-permute) encoding

Pixel-group binding strategies

Random-index sparse encoding

Continuous-class prototype updates

Future experiments will be added under an experiments/ folder.

🤝 Contributing

Contributions and discussions are welcome.
Interesting directions include:

Alternative binding/bundling operators

Optimized binary implementations

Applications to other computer vision tasks

Hybrid HDC + neural network models

📜 License

This project is open-source under the MIT License.
