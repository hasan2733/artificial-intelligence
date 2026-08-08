# Artificial Intelligence

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/hasan2733/artificial-intelligence/blob/main/LICENSE)

A personal repository for learning, practicing, and experimenting with Artificial Intelligence, Machine Learning, and Deep Learning. This repository collects Jupyter Notebooks, experiments, small projects, and utilities used while studying and prototyping AI techniques.

---

## Table of Contents

- [About](#about)
- [Topics Included](#topics-included)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## About

This repository is a personal workspace for learning and revising AI-related topics. It contains Jupyter Notebooks with step-by-step experiments, example code, and notes covering classical machine learning, deep learning, computer vision, and related research experiments.

Note: According to the repository statistics, the primary language in this repo is Jupyter Notebook (100%).

---

## Topics Included

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Artificial Neural Networks (ANN) & Convolutional Neural Networks (CNN)
- Transfer Learning
- Computer Vision
- Federated Learning
- Feature Engineering
- Image Processing
- Discrete Wavelet Transform (DWT)
- Graph Attention Networks (GAT)
- Octree Segmentation
- Research-based Experiments

---

## Tech Stack

- Python 3.8+
- Jupyter Notebook / JupyterLab (primary)
- TensorFlow & Keras
- PyTorch
- NumPy, Pandas
- Scikit-Learn
- Matplotlib, Seaborn

---

## Repository Structure

Typical layout (may vary by folder):

- notebooks/           — Jupyter notebooks for experiments
- projects/            — standalone project folders
- datasets/            — sample data or download scripts
- scripts/             — helper scripts and utilities
- README.md            — this file

---

## Setup

Recommended: create and use a virtual environment.

```bash
git clone https://github.com/hasan2733/artificial-intelligence.git
cd artificial-intelligence

# create and activate virtual env (example using venv)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

# install dependencies (if requirements.txt present)
pip install -r requirements.txt

# or install common packages manually
pip install tensorflow jupyter numpy matplotlib pandas scikit-learn seaborn
jupyter notebook
```

If you don't have a requirements.txt file, I can generate one from the notebooks and projects if you want.

---

## Usage

- Open notebooks in `notebooks/` using Jupyter Notebook or JupyterLab.
- Run project-specific scripts from each project folder (check each folder's README for details).
- For GPU experiments, ensure proper CUDA and cuDNN setup and use compatible TensorFlow/PyTorch builds.

---

## Contributing

This is primarily a personal learning repo, but contributions are welcome. If you'd like to contribute:

- Open an issue describing the proposed change.
- Submit a pull request with a clear description and reproducible steps.

Please keep contributions focused, include tests or runnable notebooks where possible, and add a short description of what the notebook/project does.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Author

**Abid Hasan**  
GitHub: https://github.com/hasan2733

---

*README updated to improve clarity, setup instructions, and structure.*
