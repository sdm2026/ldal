# Learning-Dynamics Aware Loss (LDAL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation for our paper introducing the **Learning-Dynamics Aware Loss (LDAL)** framework. 

LDAL is a novel, plug-and-play loss function designed to handle severe class imbalance in long-tailed visual recognition tasks. Moving beyond static, frequency-based reweighting, LDAL dynamically adjusts class penalties during training by leveraging real-time feature representation strength (semantic scale), intrinsic learning difficulty (Shannon entropy), and an inter-epoch regularization penalty to prevent premature convergence to head-class-dominated local minima.

##  Supported Datasets
The repository currently supports training and evaluation on the following standard long-tailed benchmarks:
- **ImageNet-LT**
- **iNaturalist 2018**
- **CIFAR-10-LT**
- **CIFAR-100-LT**
- **Tiny ImageNet-LT**

---

##  Getting Started

Follow these instructions to set up the environment and run the experiments.

### 1. Clone the Repository
```bash
git clone https://github.com/iclr-sub/ldal.git
cd ldal
```

### 2. Environment Setup
We recommend using a virtual environment to manage dependencies:
```bash
python -m venv ldal_env
```

**Activate the environment:**
- **Windows:**
  ```bash
  ldal_env\Scripts\activate
  ```
- **Unix/macOS:**
  ```bash
  source ldal_env/bin/activate
  ```

### 3. Install Dependencies
Install the required packages via `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation
Download and extract the required datasets into the `datasets/` directory. The directory structure must follow this exact hierarchy for the dataloaders to function correctly:

```text
datasets/
    ├── ImageNet_LT/
    ├── INaturalist18/
    ├── CIFAR-10/
    ├── CIFAR-100/
    └── TinyImageNet/
```
*(Note: Please refer to the specific dataset loader scripts in `dataloaders/` for exact annotation file requirements).*

---

##  Training and Evaluation

### Large-Scale Datasets (ImageNet-LT & iNaturalist 2018)
To train models on the larger datasets, utilize the `main.py` script. You can configure the dataset, backbone architecture, and hyperparameters via command-line arguments.

**Example: Training ResNet-50 on ImageNet-LT**
```bash
python main.py --dataset_name imagenet --model_name resnet50 
```

**Example: Training ResNet-50 on iNaturalist 2018**
```bash
python main.py --dataset_name inaturalist --model_name resnet50
```

#### Command-Line Arguments:
- `--dataset_name`: Target dataset (`imagenet`, `inaturalist`).
- `--model_name`: Backbone architecture (`resnet32`, `resnet50`, `resnext50`, `resnext101`).
- `--batch_size`: Batch size for training (default: `256`).
- `--num_epochs`: Total number of training epochs (default: `200`).
- `--learning_rate`: Initial learning rate (default: `0.01`).
- `--data_path`: Path to the root datasets directory (default: `datasets/`).

### Standard Benchmarks (CIFAR & Tiny ImageNet)
For rapid experimentation and evaluation on CIFAR and Tiny ImageNet, we provide dedicated Jupyter Notebooks. These notebooks contain complete end-to-end training loops and visualization code.

Start your Jupyter environment:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and execute the desired experiment:
- `cifar_10.ipynb`
- `cifar_100.ipynb`
- `tiny_imagenet.ipynb`

---

##  Repository Structure

```text
├── datasets/                   # Root directory for all dataset files (Not tracked by Git)
├── dataloaders/
│   ├── __init__.py
│   ├── ImageNet_LT/
│   ├── Inaturalist18/
│   ├── imagenet_lt_loader.py   # ImageNet-LT specific dataloader
│   └── inaturalist_loader.py   # iNaturalist 2018 specific dataloader
├── models/
│   ├── __init__.py
│   ├── resnet.py               # ResNet backbone implementations
│   └── resnext.py              # ResNeXt backbone implementations
├── notebooks/
│   ├── cifar_10.ipynb          # End-to-end experiment notebook for CIFAR-10-LT
│   ├── cifar_100.ipynb         # End-to-end experiment notebook for CIFAR-100-LT
│   └── tiny_imagenet.ipynb     # End-to-end experiment notebook for Tiny ImageNet-LT
├── utils/
│   ├── __init__.py
│   ├── ldal_loss.py            # PyTorch implementation of the LDAL function
│   └── plot_utils.py           # Visualization and metric tracking utilities
├── main.py                     # Primary training script for large-scale datasets
├── requirements.txt            # Python environment dependencies
└── README.md                   # Repository documentation
```

*(Note: Citation will be updated upon publication).*
