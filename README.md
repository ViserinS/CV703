# CV703: Swin Transformer for Image Classification

## Authors
Hazza Mahmood, Yongqiang Yu, Seung Hun (Eddie) Han  

## Overview
This repository contains the implementation of a **Swin Transformer-based image classifier** for the CV703 course. The model is trained and evaluated on a custom dataset, leveraging **PyTorch** and **Hugging Face's Transformers** library.

## Installation

### 1. Set Up the Environment
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed, then create and activate the environment:

```bash
conda env create -f environment.yml
conda activate swin-classifier-env
```

### 2. Prepare the Dataset

Ensure the dataset is downloaded and placed in the appropriate directory before running training or testing scripts.

## Usage
Training the Model

To train the model, run:
```bash
python train_swin.py
```

Then to test, run:
```bash
python test_swin.py
```

The training and testing scripts accept command-line arguments for customization. Default configurations are provided, but you can override them by specifying parameters when running the scripts.
