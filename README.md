# LabelAwareQuantization

This repository contains the code and resources for the DSC180 Capstone Project on Label-Aware Quantization. The project focuses on exploring post-training quantization techniques in neural networks with an emphasis on label awareness.

## Repository Structure

- `Quantized_Neural_Nets/`: Directory containing code and scripts related to quantized neural networks.
- `Class Embeddings & KL Divergence.ipynb`: Jupyter notebook analyzing class embeddings and Kullback-Leibler (KL) divergence.
- `cifar100_class_names.npy`: NumPy array file containing the class names for the CIFAR-100 dataset.
- `cifar100_kl_div_matrix.npy`: NumPy array file containing the KL divergence matrix for CIFAR-100 classes.
- `cifar100_subset_generation.py`: Python script for generating subsets of the CIFAR-100 dataset.

## Experimental Setup

### Objective

The primary objective of this experiment is to investigate the impact of label-aware quantization on neural network performance, particularly focusing on how class embeddings and the relationships between classes as measured by KL divergence impact quantization.

### Data

The experiments utilize the CIFAR-100 dataset, which consists of 100 classes with 600 images per class. The dataset is divided into 500 training images and 100 testing images per class.

### Methodology

1. **Class Embeddings Analysis**:
   - Use the `Class Embeddings & KL Divergence.ipynb` notebook to compute and analyze the embeddings of different classes in the CIFAR-100 dataset.
   - Calculate the KL divergence between class distributions to understand the similarity between classes.

2. **Subset Generation**:
   - Employ the `cifar100_subset_generation.py` script to create subsets of the CIFAR-100 dataset based on specific criteria, such as selecting classes with high or low KL divergence.

3. **Quantized Neural Networks**:
   - Explore the `Quantized_Neural_Nets/` directory for experiments and results on how subset homogeneity (measured by KL) impacts GPFQ Quantization of CNNs.
   - Train and evaluate quantized models on the original and subset datasets to assess performance impacts.

### Requirements

- Python 3.x
- NumPy
- Jupyter Notebook
- Additional libraries as specified in individual scripts or notebooks

### Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/saathvikpd/LabelAwareQuantization.git
   cd LabelAwareQuantization
