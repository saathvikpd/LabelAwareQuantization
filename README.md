# Label-Aware Quantization

Contributors: Saathvik Dirisala, Jessica Hung, Ari Juljulian, Yijun Luo

This repository contains the code and resources for the DSC180 Capstone Project on Label-Aware Quantization. The project focuses on exploring post-training quantization techniques in neural networks with an emphasis on label awareness. We used the GPFQ Quantization framework /(https://github.com/YixuanSeanZhou/Quantized_Neural_Nets)/ and altered it for our experiments.

## Repository Structure

- `Quantized_Neural_Nets/`: Directory containing code and scripts related to quantized neural networks.
- `Quantized_Neural_Nets/src/`: 
     - `plots/`: Plots from experiments
     - `testing.ipynb/`: Juypter notebook to run experiments and generate plots
     - `cifar100_class_names.npy`: NumPy array file containing the class names for the CIFAR-100 dataset.
     - `cifar100_kl_div_matrix.npy`: NumPy array file containing the KL divergence matrix for CIFAR-100 classes.
     - `cifar100_subset_generation.py`: Python script for generating subsets of the CIFAR-100 dataset.
- `Class Embeddings & KL Divergence.ipynb`: Jupyter notebook analyzing class embeddings and Kullback-Leibler (KL) divergence.
 
## Experimental Setup

### Objective

The primary objective of this experiment is to investigate the impact of label-aware quantization on neural network performance, particularly focusing on how class embeddings and the relationships between classes as measured by KL divergence impact quantization. Intuitively, subsets with a larger median distance between classes will be easier tasks, so quantization should not impact performance as much as with highly homogeneous subsets.

### Data

The experiments utilize the CIFAR-100 dataset, which consists of 100 classes with 600 images per class. The dataset is divided into 500 training images and 100 testing images per class.

### Methodology

1. **Class Embeddings Analysis**:
   - Use the `Class Embeddings & KL Divergence.ipynb` notebook to compute and analyze the embeddings of different classes in the CIFAR-100 dataset.
   - Calculate the KL divergence between class distributions to understand the similarity between classes.

2. **Subset Generation**:
   - Employ the `cifar100_subset_generation.py` script to create subsets of the CIFAR-100 dataset based on specific criteria, such as selecting classes with high or low KL divergence.

3. **Quantized Neural Networks**:
   - Explore the `Quantized_Neural_Nets/src/testing.ipynb` notebook for experiments and results on how subset homogeneity (measured by KL) impacts GPFQ Quantization of CNNs.
   - Train and evaluate quantized models on the original and subset datasets to assess performance impacts.
  
### Sample Result

![Plot unavailable](https://github.com/saathvikpd/LabelAwareQuantization/blob/main/Quantized_Neural_Nets/src/plots/resnet50_median.png)

All experiments:
- 10 classes per subset
- Classes selected from CIFAR100

Plot description:
1. Quantized model curve: ResNet-50 pre-trained on CIFAR100 quantized down to 4 bits using GPFQ
2. Original model curve: ResNet-50 pre-trained on CIFAR100
3. Fine-tuned model curve: ResNet-50 pre-trained on CIFAR100 and fine-tuned on subset

### Requirements

- Python 3.x
- NumPy
- PyTorch
- Jupyter Notebook
- Additional libraries as specified in individual scripts or notebooks

### Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/saathvikpd/LabelAwareQuantization.git
   cd LabelAwareQuantization
   ```

2. **Install Requirements**:
   ```bash
   cd ./Quantized_Neural_Nets
   pip install -r requirements.txt
   ```

3. **Run Experiment**:
    ```bash
   python main.py -model 'resnet50' -b 4 -bs 64 -s 1.16 -ds 'CIFAR100' -sn 10 -sc 'False'
   ```
    Function Parameters:
   - model: Model name (pulls pre-trained model from hugging-face)
   - b: Bit width or precision-level
   - bs: => Batch size
   - s: Scalar C used to determine the radius of alphabets
   - ds: Dataset /(only CIFAR100 used for our experiments/)
   - sn: Subset size
   - sc: Class similarity level for random subset generation /('True': very similar, 'False': very dissimilar, 'None': random/)
