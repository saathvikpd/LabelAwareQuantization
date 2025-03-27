# 🧠 Label-Aware Quantization

**Contributors**: Saathvik Dirisala, Jessica Hung, Ari Juljulian, Yijun Luo  
🔗 **Website (Result Visualizations)**: [https://saathvikpd.github.io/LabelAwareQuantWebsite](https://saathvikpd.github.io/LabelAwareQuantWebsite/)

This repository contains code and resources for the **DSC180 Capstone Project** on *Label-Aware Quantization*. The project explores **post-training quantization techniques** in neural networks with an emphasis on label and class similarity awareness.

We adapted the [GPFQ Quantization framework](https://github.com/YixuanSeanZhou/Quantized_Neural_Nets) for our experiments.

---

## 📂 Repository Structure

- `src/`: 
     - `plots/`: Plots from experiments
     - `testing.ipynb/`: Juypter notebook to run experiments and generate plots
     - `cifar100_class_names.npy`: NumPy array file containing the class names for the CIFAR-100 dataset.
     - `cifar100_kl_div_matrix.npy`: NumPy array file containing the KL divergence matrix for CIFAR-100 classes.
     - `cifar100_subset_generation.py`: Python script for generating subsets of the CIFAR-100 dataset.
- `logs/`: Log files (.csv)
- `Class Embeddings & KL Divergence.ipynb`: Jupyter notebook analyzing class embeddings and Kullback-Leibler (KL) divergence.
 

---

## 🧪 Experimental Setup

### 🎯 Objective

To investigate how **label-aware quantization** impacts neural network performance. Specifically, we explore how class embeddings and **inter-class relationships** (measured via KL divergence) affect model accuracy post quantization.

> Hypothesis: Subsets with **larger median distances** between classes should be easier tasks. Thus, quantization will have **less impact** on performance than with homogeneous subsets.

---

### 📚 Dataset

- **CIFAR-100**:  
  - 100 classes, 600 images per class  
  - 500 training + 100 testing images per class  

---

### 🧠 Methodology

#### 1. **Class Embeddings & KL Divergence**
- Use `Class Embeddings & KL Divergence.ipynb` to compute class embeddings
- Calculate **KL divergence** to determine similarity between class distributions

#### 2. **Subset Generation**
- Use `cifar100_subset_generation.py` to create subsets with:
  - Very similar classes (low KL)
  - Very dissimilar classes (high KL)
  - Randomly selected classes

#### 3. **Quantization Experiments**
- Run experiments in `src/testing.ipynb`
- Apply **GPFQ** quantization on ResNet-50 models
- Compare performance on full dataset vs. generated subsets

---

## 📈 Sample Result

![Quantization Result](https://github.com/saathvikpd/LabelAwareQuantization/blob/main/src/plots/resnet50_4bit_all_median.png)

### Plot Description:
1. **Quantized**: ResNet-50, quantized to 4 bits using GPFQ  
2. **Original**: ResNet-50 pre-trained on full CIFAR-100  
3. **Fine-Tuned**: Pre-trained ResNet-50, fine-tuned on selected subset  
4. **Quant + FT**: Fine-tuned model further quantized to 4 bits  

> Experiments conducted with subsets of 10 classes from CIFAR-100.

---

## 🛠️ Requirements

- Python 3.x  
- NumPy  
- PyTorch  
- Jupyter Notebook  
- Other dependencies listed in individual files or `requirements.txt`

---

## 🚀 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/saathvikpd/LabelAwareQuantization.git
cd LabelAwareQuantization
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Experiment:
```bash
python main.py -model 'resnet50' -b 4 -bs 64 -s 1.16 -ds 'CIFAR100' -sn 10 -sc 'False' -ue -mp 'Asc'
```
Function Parameters:
- model: Model name (pulls pre-trained model from hugging-face)
- b: Bit width or precision-level
- bs: Batch size
- s: Scalar C used to determine the radius of alphabets
- ds: Dataset (only CIFAR100 used for our experiments)
- sn: Subset size
- sc: Class similarity level for random subset generation ('True': very similar, 'False': very dissimilar, 'None': random)
- ue: Use existing subsets for experiments
- mp: 'Asc' => First 50% of weights in 3-bit and second 50% of weights in 5-bit; 'Desc' => First 50% of weights in 5-bit and second 50% of weights in 3-bit
