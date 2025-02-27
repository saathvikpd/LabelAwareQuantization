import os
import numpy as np

# Set parameters
subset_size = 10
sample_size = 10  # Number of experiments per configuration

# Create a mix of similar, dissimilar, and random class subsets
sc_options = ['None', 'None', 'True', 'False'] * sample_size

# Create log file if it doesn't exist
log_file = '../logs/Quantization_Log_googlenet_4bit.csv'
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))

# Run experiments
for i, sc_choice in enumerate(sc_options):
    print(f"Running experiment {i+1}/{len(sc_options)}: subset_size={subset_size}, similar_classes={sc_choice}")
    os.system(f"python main.py -model 'googlenet' -b 4 -bs 16 -s 1.16 -ds 'CIFAR100' -sn {subset_size} -sc '{sc_choice}'")