import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import pickle
from tqdm import tqdm
import timm
from scipy.optimize import curve_fit
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import pandas as pd
import matplotlib.pyplot as plt
import re

from torchvision.models.resnet import BasicBlock as tBasicBlock
from timm.models.resnet import Bottleneck as timBottleneck
from torchvision.models.resnet import Bottleneck as tBottleneck 
from torchvision.models.resnet import ResNet as tResNet
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.efficientnet import Conv2dNormActivation, SqueezeExcitation, MBConv 
from torchvision.models.mobilenetv2 import InvertedResidual

from bsconv.pytorch.common import ConvBlock
from bsconv.pytorch.mobilenet import LinearBottleneck

SUPPORTED_LAYER_TYPE = {nn.Linear, nn.Conv2d, nn.modules.conv.Conv2d, nn.modules.linear.Linear}
SUPPORTED_BLOCK_TYPE = {nn.Sequential, ConvBlock,
                        tBottleneck, timBottleneck, tBasicBlock, tResNet,
                        BasicConv2d, Inception, InceptionAux,
                        LinearBottleneck,
                        Conv2dNormActivation, SqueezeExcitation, MBConv,
                        InvertedResidual, nn.modules.container.Sequential
                        }


# SUPPORTED_LAYER_TYPE = {nn.Linear, nn.Conv2d, nn.modules.conv.Conv2d, nn.modules.linear.Linear}
# SUPPORTED_BLOCK_TYPE = {nn.Sequential, ConvBlock,
#                         tBottleneck, tBasicBlock, tResNet,
#                         BasicConv2d, Inception, InceptionAux,
#                         LinearBottleneck,
#                         Conv2dNormActivation, SqueezeExcitation, MBConv,
#                         InvertedResidual, 
#                         }

class InterruptException(Exception):
    pass


def parse_imagenet_val_labels(data_dir):
    """
    Generate labels of imagenet validation dataset
    More details, see 
    https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html
    """
    meta_path = os.path.join(data_dir, 'meta.mat')
    meta = sio.loadmat(meta_path, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
    idcs, wnids = list(zip(*meta))[:2]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}

    val_path = os.path.join(data_dir, 'ILSVRC2012_validation_ground_truth.txt')
    val_idcs = np.loadtxt(val_path) 
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
   
    label_path = os.path.join(data_dir, 'wnid_to_label.pickle')  
    with open(label_path, 'rb') as f:
        wnid_to_label = pickle.load(f)
    
    val_labels = [wnid_to_label[wnid] for wnid in val_wnids]
    return np.array(val_labels)


def test_accuracy(model, test_dl, device, topk=(1, )):
    """ 
    Compute top k accuracy on testing dataset
    """
    model.eval()
    maxk = max(topk)
    topk_count = np.zeros((len(topk), len(test_dl)))
    
    for j, (x_test, target) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            y_pred = model(x_test.to(device))
        topk_pred = torch.topk(y_pred, maxk, dim=1).indices
        target = target.to(device).view(-1, 1).expand_as(topk_pred)
        correct_mat = (target == topk_pred)

        for i, k in enumerate(topk):
            topk_count[i, j] = correct_mat[:, :k].reshape(-1).sum().item()
    
    topk_accuracy = topk_count.sum(axis=1) / len(test_dl.dataset)
    return topk_accuracy

def test_accuracy_sub(model, test_dl, subset, device, topk=(1, )):
    """ 
    Compute top k accuracy on testing dataset but only letting model pick from subset classes
    """
    model.eval()
    maxk = max(topk)
    topk_count = np.zeros((len(topk), len(test_dl)))
    
    for j, (x_test, target) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            y_pred = model(x_test.to(device))

        y_pred = y_pred[:, subset]
        topk_pred = torch.topk(y_pred, maxk, dim=1).indices.cpu().apply_(lambda x: subset[x]).to(device)
        target = target.to(device).view(-1, 1).expand_as(topk_pred)
        correct_mat = (target == topk_pred)

        for i, k in enumerate(topk):
            topk_count[i, j] = correct_mat[:, :k].reshape(-1).sum().item()

    topk_accuracy = topk_count.sum(axis=1) / len(test_dl.dataset)
    return topk_accuracy

            
def extract_layers(model, layer_list, supported_block_type=SUPPORTED_BLOCK_TYPE, supported_layer_type=SUPPORTED_LAYER_TYPE):
    '''
    Recursively obtain layers of given network
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all MLP and CNN layers
    layer_list: list
        list containing all supported layers
    '''
    for layer in model.children():
        if type(layer) in supported_block_type:
            # if sequential layer, apply recursively to layers in sequential layer
            extract_layers(layer, layer_list, supported_block_type, supported_layer_type)
        # print(type(layer)) if not list(layer.children()) else print(type(layer), "not layer")
        if not list(layer.children()) and type(layer) in supported_layer_type:
            # if leaf node, add it to list
            layer_list.append(layer)

# ======================================================================================================================================

def get_all_layers(model, layer_list, indent = 0):

    print((" " * indent), type(model), type(model) in SUPPORTED_BLOCK_TYPE, model in layer_list)
    if list(model.children()):
        for layer in model.children():
            get_all_layers(layer, layer_list, indent = indent + 1)

# ======================================================================================================================================
def get_training_dataloader_mobilenetv2(mean=CIFAR100_TRAIN_MEAN_MOBILENETV2, std=CIFAR100_TRAIN_STD_MOBILENETV2, batch_size=16, num_workers=2, shuffle=True, subset=None):
    """ return training dataloader for MOBILENETV2
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    
    if subset is not None:
        train_tar = cifar100_training.targets
        filtered_indices = [i for i, (_, label) in enumerate(cifar100_training) if label in subset]
        cifar100_training = Subset(cifar100_training, filtered_indices)
        
    cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_training_loader
    
# ======================================================================================================================================
def get_test_dataloader_mobilenetv2(mean=CIFAR100_TRAIN_MEAN_MOBILENETV2, std=CIFAR100_TRAIN_STD_MOBILENETV2, batch_size=16, num_workers=2, shuffle=False, subset=None):
    """ return training dataloader for MOBILENETV2
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    if subset is not None:
        test_tar = cifar100_test.targets
        filtered_indices = [i for i, (_, label) in enumerate(cifar100_test) if label in subset]
        cifar100_test = Subset(cifar100_test, filtered_indices)
        
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader
# ======================================================================================================================================
def fusion_layers_inplace(model, device):
    '''
    Let a convolutional layer fuse with its subsequent batch normalization layer  
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all CNN and BN layers
    '''
    model_layers = []
    extract_layers(model, model_layers, supported_layer_type = [nn.Conv2d, nn.BatchNorm2d])

    if len(model_layers) < 2:
        return 
    
    for i in range(len(model_layers)-1):
        curr_layer, next_layer = model_layers[i], model_layers[i+1]

        if isinstance(curr_layer, nn.Conv2d) and isinstance(next_layer, nn.BatchNorm2d):
            cnn_layer, bn_layer = curr_layer, next_layer
            # update the weight and bias of the CNN layer 
            bn_scaled_weight = bn_layer.weight.data / torch.sqrt(bn_layer.running_var + bn_layer.eps)
            bn_scaled_bias = bn_layer.bias.data - bn_layer.weight.data * bn_layer.running_mean / torch.sqrt(bn_layer.running_var + bn_layer.eps)
            cnn_layer.weight.data = cnn_layer.weight.data * bn_scaled_weight[:, None, None, None]
            # update the parameters in the BN layer 
            bn_layer.running_var = torch.ones(bn_layer.num_features, device=device)
            bn_layer.running_mean = torch.zeros(bn_layer.num_features, device=device)
            bn_layer.weight.data = torch.ones(bn_layer.num_features, device=device)
            bn_layer.eps = 0.

            if cnn_layer.bias is None:
                bn_layer.bias.data = bn_scaled_bias
            else:
                cnn_layer.bias.data = cnn_layer.bias.data * bn_scaled_weight + bn_scaled_bias 
                bn_layer.bias.data = torch.zeros(bn_layer.num_features, device=device)
            

def eval_sparsity(model):
    '''
    Compute the propotion of 0 in a network.
    
    Parameters
    ----------
    model: nn.Module
        The module to evaluate sparsity
    
    Returns
    -------
    A float capturing the proption of 0 in all the considered params.
    '''
    layers = []
    extract_layers(model, layers)
    supported_layers = [l for l in layers if type(l) in SUPPORTED_LAYER_TYPE]
    total_param = 0
    num_of_zero = 0
    
    for l in supported_layers:
        if l.weight is not None:
            total_param += l.weight.numel()
            num_of_zero += l.weight.eq(0).sum().item()
        if l.bias is not None:
            total_param += l.bias.numel()
            num_of_zero += l.bias.eq(0).sum().item()
    return np.around(num_of_zero / total_param, 4)

def finetune_model(load_model, train_loader, batch_size, num_epochs, learning_rate, device):
    
    model = load_model() # timm.create_model(model_path, pretrained=True)
    model.to(device)

    # Define optimizer & loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

            del images, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
        
    return model

def plot_results(csv_path, central_tendency, eval_type = "all"):
    df = pd.read_csv(csv_path)
    # df = df[df.Bits == bits].reset_index(drop = True)
    model_name = csv_path.split("_")[-2]#.split(".")[0]
    bits = csv_path.split("_")[-1].split(".")[0]

    def func(x, a, b, c):
        return (a * np.log(b * x)) + c

    replace = []
    if central_tendency.lower() == "avg":
        replace += ["Avg"]
    elif central_tendency.lower() == "median":
        replace += ["Median"]
    else:
        replace = None

    if eval_type.lower() == "all":
        replace += [""]
    elif eval_type.lower() == "sub":
        replace += [" (Pick Sub)"]
    else:
        replace = None


    cycle = ["Original", "Quantized", "Fine-Tuned", "Quant+FT"]
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']
    fig = plt.figure(figsize = (8, 5))
    for c, color in zip(cycle, colors):
    
        X, y = df[f"{replace[0]}_KL"], df[f"{c} Top1 Accuracy{replace[1]}"]

        # print(X.shape, y.shape)
        
        coefs, pcov = curve_fit(func, X, y)
        
        fitted_line = []
        for i in range(100):
            fitted_line += [func(i, *coefs).item()]
        
        
        
        plt.scatter(X, y, s = 5, c = color)
        plt.plot(range(len(fitted_line)), fitted_line, '--', c = color)
# plt.scatter(df["Avg_KL"], df["Original Top1 Accuracy"], s = 5)
# plt.plot(range(len(fitted_line_o)), fitted_line_o, '--')
# plt.scatter(df["Avg_KL"], df["Fine-Tuned Top1 Accuracy"], s = 5)
# plt.plot(range(len(fitted_line_ft)), fitted_line_ft, '--')
# plt.scatter(df["Avg_KL"], df["Quant+FT Top1 Accuracy"], s = 5)
# plt.plot(range(len(fitted_line_qft)), fitted_line_qft, '--')
    bits_num = re.findall(r"[0-9]+", bits)[0]
    
    plt.xlabel(f"{replace[0]} ICD", fontsize = 15)
    plt.ylabel("Top-1 Accuracy", fontsize = 15)
    plt.xlim(0, 100)
    plt.ylim(0.5, 1)
    plt.title(f"Performance of ResNet50 with {bits_num}-bit LAQ (Eval. on '{eval_type.capitalize()}')", fontsize = 12)
    leg = plt.legend(["Original", "-> fitted curve", "Quant", "-> fitted curve",  "FT", "-> fitted curve",  "FT + Quant", "-> fitted curve"], fontsize = 15)
    plt.savefig(f"./plots/{model_name}_{bits}_{eval_type.lower()}_{central_tendency.lower()}.png")

    return fig

    
    
