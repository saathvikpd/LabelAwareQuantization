import torch
import torchvision
import numpy as np
import os
import csv
import argparse
from datetime import datetime
from gpfq_cifar100 import QuantizeNeuralNet
from utils import test_accuracy, eval_sparsity, fusion_layers_inplace
from data_loaders import data_loader
import torch.nn as nn
from cifar100_subset_generation import generate_subset

LOG_FILE_NAME = 'logs/Quantization_Log_Random.csv'
CLASS_NAMES_PATH = "./cifar100_class_names.npy"
CLASS_NAMES = np.load(CLASS_NAMES_PATH)

# hyperparameter section
parser = argparse.ArgumentParser(description='Quantization algorithms in the paper')

parser.add_argument('--bits', '-b', default=[4], type=int, nargs='+',
                    help='number of bits for quantization')
parser.add_argument('--scalar', '-s', default=[1.16], type=float, nargs='+',
                    help='the scalar C used to determine the radius of alphabets')
parser.add_argument('--batch_size', '-bs', default=[128], type=int, nargs='+',
                    help='batch size used for quantization')
parser.add_argument('--percentile', '-p', default=[1], type=float, nargs='+',
                    help='percentile of weights')
parser.add_argument('--num_worker', '-w', default=8, type=int, 
                    help='number of workers for data loader')
parser.add_argument('--data_set', '-ds', default='ILSVRC2012',
                    help='dataset used for quantization')   
parser.add_argument('--stochastic_quantization', '-sq', action='store_true',
                    help='use stochastic quantization')
parser.add_argument('--retain_rate', '-rr', default=0.25, type=float,
                    help='subsampling probability p for convolutional layers')
parser.add_argument('--regularizer', '-reg', default=None, choices=['L0', 'L1'], 
                    help='choose the regularization mode')
parser.add_argument('--lamb', '-l', default=[0.1], type=float, nargs='+',
                    help='regularization term')
parser.add_argument('--ignore_layer', '-ig', default=[], type=int, nargs='+',
                    help='indices of unquantized layers')
parser.add_argument('-seed', default=0, type=int, help='set random seed')
parser.add_argument('--fusion', '-f', action='store_true', help='fusing CNN and BN layers')
parser.add_argument('--similar', '-sim', default=[1], type=int, help='indicate if subset is similar or dissimilar')
parser.add_argument('--num_classes', '-n', default=[15], type=int, help='determine number of classes in subset')
parser.add_argument('--model', '-m', default='resnet_20', help='model name')
parser.add_argument('--cifar100_model', '-c100_m', default='../models/cifar100_resnet20-23dac2f1.pt', type=str, help='pretrained cifar100 network')

args = parser.parse_args()


def main(b, mlp_s, cnn_s, bs, mlp_per, cnn_per, l):
    batch_size = bs  
    bits = b
    mlp_percentile = mlp_per 
    cnn_percentile = cnn_per
    mlp_scalar = mlp_s 
    cnn_scalar = cnn_s
    lamb = l
    stochastic = args.stochastic_quantization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    subset_dict = generate_subset(args.num_classes)
    classes_of_interest = subset_dict['classes']
    print(f"CLASS NAMES: {classes_of_interest}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the model to be quantized
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x0_75", pretrained=True)
    original_accuracy_table = {}
    
    model.to(device)  
    model.eval()  # turn on the evaluation mode

    if args.fusion:
        fusion_layers_inplace(model, device) # combine CNN layers and BN layers (in place fusion)
        print('CNN and BN layers are fused before quantization!\n')
    
    if stochastic:
        print(f'Quantization mode: stochastic quantization, i.e. SGPFQ')
    elif args.regularizer:
        print(f'Quantization mode: sparse quantization using {args.regularizer} norm with lambda {lamb}')
    else:
        print(f'Quantization mode: GPFQ')
        
    print('\nQuantization hyperparameters:')
    print(f'Quantizing {args.model} on {device} with\n\t  dataset: {args.data_set}, bits: {bits}, mlp_scalar: {mlp_scalar}, cnn_scalar: {cnn_scalar}, mlp_percentile: {mlp_percentile}, \
        \n\tcnn_percentile: {cnn_percentile}, retain_rate: {args.retain_rate}, batch_size: {batch_size}\n')

    # subset_dict = generate_subset(args.num_classes, similar_classes=args.similar)
    # classes_of_interest = subset_dict['classes']
    
    # load the data loader for training and testing
    subset_train_loader, subset_test_loader = data_loader(args.data_set, batch_size, args.num_worker, classes_of_interest)
    
    # quantize the neural net
    subset_quantizer = QuantizeNeuralNet(model, args.model, batch_size, 
                                    subset_train_loader, 
                                    mlp_bits=bits,
                                    cnn_bits=bits,
                                    ignore_layers=args.ignore_layer,
                                    mlp_alphabet_scalar=mlp_scalar,
                                    cnn_alphabet_scalar=cnn_scalar,
                                    mlp_percentile=mlp_percentile,
                                    cnn_percentile=cnn_percentile,
                                    reg = args.regularizer, 
                                    lamb=lamb,
                                    retain_rate=args.retain_rate,
                                    stochastic_quantization=stochastic,
                                    device = device
                                    )
    start_time = datetime.now()
    subset_quantized_model = subset_quantizer.quantize_network()
    end_time = datetime.now()
    subset_quantized_model = subset_quantized_model.to(device)

    print(f'\nTime used for quantization: {end_time - start_time}\n')

    all_train_loader, all_test_loader = data_loader(args.data_set, batch_size, args.num_worker, classes_of_interest)
    
    # quantize the neural net
    all_quantizer = QuantizeNeuralNet(model, args.model, batch_size, 
                                    all_train_loader, 
                                    mlp_bits=bits,
                                    cnn_bits=bits,
                                    ignore_layers=args.ignore_layer,
                                    mlp_alphabet_scalar=mlp_scalar,
                                    cnn_alphabet_scalar=cnn_scalar,
                                    mlp_percentile=mlp_percentile,
                                    cnn_percentile=cnn_percentile,
                                    reg = args.regularizer, 
                                    lamb=lamb,
                                    retain_rate=args.retain_rate,
                                    stochastic_quantization=stochastic,
                                    device = device
                                    )
    start_time = datetime.now()
    all_quantized_model = all_quantizer.quantize_network()
    end_time = datetime.now()
    all_quantized_model = all_quantized_model.to(device)

    print(f'\nTime used for quantization: {end_time - start_time}\n')

    saved_model_name = f'ds{args.data_set}_b{bits}_batch{batch_size}_mlpscalar{mlp_scalar}_cnnscalar{cnn_scalar}\
        _mlppercentile{mlp_percentile}_cnnpercentile{cnn_percentile}_retain_rate{args.retain_rate}\
        _reg{args.regularizer}_lambda{lamb}.pt'

    if not os.path.isdir('../quantized_models/'):
        os.mkdir('../quantized_models/')
    saved_model_dir = '../quantized_models/'+args.model
    if not os.path.isdir(saved_model_dir):
        os.mkdir(saved_model_dir)
    torch.save(subset_quantized_model, os.path.join(saved_model_dir, saved_model_name))

    topk = (1, 5)   # top-1 and top-5 accuracy
    
    print(f'\nEvaluting the original model to get its accuracy\n')
    original_topk_accuracy = test_accuracy(model, subset_test_loader, device, topk)

    subset_classes_names = [CLASS_NAMES[class_id] for class_id in classes_of_interest]

    print(f'Top-1 accuracy of {args.model} with classes {subset_classes_names} is {original_topk_accuracy[0]}.')
    print(f'Top-5 accuracy of {args.model} with classes {subset_classes_names} is {original_topk_accuracy[1]}.')
    
    start_time = datetime.now()

    print(f'\n Evaluting the SUBSET quantized model to get its accuracy\n')
    subset_topk_accuracy = test_accuracy(subset_quantized_model, subset_test_loader, device, topk)
    print(f'Top-1 accuracy of quantized {args.model} with classes {subset_classes_names} is {subset_topk_accuracy[0]}.')
    print(f'Top-5 accuracy of quantized {args.model} with classes {subset_classes_names} is {subset_topk_accuracy[1]}.')

    end_time = datetime.now()

    print(f'\nTime used for evaluation: {end_time - start_time}\n')

    start_time = datetime.now()

    print(f'\n Evaluting the ALL quantized model to get its accuracy\n')
    all_topk_accuracy = test_accuracy(all_quantized_model, subset_test_loader, device, topk)
    print(f'Top-1 accuracy of quantized {args.model} with classes {subset_classes_names} is {all_topk_accuracy[0]}.')
    print(f'Top-5 accuracy of quantized {args.model} with classes {subset_classes_names} is {all_topk_accuracy[1]}.')

    end_time = datetime.now()

    print(f'\nTime used for evaluation: {end_time - start_time}\n')
    
    original_sparsity = eval_sparsity(model)
    subset_quantized_sparsity = eval_sparsity(subset_quantized_model)
    all_quantized_sparsity = eval_sparsity(all_quantized_model)
    
    print("Sparsity: Org: {}, Subset_Quant: {}, All_Quant".format(original_sparsity, subset_quantized_sparsity, all_quantized_sparsity))
    # store the validation accuracy and parameter settings
    with open(LOG_FILE_NAME, 'a') as f:
        csv_writer = csv.writer(f)
        row = [
            args.model, args.data_set, batch_size, 
            original_topk_accuracy[0], subset_topk_accuracy[0], all_topk_accuracy[0], 
            original_topk_accuracy[1], subset_topk_accuracy[1], all_topk_accuracy[1], 
            bits, mlp_scalar, cnn_scalar, 
            mlp_percentile, cnn_percentile, stochastic,
            args.regularizer, lamb, original_sparsity, subset_quantized_sparsity, all_quantized_sparsity, 
            args.retain_rate, args.fusion, args.seed,
            args.similar,args.num_classes,subset_classes_names,
            subset_dict['max_dist'],subset_dict['avg_dist']
        ]
        csv_writer.writerow(row)


if __name__ == '__main__':

    params = [(b, s, s, bs, mlp_per, cnn_per, l) 
                            for b in args.bits
                            for s in args.scalar
                            for bs in args.batch_size
                            for mlp_per in args.percentile
                            for cnn_per in args.percentile
                            for l in args.lamb
                            ]

    # testing section
    for b, mlp_s, cnn_s, bs, mlp_per, cnn_per, l in params:
        main(b, mlp_s, cnn_s, bs, mlp_per, cnn_per, l)
