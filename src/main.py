import torch
import torchvision
import numpy as np
import os
import csv
import re
# ======================================================================================================================================
import timm
# ======================================================================================================================================
import argparse
from datetime import datetime
from quantize_neural_net import QuantizeNeuralNet
from utils import test_accuracy, eval_sparsity, fusion_layers_inplace, get_all_layers, test_accuracy_sub, finetune_model, 
get_training_dataloader_mobilenetv2, get_test_dataloader_mobilenetv2
from data_loaders import data_loader
from cifar100_subset_generation import generate_subset, class_names
import pandas as pd
from vgg_arc import vgg16
import bsconv.pytorch
from mobilenetv2 import MobileNetV2
import detectors

LOG_FILE_NAME = '../logs/Quantization_Log.csv'

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
parser.add_argument('--data_set', '-ds', default='ILSVRC2012', choices=['ILSVRC2012', 'CIFAR10', 'CIFAR100'],
                    help='dataset used for quantization')
parser.add_argument('--use_existing', '-ue', action='store_true', help='use existing subsets')
parser.add_argument('-model', default='resnet18', help='model name')
parser.add_argument('--subset_size', '-sn', default=None, type=int)
parser.add_argument('--similar_classes', '-sc', default=None, choices=["True", "False", "None"])
parser.add_argument('--mixed_precision', '-mp', default=None, choices=["Asc", "Desc", "None"])
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

args = parser.parse_args()
args.similar_classes = True if args.similar_classes == "True" else (False if args.similar_classes == "False" else None)
args.mixed_precision = True if args.mixed_precision == "Asc" else (False if args.mixed_precision == "Desc" else None)

LOG_FILE_SUB_NAME = f'../logs/Quantization_Log_{args.model}_{args.bits[0]}bit.csv'


if args.mixed_precision is not None:
    temp = LOG_FILE_SUB_NAME.split("_")
    add_tag = "mpasc" if args.mixed_precision else "mp"
    temp[-1] = add_tag + temp[-1]
    LOG_FILE_SUB_NAME = "_".join(temp)


print(LOG_FILE_SUB_NAME)

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

    num_epochs = 5  # Set the number of fine-tuning epochs
    learning_rate = 1e-4  # Fine-tuning learning rate
    
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the model to be quantized
    if args.data_set == 'ILSVRC2012':
        model = getattr(torchvision.models, args.model)(pretrained=True)

        # NOTE: refer to https://pytorch.org/vision/stable/models.html
        original_accuracy_table = {
            'alexnet': (.56522, .79066),
            'vgg16': (.71592, .90382),
            'resnet18': (.69758, .89078),
            'googlenet': (.69778, .89530),
            'resnet50': (.7613, .92862),
            'efficientnet_b1': (.7761, .93596),
            'efficientnet_b7': (.84122, .96908),
            'mobilenet_v2': (.71878, .90286)
        }

    elif args.data_set == 'CIFAR10':
        model = torch.load(os.path.join('pretrained_cifar10', args.model + '_cifar10.pt'), 
            map_location=torch.device('cpu')).module

        original_accuracy_table = {}

    elif args.data_set == 'CIFAR100' and args.model == 'resnet50':

        def load_model():
            model_path = "hf_hub:anonauthors/cifar100-timm-resnet50"
            
            model = timm.create_model(model_path, pretrained=True)

            return model

        original_accuracy_table = {}

    elif args.data_set == 'CIFAR100' and args.model == 'mobilenet':

        def load_model():

            model_path = 'cifar_mobilenetv2_w1'
            model = bsconv.pytorch.get_model(model_path, num_classes=100)

            return model

        original_accuracy_table = {}

    elif args.data_set == 'CIFAR100' and args.model == 'vgg16':


        def load_model():

            model_path = 'cifar100_best_model_VGG16_seed2023.pth'
            model = vgg16(100).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            return model

        original_accuracy_table = {}

    elif args.data_set == 'CIFAR100' and args.model == 'densenet':


        def load_model():

            model_path = "hf_hub:edadaltocg/densenet121_cifar100"
            model = timm.create_model(model_path, pretrained=True)

            return model

        original_accuracy_table = {}

    elif args.data_set == 'CIFAR100' and args.model == 'mobilenetv2':
      
        def load_model():
            model_path = "Sunday_23_February_2025_20h_14m_08s/mobilenetv2-200-regular.pth"
            model = MobileNetV2()
            model.load_state_dict(state_dict)
            return model
          
        original_accuracy_table = {}


    model = load_model()

    # print(model.state_dict()["backbone.init_conv.conv.weight"].norm())


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
    
    
    # load the data loader for training and testing
    if args.model != "mobilenetv2"
        if args.subset_size is not None:
            if not args.use_existing:
                subset_info = generate_subset(args.subset_size, similar_classes = args.similar_classes)
            else:
    
                existing_csvs = os.listdir("../logs/")
                existing_csvs = list(filter(lambda x: ".csv" in x, existing_csvs))
                chosen_path = existing_csvs[0]
    
                print(chosen_path)
                
                ref_df = pd.read_csv(f"../logs/{chosen_path}")
                
                try:
                    curr_df = pd.read_csv(LOG_FILE_SUB_NAME)
                    curr_index = curr_df.shape[0]
                except:
                    curr_index = 0
        
                subset_info = {}
                subset_info["classes"] = ref_df["Subset_Inds"].iloc[curr_index]
                subset_info["min_dist"] = ref_df["Min_KL"].iloc[curr_index]
                subset_info["max_dist"] = ref_df["Max_KL"].iloc[curr_index]
                subset_info["avg_dist"] = ref_df["Avg_KL"].iloc[curr_index]
                subset_info["median_dist"] = ref_df["Median_KL"].iloc[curr_index]
        
                subset_info["classes"] = list(map(lambda x: int(x), re.findall(r"[0-9]+", subset_info["classes"])))
        
                print(subset_info["classes"], type(subset_info["classes"]))
                
            train_loader, test_loader = data_loader(args.data_set, batch_size, args.num_worker, subset = subset_info["classes"])
        else:
            train_loader, test_loader = data_loader(args.data_set, batch_size, args.num_worker)
    else:
        if args.subset_size is not None:
            subset_info = generate_subset(args.subset_size, similar_classes = args.similar_classes)
            train_loader = get_training_dataloader_mobilenetv2(num_workers=args.num_worker, 
                                                               batch_size=batch_size,
                                                               subset=subset_info["classes"])
            test_loader = get_test_dataloader_mobilenetv2(num_workers=args.num_worker, 
                                                          batch_size=batch_size, 
                                                          subset=subset_info["classes"])
            labels_set = set(label for _, label in test_loader)
        else:
            train_loader = get_training_dataloader_mobilenetv2(num_workers=args.num_worker, 
                                                               batch_size=batch_size)
            test_loader = get_test_dataloader_mobilenetv2(num_workers=args.num_worker, 
                                                          batch_size=batch_size)
        
        
    
    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, args.model, batch_size, 
                                    train_loader, 
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
                                    device = device,
                                    mixed_precision = args.mixed_precision
                                    )
    start_time = datetime.now()
    quantized_model = quantizer.quantize_network()
    end_time = datetime.now()
    quantized_model = quantized_model.to(device)

    print(f'\nTime used for quantization: {end_time - start_time}\n')

    saved_model_name = f'ds{args.data_set}_b{bits}_batch{batch_size}_mlpscalar{mlp_scalar}_cnnscalar{cnn_scalar}\
        _mlppercentile{mlp_percentile}_cnnpercentile{cnn_percentile}_retain_rate{args.retain_rate}\
        _reg{args.regularizer}_lambda{lamb}.pt'

    if not os.path.isdir('../quantized_models/'):
        os.mkdir('../quantized_models/')
    saved_model_dir = '../quantized_models/'+args.model
    if not os.path.isdir(saved_model_dir):
        os.mkdir(saved_model_dir)
    torch.save(quantized_model, os.path.join(saved_model_dir, saved_model_name))

    topk = (1, 5)   # top-1 and top-5 accuracy
    
    if args.model in original_accuracy_table:
        print(f'\nUsing the original model accuracy from pytorch.\n')
        original_topk_accuracy = original_accuracy_table[args.model]
    else:
        print(f'\nEvaluting the original model to get its accuracy\n')
        original_topk_accuracy = test_accuracy(model, test_loader, device, topk)
        original_topk_accuracy_sub = test_accuracy_sub(model, test_loader, subset_info["classes"], device, topk)
    
    print(f'Top-1 & Top-5 accuracies of {args.model} is {original_topk_accuracy[0]} & {original_topk_accuracy[1]}.')
    print(f'Top-1 & Top-5 accuracies of {args.model}, picking from subset classes, is {original_topk_accuracy_sub[0]} & {original_topk_accuracy_sub[1]}.')
    
    start_time = datetime.now()

    print(f'\n Evaluting the quantized model to get its accuracy\n')
    topk_accuracy = test_accuracy(quantized_model, test_loader, device, topk)
    topk_accuracy_sub = test_accuracy_sub(quantized_model, test_loader, subset_info["classes"], device, topk)
    print(f'Top-1 & Top-5 accuracies of quantized {args.model} is {topk_accuracy[0]} & {topk_accuracy[1]}.')
    print(f'Top-1 & Top-5 accuracies of quantized {args.model}, picking from subset classes, is {topk_accuracy_sub[0]} & {topk_accuracy_sub[1]}.')

    end_time = datetime.now()


    print(f'\nTime used for evaluation: {end_time - start_time}\n')

    original_sparsity = eval_sparsity(model)
    quantized_sparsity = eval_sparsity(quantized_model)

    del model, quantized_model
    # del train_loader, test_loader

    print(f'\nFine-tuning the original model on subset\n')

    # train_loader, test_loader = data_loader(args.data_set, 8, args.num_worker, subset = subset_info["classes"])
    
    finetuned_model = finetune_model(load_model, train_loader, batch_size, num_epochs, learning_rate, device)

    print(f'\nEvaluting the fine-tuned model to get its accuracy\n')
    finetuned_topk_accuracy = test_accuracy(finetuned_model, test_loader, device, topk)
    finetuned_topk_accuracy_sub = test_accuracy_sub(finetuned_model, test_loader, subset_info["classes"], device, topk)
    print(f'Top-1 & Top-5 accuracies of fine-tuned {args.model} is {finetuned_topk_accuracy[0]} & {finetuned_topk_accuracy[1]}.')
    print(f'Top-1 & Top-5 accuracies of fine-tuned {args.model}, picking from subset classes, is {finetuned_topk_accuracy_sub[0]} & {finetuned_topk_accuracy_sub[1]}.')

    train_loader, test_loader = data_loader(args.data_set, batch_size, args.num_worker, subset = subset_info["classes"])

    print(f'\nQuantizing the fine-tuned model on subset\n')
    quantizer_finetuned = QuantizeNeuralNet(finetuned_model, args.model, batch_size, 
                                    train_loader, 
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
                                    device = device,
                                    mixed_precision = args.mixed_precision
                                    )

    quantized_finetuned_model = quantizer_finetuned.quantize_network()
    quantized_finetuned_model = quantized_finetuned_model.to(device)

    print(f'\n Evaluting the quantized + fine-tuned model to get its accuracy\n')
    quant_ft_topk_accuracy = test_accuracy(quantized_finetuned_model, test_loader, device, topk)
    quant_ft_topk_accuracy_sub = test_accuracy_sub(quantized_finetuned_model, test_loader, subset_info["classes"], device, topk)
    print(f'Top-1 & Top-5 accuracies of quantized + fine-tuned {args.model} is {quant_ft_topk_accuracy[0]} & {quant_ft_topk_accuracy[1]}.')
    print(f'Top-1 & Top-5 accuracies of quantized + fine-tuned {args.model}, picking from subset classes, is {quant_ft_topk_accuracy_sub[0]} & {quant_ft_topk_accuracy_sub[1]}.')
    
    
    finetuned_sparsity = eval_sparsity(finetuned_model)
    quantized_finetuned_sparsity = eval_sparsity(quantized_finetuned_model)
    
    print("Sparsity: Org: {}, Quant: {}".format(original_sparsity, quantized_sparsity))
    # store the validation accuracy and parameter settings

    if not os.path.exists(LOG_FILE_SUB_NAME):
        columns = ['Model Name', 'Dataset', 'Quantization Batch Size',
       'Original Top1 Accuracy', 'Quantized Top1 Accuracy',
       'Original Top5 Accuracy', 'Quantized Top5 Accuracy', 'Bits',
       'MLP_Alphabet_Scalar', 'CNN_Alphabet_Scalar', 'MLP_Percentile',
       'CNN_Percentile', 'Stochastic Quantization', 'Regularizer', 'Lambda',
       'Original Sparsity', 'Quantized Sparsity', 'Retain_rate', 'Fusion',
       'Seed', 'Subset_Inds', 'Subset_Classes', 'Max_KL', 'Min_KL', 'Avg_KL', 'Median_KL',
       'Classes Repeated', 'Fine-Tuned Top1 Accuracy', 'Fine-Tuned Top5 Accuracy', 'Quant+FT Top1 Accuracy', 
       'Quant+FT Top5 Accuracy', 'Original Top1 Accuracy (Pick Sub)', 'Original Top5 Accuracy (Pick Sub)', 
       'Quantized Top1 Accuracy (Pick Sub)', 'Quantized Top5 Accuracy (Pick Sub)', 
       'Fine-Tuned Top1 Accuracy (Pick Sub)', 'Fine-Tuned Top5 Accuracy (Pick Sub)',
       'Quant+FT Top1 Accuracy (Pick Sub)', 'Quant+FT Top5 Accuracy (Pick Sub)', 'Fine-Tuned Sparsity', 'Quant+FT Sparsity']
        df = pd.DataFrame(columns = columns)
        df.to_csv(LOG_FILE_SUB_NAME, index = False)

    if args.subset_size is not None:
        with open(LOG_FILE_SUB_NAME, 'a') as f:
            csv_writer = csv.writer(f)
            row = [
                args.model, args.data_set, batch_size, 
                original_topk_accuracy[0], topk_accuracy[0], 
                original_topk_accuracy[1], topk_accuracy[1], 
                bits, mlp_scalar, cnn_scalar, 
                mlp_percentile, cnn_percentile, stochastic,
                args.regularizer, lamb, original_sparsity, quantized_sparsity,
                args.retain_rate, args.fusion, args.seed, subset_info["classes"],
                list(map(lambda x: class_names[x].item(), subset_info["classes"])),
                subset_info["max_dist"], subset_info["min_dist"], subset_info["avg_dist"], 
                subset_info["median_dist"], False, finetuned_topk_accuracy[0], 
                finetuned_topk_accuracy[1], quant_ft_topk_accuracy[0], 
                quant_ft_topk_accuracy[1], original_topk_accuracy_sub[0], original_topk_accuracy_sub[1], 
                topk_accuracy_sub[0], topk_accuracy_sub[1], finetuned_topk_accuracy_sub[0], 
                finetuned_topk_accuracy_sub[1], quant_ft_topk_accuracy_sub[0], quant_ft_topk_accuracy_sub[1],
                finetuned_sparsity, quantized_finetuned_sparsity
            ]
            csv_writer.writerow(row)
    else:
        with open(LOG_FILE_NAME, 'a') as f:
            csv_writer = csv.writer(f)
            row = [
                args.model, args.data_set, batch_size, 
                original_topk_accuracy[0], topk_accuracy[0], 
                original_topk_accuracy[1], topk_accuracy[1], 
                bits, mlp_scalar, cnn_scalar, 
                mlp_percentile, cnn_percentile, stochastic,
                args.regularizer, lamb, original_sparsity, quantized_sparsity,
                args.retain_rate, args.fusion, args.seed
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
