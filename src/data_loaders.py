import torch
import torchvision
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import os
import glob
import re
import pickle
from utils import parse_imagenet_val_labels

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
# use above function and g to preserve reproducibility.

class Imagenet(Dataset):
    """
    Validation dataset of Imagenet
    """
    def __init__(self, data_dir, transform):
        # we can maybe pput this into diff files.
        self.Y = torch.from_numpy(parse_imagenet_val_labels(data_dir)).long()
        self.X_path = sorted(glob.glob(os.path.join(data_dir, 'ILSVRC2012_img_val/*.JPEG')), 
            key=lambda x: re.search('%s(.*)%s' % ('ILSVRC2012_img_val/', '.JPEG'), x).group(1))
        self.transform = transform

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, idx):
        img = Image.open(self.X_path[idx]).convert('RGB')
        y = self.Y[idx] 
        if self.transform:
            x = self.transform(img)
        return x, y


def data_loader(ds_name, batch_size, num_workers, classes_of_interest=[i for i in range(100)]): 
    """
    Prepare data loaders
    """
    if ds_name == 'CIFAR100':
        data_dir = '../data'

        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, 
            transform=transform_train)

        test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, 
            transform=transform_test)

        # Filter the indices of classes of interest
        def filter_classes(dataset, classes_of_interest):
            indices = [i for i, (_, label) in enumerate(dataset) if label in classes_of_interest]
            return Subset(dataset, indices)
            
        train_ds_filtered = filter_classes(train_ds, classes_of_interest)
        test_ds_filtered = filter_classes(test_ds, classes_of_interest)

        # Check if the subsets are empty
        if len(train_ds_filtered) == 0 or len(test_ds_filtered) == 0:
            raise ValueError("Filtered dataset is empty. Check 'classes_of_interest' and dataset labels.")
        
        train_dl = DataLoader(train_ds_filtered, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        
        test_dl = DataLoader(test_ds_filtered, shuffle=False, batch_size=min(batch_size, 1024),
                             num_workers=num_workers)

    else:
        raise Exception('Unkown dataset! This script is intended only for CIFAR100!')

    return train_dl, test_dl 
