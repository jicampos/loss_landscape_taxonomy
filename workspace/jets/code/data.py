import os 
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

# Hard code here, though normally grabbed from a yaml file
# features = ['j_zlogz', 'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_mass_mmdt', 'j_multiplicity']


def load_CIFAR10(args, kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader


def load_JETS(args, kwargs):
    # No support for subset or randomized labels at this point...

    # Data is normalized via sklearn.standardscaler per dataset right now, which is how the HAWQ models + QAP Brevitas models are trained
    # but it likely makes more sense to fit to the train data and use that mean/var to normalize train and test sets?
    #TODO - Investigate normalizing per dataset or on train set mean/var 
    
    print("Loading Datasets")
    
    file_suffix = ''
    if args.noise:
        print(f'Loading noisy dataset with {args.noise_type} {args.noise_magnitude}')
        file_suffix = f'_{args.noise_type}{args.noise_magnitude}'
    X_train = np.load(os.path.join(args.data_path, 'X_train' + file_suffix + '.npy'))
    y_train = np.load(os.path.join(args.data_path, 'y_train' + file_suffix + '.npy'))
    X_test  = np.load(os.path.join(args.data_path, 'X_test' + file_suffix + '.npy'))
    y_test  = np.load(os.path.join(args.data_path, 'y_test' + file_suffix + '.npy'))

    # Transform to torch tensor
    X_train = torch.Tensor(X_train) 
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # Create dataset and dataloaders
    train_dataset = TensorDataset(X_train, y_train) 
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, **kwargs)
    
    print("\nDataset loading complete!")

    return train_loader, test_loader


def get_loader(args):
    kwargs = {'num_workers': 10, 'pin_memory': True}
    model_arch = args.arch.split('_')[0]

    print(f'Loading dataset with train batch size {args.train_bs} and test batch size {args.test_bs}')

    if model_arch == 'RN07':
        return load_CIFAR10(args, kwargs)
    elif model_arch == 'JT':
        return load_JETS(args, kwargs)
    else:
        raise Exception(f'Model architecture {model_arch} not recognized')
