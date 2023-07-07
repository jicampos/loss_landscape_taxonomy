import os 
import argparse
from arguments import get_parser
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from models.econ.autoencoder_datamodule import AutoEncoderDataModule


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


def load_ECON(args, kwargs):
    parser = get_parser(code_type='hessian')
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./pt_autoencoder_test")
    parser.add_argument("--experiment_name", type=str, default="autoencoder")
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "auto"], default="auto"
    )
    parser.add_argument("--checkpoint", type=str, default="", help="model checkpoint")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument(
        "--quantize", 
        action="store_true", 
        default=False, 
        help="quantize model to 6-bit fixed point (1 signed bit, 1 integer bit, 4 fractional bits)"
    )

    # Add dataset-specific args
    parser = AutoEncoderDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    data_module = AutoEncoderDataModule.from_argparse_args(args)
    data_module.setup(0)
    train_dataloader = data_module.train_dataloader()
    test_dataloader = data_module.test_dataloader()
    return train_dataloader, test_dataloader


def get_loader(args):
    kwargs = {'num_workers': 10, 'pin_memory': True}
    model_arch = args.arch.split('_')[0]

    print(f'Loading dataset with train batch size {args.train_bs} and test batch size {args.test_bs}')

    if model_arch == 'RN07':
        return load_CIFAR10(args, kwargs)
    elif model_arch == 'JT':
        return load_JETS(args, kwargs)
    elif model_arch == 'ECON':
        return load_ECON(args, kwargs)
    else:
        raise Exception(f'Model architecture {model_arch} not recognized')
