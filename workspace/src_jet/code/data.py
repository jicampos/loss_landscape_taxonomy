from torchvision import datasets, transforms
import torch


def get_loader(args):

    CIFAR10_mean, CIFAR10_var = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_mean, CIFAR10_var),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_mean, CIFAR10_var),
    ])
    
    if args.random_labels:
    
        
    elif args.data_subset:

    else:
        trainset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False)
    
    return train_loader, test_loader

