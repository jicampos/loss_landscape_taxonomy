import os 
import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def get_loader(args):
    kwargs = {'num_workers': 10, 'pin_memory': True}

    print(f'Loading dataset with train batch size {args.train_bs} and test batch size {args.test_bs}')
    
    X_train = np.load(os.path.join(args.data_dir, 'train_data_inputs_64_frames_5_hops_512_fft_1024_mels_128_power_2.0.npy'))
    X_train, X_test = train_test_split(X_train, test_size=0.1)

    # Convert to torch tensors
    X_train = torch.Tensor(X_train) 
    X_test = torch.Tensor(X_test) 

    # Create dataset and dataloaders
    train_dataset = TensorDataset(X_train, X_train) 
    test_dataset = TensorDataset(X_test, X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, **kwargs)
    
    print("\nDataset loading complete!")

    return train_loader, test_loader
