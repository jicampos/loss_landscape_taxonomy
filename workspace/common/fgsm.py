from __future__ import print_function
# Standard 
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
# PyTorch/ML
import torch 
import torch.nn as nn
from sklearn.metrics import accuracy_score
# Custom 
from arguments import get_parser
from utils import *


####################################################
# Get file name from experiment ID 
####################################################
def get_filename(args, exp_id1):
    checkpoint1 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id1}.pkl")
    
    early_stopped_checkpoint1 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id1}_early_stopped_model.pkl")
    
    best_checkpoint1 = f'{args.checkpoint_folder}/net_exp_{exp_id1}_best.pkl'
    
    if args.early_stopping:
        if os.path.exists(early_stopped_checkpoint1):
            checkpoint1 = early_stopped_checkpoint1
            print("Using the early stopping model!")
        elif os.path.exists(best_checkpoint1):
            checkpoint1 = best_checkpoint1
            print("Using best model!")
    
    print(checkpoint1)
    return checkpoint1 


####################################################
# Apply Fast Gradient Sign Method (FGSM)
####################################################
def perturb_input(model, xbatch, ybatch, criterion, optimizer, epsilon=0.02):
    # Turn gradients on 
    xbatch.requires_grad = True
    # Evaluate 
    outputs = model(xbatch)
    loss = criterion(outputs, ybatch)
    optimizer.zero_grad()
    # Compute gradients 
    loss.backward() 
    # Create adverarial batch 
    xbatch = xbatch + epsilon * torch.sign(xbatch.grad)
    return xbatch


####################################################
# Import model, data loader, & criterion
####################################################
parser = get_parser(code_type='neural_eff')
args = parser.parse_args()

model_arch = args.arch.split('_')[0]
print('Importing code for', model_arch)
if model_arch == 'JT':
    sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
elif model_arch == 'ECON':
    sys.path.append(os.path.join(sys.path[0], "../econ/code")) 
elif model_arch == 'AD':
    sys.path.append(os.path.join(sys.path[0], "../ad08/code")) 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam

from model import load_checkpoint 
from data import get_loader

train_loader, test_loader = get_loader(args)

if args.train_or_test == 'train':
    eval_loader = train_loader
elif args.train_or_test == 'test':
    eval_loader = test_loader
else:
    raise Exception('Unrecognized argument for --train-or-test', args.train_or_test)


####################################################
# Apply FGSM and record accuracy 
####################################################
model_metrics = {}
for exp_id1 in range(3):
    file_name1 = get_filename(args, exp_id1)
    model = load_checkpoint(args, file_name1)    
    print(f'Computing accuracy for {file_name1}')
    print(f'Using {args.train_or_test} dataset')

    model_metrics[exp_id1] = {}
    optim = optimizer(model.parameters())
    y_pred = torch.Tensor([])
    y_true = torch.Tensor([])
    # loss = []
    loss = 0
    
    for batch in eval_loader:
        xbatch, ybatch = batch
        # Apply FGSM
        xbatch = perturb_input(model, xbatch, ybatch, criterion, optim)
        # Evalute on perturbed input 
        pred = model(xbatch)
        # loss.append(criterion(pred, ybatch).item())
        loss += criterion(pred, ybatch).item()
        
        y_pred = torch.concat([y_pred, pred])
        y_true = torch.concat([y_true, ybatch])
    
    model_metrics[exp_id1]['acc'] = accuracy_score(
        np.argmax(y_true.detach().numpy(), axis=1),
        np.argmax(y_pred.detach().numpy(), axis=1)
    )
    # model_metrics[exp_id1]['loss'] = np.array(loss).mean()
    model_metrics[exp_id1]['loss'] = loss


f = open(args.result_location, "wb")
pickle.dump(model_metrics, f)
f.close()
