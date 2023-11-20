from __future__ import print_function

import sys
sys.path.insert(1, '..')

import numpy as np
import pickle
import matplotlib.pyplot as plt

from arguments import get_parser
from model import load_checkpoint 
from utils import *
from data import get_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################################################
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


##################################################################
class EmptyLayer(nn.Module):
  def __init__(self) -> None:
     super().__init__()
    
  def forward(self, x):
    return x

def log_likelihood(outputs, labels):
    # log_probs = torch.gather(outputs, 1, labels.unsqueeze(1)).squeeze()
    log_probs = torch.gather(outputs, 1, labels.type(torch.int64))
    log_likelihood = torch.sum(log_probs)
    return log_likelihood
  
##################################################################
def fisher_trace(model, dataloader):
    """
    Calculate the Fisher trace of a PyTorch model for a given dataset.
    
    Args:
        model (nn.Module): The PyTorch model to compute the Fisher trace for.
        dataloader (DataLoader): A PyTorch DataLoader object representing the dataset to use.
        
    Returns:
        fisher_trace (float): The Fisher trace of the model.
    """
    fisher_trace = 0.0
    
    # Set the model to evaluation mode
    model.eval()
    
    # Loop over the dataset
    for inputs, labels in dataloader:
        # Move inputs and labels to the device the model is on
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero out the gradients
        model.zero_grad()
        
        # Compute the gradients of the log-likelihood with respect to the model parameters
        outputs = model(inputs)
        log_likelihood = model.log_likelihood(outputs, labels)
        log_likelihood.backward(retain_graph=True)
        
        # Compute the squared gradients of the log-likelihood with respect to the model parameters
        for name, param in model.named_parameters():
            param_grad = param.grad
            squared_param_grad = torch.square(param_grad)
            fisher_trace += torch.sum(squared_param_grad)

    # Divide by the number of examples in the dataset
    fisher_trace /= len(dataloader.dataset)
    
    return fisher_trace


##################################################################
parser = get_parser(code_type='neural_eff')
args = parser.parse_args()

fisher = []

train_loader, test_loader = get_loader(args)

# if args.train_or_test == 'train':
    # eval_loader = train_loader
# elif args.train_or_test == 'test':
    # eval_loader = test_loader
eval_loader = test_loader

for exp_id1 in range(3):
    file_name1 = get_filename(args, exp_id1)
    model = load_checkpoint(args, file_name1)   
    model.softmax = EmptyLayer()
    setattr(model, 'log_likelihood', log_likelihood)
    fisher.append(fisher_trace(model, eval_loader))

f = open(args.result_location, "wb")
pickle.dump(fisher, f)
f.close()
