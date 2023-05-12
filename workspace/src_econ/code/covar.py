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
def covariance(model):
    """
    Calculate the Fisher trace of a PyTorch model for a given dataset.
    
    Args:
        model (nn.Module): The PyTorch model to compute the Covariance trace for.
        
    Returns:
        covariance_trace (float): The Fisher trace of the model.
    """
    covariance_trace = 0.0
        
    # Compute the squared gradients of the log-likelihood with respect to the model parameters
    for name, param in model.named_parameters():
        # param_grad = param.grad
        covar_matrix = torch.cov(param)
        covariance_trace += torch.sum(covar_matrix)

    return covariance_trace


##################################################################
parser = get_parser(code_type='fisher')
args = parser.parse_args()

covar = []

for exp_id1 in range(5):
    file_name1 = get_filename(args, exp_id1)
    model = load_checkpoint(args, file_name1)   
    covar.append(covariance(model))

f = open(args.result_location, "wb")
pickle.dump(covar, f)
f.close()
