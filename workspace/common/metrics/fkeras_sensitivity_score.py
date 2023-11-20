'''
The following code is adapted from 
fkeras: A sebsitive Analysis Tool for Edge Neural Networks
TODO: add authors
https://github.com/KastnerRG/fkeras
'''

from __future__ import print_function

import os 
import sys
import numpy as np
import random
import argparse
import pickle
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from workspace.common.metrics.utils.utils import *
from arguments import get_parser



# import the model 
exp_id = 2
file_name = os.path.join('../checkpoint/different_knobs_subset_10/bs_16/normal/ECON_6b/baseline/', f"net_exp_{exp_id}_best.pkl")

sys.path.append(os.path.join(sys.path[0], "../econ/code")) 

from model import load_checkpoint

def return_model(file_name, args):
    model = load_checkpoint(args, file_name)
    return model
model = return_model(file_name, args)
# import the top-k eigenvalues and eigenvectors
# get the model file name

file_name = os.path.join('../checkpoint/different_knobs_subset_10/bs_16/normal/ECON_6b/metrics/baseline/', "hessian.pkl")

if os.path.exists(file_name):
    print("use model {0}".format(file_name))
    
hessian_file = open(file_name, 'rb')
hessian_data = pickle.load(hessian_file)
hessian_file.close()

# select the hessian metrics of the model
hessian_metrics = hessian_data[exp_id]
print(hessian_metrics)

# compute the sensitivity score

# do the bitwise rank