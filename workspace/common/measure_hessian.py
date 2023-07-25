"""The following code is adapted from 
PyHessian: Neural Networks Through the Lens of the Hessian
Z. Yao, A. Gholami, K Keutzer, M. Mahoney
https://github.com/amirgholami/PyHessian
"""

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

# Local imports 
sys.path.append(os.path.join(sys.path[0], "pyhessian/")) 
import pyhessian
from pyhessian import hessian
from utils import *
from arguments import get_parser


parser = get_parser(code_type='hessian')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

model_arch = args.arch.split('_')[0]
print('Importing code for', model_arch)
if model_arch == 'JT':
    sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
elif model_arch == 'ECON':
    sys.path.append(os.path.join(sys.path[0], "../econ/code")) 
elif model_arch == 'AD':
    sys.path.append(os.path.join(sys.path[0], "../ad08/code")) 

# Import dataloader & model 
from data import get_loader
from model import load_checkpoint


# Get data
args.train_bs = args.mini_hessian_batch_size
args.test_bs = args.mini_hessian_batch_size

train_loader, test_loader = get_loader(args)

if args.train_or_test == 'train':
    eval_loader = train_loader
elif args.train_or_test == 'test':
    eval_loader = test_loader

def return_model(file_name, args):

    model = load_checkpoint(args, file_name)
    
    return model

if model_arch == 'ECON':
    from telescope_pt import telescopeMSE8x8
    criterion = telescopeMSE8x8
elif model_arch == 'JT':
    criterion = nn.CrossEntropyLoss()  # label loss
elif model_arch == 'AD':
    criterion = nn.MSELoss()  # reconstruction loss

######################################################
# Begin the computation
######################################################

hessian_result = {}

# turn model to eval mode
for exp_id in range(3):

    # Hessian prepare steps
    
    assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
    # assert (50000 % args.hessian_batch_size == 0)
    batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

    if batch_num == 1:
        for inputs, labels in eval_loader:
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(eval_loader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break
            
    file_name = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id}.pkl")
    es_file_name = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id}_early_stopped_model.pkl")
    best_file_name = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id}_best.pkl")
    if args.early_stopping:
        if os.path.exists(es_file_name):
            file_name = es_file_name
        elif os.path.exists(best_file_name):
            file_name = best_file_name
        print("use model {0}".format(file_name))
        
    print(f'********** start the experiment on model {file_name} **********')
        
    model = return_model(file_name, args)
    model.eval()
    if batch_num == 1:
        hessian_comp = hessian(model,
                               criterion,
                               data=hessian_dataloader,
                               cuda=args.cuda)
    else:
        hessian_comp = hessian(model,
                               criterion,
                               dataloader=hessian_dataloader,
                               cuda=args.cuda)

    print('********** finish data londing and begin Hessian computation **********')

    top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=200, tol=1e-6)
    trace = hessian_comp.trace(maxIter=200, tol=1e-6)

    print('\n***Top Eigenvalues: ', top_eigenvalues)
    print('\n***Trace: ', np.mean(trace))

    hessian_result[exp_id] = {'top_eigenvalue': top_eigenvalues, 'trace': np.mean(trace)}

f = open(args.result_location, "wb")
pickle.dump(hessian_result, f)
f.close()
    
print("Save results complete!!!")

# For debugging
# python measure_hessian.py --hessian-batch-size 64 --mini-hessian-batch-size 64 --train-bs 64 --test-bs 64 --train-or-test test --checkpoint-folder /data1/jcampos/loss_landscape/workspace/checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/JT_8b/ --batch-norm --data-path /data1/jcampos/loss_landscape/data/JT --arch JT_8b  --early-stopping --cuda
# python measure_hessian.py --hessian-batch-size 64 --mini-hessian-batch-size 64 --train-bs 64 --test-bs 64 --train-or-test test --checkpoint-folder /data1/jcampos/loss_landscape/workspace/checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/JT_32b/ --batch-norm --data-path /data1/jcampos/loss_landscape/data/JT --arch JT_32b  --early-stopping --cuda --weight-precision 32 --result-location /data1/jcampos/loss_landscape/workspace/checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/JT_32b/metrics
