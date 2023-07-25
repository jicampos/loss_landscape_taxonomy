from __future__ import print_function

import os
import sys
import pickle
import torch.nn as nn

from arguments import get_parser
from utils import *

parser = get_parser(code_type='loss_acc')
args = parser.parse_args()

####################################################
# Add Path to Model & Setup Criterion
####################################################
model_arch = args.arch.split('_')[0]
print('Importing code for', model_arch)

if model_arch == 'JT':
    sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
    criterion = nn.BCELoss()
elif model_arch == 'ECON':
    sys.path.append(os.path.join(sys.path[0], "../econ/code")) 
    criterion = nn.MSELoss()
elif model_arch == 'AD':
    sys.path.append(os.path.join(sys.path[0], "../ad08/code")) 
    criterion = nn.MSELoss()

from model import load_checkpoint 
from data import get_loader

####################################################
# Get Data
####################################################
train_loader, test_loader = get_loader(args)

if args.train_or_test == "train":
    eval_loader = train_loader
elif args.train_or_test == "test":
    eval_loader = test_loader

####################################################
# Compute loss 
####################################################
results = {}
if not args.ensemble_average_acc:
    for exp_id in range(3):
        
        file_name = return_file_name_single(args, exp_id)
        model = load_checkpoint(args, file_name)
        results[exp_id] = test_acc_loss(eval_loader, model, criterion)
else:
    models = []
    for exp_id in range(3):
        file_name = return_file_name_single(args, exp_id)
        models.append(load_checkpoint(args, file_name))
    results['ensemble_average'] = test_ensemble_average(models, eval_loader)

print('Results', results)        
f = open(args.result_location, "wb")
pickle.dump(results, f)
f.close()
