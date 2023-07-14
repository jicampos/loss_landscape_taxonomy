from __future__ import print_function

import os
import sys
import numpy as np
import pickle

from arguments import get_parser
from utils import *


parser = get_parser(code_type='model_dist')
args = parser.parse_args()

model_arch = args.arch.split('_')[0]
print('Importing code for', model_arch)
if model_arch == 'JT':
    sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
elif model_arch == 'ECON':
    sys.path.append(os.path.join(sys.path[0], "../econ/code")) 

from model import load_checkpoint 


def get_params(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.data.view(-1))
    weight_flat = torch.cat(res)
    return weight_flat

def compute_distance(model1, model2):
    
    params1 = get_params(model1.encoder)
    params2 = get_params(model2.encoder)
    dist = (params1-params2).norm().item()
    
    return dist


model_distance = {}

for exp_id1 in range(3):
    
    model_distance[exp_id1] = {}
    
    for exp_id2 in range(3):
        
        file_name1, file_name2 = return_file_name(args, exp_id1, exp_id2)
        
        model1 = load_checkpoint(args, file_name1)
        model2 = load_checkpoint(args, file_name2)
        
        model_distance[exp_id1][exp_id2] = {'dist': compute_distance(model1, model2)}
        
        temp_results = {'model_distance': model_distance}
        
        f = open(args.result_location, "wb")
        pickle.dump(temp_results, f)
        f.close()
        
        
        