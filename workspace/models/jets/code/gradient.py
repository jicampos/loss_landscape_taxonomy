from __future__ import print_function

import sys
sys.path.insert(1, './code/')

import numpy as np
import pickle
import matplotlib.pyplot as plt

from arguments import get_parser
from model import load_checkpoint 
from utils import *
from data import get_loader


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


def compute_gradient_trace(model):
    gradient_traces = []
    for name, param in model.named_parameters():
        if len(param.shape) > 1:
            gradient_traces.append(param.trace().item())
    return np.array(gradient_traces).mean()


##################################################################
def get_params(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.data.view(-1))
    weight_flat = torch.cat(res)
    return weight_flat

def compute_distance(model1, model2):
    
    params1 = get_params(model1)
    params2 = get_params(model2)
    dist = (params1-params2).norm().item()
    
    return dist

##################################################################
parser = get_parser(code_type='neural_eff')
args = parser.parse_args()


model_gradients = []
train_loader, test_loader = get_loader(args)
eval_loader = test_loader  # hard coded - make dependent on args

criterion = nn.BCELoss()

for exp_id1 in range(3):
    gradient_traces = []
    file_name1 = get_filename(args, exp_id1)
    model = load_checkpoint(args, file_name1)
    print(model)
    print(f'Computing Gradients of {file_name1}')
    ## run each batch then accumate gradients 
    for inputs, targets in eval_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.float()) 
        loss.backward()
        gradient_traces.append(compute_gradient_trace(model))
    model_gradients.append(np.array(gradient_traces).mean())

print('Finish computing gradients')
print(model_gradients)
f = open(args.result_location, "wb")
pickle.dump(model_gradients, f)
f.close()

# python ./code/gradient.py --arch JT_6b --early-stopping --data-path ../../data/JT --train-bs 16 --test-bs 16 --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/ --result-location ../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/metrics/gradient.pkl 1>../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/metrics//gradient.log 2>../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/metrics//gradient.err
