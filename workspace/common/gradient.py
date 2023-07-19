from __future__ import print_function

import sys
sys.path.insert(1, './code/')

import numpy as np
import pickle
import matplotlib.pyplot as plt

from arguments import get_parser
from utils import *

parser = get_parser(code_type='neural_eff')
args = parser.parse_args()


model_arch = args.arch.split('_')[0]
print('Importing code for', model_arch)
if model_arch == 'JT':
    sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
elif model_arch == 'ECON':
    sys.path.append(os.path.join(sys.path[0], "../econ/code")) 

from model import load_checkpoint 
from data import get_loader


##################################################################
# Helper Functions
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


def get_batch_gradients(model):
    gradient_traces = []
    for name, param in model.encoder.named_parameters():
        if param.requires_grad and type(param.grad) == torch.Tensor:
            # print('Adding', name, type(param))
            gradient_traces.append(param.grad.mean())
    # print('Computing mean of', gradient_traces)
    return np.array(gradient_traces)


##################################################################
# Compute Gradient 
##################################################################
model_gradients = []
train_loader, test_loader = get_loader(args)

if args.train_or_test == 'train':
    eval_loader = train_loader
elif args.train_or_test == 'test':
    eval_loader = test_loader

if model_arch == 'JT':
    criterion = nn.CrossEntropyLoss()  
elif model_arch == 'ECON':
    from telescope_pt import telescopeMSE8x8
    criterion = telescopeMSE8x8


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
        gradient_traces.append(get_batch_gradients(model))
    model_gradients.append(np.array(gradient_traces).mean())

print('Finish computing gradients')
print(model_gradients)
f = open(args.result_location, "wb")
pickle.dump(model_gradients, f)
f.close()

# python ./code/gradient.py --arch JT_6b --early-stopping --data-path ../../data/JT --train-bs 16 --test-bs 16 --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/ --result-location ../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/metrics/gradient.pkl 1>../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/metrics//gradient.log 2>../checkpoint/different_knobs_subset_10/bs_16/bs_decay/JT_6b/metrics//gradient.err
# python ../common/gradient.py --early-stopping --arch ECON_6b --experiment-name baseline --train-or-test test --data-path ../../data/ECON/Elegun --train-bs 16 --test-bs 16 --checkpoint-folder ../checkpoint/different_knobs_subset_10/lr_0.05/normal/ECON_6b/autoencoder --result-location ../checkpoint/different_knobs_subset_10/lr_0.05/normal/ECON_6b/metrics/baseline/gradient.pkl
