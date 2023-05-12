from __future__ import print_function

import sys
sys.path.insert(1, '..')

import numpy as np
import pickle
import matplotlib.pyplot as plt

from arguments import get_parser
from model import load_checkpoint 
from utils import *
from data import get_dataset, get_loader


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
def jacobian(batch, model):

    j_trace = 0
    for n in range(batch.shape[0]):
        x = batch[n].reshape(-1)
        x.requires_grad = True
        # Calculate the Jacobian matrix
        y = model(x).reshape(-1)
        J = torch.zeros((y.shape[0], x.shape[0]))
        for i in range(y.shape[0]):
            # Compute the gradient of y[i] with respect to x
            grad = torch.autograd.grad(y[i], x, create_graph=True)[0]
            # Set the i-th row of J to the gradient
            J[i] = grad
        j_trace += J.trace()
    return j_trace


##################################################################
parser = get_parser(code_type='fisher')
args = parser.parse_args()

trainloader, testloader = get_loader(args)
for inputs, labels in testloader:
    break

# trainset, testset = get_dataset(args)
# x = torch.tensor(testset[0][0], requires_grad=True)
# x = torch.tensor([-1.6036,  0.8700,  0.5328,  0.0110,  0.4951,  0.1139, -0.1748,  0.0399,
#         -0.1748,  0.8194,  1.0731,  1.0447,  0.7836,  0.8257,  0.2496,  2.0396], requires_grad=True)
jacobian_list = []

for exp_id1 in range(5):
    file_name1 = get_filename(args, exp_id1)
    model = load_checkpoint(args, file_name1)
    jacobian_list.append(jacobian(inputs, model).item())

f = open(args.result_location, "wb")
pickle.dump(jacobian_list, f)
f.close()
