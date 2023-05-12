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

from scipy.stats import wasserstein_distance


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

  
# ##################################################################
# def emd(model, eval_loader):

#     for inputs, labels in eval_loader:
#         break

#     all_emd = []

#     for i in range(len(inputs)):
#         x = inputs[i]
#         target = labels[i].numpy()
#         y = model(x).reshape(-1).detach().numpy()

#         emd_hist = [0]
#         for j in range(1, target.shape[0]+1):
#             emd_hist.append(
#                 (y[j-1] + emd_hist[j-1]) - target[j-1]
#             )
#         all_emd.append(np.sum(emd_hist))
#     return np.average(all_emd)


# ##################################################################
# parser = get_parser(code_type='fisher')
# args = parser.parse_args()

# fisher = []

# train_loader, test_loader = get_loader(args)

# # if args.train_or_test == 'train':
#     # eval_loader = train_loader
# # elif args.train_or_test == 'test':
#     # eval_loader = test_loader
# eval_loader = test_loader

# for exp_id1 in range(5):
#     file_name1 = get_filename(args, exp_id1)
#     model = load_checkpoint(args, file_name1)   
#     fisher.append(emd(model, eval_loader))

# f = open(args.result_location, "wb")
# pickle.dump(fisher, f)
# f.close()


##################################################################
def emd(model1, model2, eval_loader):

    for inputs, labels in eval_loader:
        break

    labels = model1(inputs).detach()
    all_emd = []

    for i in range(len(inputs)):
        x = inputs[i]
        target = labels[i].numpy()
        y = model2(x).reshape(-1).detach().numpy()

        # emd_hist = [0]
        # for j in range(1, target.shape[0]+1):
        #     emd_hist.append(
        #         (y[j-1] + emd_hist[j-1]) - target[j-1]
        #     )
        # all_emd.append(np.sum(emd_hist)
        # )
        all_emd.append(wasserstein_distance(y, target))
    return np.average(all_emd)



##################################################################
parser = get_parser(code_type='fisher')
args = parser.parse_args()

fisher = []

train_loader, test_loader = get_loader(args)
eval_loader = test_loader

# for exp_id1 in range(5):
#     file_name1 = get_filename(args, exp_id1)
#     model = load_checkpoint(args, file_name1)   
#     fisher.append(emd(model, eval_loader))

# f = open(args.result_location, "wb")
# pickle.dump(fisher, f)
# f.close()



# parser = get_parser(code_type='model_dist')
# args = parser.parse_args()

model_distance = {}

for exp_id1 in range(5):
    
    model_distance[exp_id1] = {}
    
    for exp_id2 in range(5):
        
        file_name1, file_name2 = return_file_name(args, exp_id1, exp_id2)
        
        model1 = load_checkpoint(args, file_name1)
        model2 = load_checkpoint(args, file_name2)
        
        model_distance[exp_id1][exp_id2] = {'emd': emd(model1, model2, eval_loader)}
        
        temp_results = {'model_emd': model_distance}
        
        f = open(args.result_location, "wb")
        pickle.dump(temp_results, f)
        f.close()