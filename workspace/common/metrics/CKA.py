from __future__ import print_function

import os 
import sys
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

from arguments import get_parser
from utils import *
from workspace.common.metrics.utils.CKA_utils import *

# get the parsed arguments from 'arguments.py'
parser = get_parser(code_type='CKA')
args = parser.parse_args()

# get the model architecture picked from the user
model_arch = args.arch.split('_')[0]
print('Importing code for', model_arch)
# set the path to import the proper data loader and model
if model_arch == 'JT':
    sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
elif model_arch == 'ECON':
    sys.path.append(os.path.join(sys.path[0], "../econ/code")) 

# Import dataloader & model 
from data import get_loader
from model import load_checkpoint


# Get data
train_loader, test_loader = get_loader(args)

# init the structure to store the performances
representation_similarity = {}
classification_similarity = {}
cos = nn.CosineSimilarity(dim=0)

# we do it 3 times
for exp_id1 in range(3):
    
    representation_similarity[exp_id1] = {}
    classification_similarity[exp_id1] = {}
    
    for exp_id2 in range(3):
        # get the file name of two checkpoints 
        file_name1, file_name2 = return_file_name(args, exp_id1, exp_id2)
        
        # load the models
        model1 = load_checkpoint(args, file_name1)
        model2 = load_checkpoint(args, file_name2)
        
        # pick the dataset on which evaluate the CKA metric
        if args.train_or_test == "train":
            eval_loader = train_loader
        elif args.train_or_test == "test":
            eval_loader = test_loader
        else:
            raise ValueError('Invalid input.')
        
        if not args.compare_classification:
            # do not compare the classification
            
            cka_from_features_average = []
            
            # as many times as designed for CKA
            for CKA_repeat_runs in range(args.CKA_repeat_runs):

                cka_from_features = []

                latent_all_1, latent_all_2 = all_latent(model1, model2, eval_loader, num_batches=args.CKA_batches, args=args)

                for name in latent_all_1.keys():

                    print(name)

                    if args.flattenHW:
                        cka_from_features.append(feature_space_linear_cka(latent_all_1[name], latent_all_2[name]))
                    else:
                        cka_from_features.append(cka_compute(gram_linear(latent_all_1[name]), gram_linear(latent_all_2[name])))
                        
                cka_from_features_average.append(cka_from_features)
                
            # compute the average of n CKA computations
            cka_from_features_average = np.mean(np.array(cka_from_features_average), axis=0)
            
            print('cka_from_features shape')
            print(cka_from_features_average.shape)

            representation_similarity[exp_id1][exp_id2] = cka_from_features_average
        
        else:
            # compare the classification
            classification_similarity[exp_id1][exp_id2] = compare_classification(model1, model2, eval_loader, args=args, cos=cos)
        
        temp_results = {'representation_similarity': representation_similarity, 'classification_similarity': classification_similarity}
        
        # save the results on file
        f = open(args.result_location, "wb")
        pickle.dump(temp_results, f)
        f.close()
        
        
        