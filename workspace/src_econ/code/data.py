from torchvision import datasets, transforms
import torch
from jet_dataset import JetTaggingDataset

# Hard code here, though normally grabbed from a yaml file
features = ['j_zlogz', 'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_mass_mmdt', 'j_multiplicity']

def get_loader(args):
    kwargs = {'num_workers': 10, 'pin_memory': True}

    # No support for subset or randomized labels at this point...

    # Data is normalized via sklearn.standardscaler per dataset right now, which is how the HAWQ models + QAP Brevitas models are trained
    # but it likely makes more sense to fit to the train data and use that mean/var to normalize train and test sets?
    #TODO - Investigate normalizing per dataset or on train set mean/var 
    
    print("Loading Datasets")
    
    trainset = JetTaggingDataset(path="../../../datasets/train", features=features, preprocess="standardize")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, **kwargs)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, **kwargs)

    testset = JetTaggingDataset(path="../../../datasets/val", features=features, preprocess="standardize")
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **kwargs)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, **kwargs)
          
    print("\nDataset loading complete!")

    return train_loader, test_loader
    

def get_dataset(args):
    kwargs = {'num_workers': 10, 'pin_memory': True}

    # No support for subset or randomized labels at this point...

    # Data is normalized via sklearn.standardscaler per dataset right now, which is how the HAWQ models + QAP Brevitas models are trained
    # but it likely makes more sense to fit to the train data and use that mean/var to normalize train and test sets?
    #TODO - Investigate normalizing per dataset or on train set mean/var 
    
    print("Loading Datasets")
    
    trainset = JetTaggingDataset(path="../../../datasets/train", features=features, preprocess="standardize")

    testset = JetTaggingDataset(path="../../../datasets/val", features=features, preprocess="standardize")
          
    print("\nDataset loading complete!")

    return trainset, testset
