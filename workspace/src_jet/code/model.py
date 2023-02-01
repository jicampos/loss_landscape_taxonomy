import torch
from models.jt_q_three_layer import get_quantized_model 


def get_new_model(args):
    
    model = get_quantized_model(args).cuda()
    
    return model


def load_checkpoint(args, file_name):
    
    checkpoint = torch.load(file_name)
    model = get_new_model(args)
    model.load_state_dict(checkpoint)
    
    return model


