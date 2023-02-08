import torch
from models.jt_q_three_layer import get_quantized_model 
from models.jt_three_layer import get_model 

def get_new_model(args):
    if args.weight_precision is 32:
        print("using FP32 model")
        model = get_model(args).cuda()
    else:
        print("using quant model")
        model = get_quantized_model(args).cuda()
        
    return model


def load_checkpoint(args, file_name):
    
    checkpoint = torch.load(file_name)
    model = get_new_model(args)
    model.load_state_dict(checkpoint)
    
    return model


