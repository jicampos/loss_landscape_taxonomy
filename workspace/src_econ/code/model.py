import torch
from models.jt_q_three_layer import get_quantized_model 
from workspace.src_econ.code.models.autoencoder import get_model 
import re



def get_new_model(args):
    if args.weight_precision == 32:
        print("using FP32 model")
        model = get_model(args)
    else:
        print("using quant model")
        model = get_quantized_model(args)
        
    return model


class ArgFake:
    def __init__(self, w=32, b=32, a=32, bn=False, d=False):
        self.weight_precision = w
        self.bias_precision = b
        self.act_precision = a
        self.batch_norm = bn
        self.dropout = d

def update_fc_size(model, ckp, dense_layers=['dense_1', 'dense_2', 'dense_3', 'dense_4']):
    for dl in dense_layers:
        layer = getattr(model, dl)
        ckp[f"{dl}.fc_scaling_factor"] = torch.ones(layer.fc_scaling_factor.shape) * ckp[f"{dl}.fc_scaling_factor"]
    return ckp

def load_checkpoint(args, file_name):
    
   # checkpoint = torch.load(file_name)
   # model = get_new_model(args)
   # model.load_state_dict(checkpoint)
    
    result = re.search('JT_(.*)b', args.arch)
    quant = int(result.group(1))
    fake = ArgFake(w=quant,b=quant,a=quant+3)
    model = get_new_model(fake)
    
    
    checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
    
    if quant == 32:
        model.load_state_dict(checkpoint)
        #model.train()
        return model

    #set_bit_config(model, checkpoint["bit_config"], args)
    #checkpoint = checkpoint["state_dict"]
    model_dict = model.state_dict()
    modified_dict = {}
    
    updated_ckp = update_fc_size(model, checkpoint)   
    
    for key, value in checkpoint.items():
        if model_dict[key].shape != value.shape:
            print(f"mismatch: {key}: {value}")
            value = torch.tensor([value], dtype=torch.float64)
        modified_dict[key] = value
    
    model.load_state_dict(modified_dict, strict=False)
    
    return model



def load_state(model, checkpoint):
    if model.__class__.__name__ == 'ThreeLayerMLP':
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint["state_dict"])
        return

    model_dict = model.state_dict()
    modified_dict = {}
    
    updated_ckp = update_fc_size(model, checkpoint)   
    
    for key, value in checkpoint.items():
        if model_dict[key].shape != value.shape:
            print(f"mismatch: {key}: {value}")
            value = torch.tensor([value], dtype=torch.float64)
        modified_dict[key] = value
    
    model.load_state_dict(modified_dict, strict=False)
    
    return model


        

