import torch
from models.jt_q_three_layer import get_quantized_model 
from models.jt_three_layer import get_model 
import re



def get_new_model(args):
    if args.weight_precision is 32:
        print("using FP32 model")
        model = get_model(args).cuda()
    else:
        print("using quant model")
        model = get_quantized_model(args).cuda()
        
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
    fake = ArgFake(w=quant,b=quant,a=quant)
    
    
    checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
    
    if quant is 32:
        model.load_state_dict(checkpoint["state_dict"])
        #model.train()
        return

    #set_bit_config(model, checkpoint["bit_config"], args)
    model = get_new_model(fake)
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


        

