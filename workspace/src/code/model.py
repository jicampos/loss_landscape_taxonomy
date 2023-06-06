import torch
from models.jet_tagger.jt_q_three_layer import get_quantized_jettagger 
from models.jet_tagger.jt_three_layer import get_jettagger 
from models.mlperf.rn07 import get_rn07
import re



def get_new_model(args):
    model_arch = args.arch.split('_')[0]
    if model_arch == 'RN07':
        return get_rn07(args)
    if args.weight_precision is 32:
        print("using FP32 model")
        model = get_jettagger(args)#.cuda()
    else:
        print("using quant model")
        model = get_quantized_jettagger(args)#.cuda()
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
   # model.load_state_dict(checkpoint)

    result = re.search('JT_(.*)b', args.arch)
    quant = int(result.group(1))
    fake = ArgFake(w=quant,b=quant,a=quant+3)
    model = get_new_model(args)
    return model
    
    
    checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
    
    if quant is 32:
        model.load_state_dict(checkpoint)
        #model.train()
        return model

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
