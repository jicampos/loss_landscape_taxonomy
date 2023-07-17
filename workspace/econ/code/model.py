import re
import torch
from q_autoencoder import AutoEncoder


input_shapes = {
    "JT": (1,16),
    "RN07": (1,3,32,32),
    "ECON": (1, 1, 8, 8),
}

def get_new_model(model_arch, bitwidth, args):
    if model_arch == 'ECON' and bitwidth == 32:
        model = AutoEncoder(
            accelerator='auto', 
            quantize=False,
            precision=[
                bitwidth, 
                bitwidth, 
                bitwidth+3
            ],
            learning_rate=0.1,
            econ_type=args.experiment_name,
        )
    elif model_arch == 'ECON' and bitwidth < 32:
        model = AutoEncoder(
            accelerator='auto', 
            quantize=True,
            precision=[
                bitwidth, 
                bitwidth, 
                bitwidth+3
            ],
            learning_rate=0.1,
            econ_type=args.experiment_name,
        )
    return model


def load_checkpoint(args, checkpoint_filename): 
   # Find model architecture and quantization scheme  
   model_arch = args.arch.split('_')[0]
   bitwidth_str = re.search(f'{model_arch}_(.*)b', args.arch)
   bitwidth = int(bitwidth_str.group(1))

   model = get_new_model(model_arch, bitwidth, args)
   model(torch.randn(input_shapes[model_arch]))  # Update tensor shapes 
   
   # Load checkpoint 
   print('Loading checkpoint...', checkpoint_filename)
   checkpoint = torch.load(checkpoint_filename)
   model.load_state_dict(checkpoint['state_dict'])  # strict=False
   return model
