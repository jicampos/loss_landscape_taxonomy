import re
import torch
from q_autoencoder import AutoEncoder

def get_new_model(bitwidth, args):
    model = AutoEncoder(
        accelerator='auto', 
        quantize=(bitwidth < 32),
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

   model = get_new_model(bitwidth, args)
   model(torch.randn((1, 1, 8, 8)))  # Update tensor shapes 
   
   # Load checkpoint 
   print('Loading checkpoint...', checkpoint_filename)
   checkpoint = torch.load(checkpoint_filename)
   model.load_state_dict(checkpoint['state_dict'])  # strict=False
   return model
