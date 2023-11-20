from argparse import ArgumentParser

from autoencoder_datamodule import AutoEncoderDataModule

def main(args):
    # load the model
    
    #load the data loader
    
    # select the metric
    
    # compute the metric 
    
    # save the result
    
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--name_file", type=str, default=None)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--size", type=str, default="baseline")
    parser.add_argument("--bias_precision", type=int, default=8)
    parser.add_argument("--act_precision", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.0015625)
    parser.add_argument("--top_models", type=int, default=3)
    parser.add_argument("--experiment", type=int, default=0)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "tpu", "auto"], default="auto"
    )
    
    args = parser.parse_args()
    
    print(' '.join(f'{k} = {v}\n' for k, v in vars(args).items()))
    main(args)