from argparse import ArgumentParser

from autoencoder_datamodule import AutoEncoderDataModule


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--size", type=str, default="baseline")
    parser.add_argument("--weight_precision", type=int, default=8)
    parser.add_argument("--bias_precision", type=int, default=8)
    parser.add_argument("--act_precision", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.0015625)
    parser.add_argument("--top_models", type=int, default=3)
    parser.add_argument("--experiment", type=int, default=0)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "tpu", "auto"], default="auto"
    )
    # Add dataset-specific args
    parser = AutoEncoderDataModule.add_argparse_args(parser)
    # NOTE: do not activate during real training, just for debugging
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    
    args = parser.parse_args()
    
    print(' '.join(f'{k} = {v}\n' for k, v in vars(args).items()))
    main(args)