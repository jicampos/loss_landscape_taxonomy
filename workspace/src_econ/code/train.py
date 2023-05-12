import os
import time
from tqdm import tqdm
import torch
# import qtorch
import torchinfo
import numpy as np
import multiprocessing
import pytorch_lightning as pl 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from models.q_autoencoder import AutoEncoder
from models.autoencoder_datamodule import AutoEncoderDataModule
from models.utils_pt import unnormalize, emd


def test_model(model, test_loader):
    """
    Our own testing loop instead of using the trainer.test() method so that we
    can multithread EMD computation on the CPU
    """
    model.eval()
    input_calQ_list = []
    output_calQ_list = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(model.device)
            output = model(x)
            input_calQ = model.map_to_calq(x)
            output_calQ_fr = model.map_to_calq(output)
            input_calQ = torch.stack(
                [input_calQ[i] * model.val_sum[i] for i in range(len(input_calQ))]
            )  # shape = (batch_size, 48)
            output_calQ = unnormalize(
                torch.clone(output_calQ_fr), model.val_sum
            )  # ae_out
            input_calQ_list.append(input_calQ)
            output_calQ_list.append(output_calQ)
    input_calQ = np.concatenate([i_calQ.cpu() for i_calQ in input_calQ_list], axis=0)
    output_calQ = np.concatenate([o_calQ.cpu() for o_calQ in output_calQ_list], axis=0)
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        emd_list = pool.starmap(emd, zip(input_calQ, output_calQ))
    print(f"EMD computation time: {time.time() - start_time} seconds")
    average_emd = np.mean(np.array(emd_list))
    print(f"Average EMD: {average_emd}")
    return average_emd


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # ------------------------
    # 0 PREPARE DATA
    # ------------------------
    data_module = AutoEncoderDataModule.from_argparse_args(args)
    if args.process_data:
        print("Processing data...")
        data_module.process_data()
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    args.quantize = True if args.weight_precision < 32 else False
    print(f'Loading with quantize: {args.quantize}')
    model = AutoEncoder(
        accelerator=args.accelerator, 
        quantize=args.quantize,
        precision=[
            args.weight_precision, 
            args.bias_precision, 
            args.act_precision
        ],
        learning_rate=args.lr
    )

    torchinfo.summary(model, input_size=(1, 1, 8, 8))  # (B, C, H, W)

    tb_logger = pl_loggers.TensorBoardLogger(args.save_dir, name=args.experiment_name)

    # Save top-3 checkpoints based on Val/Loss
    top3_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(args.save_dir, args.experiment_name),
        filename=f"{args.experiment_name}_{args.file_prefix}" "_epoch={epoch:02d}_loss={val_loss:.3f}",
        auto_insert_metric_name=False,
    )
    print(f'Saving to dir: {os.path.join(args.save_dir, args.experiment_name)}')
    print(f'Running experiment: {args.file_prefix}')

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=tb_logger,
        callbacks=[top3_checkpoint_callback],
        fast_dev_run=args.fast_dev_run,
    )

    # ------------------------
    # 3 TRAIN MODEL
    # ------------------------
    if args.train:
        trainer.fit(model=model, datamodule=data_module)

    # ------------------------
    # 4 EVALUTE MODEL
    # ------------------------
    if args.train or args.evaluate:
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model = model.load_state_dict(checkpoint['state_dict'])
        # Need val_sum to compute EMD
        _, val_sum = data_module.get_val_max_and_sum()
        model.set_val_sum(val_sum)
        data_module.setup("test")
        test_results = test_model(model, data_module.test_dataloader())
        test_results_log = os.path.join(
            args.save_dir, args.experiment_name, args.experiment_name + "_emd.txt"
        )
        with open(test_results_log, "w") as f:
            f.write(str(test_results))
            f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./pt_autoencoder_test")
    parser.add_argument("--experiment_name", type=str, default="autoencoder")
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "auto"], default="auto"
    )
    parser.add_argument("--checkpoint", type=str, default="", help="model checkpoint")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument(
        "--quantize", 
        action="store_true", 
        default=False, 
        help="quantize model to 6-bit fixed point (1 signed bit, 1 integer bit, 4 fractional bits)"
    )

    # Add dataset-specific args
    parser = AutoEncoderDataModule.add_argparse_args(parser)

    # Add loss landscape args 
    parser.add_argument("--training-type", type=str)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--subset", type=float)
    parser.add_argument("--file-prefix", type=str)
    parser.add_argument("--mixup-alpha", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--train-bs", type=int)
    parser.add_argument("--test-bs", type=int)
    parser.add_argument("--weight-precision", type=int)
    parser.add_argument("--bias-precision", type=int)
    parser.add_argument("--act-precision", type=int)
    parser.add_argument("--one-lr-decay", action="store_true")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--ignore-incomplete-batch", action="store_true")

    args = parser.parse_args()
    main(args)
