from __future__ import print_function
import os 
import re 
import torch
import torchinfo
####################################################
# pytorch lightning 
####################################################
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
####################################################
# custom 
####################################################
from arguments import get_parser
from data import get_loader
from model import AD08

parser = get_parser(code_type='training')


def main(args):
    if os.path.exists(args.saving_folder) == False:
        os.mkdir(args.saving_folder)
    
    if args.save_final or args.no_lr_decay or args.one_lr_decay:
        if args.saving_folder == '':
            raise ('you must give a position and name to save your model')
        if args.saving_folder[-1] != '/':
            args.saving_folder += '/'

    ####################################################
    # Log Run
    ####################################################
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print("------------------------------------------------------")
    print("Experiement: {0} training for {1}".format(args.training_type, args.arch))
    print("------------------------------------------------------")

    tb_logger = pl_loggers.TensorBoardLogger(args.saving_folder)

    ####################################################
    # Load Data
    ####################################################
    train_loader, test_loader = get_loader(args)

    ####################################################
    # Setup Callbacks 
    ####################################################
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta, patience=args.patience, verbose=True, mode="min")

    top3_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=args.saving_folder,
        filename=f'net_{args.file_prefix}_best',
        auto_insert_metric_name=False,
    )
    top3_checkpoint_callback.FILE_EXTENSION = '.pkl'
    print(f'Saving to dir: {os.path.join(args.saving_folder)}')
    print(f'Running experiment: {args.file_prefix}')

    ####################################################
    # Load Model
    ####################################################
    model_arch = args.arch.split('_')[0]
    result = re.search(f'{model_arch}_(.*)b', args.arch)
    bitwidth = int(result.group(1))
    model = AD08(precision=[
        bitwidth, 
        bitwidth, 
        bitwidth+3,
        ],
        lr=args.lr
    )
    
    if args.resume:
        model.load_state_dict(torch.load(f"{args.resume}"))

    torchinfo.summary(model, (1, 64))

    ####################################################
    # Train & Evaluate
    ####################################################
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[top3_checkpoint_callback, early_stop_callback],
        )
    
    if args.train:
        trainer.fit(model, train_loader, test_loader)

    if args.train or args.evaluate:
        trainer.test(model, dataloaders=test_loader)

if __name__ == '__main__':
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--evaluation', default=False, action='store_true')
    args = parser.parse_args()
    main(args)

# python code/train.py --training-type normal --arch AD_11b --saving-folder ../chec
# kpoint/different_knobs_subset_10/lr_0.1/normal/AD_6b/ --file-prefix exp_0 --data_dir ../../data/AD/  --lr 0.001 --weight-decay 0 --train-bs 512 --test-bs 512 --weight-prec
# ision 6 --bias-precision 6 --act-precision 9  --no-lr-decay --epochs 100 --save-early-stop --min-delta 0.0001 --patience 15 --save-best --ignore-incomplete-batch
