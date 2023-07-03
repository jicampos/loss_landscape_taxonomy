import argparse

#######################################
##  0. Define experiment parameters  ##
#######################################
EXPERIMENT_TYPE = ''
MODEL = 'ECON'
TEMPERATURE = 'BATCHSIZE'
BITWDITHS = [6, 7, 8, 9, 10, 11]
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
DATAPATH = '../../data/ECON/Elegun'
TRAINING_TYPE = 'normal'
LEARNING_RATE = 0.1


# Name of script for given metric
file_lut = {
    'hessian': 'measure_hessian',
    'gradient': 'gradient',
    'CKA': 'CKA',
    'neural_eff': 'neural_eff',
    'loss_acc': 'loss_acc',
}

# Name of result file for given metric
metric_file = {
    'train': 'train_econ',
    'hessian': 'hessain',
    'gradient': 'gradient',
    'CKA': 'CKA_mixup_alpha_16',
    'neural_eff': 'neural_eff',
    'loss_acc': 'loss_acc',
}

#####################
def metric_command(batch_size, bitwidth, metric='hessian'):
    if metric == 'train':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --training-type {TRAINING_TYPE} --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} '
    elif metric == 'hessian':
        commmand = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --training-type {TRAINING_TYPE} --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} '
    elif metric == 'neural_eff':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --early-stopping --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ '
    elif metric == 'gradient':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --early-stopping --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ '
    elif metric == 'loss_acc':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH}  --train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ '
    else:
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH}  --train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping '
    return command

def metric_params(batch_size, bitwidth, metric='hessian'):
    if metric == 'train':
        parameters = f'--saving-folder ../checkpoint/different_knobs_subset_10/lr_0.1/normal/ECON_11b/ --file-prefix exp_0 --data-subset --subset 1.0 --data_dir ../../data/ECON/Elegun  --lr {LEARNING_RATE} --weight-decay 0.0005 --train-bs 1024 --test-bs 1024 --weight-precision 11 --bias-precision 11 --act-precision 14  --no-lr-decay --max_epochs 25 --save-early-stop --min-delta 0.0001 --patience 5 --save-best --train --ignore-incomplete-batch > >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/normal/ECON_11b/log_0.txt) 2> >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/normal/ECON_11b/err_0.txt >&2) '
    elif metric == 'hessian':
        remainder = 2000%batch_size
        parameters = f'--train-or-test test --hessian-batch-size {2000-remainder} --mini-hessian-batch-size {batch_size} --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping '
    elif metric == 'CKA':
        parameters = f'--train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping --mixup-CKA --mixup-alpha 16.0 --not-input '
    else:
        parameters = ''
    return parameters


#######################################
##  1. Create Bash script            ##
#######################################
def write_experiemnt():

    try:
        f = open(f'{EXPERIMENT_TYPE}.sh', "w")
    except Exception:
        raise Exception('Could not create file!')

    for batch_size in BATCH_SIZES:
        f.write(f'## {TEMPERATURE} {batch_size} ##\n')
        f.write(f'echo "Running {TEMPERATURE} {batch_size}"\n')
        for bitwidth in BITWDITHS:
            commmand= metric_command(batch_size, bitwidth, EXPERIMENT_TYPE)
            parameters = metric_params(batch_size, bitwidth, EXPERIMENT_TYPE)
            result_location = f'--result-location ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics/{metric_file[EXPERIMENT_TYPE]}.pkl 1>../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics//{metric_file[EXPERIMENT_TYPE]}.log 2>../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics//{metric_file[EXPERIMENT_TYPE]}.err\n'
            final_command = commmand+parameters+result_location
            f.write(final_command)
        f.write('\n')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=['train', 'hessian', 'gradient', 'CKA', 'neural_eff', 'loss_acc'])
    args = parser.parse_args()
    
    EXPERIMENT_TYPE = args.metric
    MODEL = args.model

    write_experiemnt()
