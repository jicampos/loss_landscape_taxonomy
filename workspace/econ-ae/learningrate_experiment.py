import argparse

#######################################
##  0. Define experiment parameters  ##
#######################################
EXPERIMENT_TYPE = ''
MODEL = 'ECON'
TEMPERATURE = 'LEARNING RATE'
SUBFOLDER = 'lr_'
BITWDITHS = [6, 7, 8, 9, 10, 11]
TEMPERATURE_SIZES = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]
DATAPATH = '../../data/ECON/Elegun'
TRAINING_TYPE = 'normal'
BATCH_SIZE = 1024
NUM_TRAILS = 3



# Name of script for given metric
file_lut = {
    'train': 'train_econ',
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
def metric_command(learning_rate, bitwidth, metric='hessian'):
    if metric == 'train':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --training-type {TRAINING_TYPE} --arch {MODEL}_{bitwidth}b --lr {learning_rate} --data-subset --subset 1.0 --data_dir {DATAPATH} --train-bs {BATCH_SIZE} --test-bs {BATCH_SIZE} '
    elif metric == 'hessian':
        commmand = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --training-type {TRAINING_TYPE} --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data_dir {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} '
    elif metric == 'neural_eff':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --early-stopping --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} --checkpoint-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{learning_rate}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ '
    elif metric == 'gradient':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --early-stopping --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} --checkpoint-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ '
    elif metric == 'loss_acc':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH}  --train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ '
    else:
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH}  --train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping '
    return command

def metric_params(learning_rate, bitwidth, metric='hessian'):
    if metric == 'train':
        parameters = f'--saving-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{learning_rate}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --file-prefix exp_0 --lr {learning_rate} --weight-decay 0.0005 --train-bs 1024 --test-bs 1024 --weight-precision 11 --bias-precision 11 --act-precision 14  --no-lr-decay --max_epochs 25 --save-early-stop --min-delta 0.0001 --patience 5 --save-best --train --ignore-incomplete-batch > >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/normal/ECON_11b/log_0.txt) 2> >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/normal/ECON_11b/err_0.txt >&2) '
    elif metric == 'hessian':
        parameters = f'--train-or-test test --hessian-batch-size {BATCH_SIZE} --mini-hessian-batch-size {BATCH_SIZE} --checkpoint-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{learning_rate}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping '
    elif metric == 'CKA':
        parameters = f'--train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping --mixup-CKA --mixup-alpha 16.0 --not-input '
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

    for learning_rate in TEMPERATURE_SIZES:
        f.write(f'## {TEMPERATURE} {learning_rate} ##\n')
        f.write(f'echo "Running {TEMPERATURE} {learning_rate}"\n')
        for bitwidth in BITWDITHS:
            commmand= metric_command(learning_rate, bitwidth, EXPERIMENT_TYPE)
            parameters = metric_params(learning_rate, bitwidth, EXPERIMENT_TYPE)
            final_command = commmand+parameters
            if EXPERIMENT_TYPE != 'train':
                result_location = f'--result-location ../checkpoint/different_knobs_subset_10/{SUBFOLDER}{learning_rate}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics/{metric_file[EXPERIMENT_TYPE]}.pkl 1>../checkpoint/different_knobs_subset_10/{SUBFOLDER}{learning_rate}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics//{metric_file[EXPERIMENT_TYPE]}.log 2>../checkpoint/different_knobs_subset_10/{SUBFOLDER}{learning_rate}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics//{metric_file[EXPERIMENT_TYPE]}.err\n'
                final_command += result_location
            f.write(final_command)
        f.write('\n')
 
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=['train', 'hessian', 'gradient', 'CKA', 'neural_eff', 'loss_acc'])
    args = parser.parse_args()
    
    EXPERIMENT_TYPE = args.metric

    write_experiemnt()
