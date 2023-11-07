import argparse

#####################
# Define the experiment parameters
EXPERIMENT_TYPE = ''
MODEL = ''
TEMPERATURE = 'BATCH SIZE'
BITWDITHS = [6, 7, 8, 9, 10, 11]
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
NOISE = True 
NOISE_TYPE = 'gaussian'  # gaussian, uniform
NOISE_MAGNITUDE = '1.0'  # 0.05, 0.1, 1.0
DATAPATH = '../../data/JT' + '/' + NOISE_TYPE
TRAINING_TYPE = 'normal'


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
    'hessian': 'hessain',
    'gradient': 'gradient',
    'CKA': 'CKA_mixup_alpha_16',
    'neural_eff': 'neural_eff',
    'loss_acc': 'loss_acc',
}

#####################
def metric_command(batch_size, bitwidth, metric='hessian'):
    if metric == 'hessian':
        command = f'python ./code/{file_lut[EXPERIMENT_TYPE]}.py --training-type {TRAINING_TYPE} --arch {MODEL}_{bitwidth}b --data-subset --subset 1.0 --data-path {DATAPATH} --train-bs {batch_size} --test-bs {batch_size} '
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
    if metric == 'hessian':
        remainder = 2000%batch_size
        parameters = f'--train-or-test test --hessian-batch-size {2000-remainder} --mini-hessian-batch-size {batch_size} --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping '
    elif metric == 'CKA':
        parameters = f'--train-or-test test --checkpoint-folder ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/ --early-stopping --mixup-CKA --mixup-alpha 16.0 --not-input '
    elif metric == 'loss_acc':
        if NOISE:
            parameters = f'--noise --noise-type {NOISE_TYPE} --noise-magnitude {NOISE_MAGNITUDE} '
        else:
            parameters = ''
    else:
        parameters = ''
    return parameters


#####################
def write_experiment():
    if NOISE:
        file_suffix = f'_{NOISE_TYPE}{NOISE_MAGNITUDE}_np'
    else: 
        file_suffix = ''

    try:
        f = open(f'{EXPERIMENT_TYPE}{file_suffix}.sh', "w")
    except Exception:
        raise Exception('Could not create file!')

    for batch_size in BATCH_SIZES:
        f.write(f'## {TEMPERATURE} {batch_size} ##\n')
        f.write(f'echo "Running {TEMPERATURE} {batch_size}"\n')
        for bitwidth in BITWDITHS:
            command = metric_command(batch_size, bitwidth, EXPERIMENT_TYPE)
            parameters = metric_params(batch_size, bitwidth, EXPERIMENT_TYPE)
            result_location = f'--result-location ../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics/{metric_file[EXPERIMENT_TYPE]}{file_suffix}.pkl 1>../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics//{metric_file[EXPERIMENT_TYPE]}{file_suffix}.log 2>../checkpoint/different_knobs_subset_10/bs_{batch_size}/{TRAINING_TYPE}/{MODEL}_{bitwidth}b/metrics//{metric_file[EXPERIMENT_TYPE]}{file_suffix}.err\n'
            final_command = command + parameters + result_location
            f.write(final_command)
        f.write('\n')

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=['hessian', 'gradient', 'CKA', 'neural_eff', 'loss_acc'])
    parser.add_argument('--model', choices=['JT', 'ECON', 'RN07'])
    # parser.add_argument('--training', choices=['normal', 'bs-decay']) TODO
    args = parser.parse_args()
    
    EXPERIMENT_TYPE = args.metric
    MODEL = args.model

    write_experiment()
