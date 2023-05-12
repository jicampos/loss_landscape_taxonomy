import os
import argparse

###################################################################
# Would you like to customize the range of load and temperature?
###################################################################
"""
    Here, selecting 'False' means choosing the default settings for load parameters and temperature parameters.
    You can also customize the load and temperature by inputing a list of load and temperature values
"""
customize_range_command = False
#customize_range_command = "--load-range customize --load 02 --temperature-range customize --temperature 128"

#Quantization Bitwidths
widths =["32", "6", "7",  "8",  "9",  "10", "11"]

def get_slurm_script(args, info):
    
    return f"""#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n {args.num_gpus} # number of tasks (i.e. processes)
#SBATCH --cpus-per-task={args.cpus_per_task} # number of cores per task
#SBATCH --gres=gpu:{args.num_gpus}
#SBATCH --nodelist={args.nodes} # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t {args.request_days}-00:00 # time requested (D-HH:MM)
#SBATCH -D {args.running_folder}
#SBATCH -o slurm_logs/{args.log_folder}/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/{args.log_folder}/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate pytorch_p36
export PYTHONUNBUFFERED=1
{info}

wait
date
"""


def customize_range(RANGE_COMMANDS):

    if customize_range_command:
        RANGE_COMMANDS = customize_range_command
    
    return RANGE_COMMANDS


def different_metrics(COMMON_COMMANDS, FILE_NAME_SUFFIX, GENERATE_SLURM_SCRIPTS,
                          TRAINING_PARAMETERS, CURVE_PARAMETERS, CKA_PARAMETERS, LOSS_ACC_PARAMETERS,
                          DIST_PARAMETERS, HESSIAN_PARAMETERS, FILE_FOLDER='.'):
    
    train_or_test_commands = f" --train-or-test {args.train_or_test}"

    # Training on the RISE machines
    if "train" in args.metrics:
        code = f"{COMMON_COMMANDS} --file-name ./bash_scripts/{FILE_FOLDER}/train_{FILE_NAME_SUFFIX}.sh {TRAINING_PARAMETERS} --check-result no_check --create-folder{GENERATE_SLURM_SCRIPTS}"
        print(f"executing {code}")
        os.system(code)
    
    # Mode connectivity code
    if "curve" in args.metrics:
        code = f"{COMMON_COMMANDS} --file-name ./bash_scripts/{FILE_FOLDER}/curve_{FILE_NAME_SUFFIX}.sh {CURVE_PARAMETERS}{train_or_test_commands} --check-result single_result --create-folder{GENERATE_SLURM_SCRIPTS}"
        os.system(code)
    
    # CKA code
    if "CKA" in args.metrics:
        code = f"{COMMON_COMMANDS} --file-name ./bash_scripts/{FILE_FOLDER}/CKA_{FILE_NAME_SUFFIX}_mixup.sh {CKA_PARAMETERS}{train_or_test_commands} --check-result single_result --create-folder{GENERATE_SLURM_SCRIPTS}"
        os.system(code)
        
    # loss_acc code
    if "loss_acc" in args.metrics:
        code = f"{COMMON_COMMANDS} --file-name ./bash_scripts/{FILE_FOLDER}/loss_acc_{FILE_NAME_SUFFIX}.sh {LOSS_ACC_PARAMETERS}{train_or_test_commands} --check-result single_result --create-folder{GENERATE_SLURM_SCRIPTS}"
        os.system(code)

    # model dist code
    if "dist" in args.metrics:
        code = f"{COMMON_COMMANDS} --file-name ./bash_scripts/{FILE_FOLDER}/dist_{FILE_NAME_SUFFIX}.sh {DIST_PARAMETERS} --check-result single_result --create-folder{GENERATE_SLURM_SCRIPTS}"
        
        os.system(code)
    
    # hessian code
    if "hessian" in args.metrics:
        code = f"{COMMON_COMMANDS} --file-name ./bash_scripts/{FILE_FOLDER}/hessian_{FILE_NAME_SUFFIX}.sh {HESSIAN_PARAMETERS}{train_or_test_commands} --check-result single_result --create-folder{GENERATE_SLURM_SCRIPTS}"
        os.system(code)
    

def writing_experiments():

    # Specific variables for each experiment
    TRAINING_PARAMETERS="--stop-epoch 100 --training-type lr_decay --lr-standard 0.1 --bs-standard 1024 --ignore-incomplete-batch --exp-start 0 --exp-num 5 --early-stop-checkpoint"
    HESSIAN_PARAMETERS=f"--experiment-type hessian --hessian-batch 1 --early-stop-checkpoint --training-type lr_decay"
    CKA_PARAMETERS="--experiment-type CKA --early-stop-checkpoint --mixup-CKA --mixup-alpha 16 --training-type lr_decay"
    DIST_PARAMETERS="--experiment-type dist --early-stop-checkpoint --training-type lr_decay"
    LOSS_ACC_PARAMETERS="--experiment-type loss_acc --early-stop-checkpoint --training-type lr_decay"

    ##########################################################
    # Training on different widths with the same subset config
    ##########################################################
    
    subsets= ["10"] * len(widths)
    subset_nums= [1.0] * len(widths)
    print(f"Subsets Value: {subsets}")
    print(f"SubsetNums Value: {subset_nums}")
    for i in range(len(widths)):

        if args.generate_slurm_scripts:
            GENERATE_SLURM_SCRIPTS = ' --slurm-commands'
        else:
            GENERATE_SLURM_SCRIPTS = ''

        WIDTH=widths[i]
        ACT_WIDTH= str(int(WIDTH)+3) # Make sure the activations don't overflow, rule of thumb is weight bits + 3
        SUBSET=subsets[i] # Dataset si
        print(f"Subset Value: {SUBSET}")
        
        ARCH=f"ECON_{WIDTH}b"
        EPOCH_NUM=int(10*1.0/subset_nums[i])*5
        #print(f"For width={WIDTH} we use {EPOCH_NUM} epochs")
        
        # Here, we change the CURVE_PARAMETERS to different number of epochs, based on the size of data
        CURVE_PARAMETERS=f"--experiment-type curve --save-frequency=100 --num-points 5 --eval-epoch {EPOCH_NUM} --epochs {EPOCH_NUM} --early-stop-checkpoint --training-type lr_decay"
        
        ###################
        # Special attention
        ###################

        FILE_NAME_SUFFIX=f"{WIDTH}b_lr_decay"
        WIDTH_COMMANDS=f"--weight-precision {WIDTH} --bias-precision {WIDTH} --act-precision {ACT_WIDTH}"
        DATA_COMMANDS="--data-type subset"
        # DATA_COMMANDS="--data-type subset"
        RANGE_COMMANDS=f"--load-range customize --load {SUBSET} --temperature-type lr --temperature-range coarse"
        RANGE_COMMANDS=customize_range(RANGE_COMMANDS)
        COMMON_COMMANDS=f"python write_script.py {WIDTH_COMMANDS} {DATA_COMMANDS} --arch {ARCH} {RANGE_COMMANDS}"
        
        different_metrics(COMMON_COMMANDS, FILE_NAME_SUFFIX,
                                GENERATE_SLURM_SCRIPTS,
                                TRAINING_PARAMETERS, 
                                CURVE_PARAMETERS, 
                                CKA_PARAMETERS, 
                                LOSS_ACC_PARAMETERS, 
                                DIST_PARAMETERS,
                                HESSIAN_PARAMETERS, 
                                FILE_FOLDER=args.script_folder_name)

if __name__ == "__main__":

    print("Starting")    
    parser = argparse.ArgumentParser(description='Jet Classifier Experiment')
    
    parser.add_argument('--script-folder-name', type=str, default = 'width_lr_decay', help='experiment name')
    parser.add_argument('--metrics', type=str, nargs='+', default=["train"], choices=["train", "curve", "CKA", "hessian", "dist", "loss_acc"],
                        help='which metrics to evaluate')
    parser.add_argument('--train-or-test', type=str, default='test', choices=['train', 'test'],
                        help='Would you like to evaluate the metrics on train or test data?')
    parser.add_argument('--generate-slurm-scripts', default=False, action='store_true',
                        help='Would you like to get the examples of slurm submission files?')

    # parameters for slurm commands
    parser.add_argument('--nodes', type=str, default = 'manchester,como', help='Experiment running nodes')
    parser.add_argument('--request-days', type=str, default = '7', help='Requested running days')
    parser.add_argument('--running-folder', type=str, default = '/work/yyaoqing/Good_vs_bad_data/src', help='running folder')
    parser.add_argument('--num-gpus', type=int, default = 8, help='Number of gpus requested')
    parser.add_argument('--cpus-per-task', type=int, default = 8, help='Number of cpus per task')
    
    args = parser.parse_args()

    if args.script_folder_name != 'width_lr_decay':
        if not os.path.exists(f'bash_scripts/{args.script_folder_name}/'):
            os.mkdir(f'bash_scripts/{args.script_folder_name}/')

    writing_experiments()

    for metric in args.metrics:

        bash_file = f'bash_scripts/{args.script_folder_name}/{metric}.sh'
        bash_file_all = f'bash_scripts/{args.script_folder_name}/{metric}_*.sh'
        if args.generate_slurm_scripts:
            if os.path.exists(bash_file):
                with open(bash_file, 'w') as f:
                    f.truncate()
            os.system(f'cat {bash_file_all} > {bash_file}')
            os.system(f'rm -f {bash_file_all}')
            with open(f'{bash_file}', 'r') as file:
                info = file.read()
                args.log_folder = f'bash_scripts/{args.script_folder_name}'
                slurm_commands = get_slurm_script(args, info)

            with open(bash_file, 'w') as file:    
                file.truncate()
                file.write(slurm_commands)
        else:
            os.system(f'cat {bash_file_all} > {bash_file}')
            os.system(f'rm -f {bash_file_all}')

