#!/bin/bash

# TODO: load the data of the Econ model if it is missing

# Constants
ADD_PRECISION=3
SAVING_FOLDER="../../checkpoint/different_knobs_subset_10"
DATA_DIR="../../../data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=4
max_epochs=25
process_data=false
no_train=false
size="baseline"
top_models=3
num_test=3
scan="all"
accelerator="auto"

# ranges of the scan 
batch_sizes=(16 32 64 128 256 512 1024)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
precisions=(2 3 4 5 6 7 8 9 10 11)

# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--process_data       Flag which specify if the data need to be processed first"
    echo "--max_epochs         Max number of epochs"
    echo "--size               Model size [baseline, small, large]"
    echo "--top_models         Number of top models to store"
    echo "--num_test           Number of time we repeat the computation"
    echo "--scan               Hyperparameter to scans [bs, lr, all]"
    echo "--accelerator        Accelerator to use during training [auto, cpu, gpu, tpu]"
    echo "--no_train            Flag which specify if the model need to be train"
}

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
    echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
    while [ $# -gt 0 ]; do
        case $1 in
            -h | --help)
                usage
                exit 0
                ;;
            --process_data)
                process_data=true
                echo "Data must be processed and stored in $DATA_FILE"
                ;;
            --no_train)
                no_train=true
                echo "The model will not be trained"
                ;;
            --max_epochs)
                if has_argument $@; then
                    max_epochs=$(extract_argument $@)
                    echo "Max number of epochs: $max_epochs"
                    shift
                fi
                ;;
            --num_workers)
                if has_argument $@; then
                    num_workers=$(extract_argument $@)
                    echo "Number of workers: $num_workers"
                    shift
                fi
                ;;
            --accelerator)
                if has_argument $@; then
                    accelerator=$(extract_argument $@)
                    echo "Training accelerator: $accelerator"
                    shift
                fi
                ;;
            --size)
                if has_argument $@; then
                    size=$(extract_argument $@)
                    echo "Size of the model: $size"
                    shift
                fi
                ;;
            --top_models)
                if has_argument $@; then
                    top_models=$(extract_argument $@)
                    echo "Model to be stores: $top_models"
                    shift
                fi
                ;;
            --num_test)
                if has_argument $@; then
                    num_test=$(extract_argument $@)
                    echo "Number of test per model: $num_test"
                    shift
                fi
                ;;
            --scan)
                if has_argument $@; then
                    scan=$(extract_argument $@)
                    echo "Scan: $scan"
                    shift
                fi
                ;;
            *)
                echo "Invalid option: $1" >&2
                usage
                exit 1
                ;;
        esac
        shift
    done
}

run_train() {
    saving_folder="$SAVING_FOLDER/$scanning"_$iter/ECON_"$precision"b/
    for i in $(eval echo "{1..$num_test}")
    do
        # create directories to retrieve informations of the training process
        timestamp=$(date +%s)\
        log_folder="$saving_folder"/log
        log_file=$log_folder"/log_"$size"_"$i"_"$timestamp".txt"
        error_folder="$saving_folder"/error
        error_file=$error_folder"/err_"$size"_"$i"_"$timestamp".txt"
        mkdir -p $log_folder
        touch $log_file
        mkdir -p $error_folder
        touch $error_file

        echo ""
        echo " BATCH SIZE $bs - LEARNING_RATE $lr - PRECISION $precision - test $i "
        echo ""
        python code/train.py \
            --saving_folder "$saving_folder" \
            --data_dir "$DATA_DIR" \
            --data_file "$DATA_FILE" \
            --batch_size $bs \
            --num_workers $num_workers \
            --accelerator $accelerator \
            `if [ $process_data == true ]; then echo "--process_data"; fi` \
            `if [ $no_train == true ]; then echo "--no_train"; fi` \
            --weight_precision $precision \
            --bias_precision $precision \
            --act_precision $(($precision + $ADD_PRECISION))   \
            --lr $lr \
            --size $size \
            --top_models $top_models \
            --experiment $i \
            --max_epochs $max_epochs \
            > >(tee -a $log_file) \
            2> >(tee -a $error_file >&2)

            echo ""
            echo "-----------------------------------------------------------"

    done
}

scan_batch_sizes() {
    lr=0.0015625    # fix the learning rate
    for bs in ${batch_sizes[*]}
    do
        iter=$bs
        run_train
    done
}

scan_learning_rates() {
    bs=500    # fix the batch size
    for lr in ${learning_rates[*]}
    do
        iter=$lr
        run_train
    done
}

# Main script execution
handle_options "$@"
# creating the directories if required
mkdir -p $DATA_DIR

#iterate over the precision
for precision in ${precisions[*]}
do
    # trainig with various batch sizes
    if [[ $scan == "bs" ]] || [[ $scan == "all" ]]; then
        scanning="bs"
        scan_batch_sizes
    fi

    # training with various learning rates
    if [[ $scan == "lr" ]] || [[ $scan == "all" ]]; then
        scanning="lr"
        scan_learning_rates
    fi
done




