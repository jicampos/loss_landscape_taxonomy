#! /bin/bash

mkdir ./data/JT

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz -P data/JT
wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz -P data/JT

tar -zxf dataset/hls4ml_LHCjet_100p_train.tar.gz -C ./data/JT
tar -zxf dataset/hls4ml_LHCjet_100p_val.tar.gz -C ./data/JT

script_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
python "${script_path}/jet_dataset.py" --config "${script_path}/config.yml" --noise --noise-type gaussian --noise-magnitude 1

echo "$script_path"

cp "${script_path}/config.yml" data/config.yml
