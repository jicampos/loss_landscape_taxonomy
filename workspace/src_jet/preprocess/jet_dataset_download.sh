#! /bin/bash

mkdir ./dataset

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz -P dataset
wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz -P dataset

tar -zxf dataset/hls4ml_LHCjet_100p_train.tar.gz -C ./dataset
tar -zxf dataset/hls4ml_LHCjet_100p_val.tar.gz -C ./dataset