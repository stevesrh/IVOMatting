#!/usr/bin/env bash
echo Which PYTHON: `which python`
CUDA_VISIBLE_DEVICES=0 python video2matting.py \
--config=/home/lab505/srh/MGMatting/code-base/config/MGMatting-DIM-Fortrain.toml \
--checkpoint=/home/lab505/srh/MGMatting/code-base/pretrain/latest_model_unzip.pth \
--data_dir=/media/lab505/Toshiba/MiVOS-main/Selected_data/0001\
--guidance-thres=170
