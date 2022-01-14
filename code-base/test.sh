#!/usr/bin/env bash
echo Which PYTHON: `which python`
CUDA_VISIBLE_DEVICES=0 python infer.py \
--config=/home/lab505/srh/MGMatting/code-base/config/MGMatting-DIM-100k.toml \
--checkpoint=/home/lab505/srh/MGMatting/code-base/pretrain/latest_model_unzip.pth \
--image-dir=/home/lab505/srh/MGMatting/code-base/test/image \
--mask-dir=/home/lab505/srh/MGMatting/code-base/test/mask \
--output=/home/lab505/srh/MGMatting/code-base/test/output \
--guidance-thres=170
