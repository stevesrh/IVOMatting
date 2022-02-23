#!/usr/bin/env bash
echo Which PYTHON: `which python`
CUDA_VISIBLE_DEVICES=0 python infer.py \
--config=/home/lab505/PycharmProjects/IVOMatting/code-base/config/MGMatting-DIM-300k.toml \
--checkpoint=/home/lab505/PycharmProjects/IVOMatting/code-base/checkpoints/MGMatting-Dconv-DIM-300k/latest_model.pth \
--image-dir=/home/lab505/PycharmProjects/IVOMatting/code-base/Combined_Dataset/Test_set/Adobe-licensed\ images/merged \
--mask-dir=/home/lab505/PycharmProjects/IVOMatting/code-base/Combined_Dataset/Test_set/Adobe-licensed\ images/trimaps \
--output=/home/lab505/PycharmProjects/IVOMatting/code-base/output/MGMatting-DIM-250k \
--guidance-thres=170
