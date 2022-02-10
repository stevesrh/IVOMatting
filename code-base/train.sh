#!/usr/bin/env bash
echo Which PYTHON: `which python`
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python main.py \
--config=config/MGMatting-Dconv-DIM-100k.toml