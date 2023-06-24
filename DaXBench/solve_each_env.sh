#!/bin/bash

if [ $(( ${1} % 2)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(( ${1} % 2)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

echo "${2}  seed${1}"

set -euC

python3 -m daxbench.algorithms.apg.apg \
    --env ${2} \
    --ep_len 3 \
    --num_envs 4 \
    --lr 1e-4 \
    --gpus 1 \
    --max_grad_norm 0.3 \
    --seed ${1} \
    --eval_freq 20
