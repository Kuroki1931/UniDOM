#!/bin/bash

if [ $(( ${1} % 2)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 2)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=2
fi

VERSION=$(( ${1}+${4}+2000 ))
NAME="${3}"
echo $NAME


set -euC

python3 -m plb.algorithms.solve \
    --algo ${2} \
    --env_name ${3} \
    --seed ${VERSION}