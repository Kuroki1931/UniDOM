#!/bin/bash

if [ $(( ${1} % 3)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 3)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 3)) -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# VERSION=$(( ${1}+${5}+1500 ))
# NAME="${3}${VERSION}"
NAME="${3}"
echo $NAME


set -euC

# python3 -m plb.algorithms.solve \
#     --algo ${2} \
#     --env_name ${NAME} \
#     --create_grid_mass

python3 -m plb.algorithms.solve \
    --algo ${2} \
    --env_name ${NAME} \
