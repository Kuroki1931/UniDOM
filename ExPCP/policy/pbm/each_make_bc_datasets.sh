#!/bin/bash

if [ $(( ${1} % 3)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(( ${1} % 3)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 3)) -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
fi

VERSION=$(( ${1}+${4}+269 ))
NAME="${3}${VERSION}"
echo $NAME


set -euC

python3 -m plb.algorithms.solve \
    --algo ${2} \
    --env_name ${NAME} \
    --seed ${VERSION}
