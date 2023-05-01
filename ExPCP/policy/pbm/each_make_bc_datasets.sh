#!/bin/bash

if [ $(( ${1} % 2)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(( ${1} % 2)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

VERSION=$(( ${1}+${4}+23 ))
NAME="${3}${VERSION}"
echo $NAME


set -euC

python3 -m plb.algorithms.solve \
    --algo ${2} \
    --env_name ${NAME} \
    --seed ${VERSION}
