#!/bin/bash

if [ $(( ${1} % 3)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(( ${1} % 3)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 3)) -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
fi

<<<<<<< HEAD
VERSION=$(( ${1}+${4}+269 ))
=======
<<<<<<< HEAD
VERSION=$(( ${1}+${4}+269 ))
=======
VERSION=$(( ${1}+${4}+23 ))
>>>>>>> f4e96ed12613cad00e894537c4d2a9bf64d7f9ae
>>>>>>> 204ce3c11db89cf2d9cc059649b606e56f87e818
NAME="${3}${VERSION}"
echo $NAME


set -euC

python3 -m plb.algorithms.solve \
    --algo ${2} \
    --env_name ${NAME} \
    --seed ${VERSION}
