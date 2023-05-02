#!/bin/bash

ALGO='action'
ENV_NAME='Move-v'

<<<<<<< HEAD
POSE_NUM=32
EACH_POSE_NUM=8
=======
<<<<<<< HEAD
POSE_NUM=32
EACH_POSE_NUM=8
=======
POSE_NUM=78
EACH_POSE_NUM=6
>>>>>>> f4e96ed12613cad00e894537c4d2a9bf64d7f9ae
>>>>>>> 204ce3c11db89cf2d9cc059649b606e56f87e818
STEPS=$(( ${POSE_NUM} / ${EACH_POSE_NUM} ))

export PYTHONPATH=../../pbm
set -euC

                                                                                     
for i in `seq 0 1 $(($STEPS-1))`
do
    # random sampling
    BASE=$(( ${i}*${EACH_POSE_NUM} ))
    seq 0 1 $((${EACH_POSE_NUM}-1)) | xargs -P ${EACH_POSE_NUM} -I{} bash each_make_bc_datasets.sh {} \
                                                                                     ${ALGO} \
                                                                                     ${ENV_NAME} \
                                                                                     ${BASE}
done