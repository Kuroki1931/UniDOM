#!/bin/bash

ALGO='action'
ENV_NAME='Rope-v'

<<<<<<< HEAD
POSE_NUM=600
EACH_POSE_NUM=40
=======
POSE_NUM=200
EACH_POSE_NUM=10
>>>>>>> 2ce5a2702b3d80a5df45827d71c8f6510b37d48f
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