#!/bin/bash

IMAGE_NAME=dmlc
CONTAINER_NAME="dmlc"
echo "$0: IMAGE=${IMAGE_NAME}"
echo "$0: CONTAINER=${CONTAINER_NAME}"

if [ ! -z $1 ]; then
    ROOMBA_IP=$1
else
    ROOMBA_IP=`hostname -I | cut -d' ' -f1`
fi

EXISTING_CONTAINER_ID=`docker ps -aq -f name=${CONTAINER_NAME}`
if [ ! -z "${EXISTING_CONTAINER_ID}" ]; then
    docker exec -it ${CONTAINER_NAME} bash
else
    xhost +
    docker run -it --rm \
        --privileged \
        --gpus all \
        --env DISPLAY=${DISPLAY} \
        --net host \
        --volume ${PWD}/catkin_ws/:/root/dmlc/catkin_ws/ \
        --volume /dev/:/dev/ \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume ${PWD}/policy/:/root/dmlc/policy/ \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} \
        bash -c "sed -i 's/TMP_IP/${ROOMBA_IP}/' ~/scripts/initialize-bash-shell.sh;
                    bash"
fi

# /home/robot_dev/kuroki/DMLC/catkin_ws/src/roomba_control/scripts
# source /opt/ros/noetic/setup.bash 

# docker exec -it okubo_pull_request_xarm-dual_1 bash    
# roscore

