#!/bin/bash

IP=`hostname -I | cut -d' ' -f1`

echo $IP
ssh -t -t txcolab "sleep 1s; \
    docker container stop realsense_ros_1; \
    docker rm realsense_ros_1; \
    NUMBER=1 ~/realsense_docker/launch_realsense.sh ${IP} rs_camera.launch serial_no:=830112070756 camera:=eye_to_hand_camera align_depth:=true"
    # NUMBER=1 ~/realsense_docker/launch_realsense.sh ${IP} rs_camera.launch serial_no:=830112070756 camera:=eye_to_hand_camera align_depth:=true filters:=pointcloud"

