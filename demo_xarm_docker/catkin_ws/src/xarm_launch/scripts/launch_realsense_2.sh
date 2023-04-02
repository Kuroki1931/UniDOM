#!/bin/sh

IP=`hostname -I | cut -d' ' -f1`

echo $IP
ssh -t -t txcolab "sleep 1s; \
    docker container stop realsense_ros_2; \
    docker rm realsense_ros_2; \
    NUMBER=2 ~/realsense_docker/launch_realsense.sh ${IP} rs_camera.launch serial_no:=033422070007 camera:=eye_on_hand_camera align_depth:=true filters:=pointcloud"
