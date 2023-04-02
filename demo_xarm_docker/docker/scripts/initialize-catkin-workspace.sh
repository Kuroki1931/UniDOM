#!/bin/bash

################################################################################

# Download package lists from Ubuntu repositories.
apt-get update

# Install system dependencies required by specific ROS packages.
# http://wiki.ros.org/rosdep
rosdep update

# Source the updated ROS environment.
source /opt/ros/melodic/setup.bash
source /root/xarm_catkin_ws/devel/setup.bash

################################################################################

# Initialize and build the Catkin workspace.
cd /root/xarm/catkin_ws/ && catkin_make -DCMAKE_BUILD_TYPE=Release

# Source the Catkin workspace.
source /root/xarm/catkin_ws/devel/setup.bash
