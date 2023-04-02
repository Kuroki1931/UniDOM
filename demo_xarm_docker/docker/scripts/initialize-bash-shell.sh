#!/bin/bash

################################################################################

# Link the default shell 'sh' to Bash.
alias sh='/bin/bash'

################################################################################

# Configure the terminal.

# Disable flow control. If enabled, inputting 'ctrl+s' locks the terminal until inputting 'ctrl+q'.
stty -ixon

################################################################################

# Configure 'umask' for giving read/write/execute permission to group members.
umask 0002

################################################################################

# Source the ROS environment.
echo "Sourcing the ROS environment from '/opt/ros/noetic/setup.bash'."
source /opt/ros/noetic/setup.bash

# Source the xarm catkin workspace
echo "Sourcing the xarm catkin workspace from '/root/xarm_catkin_ws/devel/setup.bash'."
source /root/xarm_catkin_ws/devel/setup.bash

# Source the Catkin workspace.
echo "Sourcing the Catkin workspace from '/root/xarm/catkin_ws/devel/setup.bash'."
source /root/xarm/catkin_ws/devel/setup.bash

################################################################################

# Add the Catkin workspace to the 'ROS_PACKAGE_PATH'.
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/xarm/catkin_ws/src/:/root/xarm_catkin_ws/src/
export ROS_WORKSPACE=/root/xarm/catkin_ws

################################################################################

# Set xarm/ROS network interface.

export ROS_IP=`hostname -I | cut -d' ' -f1`
echo "ROS_IP is set to '${ROS_IP}'."

export ROS_HOME=~/.ros

alias sim_mode='export ROS_MASTER_URI=http://localhost:11311; export PS1="\[[44;1;37m\]<local>\[[0m\]\w$ "'
alias xarm_mode='export ROS_MASTER_URI=http://${ROS_IP}:11311; export PS1="\[[41;1;37m\]<xarm>\[[0m\]\w$ "'

################################################################################

# Move to the working directory.
cd /root/xarm/
