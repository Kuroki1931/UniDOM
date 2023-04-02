#!/bin/bash

################################################################################


PROMPT_START=$'\e[4m\e[1m'
PROMPT_END=$'\e[0m'
KEY_START=$'\e[7m'
KEY_END=$'\e[0m'${PROMPT_START}
IP=""
PROMPT="${PROMPT_START}Run '$1'? Press:"$'\n'"'${KEY_START}r${KEY_END}' to run with the robot(192.168.10.211),"$'\n'"'${KEY_START}l${KEY_END}' to run with the 2nd robot(192.168.10.210),"$'\n'"'${KEY_START}s${KEY_END}' to run in the simulator,"$'\n'"'${KEY_START}c${KEY_END}' to enter a child shell,"$'\n'"'${KEY_START}q${KEY_END}' to quit.${PROMPT_END}"$'\n'
while true; do
  read -n 1 -s -p "${PROMPT}" input;
  if [ "${input}" = "r" ]; then
    xarm_mode;
    LAUNCH=weblab_real_default.launch
    IP=robot_ip:=192.168.10.211
  elif [ "${input}" = "l" ]; then
    xarm_mode;
    LAUNCH=weblab_real_default.launch 
    IP=robot_ip:=192.168.10.210
  elif [ "${input}" = "s" ]; then
    sim_mode;
    LAUNCH=weblab_gazebo_default.launch 
  elif [ "${input}" = "q" ]; then
    break;
  elif [ "${input}" = "c" ]; then
    cat <<EOF

Starting a new shell process.
You will return to the above prompt when you exit from this shell.
Note: The new process does not inherit the mode ('xarm_mode' or 'sim_mode') from the previously executed 'roslaunch' process.

EOF
    bash -i
    continue;
  else
    continue;
  fi;
  echo "ROS_MASTER_URI: ${ROS_MASTER_URI}";
  roslaunch xarm_launch ${LAUNCH} ${IP};
  echo "" # Display an empty line.
done

cat <<EOF

Starting a new shell process.
Note: The new process does not inherit the mode ('xarm_mode' or 'sim_mode') from the previously executed 'roslaunch' process.

EOF

exec bash -i
